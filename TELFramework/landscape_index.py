import os
import torch
import tqdm
import numpy as np
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def _as_float32_numpy(x):
    import numpy as np
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    return x


def _load_cluster_assigner(outputs_dir, prefer_faiss=True, use_gpu=True):
    """Load a cluster assignment function from outputs.

    Priority:
    1) FAISS centroids index (faiss_centroids.index)
    2) legacy sklearn model (kmeans_model.pkl)

    Returns:
        assign_fn: callable(features)->labels (np.int64)
        k: int number of clusters
        backend: str
    """
    import json

    faiss_index_path = os.path.join(outputs_dir, 'faiss_centroids.index')
    faiss_meta_path = os.path.join(outputs_dir, 'faiss_kmeans_meta.json')
    sklearn_path = os.path.join(outputs_dir, 'kmeans_model.pkl')

    if prefer_faiss and os.path.exists(faiss_index_path):
        import faiss  # type: ignore

        index = faiss.read_index(faiss_index_path)
        k = int(getattr(index, 'ntotal', 0))
        if os.path.exists(faiss_meta_path):
            try:
                with open(faiss_meta_path, 'r') as f:
                    meta = json.load(f)
                k = int(meta.get('k', k))
            except Exception:
                pass

        # Optional GPU move (faiss-gpu)
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print(f'Using GPU to Kmeans')
            except Exception:
                # 如果 GPU 资源不可用，退回 CPU index
                pass

        def assign_fn(feats):
            x = _as_float32_numpy(feats)
            _, labels = index.search(x, 1)
            return labels.reshape(-1).astype(np.int64, copy=False)

        return assign_fn, k, 'faiss'

    # fallback: sklearn
    import joblib

    if not os.path.exists(sklearn_path):
        raise FileNotFoundError(
            f"No clustering model found in {outputs_dir}. Expected {faiss_index_path} or {sklearn_path}."
        )

    kmeans = joblib.load(sklearn_path)
    k = int(getattr(kmeans, 'n_clusters', 0))

    def assign_fn(feats):
        x = _as_float32_numpy(feats)
        return kmeans.predict(x).astype(np.int64, copy=False)

    return assign_fn, k, 'sklearn'


def _to_numpy(x):
    import numpy as np
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _infer_patch_size(coords_np):
    """Infer patch stride from (x, y) coords."""
    import numpy as np
    xs = np.unique(coords_np[:, 0])
    ys = np.unique(coords_np[:, 1])
    dx = np.diff(np.sort(xs))
    dy = np.diff(np.sort(ys))
    dx = dx[dx > 0]
    dy = dy[dy > 0]
    candidates = []
    if dx.size:
        candidates.append(float(dx.min()))
    if dy.size:
        candidates.append(float(dy.min()))
    if not candidates:
        return 1.0
    return float(min(candidates))

def _build_sparse_nodes(clusters, coords):
    """Rasterize coords to an integer grid and build a sparse node map.

    Returns:
        nodes: dict[(row, col)] -> label_index (0..K-1)
        unique_labels: original sorted unique cluster ids
        label_index: (N,) label index for each input point
        patch_size: inferred patch stride
        rx, ry: (N,) integer grid indices for each input point
    """
    import numpy as np

    clusters_np = _to_numpy(clusters).reshape(-1)
    coords_np = _to_numpy(coords)
    if coords_np.ndim != 2 or coords_np.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords_np.shape}")
    if clusters_np.shape[0] != coords_np.shape[0]:
        raise ValueError(
            f"clusters and coords length mismatch: {clusters_np.shape[0]} vs {coords_np.shape[0]}"
        )
    if clusters_np.shape[0] == 0:
        return {}, np.asarray([]), np.asarray([], dtype=np.int32), 1.0, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    unique_labels = np.unique(clusters_np)
    label_index = np.searchsorted(unique_labels, clusters_np).astype(np.int32)

    coords_f = coords_np.astype(np.float64, copy=False)
    patch_size = _infer_patch_size(coords_f)
    if patch_size <= 0:
        raise ValueError(f"Invalid inferred patch_size={patch_size}")

    x0 = float(coords_f[:, 0].min())
    y0 = float(coords_f[:, 1].min())
    gx = (coords_f[:, 0] - x0) / patch_size
    gy = (coords_f[:, 1] - y0) / patch_size
    rx = np.rint(gx).astype(np.int64)
    ry = np.rint(gy).astype(np.int64)

    nodes = {}
    for r, c, lab in zip(ry.tolist(), rx.tolist(), label_index.tolist()):
        nodes[(int(r), int(c))] = int(lab)  # 若坐标重复，后者覆盖前者

    return nodes, unique_labels, label_index, float(patch_size), rx, ry

def _filter_small_patches(nodes, min_patch_cells=4):
    """Remove connected components (4-neighborhood) smaller than min_patch_cells.

    Args:
        nodes: dict[(row, col)] -> label_index
        min_patch_cells: int. Components with size < min_patch_cells are removed.

    Returns:
        filtered_nodes: dict[(row, col)] -> label_index
    """
    try:
        min_patch_cells = 1 if min_patch_cells is None else int(min_patch_cells)
    except Exception:
        min_patch_cells = 1
    if min_patch_cells <= 1 or not nodes:
        return nodes

    visited = set()
    to_drop = set()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for start_rc, start_lab in nodes.items():
        if start_rc in visited:
            continue
        visited.add(start_rc)
        stack = [start_rc]
        comp = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                nb_lab = nodes.get(nb, None)
                if nb_lab is None or nb_lab != start_lab:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        if len(comp) < min_patch_cells:
            to_drop.update(comp)

    if not to_drop:
        return nodes
    return {rc: lab for rc, lab in nodes.items() if rc not in to_drop}

def _filter_points_by_min_patch_cells(clusters, coords, min_patch_cells=1):
    """Filter out points belonging to tiny connected components.

    Connectivity rule:
    - rasterize coords to integer grid with inferred patch stride
    - 4-neighborhood adjacency
    - a "patch" is a connected component with the same cluster label

    Returns:
        clusters_f: (M,) numpy array
        coords_f: (M, 2) numpy array
    """
    import numpy as np

    try:
        min_patch_cells = 1 if min_patch_cells is None else int(min_patch_cells)
    except Exception:
        min_patch_cells = 1

    clusters_np = _to_numpy(clusters).reshape(-1)
    coords_np = _to_numpy(coords)
    if min_patch_cells <= 1 or clusters_np.size == 0:
        return clusters_np, coords_np

    nodes, unique_labels, label_index, _, rx, ry = _build_sparse_nodes(clusters_np, coords_np)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return np.asarray([], dtype=clusters_np.dtype), np.asarray([], dtype=coords_np.dtype).reshape(0, 2)

    keep_mask = np.zeros((clusters_np.shape[0],), dtype=bool)
    for idx, (r, c, lab) in enumerate(zip(ry.tolist(), rx.tolist(), label_index.tolist())):
        v = nodes.get((int(r), int(c)), None)
        if v is not None and int(v) == int(lab):
            keep_mask[idx] = True

    return clusters_np[keep_mask], coords_np[keep_mask]

def cont_index(clusters, coords, min_patch_cells=1):
    """计算聚集度指数（Contagion Index）。
    这里“斑块”定义为：空间上相邻（4 邻域：上下左右）的、且 cluster 相同的一组 patch。
    计算方法参考景观生态学中的 Contagion Index 定义，适当调整以适配 patch 集合：

        $CONT = 1 + \sum_{i=1}^K \sum_{j=1}^K (p_{ij} \log p_{ij}) / (2 \log K)$
    """
    import numpy as np

    nodes, unique_labels, label_index, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    k = int(unique_labels.shape[0])
    if k <= 1:
        # 只有一个类别时，聚集度理论上达到最大；同时公式分母 2*log(1)=0。
        return 1.0

    if len(nodes) <= 1:
        return 0.0

    # 统计 4 邻域的“有向”邻接频次 g_ij（遍历每个 cell 的上下左右邻居，边会自然计数两次）。
    g = np.zeros((k, k), dtype=np.int64)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for (r, c), lab_i in nodes.items():
        for dr, dc in dirs:
            nb = (r + dr, c + dc)
            lab_j = nodes.get(nb, None)
            if lab_j is None:
                continue
            g[int(lab_i), int(lab_j)] += 1

    total_adj = int(g.sum())
    if total_adj <= 0:
        return 0.0

    p = g.astype(np.float64) / float(total_adj)
    mask = p > 0
    s = float(np.sum(p[mask] * np.log(p[mask])))
    cont = 1.0 + s / (2.0 * float(np.log(k)))
    return float(cont)

def largest_patch_index(clusters, coords, min_patch_cells=1):
    """计算最大斑块指数（Largest Patch Index, LPI）。

    这里“斑块”定义为：空间上相邻（4 邻域：上下左右）的、且 cluster 相同的一组 patch。
    对每个 cluster：

        $LPI = A_{max} / A_{total}$

    其中 $A_{max}$ 是该 cluster 内最大斑块的面积，$A_{total}$ 是整张 slide（输入 patch 集合）的总面积。
    由于所有 patch 面积相同，等价于：最大斑块 patch 数 / 总 patch 数。

    Args:
        clusters: (N,) 每个 patch 的 cluster id（numpy/torch/list 均可）。
        coords: (N, 2) 每个 patch 的左上角坐标 (x, y)（numpy/torch 均可）。

    Returns:
        dict: {cluster_id: lpi}。
    """
    import numpy as np

    nodes, unique_labels, _, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    k = int(unique_labels.shape[0])

    total_cells = int(len(nodes))
    if total_cells <= 0:
        return 0.0

    visited = set()
    max_cells_by_lab = np.zeros((k,), dtype=np.int64)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for start_rc, start_lab in nodes.items():
        if start_rc in visited:
            continue
        visited.add(start_rc)

        # DFS/BFS 找同 label 的连通分量
        stack = [start_rc]
        cell_count = 0
        while stack:
            r, c = stack.pop()
            cell_count += 1
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                nb_lab = nodes.get(nb, None)
                if nb_lab is None or nb_lab != start_lab:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        if cell_count > max_cells_by_lab[int(start_lab)]:
            max_cells_by_lab[int(start_lab)] = int(cell_count)

    # 由于所有 patch 面积一致，面积比值等价于 cell 数比值。
    max_lpi = float(max_cells_by_lab.max()) / float(total_cells) if total_cells > 0 else 0.0
    return float(max_lpi)


def largest_patch_index_by_class(clusters, coords, min_patch_cells=1):
    """按 cluster 分别计算 LPI（最大斑块面积 / 景观总面积）。"""
    import numpy as np

    nodes, unique_labels, _, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    k = int(unique_labels.shape[0])
    if not nodes or k <= 0:
        return {}

    total_cells = int(len(nodes))
    visited = set()
    max_cells_by_lab = np.zeros((k,), dtype=np.int64)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for start_rc, start_lab in nodes.items():
        if start_rc in visited:
            continue
        visited.add(start_rc)
        stack = [start_rc]
        cell_count = 0
        while stack:
            r, c = stack.pop()
            cell_count += 1
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                nb_lab = nodes.get(nb, None)
                if nb_lab is None or nb_lab != start_lab:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        if cell_count > max_cells_by_lab[int(start_lab)]:
            max_cells_by_lab[int(start_lab)] = int(cell_count)

    out = {}
    for lab_i in range(k):
        out[unique_labels[lab_i].item()] = float(max_cells_by_lab[lab_i]) / float(total_cells)
    return out

def patch_shape_index(clusters, coords, max_clusters=5, min_patch_cells=1):
    """计算“斑块形状指数”(Shape Index) 的 cluster 级均值（面积加权）。

    这里“斑块”定义为：空间上相邻（4 邻域：上下左右）的、且 cluster 相同的一组 patch。
    对每个斑块，按栅格景观常用定义计算形状指数：

        $SI = 0.25 * P / sqrt(A)$

    其中 P 为斑块周长，A 为斑块面积。若 patch 为正方形且边长为 s，则：
    - A = n_cells * s^2
    - P = n_boundary_edges * s

    该定义保证任意正方形斑块 SI=1。

    Args:
        clusters: (N,) 每个 patch 的 cluster id（numpy/torch/list 均可）。
        coords: (N, 2) 每个 patch 的左上角坐标 (x, y)（numpy/torch 均可）。
        max_clusters: 若 >0，则只返回样本数最多的前 max_clusters 个 cluster 的均值；
            若 <=0 或 None，则返回全部 cluster。

    Returns:
        dict: {cluster_id: area_weighted_mean_shape_index}，对同一 cluster 的多个斑块按面积加权取均值。
    """
    import numpy as np
    from collections import defaultdict

    nodes, unique_labels, label_index, patch_size, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    k = int(unique_labels.shape[0])

    if not nodes:
        return {}

    visited = set()
    # lab_index -> [(shape_index, area), ...]
    patch_shapes_by_lab = defaultdict(list)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for start_rc, start_lab in nodes.items():
        if start_rc in visited:
            continue
        visited.add(start_rc)

        # DFS/BFS 找同 label 的连通分量
        stack = [start_rc]
        cell_count = 0
        boundary_edges = 0

        while stack:
            r, c = stack.pop()
            cell_count += 1
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                nb_lab = nodes.get(nb, None)
                if nb_lab is None:
                    boundary_edges += 1
                    continue
                if nb_lab != start_lab:
                    boundary_edges += 1
                    continue
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        area = float(cell_count) * (patch_size ** 2)
        perimeter = float(boundary_edges) * patch_size
        shape_index = float('nan')
        if area > 0:
            shape_index = 0.25 * perimeter / float(np.sqrt(area))
        patch_shapes_by_lab[int(start_lab)].append((shape_index, area))

    # 计算每个 cluster 的“斑块形状指数面积加权均值”。
    means_all = {}
    for lab_i in range(k):
        pairs = patch_shapes_by_lab.get(lab_i, [])
        if not pairs:
            continue
        si = np.asarray([p[0] for p in pairs], dtype=np.float64)
        ar = np.asarray([p[1] for p in pairs], dtype=np.float64)
        mask = np.isfinite(si) & np.isfinite(ar) & (ar > 0)
        if not np.any(mask):
            continue
        aw_mean = float(np.sum(si[mask] * ar[mask]) / float(np.sum(ar[mask])))
        means_all[unique_labels[lab_i].item()] = aw_mean

    # 如需，仅返回出现频次最高的前 max_clusters 个 cluster。
    if max_clusters is None or int(max_clusters) <= 0:
        return means_all
    max_clusters = int(max_clusters)
    counts = np.bincount(label_index.astype(np.int64), minlength=k)
    order = np.argsort(-counts)
    keep = set(unique_labels[order[:max_clusters]].tolist())
    return {cid: v for cid, v in means_all.items() if cid in keep}

def splitting_index(clusters, coords, min_patch_cells=1):
    """计算 Splitting Index（SPLIT，分割指数）。

    经典（FRAGSTATS）定义的景观尺度 SPLIT：

        $SPLIT = A^2 / \sum_{p=1}^n a_p^2$

    其中：
    - $A$ 为整个景观面积（这里等价于所有保留 patch 的总面积）
    - $a_p$ 为第 p 个斑块（同类连通分量）的面积
    - $n$ 为斑块数

    直觉：
    - 若只有 1 个斑块，则 $SPLIT=1$（最不破碎）
    - 在总面积固定时，斑块越碎、越多，$\sum a_p^2$ 越小，SPLIT 越大

    注意：这里“斑块”仍按 4 邻域、且 cluster 相同的连通分量定义。

    Returns:
        float: SPLIT（>=1），若无有效 patch 返回 0.0。
    """
    import numpy as np

    nodes, unique_labels, _, patch_size, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return 0.0

    visited = set()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    patch_areas = []

    for start_rc, start_lab in nodes.items():
        if start_rc in visited:
            continue
        visited.add(start_rc)
        stack = [start_rc]
        cell_count = 0
        while stack:
            r, c = stack.pop()
            cell_count += 1
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                nb_lab = nodes.get(nb, None)
                if nb_lab is None or nb_lab != start_lab:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        patch_areas.append(float(cell_count) * (patch_size ** 2))

    if not patch_areas:
        return 0.0

    areas = np.asarray(patch_areas, dtype=np.float64)
    A = float(np.sum(areas))
    denom = float(np.sum(areas ** 2))
    if A <= 0 or denom <= 0:
        return 0.0
    return float((A ** 2) / denom)


def patch_density(clusters, coords, min_patch_cells=1):
    """计算 Patch Density（PD，斑块密度）。

    景观尺度常用定义：

        $PD = N / A$

    - $N$：斑块数（这里斑块=同类、4 邻域连通分量）
    - $A$：景观总面积（这里等价于保留 patch 的总面积）

    直觉：在总面积固定时，斑块越多越碎，PD 越大。

    Returns:
        float: PD（单位为 1/area；若无有效 patch 返回 0.0）。
    """
    nodes, _, _, patch_size, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return 0.0

    visited = set()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    n_patches = 0

    for start_rc, start_lab in nodes.items():
        if start_rc in visited:
            continue
        n_patches += 1
        visited.add(start_rc)
        stack = [start_rc]
        while stack:
            r, c = stack.pop()
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                nb_lab = nodes.get(nb, None)
                if nb_lab is None or nb_lab != start_lab:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

    total_area = float(len(nodes)) * float(patch_size ** 2)
    if total_area <= 0:
        return 0.0
    return float(n_patches) / total_area


def patch_density_by_class(clusters, coords, min_patch_cells=1):
    """按 cluster 分别计算 PD（该类斑块数 / 景观总面积）。"""
    nodes, unique_labels, _, patch_size, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return {}

    visited = set()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    n_patches_by_lab = {int(i): 0 for i in range(int(unique_labels.shape[0]))}

    for start_rc, start_lab in nodes.items():
        if start_rc in visited:
            continue
        n_patches_by_lab[int(start_lab)] = n_patches_by_lab.get(int(start_lab), 0) + 1
        visited.add(start_rc)
        stack = [start_rc]
        while stack:
            r, c = stack.pop()
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                nb_lab = nodes.get(nb, None)
                if nb_lab is None or nb_lab != start_lab:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

    total_area = float(len(nodes)) * float(patch_size ** 2)
    if total_area <= 0:
        return {}

    out = {}
    for lab_i in range(int(unique_labels.shape[0])):
        out[unique_labels[lab_i].item()] = float(n_patches_by_lab.get(lab_i, 0)) / total_area
    return out

def iji_index(clusters, coords, min_patch_cells=1):
    """计算 Interspersion & Juxtaposition Index（IJI，交错并置指数）。

    IJI 衡量“不同类别之间的接触边界”是否均匀分布：
    - 若异类边界主要集中在少数类别对之间，IJI 低
    - 若异类边界在多种类别对之间更均匀出现，IJI 高

    常用定义（景观尺度，归一化到 0-100）：

        $IJI = \frac{-\sum_{i<j} p_{ij}\ln p_{ij}}{\ln(M)} \times 100$

    其中：
    - $p_{ij}$ 为类别 i 与 j 的异类相邻边界在所有异类边界中的比例
    - $M$ 为“可能的类别对数量”。若景观中实际存在 $m$ 个类别，则 $M=m(m-1)/2$

    实现细节：
    - 采用 4 邻域
    - 只统计异类相邻（i!=j）
    - 为避免重复计数，只检查每个 cell 的“右邻”和“下邻”

    Returns:
        float: IJI（0-100），若无异类边界或不足以定义则返回 0.0。
    """
    import numpy as np

    nodes, unique_labels, _, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    k = int(unique_labels.shape[0])
    if k <= 1 or not nodes:
        return 0.0

    # upper-triangular undirected adjacency counts for i<j
    e = np.zeros((k, k), dtype=np.int64)
    dirs = ((0, 1), (1, 0))  # right, down
    for (r, c), lab_i in nodes.items():
        li = int(lab_i)
        for dr, dc in dirs:
            nb = (r + dr, c + dc)
            lab_j = nodes.get(nb, None)
            if lab_j is None:
                continue
            lj = int(lab_j)
            if li == lj:
                continue
            a, b = (li, lj) if li < lj else (lj, li)
            e[a, b] += 1

    vals = e[np.triu_indices(k, k=1)]
    total = float(np.sum(vals))
    if total <= 0:
        return 0.0

    # Number of classes actually present after filtering.
    present = np.zeros((k,), dtype=np.int64)
    for lab in nodes.values():
        present[int(lab)] += 1
    m_classes = int(np.sum(present > 0))
    m_pairs_possible = int(m_classes * (m_classes - 1) // 2)

    # Special-case: only 2 classes -> only 1 possible pair -> normalization denom log(1)=0.
    # By convention, if there exists any heterotypic adjacency, IJI is maximal (100).
    if m_pairs_possible <= 1:
        return 100.0 if total > 0 else 0.0

    nz = vals[vals > 0]
    p = nz.astype(np.float64) / total
    H = -float(np.sum(p * np.log(p)))
    return float((H / float(np.log(m_pairs_possible))) * 100.0)


def pladj_index(clusters, coords, min_patch_cells=1):
    """计算 PLADJ（Percentage of Like Adjacencies，同类相邻比例）。

    定义：在所有“相邻且都存在的边”（4 邻域）中，同类相邻边所占比例：

        $PLADJ = \frac{g_{like}}{g_{total}}\times 100$

    其中：
    - $g_{total}$ 统计所有相邻对（只看右邻和下邻，避免重复计数）
    - $g_{like}$ 统计其中标签相同的相邻对

    直觉：
    - 同类更聚集、边界更少时，PLADJ 趋近 100
    - 镶嵌更破碎、异类交错更多时，PLADJ 降低

    Returns:
        float: 0-100，若无相邻对返回 0.0。
    """
    nodes, _, _, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return 0.0

    like = 0
    total = 0
    dirs = ((0, 1), (1, 0))  # right, down
    for (r, c), lab_i in nodes.items():
        li = int(lab_i)
        for dr, dc in dirs:
            nb = (r + dr, c + dc)
            lab_j = nodes.get(nb, None)
            if lab_j is None:
                continue
            total += 1
            if li == int(lab_j):
                like += 1
    if total <= 0:
        return 0.0
    return float(like) / float(total) * 100.0


def pladj_index_by_class(clusters, coords, min_patch_cells=1):
    """按 cluster 分别计算 PLADJ（该类同类相邻边占比，0-100）。"""
    nodes, unique_labels, _, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return {}

    like_by_lab = {int(i): 0 for i in range(int(unique_labels.shape[0]))}
    total_by_lab = {int(i): 0 for i in range(int(unique_labels.shape[0]))}

    dirs = ((0, 1), (1, 0))  # right, down
    for (r, c), lab_i in nodes.items():
        li = int(lab_i)
        for dr, dc in dirs:
            nb = (r + dr, c + dc)
            lab_j = nodes.get(nb, None)
            if lab_j is None:
                continue
            lj = int(lab_j)
            if li == lj:
                like_by_lab[li] = like_by_lab.get(li, 0) + 1
                total_by_lab[li] = total_by_lab.get(li, 0) + 1
            else:
                total_by_lab[li] = total_by_lab.get(li, 0) + 1
                total_by_lab[lj] = total_by_lab.get(lj, 0) + 1

    out = {}
    for lab_i in range(int(unique_labels.shape[0])):
        den = float(total_by_lab.get(lab_i, 0))
        out[unique_labels[lab_i].item()] = float(like_by_lab.get(lab_i, 0)) / den * 100.0 if den > 0 else 0.0
    return out


def _extract_patches_for_connectivity(clusters, coords, min_patch_cells=1):
    """提取斑块（连通分量）并返回其质心与面积（用于连通性/隔离度指标）。

    Returns:
        patch_labels: (P,) 原始 cluster id（不是 0..K-1 索引）
        patch_centroids: (P,2) 质心坐标 (x,y)，与输入 coords 同单位
        patch_areas: (P,) 斑块面积（与 coords 单位一致的面积单位）
        patch_size: float 推断的 patch 边长
    """
    import numpy as np

    nodes, unique_labels, _, patch_size, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64).reshape(0, 2),
            np.asarray([], dtype=np.float64),
            float(patch_size),
        )

    coords_np = _to_numpy(coords).astype(np.float64, copy=False)
    x0 = float(coords_np[:, 0].min())
    y0 = float(coords_np[:, 1].min())

    visited = set()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    patch_labels = []
    patch_centroids = []
    patch_areas = []

    for start_rc, start_lab in nodes.items():
        if start_rc in visited:
            continue
        visited.add(start_rc)
        stack = [start_rc]
        rs = []
        cs = []
        cell_count = 0
        while stack:
            r, c = stack.pop()
            rs.append(int(r))
            cs.append(int(c))
            cell_count += 1
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                nb_lab = nodes.get(nb, None)
                if nb_lab is None or nb_lab != start_lab:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        # Convert grid indices back to coordinate space. Use cell centers.
        r_mean = float(np.mean(rs))
        c_mean = float(np.mean(cs))
        cx = x0 + (c_mean + 0.5) * float(patch_size)
        cy = y0 + (r_mean + 0.5) * float(patch_size)
        area = float(cell_count) * float(patch_size ** 2)

        patch_labels.append(unique_labels[int(start_lab)].item())
        patch_centroids.append((cx, cy))
        patch_areas.append(area)

    return (
        np.asarray(patch_labels),
        np.asarray(patch_centroids, dtype=np.float64),
        np.asarray(patch_areas, dtype=np.float64),
        float(patch_size),
    )


def enn_mean(clusters, coords, min_patch_cells=1):
    """ENN（Euclidean Nearest-Neighbor Distance）的景观尺度均值。

    计算过程：
    1) 先按“同类 + 4 邻域”提取每个斑块（连通分量），得到每个斑块的质心
    2) 对每个斑块，计算其到“同一类别的其他斑块”的最近质心距离
    3) 对所有可计算的斑块取平均

    Returns:
        float: 平均最近邻距离（与 coords 同长度单位）；若不足以计算返回 0.0。
    """
    import numpy as np

    labs, centroids, _, _ = _extract_patches_for_connectivity(clusters, coords, min_patch_cells=min_patch_cells)
    if centroids.shape[0] < 2:
        return 0.0

    dists_all = []
    for lab in np.unique(labs):
        idx = np.where(labs == lab)[0]
        if idx.size < 2:
            continue
        pts = centroids[idx]
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            d, _ = tree.query(pts, k=2)
            d_nn = d[:, 1]
        except Exception:
            # Fallback O(n^2)
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            dist += np.eye(dist.shape[0]) * 1e18
            d_nn = dist.min(axis=1)
        dists_all.append(d_nn)

    if not dists_all:
        return 0.0
    return float(np.mean(np.concatenate(dists_all)))


def enn_mean_by_class(clusters, coords, min_patch_cells=1):
    """按 cluster 分别计算 ENN 均值。"""
    import numpy as np

    labs, centroids, _, _ = _extract_patches_for_connectivity(clusters, coords, min_patch_cells=min_patch_cells)
    if centroids.shape[0] < 2:
        return {}

    out = {}
    for lab in np.unique(labs):
        idx = np.where(labs == lab)[0]
        if idx.size < 2:
            out[lab.item() if hasattr(lab, 'item') else lab] = 0.0
            continue
        pts = centroids[idx]
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            d, _ = tree.query(pts, k=2)
            d_nn = d[:, 1]
        except Exception:
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            dist += np.eye(dist.shape[0]) * 1e18
            d_nn = dist.min(axis=1)
        out[lab.item() if hasattr(lab, 'item') else lab] = float(np.mean(d_nn)) if d_nn.size else 0.0
    return out


def prox_mean(clusters, coords, search_radius=None, min_patch_cells=1):
    """PROX（Proximity Index）的景观尺度均值。

    常见思想：对每个斑块 i，在搜索半径 R 内累加同类斑块 j 的面积并按距离衰减：

        $PROX_i = \sum_{j\neq i,\ d_{ij}\le R} \frac{a_j}{d_{ij}^2}$

    这里用质心距离 $d_{ij}$ 近似，最后对所有斑块取平均：$PROX = mean_i(PROX_i)$。

    Args:
        search_radius: 搜索半径 R（与 coords 同单位）。None 时默认取 10*patch_size。

    Returns:
        float: PROX 均值；若无有效斑块返回 0.0。
    """
    import numpy as np

    labs, centroids, areas, patch_size = _extract_patches_for_connectivity(
        clusters, coords, min_patch_cells=min_patch_cells
    )
    P = centroids.shape[0]
    if P == 0:
        return 0.0

    R = float(10.0 * patch_size if search_radius is None else search_radius)
    if R <= 0:
        return 0.0

    prox_vals = np.zeros((P,), dtype=np.float64)
    for lab in np.unique(labs):
        idx = np.where(labs == lab)[0]
        if idx.size < 2:
            continue
        pts = centroids[idx]
        ars = areas[idx]
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            neighbors = tree.query_ball_point(pts, r=R)
            for ii, nb in enumerate(neighbors):
                s = 0.0
                for jj in nb:
                    if jj == ii:
                        continue
                    d = float(np.linalg.norm(pts[ii] - pts[jj]))
                    if d <= 0:
                        continue
                    s += float(ars[jj]) / (d ** 2)
                prox_vals[idx[ii]] = s
        except Exception:
            # Fallback O(n^2)
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            for ii in range(dist.shape[0]):
                mask = (dist[ii] > 0) & (dist[ii] <= R)
                if not np.any(mask):
                    continue
                prox_vals[idx[ii]] = float(np.sum(ars[mask] / (dist[ii, mask] ** 2)))

    return float(np.mean(prox_vals))


def prox_mean_by_class(clusters, coords, search_radius=None, min_patch_cells=1):
    """按 cluster 分别计算 PROX 均值。"""
    import numpy as np

    labs, centroids, areas, patch_size = _extract_patches_for_connectivity(
        clusters, coords, min_patch_cells=min_patch_cells
    )
    P = centroids.shape[0]
    if P == 0:
        return {}

    R = float(10.0 * patch_size if search_radius is None else search_radius)
    if R <= 0:
        return {}

    out = {}
    for lab in np.unique(labs):
        idx = np.where(labs == lab)[0]
        key = lab.item() if hasattr(lab, 'item') else lab
        if idx.size < 2:
            out[key] = 0.0
            continue
        pts = centroids[idx]
        ars = areas[idx]
        vals = np.zeros((idx.size,), dtype=np.float64)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            neighbors = tree.query_ball_point(pts, r=R)
            for ii, nb in enumerate(neighbors):
                s = 0.0
                for jj in nb:
                    if jj == ii:
                        continue
                    d = float(np.linalg.norm(pts[ii] - pts[jj]))
                    if d <= 0:
                        continue
                    s += float(ars[jj]) / (d ** 2)
                vals[ii] = s
        except Exception:
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            for ii in range(dist.shape[0]):
                mask = (dist[ii] > 0) & (dist[ii] <= R)
                if np.any(mask):
                    vals[ii] = float(np.sum(ars[mask] / (dist[ii, mask] ** 2)))
        out[key] = float(np.mean(vals)) if vals.size else 0.0
    return out


def connectance_index(clusters, coords, threshold=None, min_patch_cells=1):
    """CONNECT（Connectance Index，连通度指数）。

    计算过程（景观尺度）：
    1) 提取每个斑块质心
    2) 对同一类别内的所有斑块对 (i,j)，若质心距离 $d_{ij} \le T$ 则认为“连通”
    3) 计算连通对数占所有可能同类对数的比例：

        $CONNECT = \frac{\#connected\ pairs}{\#possible\ pairs}\times 100$

    Args:
        threshold: 距离阈值 T（与 coords 同单位）。None 时默认取 10*patch_size。

    Returns:
        float: 0-100；若无可比较对返回 0.0。
    """
    import numpy as np

    labs, centroids, _, patch_size = _extract_patches_for_connectivity(
        clusters, coords, min_patch_cells=min_patch_cells
    )
    if centroids.shape[0] < 2:
        return 0.0

    T = float(10.0 * patch_size if threshold is None else threshold)
    if T <= 0:
        return 0.0

    connected = 0
    possible = 0
    for lab in np.unique(labs):
        idx = np.where(labs == lab)[0]
        n = int(idx.size)
        if n < 2:
            continue
        possible += n * (n - 1) // 2
        pts = centroids[idx]
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            pairs = tree.query_pairs(r=T)
            connected += len(pairs)
        except Exception:
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            tri = np.triu(dist, k=1)
            connected += int(np.sum((tri > 0) & (tri <= T)))

    if possible <= 0:
        return 0.0
    return float(connected) / float(possible) * 100.0


def connectance_index_by_class(clusters, coords, threshold=None, min_patch_cells=1):
    """按 cluster 分别计算 CONNECT（0-100）。"""
    import numpy as np

    labs, centroids, _, patch_size = _extract_patches_for_connectivity(
        clusters, coords, min_patch_cells=min_patch_cells
    )
    if centroids.shape[0] < 2:
        return {}

    T = float(10.0 * patch_size if threshold is None else threshold)
    if T <= 0:
        return {}

    out = {}
    for lab in np.unique(labs):
        idx = np.where(labs == lab)[0]
        key = lab.item() if hasattr(lab, 'item') else lab
        n = int(idx.size)
        if n < 2:
            out[key] = 0.0
            continue
        possible = n * (n - 1) // 2
        pts = centroids[idx]
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            connected = len(tree.query_pairs(r=T))
        except Exception:
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            tri = np.triu(dist, k=1)
            connected = int(np.sum((tri > 0) & (tri <= T)))
        out[key] = float(connected) / float(possible) * 100.0 if possible > 0 else 0.0
    return out


def edge_density(clusters, coords, min_patch_cells=1):
    """计算边缘密度 (Edge Density, ED)。"""
    nodes, _, _, patch_size, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return 0.0

    boundary_edges = 0
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for (r, c), lab in nodes.items():
        for dr, dc in dirs:
            nb = (r + dr, c + dc)
            # 如果邻居不存在，或者是不同类的，则为异质边界
            if nb not in nodes:
                boundary_edges += 1
            elif nodes[nb] != lab:
                boundary_edges += 1

    total_area = float(len(nodes)) * float(patch_size ** 2)
    if total_area <= 0:
        return 0.0
    return float(boundary_edges) * float(patch_size) / total_area


def edge_density_by_class(clusters, coords, min_patch_cells=1):
    """按 cluster 分别计算 ED（该类边界长度 / 景观总面积）。"""
    nodes, unique_labels, _, patch_size, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return {}

    boundary_by_lab = {int(i): 0 for i in range(int(unique_labels.shape[0]))}
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for (r, c), lab in nodes.items():
        li = int(lab)
        for dr, dc in dirs:
            nb = (r + dr, c + dc)
            if nb not in nodes or nodes[nb] != lab:
                boundary_by_lab[li] = boundary_by_lab.get(li, 0) + 1

    total_area = float(len(nodes)) * float(patch_size ** 2)
    if total_area <= 0:
        return {}

    out = {}
    for lab_i in range(int(unique_labels.shape[0])):
        out[unique_labels[lab_i].item()] = float(boundary_by_lab.get(lab_i, 0)) * float(patch_size) / total_area
    return out


def simpson_diversity_index(clusters, coords, min_patch_cells=1):
    """计算辛普森多样性指数 (Simpson's Diversity Index, SIDI)。"""
    nodes, _, _, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return 0.0

    from collections import Counter
    counts = Counter(nodes.values())
    total = sum(counts.values())
    if total <= 1:
        return 0.0

    sum_pi_sq = sum((float(v) / total) ** 2 for v in counts.values())
    return float(1.0 - sum_pi_sq)


def area_cv(clusters, coords, min_patch_cells=1):
    """计算斑块面积变异系数 (Patch Area Coefficient of Variation, AREA_CV)。"""
    import numpy as np
    _, _, areas, _ = _extract_patches_for_connectivity(clusters, coords, min_patch_cells=min_patch_cells)
    if len(areas) < 2:
        return 0.0
    mean_area = float(np.mean(areas))
    if mean_area <= 0:
        return 0.0
    std_area = float(np.std(areas, ddof=0))
    return float(std_area / mean_area * 100.0)


def area_cv_by_class(clusters, coords, min_patch_cells=1):
    """按 cluster 分别计算 AREA_CV（0-100）。"""
    import numpy as np

    labs, _, areas, _ = _extract_patches_for_connectivity(
        clusters, coords, min_patch_cells=min_patch_cells
    )
    if len(areas) == 0:
        return {}

    out = {}
    for lab in np.unique(labs):
        key = lab.item() if hasattr(lab, 'item') else lab
        a = areas[labs == lab]
        if len(a) < 2:
            out[key] = 0.0
            continue
        mean_a = float(np.mean(a))
        if mean_a <= 0:
            out[key] = 0.0
            continue
        out[key] = float(np.std(a, ddof=0) / mean_a * 100.0)
    return out


def aggregation_index(clusters, coords, min_patch_cells=1):
    """计算聚集指数 (Aggregation Index, AI)。"""
    import numpy as np
    nodes, _, _, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return 0.0

    like_adj = {}
    total_cells = {}
    dirs = ((0, 1), (1, 0))  # 只看右和下，避免重复计数
    for (r, c), lab in nodes.items():
        total_cells[lab] = total_cells.get(lab, 0) + 1
        for dr, dc in dirs:
            nb = (r + dr, c + dc)
            if nb in nodes and nodes[nb] == lab:
                like_adj[lab] = like_adj.get(lab, 0) + 1

    sum_gii = 0
    sum_max_gii = 0
    for lab, n in total_cells.items():
        gii = like_adj.get(lab, 0)
        # 对于n个cell的斑块，其可能的最大同类相接边数为 2*n - ceil(2*sqrt(n))
        max_gii = 2 * n - int(np.ceil(2.0 * np.sqrt(n)))
        sum_gii += gii
        sum_max_gii += max_gii

    if sum_max_gii <= 0:
        return 0.0
    return float(sum_gii) / float(sum_max_gii) * 100.0


def aggregation_index_by_class(clusters, coords, min_patch_cells=1):
    """按 cluster 分别计算 AI（0-100）。"""
    import numpy as np

    nodes, unique_labels, _, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    if not nodes:
        return {}

    like_adj = {}
    total_cells = {}
    dirs = ((0, 1), (1, 0))  # 只看右和下，避免重复计数
    for (r, c), lab in nodes.items():
        total_cells[lab] = total_cells.get(lab, 0) + 1
        for dr, dc in dirs:
            nb = (r + dr, c + dc)
            if nb in nodes and nodes[nb] == lab:
                like_adj[lab] = like_adj.get(lab, 0) + 1

    out = {}
    for lab_i in range(int(unique_labels.shape[0])):
        n = int(total_cells.get(lab_i, 0))
        gii = int(like_adj.get(lab_i, 0))
        max_gii = 2 * n - int(np.ceil(2.0 * np.sqrt(n))) if n > 0 else 0
        out[unique_labels[lab_i].item()] = float(gii) / float(max_gii) * 100.0 if max_gii > 0 else 0.0
    return out


def _append_cluster_level_metrics(target_dict, metric_name, metric_by_cluster, num_clusters):
    """把 dict 形式的 cluster 指标写入结果字典，缺失 cluster 自动补 0。"""
    for i in range(int(num_clusters)):
        target_dict[f'{metric_name}_{i}'] = float(metric_by_cluster.get(i, 0.0))


def pixel_blocks_total(clusters, coords, min_patch_cells=1):
    """统计 slide 的像素块总数（栅格 cell 数，而非连通斑块数）。

    说明：先按坐标栅格化，再应用 min_patch_cells 小连通域过滤，
    返回保留下来的 cell 数量。
    """
    nodes, _, _, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    return int(len(nodes))


def cohesion_index(clusters, coords, min_patch_cells=1):
    """计算斑块内聚力指数 (Patch Cohesion Index, COHESION)。"""
    import numpy as np
    nodes, _, _, _, _, _ = _build_sparse_nodes(clusters, coords)
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)
    Z = len(nodes)
    if Z <= 1:
        return 0.0

    visited = set()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    sum_p = 0.0
    sum_p_sqrt_a = 0.0

    for start_rc, start_lab in nodes.items():
        if start_rc in visited:
            continue
        visited.add(start_rc)
        stack = [start_rc]
        cell_count = 0
        boundary_edges = 0

        while stack:
            r, c = stack.pop()
            cell_count += 1
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                if nb not in nodes or nodes[nb] != start_lab:
                    boundary_edges += 1
                elif nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        p = float(boundary_edges)
        a = float(cell_count)
        sum_p += p
        sum_p_sqrt_a += p * np.sqrt(a)

    if sum_p_sqrt_a <= 0:
        return 0.0

    cohesion = (1.0 - sum_p / sum_p_sqrt_a) / (1.0 - 1.0 / np.sqrt(Z)) * 100.0
    return float(max(0.0, cohesion))


def _dominant_cluster_ratio(clusters, num_clusters):
    """Return (max_ratio, max_cluster_id) for a slide.

    Args:
        clusters: (N,) cluster labels (after any filtering).
        num_clusters: total number of clusters (K).

    Returns:
        max_ratio: float in [0,1]
        max_cluster: int
    """
    import numpy as np

    clusters_np = _to_numpy(clusters).reshape(-1)
    total = int(clusters_np.shape[0])
    if total <= 0 or int(num_clusters) <= 0:
        return 0.0, -1
    counts = np.bincount(clusters_np.astype(np.int64, copy=False), minlength=int(num_clusters))
    max_cluster = int(np.argmax(counts))
    max_ratio = float(counts[max_cluster]) / float(total) if total > 0 else 0.0
    return max_ratio, max_cluster

def draw_slide_landscape(clusters, coords, save_path, max_clusters=5, color=None, min_patch_cells=1):

    """Draw a slide-level landscape (mosaic) of clustered patches.

    Args:
        clusters: (N,) cluster id for each patch. Can be numpy/torch/list.
        coords: (N, 2) top-left (x, y) for each square patch. Can be numpy/torch.
        save_path: Output file path (e.g. "/tmp/a.png") or a directory.
        max_clusters: Max number of cluster labels shown in legend. If <= 0, show all.
        color: Colors for clusters.
            - None: use default palette.
            - list/tuple: colors in the order of sorted unique cluster ids.
            - dict: map from original cluster id (value in `clusters`) -> color.

    Returns:
        The written image filepath.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    nodes, unique_labels, label_index, patch_size, rx_int, ry_int = _build_sparse_nodes(clusters, coords)
    
    all_nodes = nodes.copy()

    # Optional: drop tiny connected components (treat as background).
    nodes = _filter_small_patches(nodes, min_patch_cells=min_patch_cells)

    k = int(unique_labels.shape[0])

    if k <= 20:
        base = plt.get_cmap('tab20')
        default_colors = base(np.linspace(0, 1, max(k, 1)))
    else:
        base = plt.get_cmap('gist_ncar')
        default_colors = base(np.linspace(0, 1, k))

    # Apply user-provided palette if given.
    colors = default_colors.copy()
    if color is not None:
        if isinstance(color, dict):
            for i, lab in enumerate(unique_labels.tolist()):
                if lab in color:
                    colors[i] = mcolors.to_rgba(color[lab])
        else:
            seq = list(color)
            for i in range(min(k, len(seq))):
                colors[i] = mcolors.to_rgba(seq[i])
    cmap = ListedColormap(colors)
    # Missing cells will be masked; render them transparent.
    try:
        cmap.set_bad((0, 0, 0, 0))
    except Exception:
        pass

    # Choose which cluster ids to show in legend (by frequency).
    counts = np.bincount(label_index.astype(np.int64), minlength=k)
    order = np.argsort(-counts)
    if max_clusters is None:
        max_clusters = 0
    show_n = int(k if int(max_clusters) <= 0 else min(int(max_clusters), k))
    legend_indices = order[:show_n]
    legend_indices = legend_indices[np.argsort(legend_indices)]  # stable by id
    legend_handles = [
        Patch(facecolor=colors[int(i)], edgecolor='none', label=str(unique_labels[int(i)]))
        for i in legend_indices
    ]

    # Resolve output filepath.
    save_path = os.fspath(save_path)
    if save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf', '.svg')):
        out_file = save_path
        os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    else:
        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, 'landscape.png')

    # Try fast rasterization to a regular grid (common for patch tiling).
    coords_np = _to_numpy(coords)
    coords_f = coords_np.astype(np.float64, copy=False)
    x0 = float(coords_f[:, 0].min())
    y0 = float(coords_f[:, 1].min())

    gx = (coords_f[:, 0] - x0) / patch_size
    gy = (coords_f[:, 1] - y0) / patch_size
    rx = np.rint(gx)
    ry = np.rint(gy)

    regular = (np.max(np.abs(gx - rx)) <= 1e-3) and (np.max(np.abs(gy - ry)) <= 1e-3)

    if regular:
        cols = int(rx.max()) + 1
        rows = int(ry.max()) + 1
        grid = np.full((rows, cols), fill_value=-1, dtype=np.int32)
        grid_small = np.full((rows, cols), fill_value=-1, dtype=np.int32)
        
        # Fill from all sparse nodes; removed cells go to grid_small, kept to grid
        for (r, c), lab in all_nodes.items():
            if 0 <= int(r) < rows and 0 <= int(c) < cols:
                if (r, c) in nodes:
                    grid[int(r), int(c)] = int(lab)
                else:
                    grid_small[int(r), int(c)] = int(lab)
                    
        grid_masked = np.ma.masked_where(grid < 0, grid)
        grid_small_masked = np.ma.masked_where(grid_small < 0, grid_small)

        aspect = cols / max(rows, 1)
        fig_w = float(np.clip(8.0 * aspect, 4.0, 14.0))
        fig_h = float(np.clip(8.0 / max(aspect, 1e-6), 4.0, 14.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
        # Use pcolormesh so we can add subtle cell edges efficiently.
        x_edges = x0 + np.arange(cols + 1) * patch_size
        y_edges = y0 + np.arange(rows + 1) * patch_size
        
        # 先绘制透明小斑块
        mesh_small = ax.pcolormesh(
            x_edges, y_edges, grid_small_masked, cmap=cmap, shading='flat',
            edgecolors=(0, 0, 0, 0.12), linewidth=0.15, antialiased=False, alpha=0.3
        )
        
        # 再绘制主体斑块
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            grid_masked,
            cmap=cmap,
            shading='flat',
            edgecolors=(0, 0, 0, 0.12),
            linewidth=0.15,
            antialiased=False,
        )
        ax.set_xlim(x0, x0 + cols * patch_size)
        ax.set_ylim(y0 + rows * patch_size, y0)  # y increases downward in slide coords
        ax.set_axis_off()
        ax.set_aspect('equal')

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title='cluster',
                loc='upper left',
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                frameon=True,
                framealpha=0.6,
                fontsize=7,
                title_fontsize=8,
            )

        fig.tight_layout(pad=0)
        fig.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return out_file

    # Fallback: irregular coords -> draw rectangles.
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    # Keep only points whose raster cell survived filtering.
    rx_i = rx.astype(np.int64)
    ry_i = ry.astype(np.int64)
    keep_mask = np.zeros((coords_f.shape[0],), dtype=bool)
    drop_mask = np.zeros((coords_f.shape[0],), dtype=bool)
    
    for idx, (r, c, lab) in enumerate(zip(ry_i.tolist(), rx_i.tolist(), label_index.tolist())):
        v = nodes.get((int(r), int(c)), None)
        if v is not None and int(v) == int(lab):
            keep_mask[idx] = True
        else:
            v_all = all_nodes.get((int(r), int(c)), None)
            if v_all is not None and int(v_all) == int(lab):
                drop_mask[idx] = True

    coords_keep = coords_f[keep_mask]
    labs_keep = label_index.astype(np.int32)[keep_mask]
    coords_drop = coords_f[drop_mask]
    labs_drop = label_index.astype(np.int32)[drop_mask]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)

    if len(coords_drop) > 0:
        rects_drop = [
            Rectangle((float(x), float(y)), patch_size, patch_size)
            for x, y in coords_drop
        ]
        pc_drop = PatchCollection(rects_drop, cmap=cmap, edgecolor=(0, 0, 0, 0.12), linewidth=0.15, alpha=0.3)
        pc_drop.set_array(labs_drop.astype(np.float32))
        ax.add_collection(pc_drop)

    if len(coords_keep) > 0:
        rects = [
            Rectangle((float(x), float(y)), patch_size, patch_size)
            for x, y in coords_keep
        ]
        pc = PatchCollection(rects, cmap=cmap, edgecolor=(0, 0, 0, 0.12), linewidth=0.15)
        pc.set_array(labs_keep.astype(np.float32))
        ax.add_collection(pc)
        
    ax.set_xlim(coords_f[:, 0].min(), coords_f[:, 0].max() + patch_size)
    ax.set_ylim(coords_f[:, 1].max() + patch_size, coords_f[:, 1].min())
    ax.set_axis_off()
    ax.set_aspect('equal')

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title='cluster',
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.6,
            fontsize=7,
            title_fontsize=8,
        )
    fig.tight_layout(pad=0)
    fig.savefig(out_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return out_file

if __name__ == "__main__":
    import argparse
    import json
    from cluster import load_data, load_h5

    parser = argparse.ArgumentParser(description='Calculate landscape indices with a JSON config file.')
    parser.add_argument(
        '--config',
        default='config_ccRCC.json',
        help='Path to config JSON file (default: config_ccRCC.json)',
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    slide_path, time, event = load_data(config['train']['file_path'], config['train']['slide_folder'])

    # 过滤小斑块（按连通分量包含的 patch 数计）。默认 1 表示不过滤。
    min_patch_cells = int(config.get('min_patch_cells', 1))
    print(f"Using min_patch_cells={min_patch_cells} for filtering small patches.")

    folder_name = config['train']['name'] + '_' + config['test']['name']

    outputs_dir = os.path.join('outputs', folder_name)
    assign_clusters, num_clusters, backend = _load_cluster_assigner(outputs_dir, prefer_faiss=True, use_gpu=True)
    print(f'Loaded clustering backend={backend}, k={num_clusters} from {outputs_dir}')

    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']  # 对应前8个cluster的颜色

    # 若某一类占比过高，则跳过该 slide（避免“单一组织/背景占绝大多数”导致指标失真）。
    # 设为 >=1.0 可关闭。
    dominant_ratio_thresh = float(config.get('dominant_ratio_thresh', 0.95))
    if dominant_ratio_thresh < 0:
        dominant_ratio_thresh = 0.0
    print(f"Using dominant_ratio_thresh={dominant_ratio_thresh} (skip if max_ratio >= thresh)")

    p = []
    kept_time = []
    kept_event = []
    it = zip(slide_path, time, event)
    for path, t, e in tqdm.tqdm(it, total=len(slide_path)):
        features, coords = load_h5(path)
        clusters = assign_clusters(features)
        clusters_f, coords_f = _filter_points_by_min_patch_cells(clusters, coords, min_patch_cells=min_patch_cells)

        if len(clusters_f) == 0:
            cluster_idx = {f'ratio_{i}': 0.0 for i in range(num_clusters)}
            cluster_idx.update({f'shape_idx_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'area_cv_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'lpi_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'ed_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'ai_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'pd_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'pladj_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'enn_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'prox_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'connect_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx['shannon_idx'] = 0.0
            cluster_idx['sidi'] = 0.0
            cluster_idx['area_cv'] = 0.0
            cluster_idx['ed'] = 0.0
            cluster_idx['ai'] = 0.0
            cluster_idx['cohesion'] = 0.0
            cluster_idx['lpi'] = 0.0
            cluster_idx['cont'] = 0.0
            cluster_idx['split'] = 0.0
            cluster_idx['iji'] = 0.0
            cluster_idx['pd'] = 0.0
            cluster_idx['pladj'] = 0.0
            cluster_idx['enn'] = 0.0
            cluster_idx['prox'] = 0.0
            cluster_idx['connect'] = 0.0
            cluster_idx['pixel_blocks_total'] = 0
            p.append(cluster_idx)
            kept_time.append(t)
            kept_event.append(e)
            print(f"Slide {path} has no patches after filtering; assigned all metrics to 0.")
            continue

        if dominant_ratio_thresh < 1.0:
            max_ratio, max_cluster = _dominant_cluster_ratio(clusters_f, num_clusters)
            if max_ratio >= dominant_ratio_thresh:
                print(
                    f"Skip slide {path} because cluster {max_cluster} dominates: {max_ratio:.4f} >= {dominant_ratio_thresh}"
                )
                continue

        out = draw_slide_landscape(
            clusters,
            coords,
            save_path='outputs/' + folder_name + '/train/'+path.split('/')[-1].split('.')[0]+'.png',
            max_clusters=num_clusters,
            color=color,
            min_patch_cells=min_patch_cells,
        )
        shape_idx = patch_shape_index(clusters_f, coords_f, max_clusters=num_clusters, min_patch_cells=min_patch_cells)
        # 计算每个cluster的比例，保存
        total = len(clusters_f)
        cluster_idx = {}
        for i in range(num_clusters):
            count = (clusters_f == i).sum()
            ratio = count / total
            cluster_idx[f'ratio_{i}'] = ratio
            if i in shape_idx:
                cluster_idx[f'shape_idx_{i}'] = shape_idx[i]
            else:
                cluster_idx[f'shape_idx_{i}'] = 0
        # 计算shannon index，保存
        from scipy.stats import entropy
        shannon_idx = entropy([(clusters_f == i).sum() for i in range(num_clusters)])
        cluster_idx['shannon_idx'] = shannon_idx
        cluster_idx['sidi'] = simpson_diversity_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['area_cv'] = area_cv(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['ed'] = edge_density(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['ai'] = aggregation_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['cohesion'] = cohesion_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['lpi'] = largest_patch_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['cont'] = cont_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['split'] = splitting_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['iji'] = iji_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['pd'] = patch_density(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['pladj'] = pladj_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['enn'] = enn_mean(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['prox'] = prox_mean(clusters_f, coords_f, search_radius=config.get('prox_radius', None), min_patch_cells=min_patch_cells)
        cluster_idx['connect'] = connectance_index(clusters_f, coords_f, threshold=config.get('connect_thresh', None), min_patch_cells=min_patch_cells)
        cluster_idx['pixel_blocks_total'] = pixel_blocks_total(clusters_f, coords_f, min_patch_cells=min_patch_cells)

        _append_cluster_level_metrics(cluster_idx, 'area_cv', area_cv_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'lpi', largest_patch_index_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'ed', edge_density_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'ai', aggregation_index_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'pd', patch_density_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'pladj', pladj_index_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'enn', enn_mean_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'prox', prox_mean_by_class(clusters_f, coords_f, search_radius=config.get('prox_radius', None), min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'connect', connectance_index_by_class(clusters_f, coords_f, threshold=config.get('connect_thresh', None), min_patch_cells=min_patch_cells), num_clusters)

        cluster_idx['slide_id'] = path.split('/')[-1]
        # 去除最后的.h5
        cluster_idx['slide_id'] = cluster_idx['slide_id'].replace('.h5', '')
        p.append(cluster_idx)
        kept_time.append(t)
        kept_event.append(e)
    import pandas as pd
    df = pd.DataFrame(p)
    df['Time'] = kept_time
    df['Event'] = kept_event
    df.to_csv('outputs/' + folder_name + '/cluster_ratios.csv', index=False)

    # 在test集上做同样的处理
    slide_path, time, event = load_data(config['test']['file_path'], config['test']['slide_folder'])
    p = []
    kept_time = []
    kept_event = []
    it = zip(slide_path, time, event)
    for path, t, e in tqdm.tqdm(it, total=len(slide_path)):
        features, coords = load_h5(path)
        clusters = assign_clusters(features)
        clusters_f, coords_f = _filter_points_by_min_patch_cells(clusters, coords, min_patch_cells=min_patch_cells)

        if len(clusters_f) == 0:
            cluster_idx = {f'ratio_{i}': 0.0 for i in range(num_clusters)}
            cluster_idx.update({f'shape_idx_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'area_cv_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'lpi_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'ed_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'ai_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'pd_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'pladj_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'enn_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'prox_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx.update({f'connect_{i}': 0.0 for i in range(num_clusters)})
            cluster_idx['shannon_idx'] = 0.0
            cluster_idx['sidi'] = 0.0
            cluster_idx['area_cv'] = 0.0
            cluster_idx['ed'] = 0.0
            cluster_idx['ai'] = 0.0
            cluster_idx['cohesion'] = 0.0
            cluster_idx['lpi'] = 0.0
            cluster_idx['cont'] = 0.0
            cluster_idx['split'] = 0.0
            cluster_idx['iji'] = 0.0
            cluster_idx['pd'] = 0.0
            cluster_idx['pladj'] = 0.0
            cluster_idx['enn'] = 0.0
            cluster_idx['prox'] = 0.0
            cluster_idx['connect'] = 0.0
            cluster_idx['pixel_blocks_total'] = 0
            p.append(cluster_idx)
            kept_time.append(t)
            kept_event.append(e)
            continue

        if dominant_ratio_thresh < 1.0:
            max_ratio, max_cluster = _dominant_cluster_ratio(clusters_f, num_clusters)
            if max_ratio >= dominant_ratio_thresh:
                print(
                    f"Skip slide {path} because cluster {max_cluster} dominates: {max_ratio:.4f} >= {dominant_ratio_thresh}"
                )
                continue

        out = draw_slide_landscape(
            clusters,
            coords,
            save_path='outputs/' + folder_name + '/test/'+path.split('/')[-1].split('.')[0]+'.png',
            max_clusters=num_clusters,
            color=color,
            min_patch_cells=min_patch_cells,
        )
        shape_idx = patch_shape_index(clusters_f, coords_f, max_clusters=num_clusters, min_patch_cells=min_patch_cells)
        # 计算每个cluster的比例，保存
        total = len(clusters_f)
        cluster_idx = {}
        for i in range(num_clusters):
            count = (clusters_f == i).sum()
            ratio = count / total
            cluster_idx[f'ratio_{i}'] = ratio
            if i in shape_idx:
                cluster_idx[f'shape_idx_{i}'] = shape_idx[i]
            else:
                cluster_idx[f'shape_idx_{i}'] = 0
        # 计算shannon index，保存
        from scipy.stats import entropy
        shannon_idx = entropy([(clusters_f == i).sum() for i in range(num_clusters)])
        cluster_idx['shannon_idx'] = shannon_idx
        cluster_idx['Pielou'] = shannon_idx / np.log(num_clusters) if num_clusters > 1 else 0.0
        cluster_idx['sidi'] = simpson_diversity_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['area_cv'] = area_cv(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['ed'] = edge_density(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['ai'] = aggregation_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['cohesion'] = cohesion_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['lpi'] = largest_patch_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['cont'] = cont_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['split'] = splitting_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['iji'] = iji_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['pd'] = patch_density(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['pladj'] = pladj_index(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['enn'] = enn_mean(clusters_f, coords_f, min_patch_cells=min_patch_cells)
        cluster_idx['prox'] = prox_mean(clusters_f, coords_f, search_radius=config.get('prox_radius', None), min_patch_cells=min_patch_cells)
        cluster_idx['connect'] = connectance_index(clusters_f, coords_f, threshold=config.get('connect_thresh', None), min_patch_cells=min_patch_cells)
        cluster_idx['pixel_blocks_total'] = pixel_blocks_total(clusters_f, coords_f, min_patch_cells=min_patch_cells)

        _append_cluster_level_metrics(cluster_idx, 'area_cv', area_cv_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'lpi', largest_patch_index_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'ed', edge_density_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'ai', aggregation_index_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'pd', patch_density_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'pladj', pladj_index_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'enn', enn_mean_by_class(clusters_f, coords_f, min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'prox', prox_mean_by_class(clusters_f, coords_f, search_radius=config.get('prox_radius', None), min_patch_cells=min_patch_cells), num_clusters)
        _append_cluster_level_metrics(cluster_idx, 'connect', connectance_index_by_class(clusters_f, coords_f, threshold=config.get('connect_thresh', None), min_patch_cells=min_patch_cells), num_clusters)

        cluster_idx['slide_id'] = path.split('/')[-1]
        # 去除最后的.h5
        cluster_idx['slide_id'] = cluster_idx['slide_id'].replace('.h5', '')
        p.append(cluster_idx)
        kept_time.append(t)
        kept_event.append(e)
    df = pd.DataFrame(p)
    df['Time'] = kept_time
    df['Event'] = kept_event
    df.to_csv('outputs/' + folder_name + '/cluster_ratios_test.csv', index=False)
