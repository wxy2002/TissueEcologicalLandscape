import torch
import tqdm
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def _as_float32_numpy(x):
    import numpy as np
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    return x


def _try_import_faiss():
    try:
        import faiss  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "faiss 未安装或不可用（需要 faiss-gpu / faiss-cpu）。"
        ) from e
    return faiss

def load_config(path='config.json'):
    import json
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def load_data(path, slide_folder):
    import pandas as pd
    data = pd.read_csv(path)
    slide_name = data['slide_id'].values
    # 添加slide_folder路径
    slide_path = [slide_folder + name + '.h5' for name in slide_name]
    time = data['Time'].values
    event = data['Event'].values
    return slide_path, time, event

def load_h5(slide_path):
    import h5py
    with h5py.File(slide_path, 'r') as f:
        features = f['feats'][:]
        coords = f['coords'][:]
    features = torch.tensor(features, dtype=torch.float32)
    coords = torch.tensor(coords, dtype=torch.float32)
    # print(f'Loaded {slide_path}: features shape {features.shape}, coords shape {coords.shape}')
    return features, coords

def draw_cluster_tsne(features, clusters, save_path):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    # sklearn 更稳定地处理 numpy 输入
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(clusters, torch.Tensor):
        clusters = clusters.detach().cpu().numpy()
    # 先随机选取10000个点进行降维，避免内存不足
    if features.shape[0] > 10000:
        idx = torch.randperm(features.shape[0])[:10000].cpu().numpy()
        features = features[idx]
        clusters = clusters[idx]
    # 降维到2维，并且不同cluster用不同颜色表示
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(10, 10))
    for i in range(clusters.max() + 1):
        plt.scatter(features_2d[clusters == i, 0], features_2d[clusters == i, 1], label='Cluster ' + str(i))
    plt.legend()
    plt.savefig(save_path + '/cluster_tsne.png')
    plt.close()

def _faiss_kmeans_sse(x_np, k, niter=25, seed=0, max_points_per_centroid=5000, use_gpu=True):
    """Return (centroids, labels, sse) using FAISS KMeans.

    Notes:
    - x_np must be float32 contiguous.
    - If faiss-gpu is available and use_gpu=True, it will run on GPU.
    """
    faiss = _try_import_faiss()
    d = x_np.shape[1]
    kmeans = faiss.Kmeans(
        d,
        k,
        niter=niter,
        nredo=1,
        verbose=False,
        seed=seed,
        gpu=bool(use_gpu),
        max_points_per_centroid=int(max_points_per_centroid),
    )
    kmeans.train(x_np)
    # kmeans.index is an index over centroids (CPU or GPU depending on build)
    distances, labels = kmeans.index.search(x_np, 1)
    sse = float(distances.sum())
    centroids = kmeans.centroids.reshape(k, d)
    return centroids, labels.reshape(-1), sse


def find_best_k(
    features,
    k_min=2,
    k_max=10,
    batch_size=1024,
    max_iter=200,
    sample_max=50000,
    sample_seed=0,
    backend='faiss',
    use_gpu=True,
):
    from kneed import KneeLocator
    from tqdm import trange
    import numpy as np

    features_np = _as_float32_numpy(features)

    sse = []
    k_range = list(range(k_min, k_max + 1))

    if backend == 'faiss':
        niter = min(int(max_iter), 50)
        for k in trange(k_min, k_max + 1):
            _, _, inertia = _faiss_kmeans_sse(
                features_np,
                k,
                niter=niter,
                seed=0,
                max_points_per_centroid=5000,
                use_gpu=use_gpu,
            )
            sse.append(inertia)
    else:
        from sklearn.cluster import KMeans

        for k in trange(k_min, k_max + 1):
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                random_state=0,
                max_iter=max_iter,
                n_init='auto',
            )
            kmeans.fit(features_np)
            sse.append(float(kmeans.inertia_))

    kn = KneeLocator(k_range, sse, curve='convex', direction='decreasing')
    best_k = kn.knee if kn.knee is not None else k_min
    return best_k

def cluster_kmeans_pp(
    features,
    n_clusters,
    save_path,
    batch_size=1024,
    max_iter=200,
    backend='faiss',
    use_gpu=True,
):
    import json
    import numpy as np

    if n_clusters <= 0:
        n_clusters = find_best_k(
            features,
            batch_size=batch_size,
            max_iter=max_iter,
            backend=backend,
            use_gpu=use_gpu,
        )
    print('Best k:', n_clusters)

    features_np = _as_float32_numpy(features)

    if backend == 'faiss':
        centroids, labels, sse = _faiss_kmeans_sse(
            features_np,
            int(n_clusters),
            niter=int(max_iter),
            seed=0,
            max_points_per_centroid=10000,
            use_gpu=use_gpu,
        )

        # 保存：centroids + faiss index（用于 test 时快速 assign）
        np.save(os.path.join(save_path, 'faiss_centroids.npy'), centroids)
        meta = {
            'backend': 'faiss',
            'k': int(n_clusters),
            'd': int(features_np.shape[1]),
            'sse': float(sse),
            'use_gpu': bool(use_gpu),
        }
        with open(os.path.join(save_path, 'faiss_kmeans_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        faiss = _try_import_faiss()
        index = faiss.IndexFlatL2(int(features_np.shape[1]))
        index.add(centroids.astype(np.float32))
        faiss.write_index(index, os.path.join(save_path, 'faiss_centroids.index'))

        draw_cluster_tsne(features_np, labels, save_path)

    else:
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            random_state=0,
            max_iter=max_iter,
            n_init='auto',
        )
        kmeans.fit(features_np)
        import joblib

        joblib.dump(kmeans, save_path + '/kmeans_model.pkl')
        draw_cluster_tsne(features_np, kmeans.labels_, save_path)

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run clustering with a JSON config file.')
    parser.add_argument(
        '--config',
        default='config_ccRCC.json',
        help='Path to config JSON file (default: config_ccRCC.json)',
    )
    args = parser.parse_args()

    set_seed(2002)
    config = load_config(path=args.config)
    save_path = os.path.join('outputs', config['train']['name'] + '_' + config['test']['name'])
    os.makedirs(save_path, exist_ok=True)
    slide_path, time, event = load_data(config['train']['file_path'], config['train']['slide_folder'])
    f = []
    c = []
    for path in tqdm.tqdm(slide_path):
        features, coords = load_h5(path)
        f.append(features)
        c.append(coords)
    # 拼成二维数组
    f = torch.cat(f, dim=0)
    c = torch.cat(c, dim=0)
    print(f.shape, c.shape)
    # 使用 faiss-gpu 加速聚类（跳过安装过程，假设环境已有 faiss-gpu）
    cluster_kmeans_pp(f, n_clusters=-1, save_path=save_path, 
                      backend='faiss', use_gpu=True)
