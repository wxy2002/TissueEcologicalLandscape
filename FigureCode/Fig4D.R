# -----GSEA分析：探索景观指标area_cv高低分组的相关通路-----
# 用户需自行提供分组文件 pheno_file，包含列：
#   sample_id: TCGA样本ID（如 TCGA-CW-5584）
#   group:     分组（如 "High", "Low" 或 1/0）
#   （可选 Time, Event 用于生存分组）
#
# TPM 文件: data/KIRC_GSEA/TCGA-KIRC.star_tpm.tsv
#   Ensembl_ID 格式: ENSG00000000003.15
#
# 基因集: MSigDB Hallmark (h.all) 和 C2 (Reactome)

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(tibble)
  library(ggplot2)
  library(clusterProfiler)
  library(enrichplot)
  library(ggpubr)
})

# ----------------------------- 配置修改区 -----------------------------
tpm_file <- "data/KIRC_GSEA/TCGA-KIRC.star_tpm.tsv"
pheno_file <- "out/ccRCC/train_core_agg.csv"  # <<<< 修改为你的分组文件路径
out_dir <- "out/GSEA"

# 分组列名（你的分组文件中标识分组的列名）
group_col <- "group"  # 你的分组列名，值为 "High" / "Low" 或其他两水平因子

# 基因集 ===============================================================
# Hallmark gmt 文件（推荐，50个精选通路，解释性强）
gmt_file <- "data/KIRC_GSEA/h.all.v2026.1.Hs.entrez.gmt"

# 输出文件名前缀
out_prefix <- "KIRC_area_cv"

# 指定基因名（用于差异表达分析）
target_gene <- "SCNN1G"  # <<<< 修改为你感兴趣的基因名（Symbol）
# ---------------------------------------------------------------------

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ----------------------------- 1. 读取TPM表达矩阵 -----------------------------
message("[1/9] 读取 TPM 表达矩阵 ...")
tpm <- read_tsv(tpm_file, show_col_types = FALSE)

tpm <- tpm %>%
  mutate(gene = str_remove(Ensembl_ID, "\\.[0-9]+$")) %>%
  dplyr::select(-Ensembl_ID) %>%
  group_by(gene) %>%
  summarise(across(everything(), mean), .groups = "drop") %>%
  column_to_rownames("gene")

message("  -> 基因数: ", nrow(tpm), ", 样本数: ", ncol(tpm))

# ----------------------------- 2. 读取分组信息 -----------------------------
message("[2/9] 读取分组信息 ...")
pheno <- read_csv(pheno_file, show_col_types = FALSE) %>%
  dplyr::select(1, !!sym(group_col))
colnames(pheno)[1] <- "sample_id"

pheno <- pheno %>%
  mutate(sample_id = str_sub(sample_id, 1, 12))

tpm_samples <- colnames(tpm) %>% str_sub(1, 12)
colnames(tpm) <- tpm_samples

common <- intersect(colnames(tpm), pheno$sample_id)
message("  -> 匹配到 ", length(common), " 个样本")

if (length(common) < 10) {
  stop("匹配样本太少 (<10)，请检查 sample_id 格式")
}

tpm <- tpm[, common, drop = FALSE]
pheno <- pheno %>% filter(sample_id %in% common) %>% distinct(sample_id, .keep_all = TRUE)
pheno <- pheno[match(colnames(tpm), pheno$sample_id), ]

pheno[[group_col]] <- as.factor(pheno[[group_col]])
# 确保 High risk / High 是 group_levels[2]（高分组）
if (any(str_detect(levels(pheno[[group_col]]), regex("high", ignore_case = TRUE)))) {
  high_idx <- str_detect(levels(pheno[[group_col]]), regex("high", ignore_case = TRUE))
  low_idx <- !high_idx
  pheno[[group_col]] <- relevel(pheno[[group_col]], levels(pheno[[group_col]])[low_idx])
}
group_levels <- levels(pheno[[group_col]])
message("  -> 分组水平: ", paste(group_levels, collapse = ", "))
message("  -> 每组样本数: ", paste(table(pheno[[group_col]]), collapse = ", "))

# ----------------------------- 3. 基因ID转换: Ensembl -> Entrez -----------------------------
message("[3/9] 基因ID转换 Ensembl -> Entrez ...")
gene_ensembl <- rownames(tpm)

suppressPackageStartupMessages(library(org.Hs.eg.db))
gene_map <- bitr(gene_ensembl,
                 fromType = "ENSEMBL",
                 toType = "ENTREZID",
                 OrgDb = org.Hs.eg.db
)

gene_map <- gene_map %>%
  group_by(ENTREZID) %>%
  slice_max(rowMeans(tpm[ENSEMBL, , drop = FALSE]), n = 1, with_ties = FALSE) %>%
  ungroup()

tpm <- tpm[gene_map$ENSEMBL, , drop = FALSE]
rownames(tpm) <- gene_map$ENTREZID
message("  -> 成功映射 ", nrow(tpm), " 个基因")

# ----------------------------- 4. 计算基因排序指标 -----------------------------
message("[4/9] 计算基因排序列表 (logFC) ...")

group_levels <- levels(pheno[[group_col]])
group_mat <- split(seq_len(ncol(tpm)), pheno[[group_col]])
mean_high <- rowMeans(tpm[, group_mat[[group_levels[2]]], drop = FALSE])
mean_low  <- rowMeans(tpm[, group_mat[[group_levels[1]]], drop = FALSE])
gene_logFC <- log2((mean_high + 1) / (mean_low + 1))  # log2比值，添加pseudocount避免除零

gene_list <- sort(gene_logFC, decreasing = TRUE)
message("  -> 排序列表长度: ", length(gene_list))

# ----------------------------- 5. 读取 GMT 基因集 -----------------------------
message("[5/9] 读取 GMT 基因集 ...")

read_gmt <- function(gmt_path) {
  lines <- read_lines(gmt_path)
  parts_list <- str_split(lines, "\t")
  bind_rows(lapply(parts_list, function(p) {
    data.frame(gs_name = p[1], entrez_gene = p[3:length(p)], stringsAsFactors = FALSE)
  }))
}

gs_df <- read_gmt(gmt_file)

clean_geneset_name <- function(x) {
  x <- str_remove(x, "^HALLMARK_")
  x <- str_replace_all(x, "_", " ")
  x <- str_to_title(x)
  x
}

gs_df$gs_name <- clean_geneset_name(gs_df$gs_name)
gs_terms <- split(gs_df$entrez_gene, gs_df$gs_name)
message("  -> 基因集数量: ", length(gs_terms))

# ----------------------------- 6. 运行 GSEA (多阈值策略) -----------------------------
message("[6/9] 运行 GSEA ...")

run_gsea <- function(p_cutoff) {
  GSEA(
    geneList = gene_list,
    TERM2GENE = gs_df[, c("gs_name", "entrez_gene")],
    minGSSize = 10,
    maxGSSize = 500,
    pvalueCutoff = p_cutoff,
    pAdjustMethod = "BH",
    eps = 0,
    seed = 42
  )
}

gsea_res <- run_gsea(0.05)
n_sig <- sum(gsea_res@result$p.adjust < 0.05)
n_up <- sum(gsea_res@result$p.adjust < 0.05 & gsea_res@result$NES > 0)
n_down <- sum(gsea_res@result$p.adjust < 0.05 & gsea_res@result$NES < 0)

message("  ========== GSEA HALLMARK 结果摘要 ==========")
message("  显著通路 (q<0.05): ", n_sig)
message("    正富集 (NES>0, ", group_levels[2], " 上调): ", n_up)
message("    负富集 (NES<0, ", group_levels[1], " 上调): ", n_down)

if (n_sig < 3) {
  message("  -> 显著通路不足3个，放宽阈值至 q<0.25 (GSEA默认推荐) ...")
  gsea_res <- run_gsea(0.25)
  n_sig <- sum(gsea_res@result$p.adjust < 0.25)
  n_up <- sum(gsea_res@result$p.adjust < 0.25 & gsea_res@result$NES > 0)
  n_down <- sum(gsea_res@result$p.adjust < 0.25 & gsea_res@result$NES < 0)
  message("    放宽后富集 (q<0.25): ", n_sig, " (", group_levels[2], " 上调: ", n_up, ", ", group_levels[1], " 上调: ", n_down, ")")
}

message("  ============================================")

# 列出每条显著通路的 NES 和方向
sig_df <- gsea_res@result %>% filter(p.adjust < ifelse(n_sig < 3 && sum(gsea_res@result$p.adjust < 0.05) < 3 && n_sig > 0, 0.25, 0.05))
sig_df <- sig_df %>% arrange(desc(NES))
message("  富集方向检查:")
for (i in seq_len(nrow(sig_df))) {
  arrow <- ifelse(sig_df$NES[i] > 0,
                  paste0("UP (", group_levels[2], " 上调)"),
                  paste0("DN (", group_levels[1], " 上调)"))
  message("    ", sig_df$ID[i], "  NES=", round(sig_df$NES[i], 2), "  q=", format(sig_df$p.adjust[i], digits = 3, scientific = TRUE), "  ", arrow)
}

# ----------------------------- 7. 指定基因的差异表达分析 -----------------------------
message("[7/9] 指定基因的差异表达分析 ...")

# 将基因Symbol转换为Entrez ID
targt_gene <- 'PLA2G2A'
target_gene_entrez <- bitr(target_gene, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)

if (nrow(target_gene_entrez) == 0) {
  message("  -> 警告: 基因 '", target_gene, "' 未找到对应的Entrez ID，跳过差异分析")
} else {
  # 取第一个匹配的Entrez ID
  target_entrez <- target_gene_entrez$ENTREZID[1]
  
  # 检查该基因是否在tpm矩阵中
  if (!target_entrez %in% rownames(tpm)) {
    message("  -> 警告: 基因 '", target_gene, "' (Entrez ID: ", target_entrez, ") 不在表达矩阵中，跳过差异分析")
  } else {
    # 提取该基因的表达数据
    expr_vec <- tpm[target_entrez, ]
    
    # 创建数据框
    target_expr_df <- data.frame(
      sample_id = names(expr_vec),
      expression = as.numeric(expr_vec),
      stringsAsFactors = FALSE
    )
    
    # 合并分组信息
    target_expr_df <- target_expr_df %>%
      left_join(pheno %>% dplyr::select(sample_id, !!sym(group_col)), by = "sample_id") %>%
      filter(!is.na(!!sym(group_col)))
    
    # 确保分组因子水平
    target_expr_df[[group_col]] <- factor(target_expr_df[[group_col]], levels = group_levels)
    
    # 进行Wilcoxon秩和检验
    group1_expr <- target_expr_df$expression[target_expr_df[[group_col]] == group_levels[1]]
    group2_expr <- target_expr_df$expression[target_expr_df[[group_col]] == group_levels[2]]
    
    wilcox_test <- wilcox.test(group1_expr, group2_expr)
    p_value <- wilcox_test$p.value
    
    # 计算log2FC（以group1为参照）
    mean_group1 <- mean(group1_expr, na.rm = TRUE)
    mean_group2 <- mean(group2_expr, na.rm = TRUE)
    log2fc <- log2((mean_group2 + 1) / (mean_group1 + 1))  # 添加pseudocount避免除零
    
    # 输出统计结果
    stat_df <- data.frame(
      Gene = target_gene,
      EntrezID = target_entrez,
      Group1 = group_levels[1],
      Group2 = group_levels[2],
      Mean_Group1 = mean_group1,
      Mean_Group2 = mean_group2,
      log2FC = log2fc,
      Wilcoxon_p = p_value,
      stringsAsFactors = FALSE
    )
    
    write_csv(stat_df, file.path(out_dir, paste0(out_prefix, "_", target_gene, "_diff_expr.csv")))
    message("  -> 统计结果已保存: ", target_gene, " (Wilcoxon p = ", format(p_value, digits = 3, scientific = TRUE), ")")
    
    # 绘制箱线图
    p_box <- ggplot(target_expr_df, aes(x = .data[[group_col]], y = expression, fill = .data[[group_col]])) +
      geom_boxplot(width = 0.6, outlier.shape = NA) +
      geom_jitter(width = 0.2, size = 1.5, alpha = 0.6) +
      scale_fill_manual(values = c("#2166AC", "#B2182B")) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "none") +
      labs(title = paste0("Expression of ", target_gene, " by Group"),
           subtitle = paste0("Wilcoxon p = ", format(p_value, digits = 3, scientific = TRUE)),
           x = "Group",
           y = "Expression (TPM)")
    
    ggsave(file.path(out_dir, paste0(out_prefix, "_", target_gene, "_boxplot.pdf")),
           p_box, width = 6, height = 5)
    message("  -> 箱线图已保存")
  }
}

# ----------------------------- 8. 可视化与输出 -----------------------------
message("[8/9] 输出结果 ...")
message("[8/9] 输出结果 ...")

result_df <- gsea_res@result %>%
  arrange(p.adjust) %>%
  dplyr::select(ID, NES, pvalue, p.adjust, setSize, leading_edge)
write_csv(result_df, file.path(out_dir, paste0(out_prefix, "_GSEA_results.csv")))
message("  -> 结果表已保存")

gene_list_df <- data.frame(
  ENTREZID = names(gene_list),
  logFC = gene_list,
  ENSEMBL = gene_map$ENSEMBL[match(names(gene_list), gene_map$ENTREZID)]
)
write_csv(gene_list_df, file.path(out_dir, paste0(out_prefix, "_gene_ranking.csv")))
message("  -> 基因排序列表已保存")

# 输出 top100 差异表达基因（按|logFC|排序）
top100_df <- gene_list_df %>% 
  arrange(desc(abs(logFC))) %>%
  slice_head(n = 100)
gene_symbol <- bitr(top100_df$ENTREZID, fromType = "ENTREZID", toType = "SYMBOL", OrgDb = org.Hs.eg.db)
top100_df <- top100_df %>% left_join(gene_symbol, by = "ENTREZID") %>% dplyr::select(ENTREZID, SYMBOL, ENSEMBL, logFC)
write_csv(top100_df, file.path(out_dir, paste0(out_prefix, "_top100_DEG.csv")))
message("  -> Top100 差异表达基因已保存")

if (nrow(gsea_res@result) == 0) {
  message("  -> 无显著结果，跳过绘图")
  quit(save = "no")
}

# 7a. 山脊图
p_top <- min(nrow(gsea_res@result), 30)
p_ridge <- ridgeplot(gsea_res, showCategory = p_top) +
  ggtitle(paste0("GSEA: Hallmark (", group_levels[2], " vs ", group_levels[1], ")")) +
  xlab("logFC")
ggsave(file.path(out_dir, paste0(out_prefix, "_ridgeplot.pdf")),
       p_ridge, width = 10, height = max(5, p_top * 0.3))

# 7b. GSEA Dot Plot (替代经典曲线)
message("  -> 绘制 GSEA Dot Plot ...")

# 从GSEA结果中提取这些通路
dot_df <- gsea_res@result %>%
  dplyr::select(ID, NES, p.adjust, setSize) %>%
  mutate(
    neg_log10_fdr = -log10(p.adjust + 1e-300),
    Direction = ifelse(NES > 0, paste0("Enriched in ", group_levels[2]),
                       paste0("Enriched in ", group_levels[1]))
  )

# 如果指定通路不在结果中，用top通路替代
if (nrow(dot_df) < 5) {
  dot_df <- gsea_res@result %>%
    filter(p.adjust < 0.25) %>%
    arrange(desc(abs(NES))) %>%
    slice_head(n = 8) %>%
    dplyr::select(ID, NES, p.adjust, setSize) %>%
    mutate(
      neg_log10_fdr = -log10(p.adjust + 1e-300),
      Direction = ifelse(NES > 0, paste0("Enriched in ", group_levels[2]),
                         paste0("Enriched in ", group_levels[1]))
    )
}

# 按NES排序
dot_df <- dot_df %>% arrange(NES)
dot_df$ID <- factor(dot_df$ID, levels = dot_df$ID)

# 定义通路类别注释
inflammatory_paths <- c("Tnfa Signaling Via Nf-Kb", "Il6-Jak-Stat3 Signaling", 
                        "Kras Signaling Dn", "Uv Response Dn")
metabolic_paths <- c("Oxidative Phosphorylation", "Heme Metabolism", "Adipogenesis")

dot_df <- dot_df %>%
  mutate(Category = case_when(
    ID %in% inflammatory_paths ~ "Inflammatory programs",
    ID %in% metabolic_paths ~ "Metabolic programs",
    TRUE ~ "Other"
  ))

# 颜色方案
color_vals <- c("#B2182B", "#2166AC")
names(color_vals) <- c(paste0("Enriched in ", group_levels[2]),
                       paste0("Enriched in ", group_levels[1]))

# 绘制dot plot
p_dot <- ggplot(dot_df, aes(x = NES, y = ID)) +
  geom_point(aes(size = neg_log10_fdr, color = Direction), alpha = 0.85) +
  scale_color_manual(values = color_vals) +
  scale_size_continuous(
    name = expression(-log[10](FDR)),
    range = c(3, 10),
    limits = c(min(dot_df$neg_log10_fdr) * 0.9, max(dot_df$neg_log10_fdr) * 1.1)
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "right",
    legend.box = "vertical",
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 13, face = "bold")
  ) +
  labs(
    title = paste0("GSEA: Hallmark Pathways (", group_levels[2], " vs ", group_levels[1], ")"),
    x = "Normalized Enrichment Score (NES)",
    y = NULL,
    color = "Direction"
  )
  # coord_cartesian(xlim = c(min(dot_df$NES) * 1.3, max(dot_df$NES) * 1.3))
p_dot

ggsave(file.path(out_dir, paste0(out_prefix, "_GSEA_dotplot.pdf")),
       p_dot, width = 8, height = 6)
message("  -> GSEA Dot Plot 已保存")

# 7c. 通路与样本 GSVA 热图 —— 按组间差异排序
message("  -> 绘制通路-样本热图 ...")
suppressPackageStartupMessages(library(GSVA))

# 用所有 Hallmark 通路做 GSVA
hallmark_genes <- gs_terms
hallmark_genes <- lapply(hallmark_genes, function(g) intersect(g, rownames(tpm)))
hallmark_genes <- hallmark_genes[sapply(hallmark_genes, length) >= 10]
message("    用于 GSVA 的通路数: ", length(hallmark_genes))

suppressPackageStartupMessages(library(GSVA))
gsva_param <- gsvaParam(as.matrix(tpm), hallmark_genes,
                        minSize = 10, maxSize = 500)
gsva_score <- gsva(gsva_param)

# 对每条通路做 Wilcoxon 检验，按差异大小排序
gsva_df <- as.data.frame(t(gsva_score)) %>%
  rownames_to_column("sample_id") %>%
  left_join(pheno %>% dplyr::select(sample_id, !!sym(group_col)), by = "sample_id")

group1 <- group_levels[1]
group2 <- group_levels[2]

wilcox_res <- lapply(rownames(gsva_score), function(pw) {
  vals <- gsva_df[[pw]]
  test <- wilcox.test(vals[gsva_df[[group_col]] == group1],
                      vals[gsva_df[[group_col]] == group2])
  data.frame(Pathway = pw,
             mean_diff = mean(vals[gsva_df[[group_col]] == group2]) -
               mean(vals[gsva_df[[group_col]] == group1]),
             p_value = test$p.value,
             stringsAsFactors = FALSE)
}) %>% bind_rows() %>%
  mutate(p_adjust = p.adjust(p_value, method = "BH"))

# 筛选显著通路 (p.adjust < 0.25 或 raw p < 0.05)
gsva_sig <- wilcox_res %>%
  filter(p_adjust < 0.25 | p_value < 0.05) %>%
  arrange(desc(mean_diff))

# 如果 GSVA 差异显著通路太少，改用 GSEA 显著通路列表
if (nrow(gsva_sig) < 3) {
  sig_paths <- gsea_res@result$ID[gsea_res@result$p.adjust < 0.25]
  gsva_sig <- wilcox_res %>% filter(Pathway %in% sig_paths) %>% arrange(desc(mean_diff))
}

# 确定热图显示的通路：差异显著的通路 + GSEA 显著通路取并集
sig_paths_gsea <- gsea_res@result$ID[gsea_res@result$p.adjust < 0.25]
sig_paths_gsva <- gsva_sig$Pathway
plot_paths <- union(sig_paths_gsea, sig_paths_gsva)

if (length(plot_paths) < 3) {
  # 都没显著时展示 top 20
  plot_paths <- wilcox_res %>% arrange(p_value) %>% slice_head(n = 20) %>% pull(Pathway)
}

# 按 mean_diff 排序，使热图中上下两组差异最明显
pathway_order <- wilcox_res %>%
  filter(Pathway %in% plot_paths) %>%
  arrange(mean_diff) %>%
  pull(Pathway)

# 准备热图数据，样本按分组排序
gsva_df_plot <- gsva_df %>%
  arrange(.data[[group_col]])

sample_order <- gsva_df_plot$sample_id

gsva_long <- gsva_df_plot %>%
  dplyr::select(sample_id, !!sym(group_col), all_of(pathway_order)) %>%
  pivot_longer(-c(sample_id, !!sym(group_col)), names_to = "Pathway", values_to = "GSVA") %>%
  mutate(Sample = factor(sample_id, levels = sample_order),
         Pathway = factor(Pathway, levels = pathway_order))

# 添加标注：GSEA 显著 / GSVA 显著
gsva_long <- gsva_long %>%
  mutate(
    gsea_sig = Pathway %in% sig_paths_gsea,
    gsva_sig = Pathway %in% sig_paths_gsva,
    anno = case_when(
      gsea_sig & gsva_sig ~ "*",
      gsea_sig ~ "~",
      gsva_sig ~ "^",
      TRUE ~ ""
    ),
    Pathway_label = paste0(Pathway, " ", anno),
    Pathway_label = factor(Pathway_label, levels = paste0(pathway_order, " ",
                                                          if_else(pathway_order %in% sig_paths_gsea & pathway_order %in% sig_paths_gsva, "*",
                                                                  if_else(pathway_order %in% sig_paths_gsea, "~",
                                                                          if_else(pathway_order %in% sig_paths_gsva, "^", "")))))
  )

p_path_heat <- ggplot(gsva_long, aes(Sample, Pathway_label, fill = GSVA)) +
  geom_tile() +
  facet_grid(~ .data[[group_col]], scales = "free_x", space = "free") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0) +
  theme_minimal(base_size = 8) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid = element_blank(),
        strip.text = element_text(size = 10, face = "bold"),
        axis.text.y = element_text(size = 7)) +
  labs(title = "Hallmark Pathway GSVA Activity (sorted by group difference)",
       subtitle = paste0(group2, " vs ", group1,
                         "  |  * GSEA+Wilcoxon sig  ~ GSEA only  ^ Wilcoxon only"),
       x = "Sample", y = "Pathway", fill = "GSVA score")

ggsave(file.path(out_dir, paste0(out_prefix, "_hallmark_heatmap.pdf")),
       p_path_heat, width = 14, height = max(6, length(pathway_order) * 0.28))

message("===== GSEA 分析完成! =====")
message("输出目录: ", normalizePath(out_dir))

# 8d. 火山图：展示差异表达基因
message("  -> 绘制火山图 ...")

# 选择top1000个基因（按|logFC|排序）
top_n_volcano <- min(1000, nrow(gene_list_df))
volcano_genes <- gene_list_df %>% 
  arrange(desc(abs(logFC))) %>%
  slice_head(n = top_n_volcano)

# 计算每个基因的p-value（使用t检验）
group1_idx <- which(pheno[[group_col]] == group_levels[1])
group2_idx <- which(pheno[[group_col]] == group_levels[2])

p_values <- sapply(volcano_genes$ENTREZID, function(gene) {
  if (!gene %in% rownames(tpm)) return(NA)
  expr <- as.numeric(tpm[gene, ])
  group1_expr <- expr[group1_idx]
  group2_expr <- expr[group2_idx]
  test <- t.test(group1_expr, group2_expr)
  return(test$p.value)
})

volcano_genes$p_value <- p_values
volcano_genes$p_adjust <- p.adjust(p_values, method = "BH")

# 添加基因符号
gene_symbol_volcano <- bitr(volcano_genes$ENTREZID, fromType = "ENTREZID", toType = "SYMBOL", OrgDb = org.Hs.eg.db)
volcano_genes <- volcano_genes %>% left_join(gene_symbol_volcano, by = "ENTREZID")

# 标记目标基因
target_entrez_id <- bitr(target_gene, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)$ENTREZID[1]
volcano_genes$is_target <- volcano_genes$ENTREZID == target_entrez_id

# 绘制火山图
p_volcano <- ggplot(volcano_genes, aes(x = logFC, y = -log10(p_value))) +
  geom_point(aes(color = is_target), size = 1.5, alpha = 0.6) +
  scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "red"), guide = "none") +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey50") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50") +
  geom_text(data = volcano_genes %>% filter(is_target), 
            aes(label = SYMBOL), color = "red", size = 3, vjust = -0.5) +
  theme_minimal(base_size = 12) +
  labs(title = "Volcano Plot: Differential Expression",
       subtitle = paste0("Top ", top_n_volcano, " genes by |logFC|"),
       x = "log2 Fold Change",
       y = "-log10(p-value)")

ggsave(file.path(out_dir, paste0(out_prefix, "_volcano_plot.pdf")),
       p_volcano, width = 8, height = 6)
message("  -> 火山图已保存")

# ----------------------------- 8. 生存分析：基于基因表达中位数分组 -----------------------------
message("\n[9/9] 生存分析 ...")

# ！！！用户需修改以下路径和参数！！！
surv_expr_file <- "data/KIRC_GSEA/mRNA Expression TPM z-scores.txt"  # <<<< 基因表达文件路径（txt，含 SAMPLE_ID 和表达量列）
surv_info_file <- "data/ccRCC/CPTAC_ccRCC_OS.csv"    # <<<< 生存信息文件路径（txt，含 SAMPLE_ID、Time、Event）
surv_gene_col <- "SCNN1G"  # <<<< 表达量列名
surv_time_col <- "Time"         # <<<< 生存时间列名
surv_event_col <- "Event"       # <<<< 生存事件列名

suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(survminer))

# 读取基因表达文件
library(data.table)
expr_df <- fread(surv_expr_file)
colnames(expr_df)[2] <- "sample_id"

# 截取前12位样本ID匹配
expr_df <- expr_df %>%
  mutate(sample_id = str_sub(sample_id, 1, 9))

# 表达量列转为数值
expr_df <- expr_df %>%
  mutate(!!sym(surv_gene_col) := as.numeric(!!sym(surv_gene_col)))

# 相同样本取均值
expr_df <- expr_df %>%
  group_by(sample_id) %>%
  summarise(!!sym(surv_gene_col) := mean(!!sym(surv_gene_col), na.rm = TRUE), .groups = "drop")

# 读取生存信息文件
surv_info <- read.csv(surv_info_file)
surv_info <- surv_info %>%
  dplyr::select(slide_id, !!sym(surv_time_col), !!sym(surv_event_col))
colnames(surv_info)[1] <- "sample_id"
surv_info <- surv_info %>%
  mutate(sample_id = str_sub(sample_id, 1, 9))

# 合并表达与生存信息
surv_df <- surv_info %>%
  inner_join(expr_df, by = "sample_id") %>%
  filter(!is.na(!!sym(surv_gene_col)) & !is.na(!!sym(surv_time_col)) & !is.na(!!sym(surv_event_col))) %>%
  distinct(sample_id, .keep_all = TRUE)

message("  -> 匹配到 ", nrow(surv_df), " 个样本")

# 基于中位数分组
median_val <- survminer::surv_cutpoint(surv_df, time = surv_time_col, event = surv_event_col, variables = surv_gene_col)$cutpoint[[1]]
surv_df <- surv_df %>%
  mutate(Group = ifelse(!!sym(surv_gene_col) >= median_val, "High", "Low"))
surv_df$Group <- factor(surv_df$Group, levels = c("Low", "High"))

message("  -> 分组: Low=", sum(surv_df$Group == "Low"), ", High=", sum(surv_df$Group == "High"))

# 拟合生存模型
surv_obj <- Surv(time = surv_df[[surv_time_col]], event = surv_df[[surv_event_col]])
fit <- survfit(surv_obj ~ Group, data = surv_df)

# Log-rank 检验
logrank <- survdiff(surv_obj ~ Group, data = surv_df)
p_val <- 1 - pchisq(logrank$chisq, df = 1)
message("  -> Log-rank p = ", format(p_val, digits = 3, scientific = TRUE))

# 绘制 KM 生存曲线
ggsurvplot(fit, data = surv_df,
           pval = TRUE,
           pval.method = TRUE,
           risk.table = TRUE,
           risk.table.col = "strata",
           palette = c("#2166AC", "#B2182B"),
           xlab = "Time (Days)",
           ylab = "Overall Survival Probability",
           title = paste0("Kaplan-Meier Curve: ", surv_gene_col, " (Median Split)"),
           legend.title = surv_gene_col,
           legend.labs = c("Low", "High"),
           ggtheme = theme_minimal())

ggsave(file.path(out_dir, paste0(out_prefix, "_", surv_gene_col, "_survival.pdf")),
       print(p_surv), width = 8, height = 6)

message("===== 生存分析完成! =====")
message("输出: ", file.path(out_dir, paste0(out_prefix, "_", surv_gene_col, "_survival.pdf")))

#