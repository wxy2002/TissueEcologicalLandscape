# -----免疫浸润与景观指标相关性分析-----
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(ggpubr)
})

ciber_path <- "data/infiltration_estimation_for_tcga.csv"
cancers <- c("ccRCC", 'GBM', 'LSCC')

land_cols_of_interest <- c(
  "shannon_idx", "sidi", "area_cv", "ed", "ai", "cohesion",
  "lpi", "cont", "split", "iji", "pd", "pladj", "enn", "prox",
  "connect"
)

metric_groups_all <- list(
  "Fragmentation heterogeneity" = c("pd", "split"),
  "Size heterogeneity"          = c("area_cv", "lpi"),
  "Boundary complexity"         = c("ed", "iji"),
  "Aggregation"                 = c("cont", "ai", "pladj", "cohesion"),
  "Connectivity"                = c("enn", "prox", "connect"),
  "Diversity"                   = c("shannon_idx", "sidi")
)

display_name <- c(
  shannon_idx = "SHDI", sidi = "SIDI", area_cv = "AREA_CV",
  ed = "ED", ai = "AI", cohesion = "COHESION", lpi = "LPI",
  cont = "CONT", split = "SPLIT", iji = "IJI", pd = "PD",
  pladj = "PLADJ", enn = "ENN", prox = "PROX", connect = "CONNECT"
)

group_colors <- c(
  "Fragmentation heterogeneity" = "#E41A1C",
  "Size heterogeneity"          = "#FF7F00",
  "Boundary complexity"         = "#984EA3",
  "Aggregation"                 = "#377EB8",
  "Connectivity"                = "#4DAF4A",
  "Diversity"                   = "#00CDCD"
)

immune_groups_all <- list(
  "Lymphoid" = c(
    "B cells naive", "B cells memory",
    "T cells CD4+ naive", "T cells CD4+ memory resting", "T cells CD4+ memory activated",
    "T cells CD4+ follicular helper", "T cells regulatory (Tregs)",
    "T cells gamma delta", "T cells CD8+",
    "NK cells resting", "NK cells activated",
    "Plasma cells"
  ),
  "Myeloid" = c(
    "Monocytes",
    "Macrophages M0", "Macrophages M1", "Macrophages M2",
    "Dendritic cells resting", "Dendritic cells activated",
    "Mast cells resting", "Mast cells activated"
  ),
  "Granulocytes" = c(
    "Neutrophils", "Eosinophils"
  )
)
ciber <- read.csv(ciber_path)
colnames(ciber)[1] <- 'SampleID'

id_candidates <- c("SampleID", "sample", "id", "patient_id", "PatientID", "slide_id")
id_col <- id_candidates[id_candidates %in% names(ciber)][1]
if (is.na(id_col)) {
  stop("CIBERSORT里找不到样本ID列，请手动指定 id_col。")
}

ciber <- ciber %>%
  mutate(
    sample_id = .data[[id_col]] %>%
      str_replace_all("\\.", "-") %>%
      str_replace("-[^-]+$", "") %>%
      substr(1, 12)
  )

ciber_numeric <- ciber %>%
  select(sample_id, where(is.numeric)) %>%
  select(-any_of(c("P.value", "Correlation", "RMSE"))) %>%
  dplyr::select(sample_id, matches("CIBERSORT\\.ABS$"))

# 去除列名的后缀
colnames(ciber_numeric) <- str_replace_all(names(ciber_numeric), "_CIBERSORT\\.ABS$", "")
colnames(ciber_numeric) <- str_replace_all(names(ciber_numeric), "\\.", " ")

all_merged <- bind_rows(lapply(cancers, function(cn) {
  land_path <- file.path("data", cn, paste0("TCGA_", cn, "_OS.csv"))
  if (!file.exists(land_path)) {
    warning(cn, " file not found, skipping")
    return(NULL)
  }
  land <- read_csv(land_path, show_col_types = FALSE) %>%
    mutate(sample_id = slide_id %>% substr(1, 12))
  merged <- inner_join(land, ciber_numeric, by = "sample_id")
  if (nrow(merged) < 5) {
    warning(cn, ": merged < 5 samples, skipping")
    return(NULL)
  }
  merged$cancer_type <- cn
  merged
}))

if (is.null(all_merged) || nrow(all_merged) < 5) {
  stop("合并样本太少，无法分析。")
}

immune_names <- setdiff(names(ciber_numeric), "sample_id")
immune_names_clean <- str_replace_all(immune_names, "\\.", " ")

all_cor_long <- bind_rows(lapply(cancers, function(cn) {
  sub <- all_merged %>% filter(cancer_type == cn)
  if (nrow(sub) < 5) return(NULL)

  land_sub <- sub %>% select(any_of(land_cols_of_interest))
  immune_sub <- sub %>% select(any_of(immune_names))
  names(immune_sub) <- str_replace_all(names(immune_sub), "\\.", " ")

  cor_sub <- cor(land_sub, immune_sub, use = "pairwise.complete.obs", method = "spearman")
  p_sub <- matrix(NA_real_, nrow = ncol(land_sub), ncol = ncol(immune_sub))
  rownames(p_sub) <- colnames(land_sub)
  colnames(p_sub) <- colnames(immune_sub)

  for (i in seq_len(ncol(land_sub))) {
    for (j in seq_len(ncol(immune_sub))) {
      ct <- suppressWarnings(cor.test(land_sub[[i]], immune_sub[[j]], method = "spearman"))
      p_sub[i, j] <- ct$p.value
    }
  }

  as.data.frame(as.table(cor_sub)) %>%
    rename(var1 = Var1, var2 = Var2, cor = Freq) %>%
    left_join(
      as.data.frame(as.table(p_sub)) %>%
        rename(var1 = Var1, var2 = Var2, p = Freq),
      by = c("var1", "var2")
    ) %>%
    mutate(cancer_type = cn)
}))

sig_long <- all_cor_long %>%
  filter(!is.na(cor)) %>%
  mutate(
    sig = p < 0.05,
    label = case_when(
      p < 0.001 ~ "***",
      p < 0.01  ~ "**",
      p < 0.05  ~ "*",
      TRUE ~ ""
    )
  )

present_metrics <- unique(sig_long$var1)
metric_groups <- lapply(metric_groups_all, function(m) intersect(m, present_metrics))
metric_groups <- metric_groups[lengths(metric_groups) > 0]
group_order <- names(metric_groups)
metric_x_order <- unlist(metric_groups, use.names = FALSE)

present_immune <- unique(sig_long$var2)
immune_groups <- lapply(immune_groups_all, function(m) intersect(m, present_immune))
immune_groups <- immune_groups[lengths(immune_groups) > 0]
immune_group_order <- names(immune_groups)
immune_y_order <- unlist(immune_groups, use.names = FALSE)

sig_long <- sig_long %>%
  mutate(
    Group = factor(
      names(metric_groups)[match(var1, unlist(metric_groups))],
      levels = group_order
    ),
    ImmuneGroup = factor(
      names(immune_groups)[match(var2, unlist(immune_groups))],
      levels = immune_group_order
    ),
    Label = display_name[var1],
    var1_f = factor(var1, levels = metric_x_order),
    var2_f = factor(var2, levels = immune_y_order)
  )

sig_long$cancer_type <- factor(sig_long$cancer_type, levels = cancers)

p_summary <- ggplot(sig_long, aes(var1_f, var2)) +
  geom_tile(aes(fill = cor), width = 1, height = 1, color = "black", linewidth = 0.1) +
  geom_text(aes(label = label), size = 3.5, color = "gray20", fontface = "bold") +
  scale_x_discrete(labels = display_name, expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  scale_fill_gradient2(low = "#2c7bb6", mid = "white", high = "#d7191c", midpoint = 0, limits = c(-0.3, 0.3)) +
  facet_grid(~ cancer_type, scales = "free", space = "free") +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8),
    axis.text.y = element_text(size = 8),
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold", size = 11),
    plot.margin = margin(5, 10, 5, 15),
    plot.caption = element_text(hjust = 0, size = 9)
  ) +
  labs(
    x = "Landscape Metric", y = "Immune Cell", fill = "Spearman r",
    caption = "* p < 0.05, ** p < 0.01, *** p < 0.001"
  )

out_file <- file.path("out", "heatmap_summary_sig.pdf")
ggsave(out_file, p_summary, width = 20, height = 10)
message("Saved: ", out_file)