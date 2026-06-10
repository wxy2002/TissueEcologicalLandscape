ciber_path <- "data/infiltration_estimation_for_tcga.csv"
ciber <- read.csv(ciber_path)
colnames(ciber)[1] <- 'SampleID'
group_path <- "out/ccRCC/train_core_agg.csv"
group_slide_col <- "patient_id"
group_group_col <- "group"

id_candidates <- c("SampleID", "sample", "id", "patient_id", "PatientID", "slide_id")
id_col <- id_candidates[id_candidates %in% names(ciber)][1]
if (is.na(id_col)) stop("CIBERSORT里找不到样本ID列，请手动指定 id_col。")

ciber <- ciber %>%
  dplyr::mutate(
    sample_id = .data[[id_col]] %>%
      str_replace_all("\\.", "-") %>%
      str_replace("-[^-]+$", "") %>%
      substr(1, 12)
  )
ciber_numeric <- ciber %>%
  dplyr::select(sample_id, where(is.numeric)) %>%
  dplyr::select(-any_of(c("P.value", "Correlation", "RMSE"))) %>%
  dplyr::select(sample_id, matches("CIBERSORT\\.ABS$"))
colnames(ciber_numeric) <- str_replace_all(names(ciber_numeric), "_CIBERSORT\\.ABS$", "")
colnames(ciber_numeric) <- str_replace_all(names(ciber_numeric), "\\.", " ")

group_df <- read_csv(group_path, show_col_types = FALSE) %>%
  dplyr::mutate(sample_id = .data[[group_slide_col]] %>% substr(1, 15)) %>%
  dplyr::select(sample_id, group = all_of(group_group_col))

immune_names <- setdiff(names(ciber_numeric), "sample_id")
immune_by_group <- inner_join(group_df, ciber_numeric, by = "sample_id") %>%
  pivot_longer(cols = -c(sample_id, group), names_to = "cell_type", values_to = "score")

mean_by_group <- immune_by_group %>%
  group_by(cell_type, group) %>%
  summarise(mean_score = mean(score, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = group, values_from = mean_score, names_prefix = "mean_")

sig_cells <- immune_by_group %>%
  group_by(cell_type) %>%
  summarise(p = wilcox.test(score ~ group)$p.value, .groups = "drop") %>%
  left_join(mean_by_group, by = "cell_type") %>%
  mutate(significant = ifelse(p < 0.05, "Yes", "No")) %>%
  arrange(p)

mean_col_names <- grep("^mean_", names(sig_cells), value = TRUE)
sig_cells$higher_in <- NA_character_
for (i in seq_len(nrow(sig_cells))) {
  if (sig_cells$significant[i] == "Yes") {
    vals <- sig_cells[i, mean_col_names, drop = TRUE]
    sig_cells$higher_in[i] <- mean_col_names[which.max(vals)]
  }
}

write_csv(sig_cells, file.path("out", "immune_group_comparison.csv"))
message("Saved: out/immune_group_comparison.csv")

# 自动筛选显著细胞类型 (p < 0.05)
target_cells <- sig_cells %>%
  filter(significant == "Yes") %>%
  pull(cell_type)

immune_sig <- immune_by_group %>%
  filter(cell_type %in% target_cells) %>%
  mutate(cell_type = factor(cell_type, levels = target_cells))

# 定义颜色：Low PD / High AREA_CV 为红色，Other 为灰色
group_colors <- c("Low PD & High area_cv" = "#E41A1C", "Other" = "#999999")

# 获取P值用于标注
pvals <- sig_cells %>%
  filter(cell_type %in% target_cells)

p_group <- ggplot(immune_sig, aes(x = group, y = score, fill = group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7, width = 0.5) +
  geom_jitter(width = 0.1, size = 1.2, alpha = 0.3) +
  facet_wrap(~ cell_type, scales = "free_y", ncol = 3) +
  stat_compare_means(method = "wilcox.test", label = "p.format", size = 3.2, vjust = -0.5) +
  scale_fill_manual(values = group_colors) +
  scale_y_continuous(expand = expansion(mult = c(0.15, 0.2))) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
    axis.text.y = element_text(size = 9),
    strip.text = element_text(face = "bold", size = 11),
    panel.grid.minor = element_blank(),
    legend.position = "none"
  ) +
  labs(x = NULL, y = "Estimated immune cell fraction")
p_group

out_group <- file.path("out", "Fig4E_immune_boxplot.pdf")
n_cells <- length(target_cells)
ggsave(out_group, p_group, width = 6, height = 4)
message("Saved: ", out_group)
