library(tidyverse)
library(patchwork)

land_cols_of_interest <- c(
  "shannon_idx", "sidi", "area_cv", "ed", "ai", "cohesion",
  "lpi", "cont", "split", "iji", "pd", "pladj", "enn", "prox",
  "connect"
)

extract_base_metric <- function(feature) {
  str_replace(feature, "_\\d+$", "")
}

read_and_aggregate <- function(path) {
  read.csv(path) %>%
    mutate(Base = extract_base_metric(Feature)) %>%
    filter(Base %in% land_cols_of_interest) %>%
    group_by(Base) %>%
    summarise(MeanSHAP = sum(MeanSHAP), .groups = "drop") %>%
    mutate(MeanSHAP = MeanSHAP / sum(MeanSHAP))
}

df_ccRCC <- read_and_aggregate('out/ccRCC/shap_importance.csv')
df_GBM   <- read_and_aggregate('out/GBM/shap_importance.csv')
df_LSCC  <- read_and_aggregate('out/LSCC/shap_importance.csv')

heatmap_long <- bind_rows(
  df_ccRCC %>% mutate(Cancer = "ccRCC"),
  df_GBM   %>% mutate(Cancer = "GBM"),
  df_LSCC  %>% mutate(Cancer = "LSCC")
)

metric_groups_all <- list(
  "Fragmentation heterogeneity"    = c("pd", "split"),
  "Size heterogeneity" = c("area_cv", "lpi"),
  "Boundary complexity" = c("ed", "iji"),
  "Aggregation"                    = c("cont", "ai", "pladj", "cohesion"),
  "Connectivity"                   = c("enn", "prox", "connect"),
  "Diversity"                      = c("shannon_idx", "sidi")
)

present_metrics <- unique(heatmap_long$Base)

metric_groups <- lapply(metric_groups_all, function(m) intersect(m, present_metrics))
metric_groups <- metric_groups[lengths(metric_groups) > 0]

display_name <- c(
  shannon_idx = "SHDI", sidi = "SIDI", area_cv = "AREA_CV",
  ed = "ED", ai = "AI", cohesion = "COHESION", lpi = "LPI",
  cont = "CONT", split = "SPLIT", iji = "IJI", pd = "PD",
  pladj = "PLADJ", enn = "ENN", prox = "PROX", connect = "CONNECT"
)

group_colors <- c(
  "Fragmentation heterogeneity"    = "#E41A1C",
  "Size heterogeneity" = "#FF7F00",
  "Boundary complexity" = "#984EA3",
  "Aggregation"                    = "#377EB8",
  "Connectivity"                   = "#4DAF4A",
  "Diversity"                      = "#00CDCD"
)

group_order <- names(metric_groups)
metric_y_order <- unlist(metric_groups, use.names = FALSE)

heatmap_long <- heatmap_long %>%
  mutate(
    Group = factor(
      names(metric_groups)[match(Base, unlist(metric_groups))],
      levels = group_order
    ),
    Label = display_name[Base],
    Base_f = factor(Base, levels = metric_y_order)
  )

ann_df <- data.frame(
  y     = factor(metric_y_order, levels = metric_y_order),
  Group = factor(rep(group_order, lengths(metric_groups)), levels = group_order)
)

p_ann <- ggplot(ann_df, aes(x = 1, y = y, fill = Group)) +
  geom_tile(width = 1) +
  scale_y_discrete(limits = metric_y_order, labels = NULL, expand = expansion(add = 0.6)) +
  scale_fill_manual(values = group_colors, name = "Ecological category") +
  labs(x = NULL, y = NULL) +
  theme_void() +
  theme(
    legend.position  = "right",
    legend.direction = "vertical",
    legend.text      = element_text(size = 9, family = "sans"),
    legend.title     = element_text(size = 10, face = "bold", family = "sans"),
    legend.key.size  = unit(0.45, "cm"),
    plot.margin      = margin(3, 0, 3, 0)
  )

p_heat <- ggplot(heatmap_long, aes(x = Cancer, y = Base_f, fill = MeanSHAP)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.3f", MeanSHAP)), size = 3.5, family = "sans") +
  scale_y_discrete(limits = metric_y_order, labels = display_name, expand = expansion(add = 0.6)) +
  scale_fill_gradient(low = "white", high = "firebrick3") +
  labs(x = NULL, y = NULL, fill = "MeanSHAP") +
  theme_minimal() +
  theme(
    axis.text.x  = element_text(size = 11, family = "sans"),
    axis.text.y  = element_text(size = 10, family = "sans"),
    axis.ticks.y = element_blank(),
    panel.grid   = element_blank(),
    plot.margin  = margin(3, 5, 3, 5),
    legend.title = element_text(size = 10, face = "bold", family = "sans"),
    legend.text  = element_text(size = 9, family = "sans")
  )

fig3a <- p_ann + p_heat + plot_layout(widths = c(0.06, 1), guides = "collect")
fig3a

top5_df <- heatmap_long %>%
  group_by(Cancer, Base) %>%
  summarise(MeanSHAP = mean(MeanSHAP), Group = first(Group), .groups = "drop") %>%
  group_by(Cancer) %>%
  slice_max(MeanSHAP, n = 5) %>%
  arrange(Cancer, desc(MeanSHAP)) %>%
  mutate(
    Label = display_name[Base],
    y_lbl = factor(paste(Cancer, Label, sep = "_"), levels = rev(paste(Cancer, Label, sep = "_")))
  )

fig3b <- ggplot(top5_df, aes(x = MeanSHAP, y = y_lbl, color = Group)) +
  geom_segment(aes(xend = 0, yend = y_lbl), linewidth = 0.6) +
  geom_point(size = 3.5) +
  facet_wrap(~ Cancer, scales = "free_y", nrow = 1) +
  scale_y_discrete(labels = setNames(top5_df$Label, top5_df$y_lbl)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.1))) +
  scale_color_manual(values = group_colors, guide = "none") +
  labs(x = "Mean |SHAP|", y = NULL) +
  theme_minimal() +
  theme(
    strip.text         = element_text(size = 11, face = "bold", family = "sans"),
    axis.text.y        = element_text(size = 9, family = "sans"),
    axis.text.x        = element_text(size = 9, family = "sans"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    plot.margin        = margin(5, 10, 5, 10)
  )
fig3b
