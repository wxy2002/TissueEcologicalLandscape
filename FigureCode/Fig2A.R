library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

# Raw data from Table 2
raw <- data.frame(
  Cancer = c("ccRCC","ccRCC","GBM","GBM","LSCC","LSCC","Mean","Mean"),
  Model  = c("TEL","CLAM","TEL","CLAM","TEL","CLAM","TEL","CLAM"),
  Train_Cindex = c("0.761(0.726-0.798)","0.846(0.817-0.873)",
                    "0.666(0.628-0.703)","0.734(0.699-0.766)",
                    "0.742(0.703-0.781)","0.635(0.590-0.677)",
                    "0.723","0.738"),
  Test_Cindex  = c("0.669(0.580-0.763)","0.704(0.606-0.793)",
                    "0.573(0.516-0.631)","0.542(0.479-0.601)",
                    "0.632(0.532-0.728)","0.602(0.491-0.704)",
                    "0.625","0.616"),
  Train_AUC    = c("0.793(0.746-0.836)","0.876(0.842-0.905)",
                    "0.710(0.660-0.759)","0.805(0.763-0.842)",
                    "0.778(0.729-0.823)","0.660(0.604-0.715)",
                    "0.760","0.780"),
  Test_AUC     = c("0.694(0.591-0.795)","0.708(0.582-0.822)",
                    "0.612(0.527-0.693)","0.580(0.485-0.665)",
                    "0.675(0.555-0.787)","0.628(0.499-0.754)",
                    "0.660","0.639"),
  stringsAsFactors = FALSE
)

parse_ci <- function(x) {
  x <- trimws(x)
  if (grepl("(", x, fixed = TRUE)) {
    val <- as.numeric(sub("\\(.*", "", x))
    ci  <- sub(".*\\((.*)\\).*", "\\1", x)
    parts <- strsplit(ci, "-")[[1]]
    c(val = val, lower = as.numeric(parts[1]), upper = as.numeric(parts[2]))
  } else {
    c(val = as.numeric(x), lower = NA, upper = NA)
  }
}

# Build tidy long data
build_df <- function(col_name) {
  parsed <- lapply(raw[[col_name]], parse_ci)
  data.frame(
    Cancer = raw$Cancer,
    Model  = raw$Model,
    val    = sapply(parsed, `[[`, "val"),
    lower  = sapply(parsed, `[[`, "lower"),
    upper  = sapply(parsed, `[[`, "upper"),
    stringsAsFactors = FALSE
  )
}

df <- bind_rows(
  build_df("Train_Cindex") %>% mutate(Set = "Train", Metric = "C-index"),
  build_df("Test_Cindex")  %>% mutate(Set = "Test",  Metric = "C-index"),
  build_df("Train_AUC")    %>% mutate(Set = "Train", Metric = "AUC"),
  build_df("Test_AUC")     %>% mutate(Set = "Test",  Metric = "AUC")
) %>%
  mutate(
    Cancer = factor(Cancer, levels = c("ccRCC", "GBM", "LSCC", "Mean")),
    Model  = factor(Model,  levels = c("TEL", "CLAM")),
    Set    = factor(Set,    levels = c("Train", "Test")),
    Metric = factor(Metric, levels = c("C-index", "AUC"))
  )

color_tel  <- "#2166AC"
color_clam <- "#999999"

build_bar <- function(metric_label) {
  p <- ggplot(
    df %>% filter(Metric == metric_label),
    aes(x = Cancer, y = val, fill = Model)
  ) +
    geom_col(
      position = position_dodge(width = 0.75),
      width = 0.65, color = "white", linewidth = 0.2
    ) +
    geom_errorbar(
      aes(ymin = lower, ymax = upper),
      position = position_dodge(width = 0.75),
      width = 0.2, linewidth = 0.4
    ) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
    facet_wrap(~ Set, nrow = 1) +
    scale_fill_manual(values = c("TEL" = color_tel, "CLAM" = color_clam)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(x = NULL, y = metric_label) +
    theme_classic(base_size = 11) +
    theme(
      legend.position  = "top",
      legend.title     = element_blank(),
      strip.text       = element_text(face = "bold", size = 12),
      axis.text.x      = element_text(size = 10, face = "bold"),
      axis.title.y     = element_text(face = "bold"),
      panel.spacing    = unit(1.2, "lines"),
      plot.title       = element_text(hjust = 0.5, face = "bold")
    )
  return(p)
}

p_cindex <- build_bar("C-index") + ggtitle("C-index")
p_auc    <- build_bar("AUC")    + ggtitle("AUC")

p_combined <- p_cindex / p_auc + plot_layout(guides = "collect") &
  theme(legend.position = "top")

ggsave("out/Fig2A_Cindex_AUC_barplot.pdf", p_combined, width = 10, height = 7)
ggsave("out/Fig2A_Cindex_AUC_barplot.png", p_combined, width = 10, height = 7, dpi = 300)

cat("Figure saved to out/Fig2A_Cindex_AUC_barplot.pdf and .png\n")
