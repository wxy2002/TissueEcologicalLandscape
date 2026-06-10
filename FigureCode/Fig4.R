library(tidyverse)
library(survival)
library(survminer)
library(patchwork)

df_tcga <- read.csv('out/ccRCC/train_core_agg.csv')
df_cptac <- read.csv('out/ccRCC/test_core_agg.csv')

cutoff_pd <- survminer::surv_cutpoint(df_tcga, time = 'Time', event = 'Event', variables = 'pd')$cutpoint$cutpoint
cutoff_area_cv <- survminer::surv_cutpoint(df_tcga, time = 'Time', event = 'Event', variables = 'area_cv')$cutpoint$cutpoint

# Assign phenotype groups
df_tcga$phenotype <- ifelse(df_tcga$pd < cutoff_pd & df_tcga$area_cv > cutoff_area_cv,
                            'LPD/HACV', 'Other')
df_cptac$phenotype <- ifelse(df_cptac$pd < cutoff_pd & df_cptac$area_cv > cutoff_area_cv,
                             'LPD/HACV', 'Other')

df_tcga$phenotype <- factor(df_tcga$phenotype, levels = c('Other', 'LPD/HACV'))
df_cptac$phenotype <- factor(df_cptac$phenotype, levels = c('Other', 'LPD/HACV'))

# === Figure 4A: Scatter plot ===
p_4a <- ggplot(df_tcga, aes(x = pd, y = area_cv)) +
  geom_point(aes(color = phenotype), size = 2.5, alpha = 0.8) +
  scale_color_manual(values = c('LPD/HACV' = '#E41A1C', 'Other' = 'grey70'),
                     labels = c('Other', 'Low PD / High AREA_CV')) +
  geom_hline(yintercept = cutoff_area_cv, linetype = 'dashed', color = 'grey40', linewidth = 0.5) +
  geom_vline(xintercept = cutoff_pd, linetype = 'dashed', color = 'grey40', linewidth = 0.5) +
  annotate('text', x = min(df_tcga$pd) + 0.1, y = max(df_tcga$area_cv) - 0.1,
           label = 'High-risk spatial phenotype\nLow PD / High AREA_CV',
           hjust = 0.2, vjust = 3, size = 4, fontface = 'bold', color = '#E41A1C') +
  labs(x = 'PD (More patches)', y = "AREA_CV (Greater patch-size unevenness)", color = NULL) +
  theme_classic(base_size = 13) +
  theme(legend.position = 'bottom',
        legend.text = element_text(size = 10))
p_4a

# === Figure 4B/C: KM curves ===

# HR calculation function
calc_hr <- function(data) {
  cox_fit <- coxph(Surv(Time, Event) ~ phenotype, data = data)
  hr <- exp(coef(cox_fit))
  ci <- exp(confint(cox_fit))
  p_val <- summary(cox_fit)$coefficients[, 'Pr(>|z|)']
  list(hr = hr, lower = ci[1], upper = ci[2], p = p_val)
}

hr_tcga <- calc_hr(df_tcga)
hr_cptac <- calc_hr(df_cptac)

cat(sprintf('TCGA:   HR = %.2f, 95%% CI %.2f-%.2f, P = %.2e\n',
            hr_tcga$hr, hr_tcga$lower, hr_tcga$upper, hr_tcga$p))
cat(sprintf('CPTAC:  HR = %.2f, 95%% CI %.2f-%.2f, P = %.2e\n',
            hr_cptac$hr, hr_cptac$lower, hr_cptac$upper, hr_cptac$p))

# KM plot function
plot_km <- function(data, title, hr_result) {
  fit <- survfit(Surv(Time, Event) ~ phenotype, data = data)

  hr_text <- sprintf('HR = %.2f, 95%% CI %.2f-%.2f\nLog-rank P = %.2e',
                     hr_result$hr, hr_result$lower, hr_result$upper, hr_result$p)

  p <- ggsurvplot(fit, data = data,
                  pval = FALSE,
                  risk.table = TRUE,
                  risk.table.fontsize = 3,
                  risk.table.y.text = FALSE,
                  palette = c('grey60', '#E41A1C'),
                  legend.title = '',
                  legend.labs = c('Other', 'LPD/HACV'),
                  title = title,
                  xlab = 'Time, days',
                  ylab = 'Overall survival probability',
                  ggtheme = theme_classic(base_size = 12))

  p$plot <- p$plot +
    annotate('text', x = 0, y = 0.05, label = hr_text,
             hjust = 0, size = 3.5, fontface = 'italic')

  p
}

p_km_tcga <- plot_km(df_tcga, 'TCGA', hr_tcga)
p_km_cptac <- plot_km(df_cptac, 'CPTAC', hr_cptac)

# Save outputs
ggsave('out/ccRCC/Fig4A_scatter.pdf', p_4a, width = 6, height = 5.5)

# Combined KM plot: side by side, 1:2 height:width ratio
pdf('out/ccRCC/Fig4BC_KM.pdf', width = 15, height = 5)
arrange_ggsurvplots(list(p_km_tcga, p_km_cptac), ncol = 2, nrow = 1)
dev.off()
