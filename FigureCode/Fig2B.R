library(tidyverse)
library(survival)
library(timeROC)
library(survminer)
library(patchwork)

cancer_name <- "ccRCC"

df_train_dl <- read.csv(sprintf('out/%s/train_patient_dl.csv', cancer_name))
df_test_dl <- read.csv(sprintf('out/%s/test_patient_dl.csv', cancer_name))
df_train_xgb <- read.csv(sprintf('out/%s/train_patient_xgb.csv', cancer_name))
df_test_xgb <- read.csv(sprintf('out/%s/test_patient_xgb.csv', cancer_name))

# KM curves for XGB with optimal cutoff
cutpoint <- surv_cutpoint(df_train_xgb, time = "Time", event = "Event", 
                          variables = "risk_patient")
cutoff_val <- cutpoint$cutpoint$cutpoint
cat(sprintf("\nOptimal cutoff: %.4f\n", cutoff_val))

# Function to calculate HR and 95% CI
calc_hr <- function(data, cutoff) {
  data$Risk <- ifelse(data$risk_patient >= cutoff, "High Risk", "Low Risk")
  data$Risk <- factor(data$Risk, levels = c("Low Risk", "High Risk"))
  cox_fit <- coxph(Surv(Time, Event) ~ Risk, data = data)
  hr <- exp(coef(cox_fit))
  ci <- exp(confint(cox_fit))
  p <- summary(cox_fit)$coefficients[, "Pr(>|z|)"]
  list(hr = hr, lower = ci[1], upper = ci[2], p = p)
}

# Calculate HR for train and test
hr_train <- calc_hr(df_train_xgb, cutoff_val)
hr_test <- calc_hr(df_test_xgb, cutoff_val)

cat(sprintf("\n=== HR Results ===\n"))
cat(sprintf("Train: HR = %.2f, 95%% CI %.2f-%.2f, P = %.4f\n", 
            hr_train$hr, hr_train$lower, hr_train$upper, hr_train$p))
cat(sprintf("Test: HR = %.2f, 95%% CI %.2f-%.2f, P = %.4f\n", 
            hr_test$hr, hr_test$lower, hr_test$upper, hr_test$p))

# KM plot function with HR, standardized labels, and colors
plot_km <- function(data, cutoff, title, hr_result) {
  data$Risk <- ifelse(data$risk_patient >= cutoff, "High Risk", "Low Risk")
  data$Risk <- factor(data$Risk, levels = c("Low Risk", "High Risk"))
  fit <- survfit(Surv(Time, Event) ~ Risk, data = data)
  
  # Format HR text
  hr_text <- sprintf("HR = %.2f, 95%% CI %.2f-%.2f\nLog-rank P = %.2e",
                     hr_result$hr, hr_result$lower, hr_result$upper, hr_result$p)
  
  p <- ggsurvplot(fit, data = data,
                  pval = FALSE,
                  risk.table = TRUE,
                  risk.table.fontsize = 3,
                  risk.table.y.text = FALSE,
                  palette = c("#2B6CB0", "#C0392B"),
                  legend.title = "",
                  legend.labs = c("Low Risk", "High Risk"),
                  title = title,
                  xlab = "Time, days",
                  ylab = "Overall survival probability",
                  ggtheme = theme_classic(base_size = 12))
  
  # Add HR annotation
  p$plot <- p$plot + 
    annotate("text", x = 0, y = 0.1, label = hr_text, hjust = 0, size = 3.5, fontface = "italic")
  
  p
}

p_train <- plot_km(df_train_xgb, cutoff_val, sprintf("TCGA Train", cancer_name), hr_train)
p_test <- plot_km(df_test_xgb, cutoff_val, sprintf("CPTAC Test", cancer_name), hr_test)

# 保存图像为pdf，要求高9宽3
pdf(sprintf("out/%s/KM_curves_train.pdf", cancer_name), width = 4, height = 7)
print(p_train)
dev.off()
pdf(sprintf("out/%s/KM_curves_test.pdf", cancer_name), width = 4, height = 7)
print(p_test)
dev.off()

p_test
