library(purrr)
library(readr)
library(tidyverse)

# Get a list of all .RDS files in the directory
# files <- list.files(path = "data/progressive_sampling/", pattern = "\\.RDS$", full.names = TRUE)
files <- list.files(path = "../data/progressive_sampling/", pattern = "\\.RDS$", full.names = TRUE)

# Read each file and combine them into a single data frame
results_df <- map_df(files, read_rds)

results_df <- results_df %>%
    group_by(sample_size, model) %>%
    mutate(run = row_number()) %>%
    ungroup()

# Calculate lambda for each sample size
lambda_by_size <- results_df %>%
  # Group by sample size
  group_by(sample_size) %>%
  # Calculate lambda = cov(ground_truth, ground_truth - yhat_adv_corr)/var(ground_truth - yhat_adv_corr)
  summarize(
    lambda = cov(
      estimate[model == "tree_cover"], 
      estimate[model == "tree_cover"] - estimate[model == "yhat_adv_corr"]
    ) / var(
      estimate[model == "tree_cover"] - estimate[model == "yhat_adv_corr"]
    )
  ) %>%
  # Arrange by sample size for better readability
  arrange(desc(sample_size))
# Create a new model called adv_tuned using the lambda values
# First, extract the ground truth data
ground_truth_data <- results_df %>%
  filter(model == "tree_cover") %>%
  select(sample_size, run, estimate) %>%
  rename(tree_cover_estimate = estimate)

# Extract the adversarial model data
adv_corr_data <- results_df %>%
  filter(model == "yhat_adv_corr") %>%
  select(sample_size, run, estimate) %>%
  rename(adv_corr_estimate = estimate)

# Combine with lambda values and calculate tuned estimates
tuned_data <- ground_truth_data %>%
  left_join(adv_corr_data, by = c("sample_size", "run")) %>%
  left_join(lambda_by_size, by = "sample_size") %>%
  mutate(
    adv_tuned_estimate = (1 - lambda) * tree_cover_estimate + 
                         lambda * adv_corr_estimate,
    model = "adv_tuned",
    estimate = adv_tuned_estimate
  ) %>%
  select(sample_size, run, model, estimate)

# Add the tuned model back to the original results
results_df <- bind_rows(results_df, tuned_data)





results_df %>% 
  filter(sample_size == 10000,
         model %in% c(
            # "tree_cover", 
            "yhat_lr", 
            # "yhat_adv", 
            # "yhat_adv_corr", 
            # "yhat_lr_biascorrect", 
            "imputation", 
            # "ground_truth", 
            # "yhat_adv_mg", 
                      "ptd", 
                    #   "ipd_ppi_plusplus",
                    #   "ipd_postpi_analytic",
                      "ipd_postpi_boot",
                      "ipd_ppi",
                    #   "ipd_pspa",
                      
                      # "yhat_lr", (already included above)
                      "adv_tuned")) %>%
  ggplot(aes(x = estimate, color = model)) +
  geom_density(linewidth = 0.85, alpha = 0) +
  scale_color_discrete(labels = c("yhat_lr" = "Baseline", 
                                  "yhat_adv" = "SLR Adversary", 
                                  "yhat_adv_corr" = "Correlational adversary", 
                                  "yhat_lr_biascorrect" = "Bias Correction",
                                  "imputation" = "Multiple Imputation",
                                  "ground_truth" = "Ground truth Only",
                                  "yhat_adv_mg" = "Adversarial MG",
                                  "ptd" = "PTD",
                                  "ipd_ppi_plusplus" = "PPI++",
                                  "ipd_postpi_analytic" = "PostPI Analytic",
                                  "ipd_postpi_boot" = "PostPI Boot",
                                  "ipd_ppi" = "PPI",
                                  "ipd_pspa" = "PSPA",
                                  "tree_cover" = "Tree cover",
                                  "adv_tuned" = "Adversarial Tuned"),
                       name = "") +
  theme_minimal() + theme(text = element_text(size = 20),
        legend.position = c(0.3, 0.9))

ggsave(filename = "figures/baseline_vs_adversarial_progressive_all_after10k.png", width = 8, height = 8)

results_df_summary <- results_df %>%
  # Filter to include all available models
  filter(model %in% c("yhat_lr", "yhat_adv", "yhat_adv_corr", "yhat_lr_biascorrect", 
                      "yhat_adv_mg", "imputation", "ground_truth", "tree_cover",
                      "ipd_postpi_analytic", "ipd_postpi_boot", "ipd_ppi", 
                      "ipd_ppi_plusplus", "ipd_pspa", "ptd", "adv_tuned")) %>%
  group_by(sample_size, model) %>%
  summarise(lower_bound = mean(estimate) - 2*sd(estimate), 
            upper_bound = mean(estimate) + 2*sd(estimate), 
            mean_estimate = mean(estimate), 
            .groups = 'drop') %>%
  # Create factor with all models in desired order
  mutate(modelfactor = factor(model, levels = c("yhat_lr", "yhat_adv", "yhat_adv_corr", 
                                               "yhat_lr_biascorrect", "yhat_adv_mg", 
                                               "imputation", "ground_truth", "tree_cover",
                                               "ipd_postpi_analytic", "ipd_postpi_boot", 
                                               "ipd_ppi", "ipd_ppi_plusplus", "ipd_pspa", "ptd", "adv_tuned")))

results_df_adv <- results_df %>%
  # Filter to include all available models
  filter(model %in% c("yhat_lr", "yhat_adv", "yhat_adv_corr", "yhat_lr_biascorrect", 
                      "yhat_adv_mg", "imputation", "ground_truth", "tree_cover",
                      "ipd_postpi_analytic", "ipd_postpi_boot", "ipd_ppi", 
                      "ipd_ppi_plusplus", "ipd_pspa", "ptd", "adv_tuned")) %>%
  # Create factor with all models in desired order
  mutate(modelfactor = factor(model, levels = c("yhat_lr", "yhat_adv", "yhat_adv_corr", 
                                               "yhat_lr_biascorrect", "yhat_adv_mg", 
                                               "imputation", "ground_truth", "tree_cover",
                                               "ipd_postpi_analytic", "ipd_postpi_boot", 
                                               "ipd_ppi", "ipd_ppi_plusplus", "ipd_pspa", "ptd", "adv_tuned")))


ggplot() +
  geom_line(data = filter(results_df_adv, model %in% c("yhat_lr", "adv_tuned", "ipd_ppi", "ptd", "ipd_postpi_boot", "imputation")), 
            aes(x = sample_size, y = estimate, color = model, group = run), alpha = 0.1) + # Add a line for each run, but make it very faint
  geom_line(data = filter(results_df_summary, model %in% c("yhat_lr", "adv_tuned", "ipd_ppi", "ptd", "ipd_postpi_boot", "imputation")), 
            aes(x = sample_size, y = mean_estimate, color = model)) +
  geom_line(data = filter(results_df_summary, model %in% c("yhat_lr", "adv_tuned", "ipd_ppi", "ptd", "ipd_postpi_boot", "imputation")), 
            aes(x = sample_size, y = lower_bound, color = model), size = 1.5) + # Add lower bound of 2 standard deviations
  geom_line(data = filter(results_df_summary, model %in% c("yhat_lr", "adv_tuned", "ipd_ppi", "ptd", "ipd_postpi_boot", "imputation")), 
            aes(x = sample_size, y = upper_bound, color = model), size = 1.5) + # Add upper bound of 2 standard deviations
  labs(x = "Sample Size", y = "Estimated relationship") +
  facet_wrap(~modelfactor, ncol = 2, 
             labeller = as_labeller(c(yhat_lr = "Baseline Model",
                                      adv_tuned = "Adversarial Tuned",
                                      ipd_ppi = "PPI",
                                      ptd = "PTD",
                                      ipd_postpi_boot = "PostPI Boot",
                                      imputation = "Multiple Imputation"))) +
  theme_minimal() + theme(legend.position = "none", text = element_text(size = 20)) +
  ylim(-0.05, 0.05)
ggsave(filename = "figures/baseline_vs_adversarial_progressive_6models.png", width = 8, height = 8)






# Create a summary dataframe with mean and confidence intervals for each model across multiple sample sizes
summary_df <- results_df %>% 
  filter(sample_size %in% c(600, 1800, 5000),
         model %in% c(
            # "tree_cover", 
            "yhat_lr", 
            # "yhat_adv", 
            # "yhat_adv_corr", 
            # "yhat_lr_biascorrect", 
            "imputation", 
            # "ground_truth", 
            # "yhat_adv_mg", 
                      "ptd", 
                    #   "ipd_ppi_plusplus",
                    #   "ipd_postpi_analytic",
                      "ipd_postpi_boot",
                      "ipd_ppi",
                    #   "ipd_pspa",
                      
                      # "yhat_lr", (already included above)
                      "adv_tuned")) %>%
  group_by(sample_size, model) %>%
  summarise(
    mean_estimate = mean(estimate, na.rm = TRUE),
    lower_ci = mean_estimate - 1.96 * sd(estimate, na.rm = TRUE) / sqrt(n()),
    upper_ci = mean_estimate + 1.96 * sd(estimate, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  # Convert sample_size to factor for faceting
  mutate(sample_size = factor(sample_size, levels = c(600, 1800, 5000)))

# For each sample size, reorder models by mean estimate
summary_df <- summary_df %>%
  group_by(sample_size) %>%
  mutate(model = factor(model, levels = model[order(mean_estimate)])) %>%
  ungroup()

# Create the faceted plot with models on y-axis and estimates on x-axis
model_ci_plot <- ggplot(summary_df, aes(x = mean_estimate, y = model)) +
  geom_vline(xintercept = 0, color = "red", linetype = "dotted", linewidth = 0.8) +
  geom_point(size = 3) +
  geom_segment(aes(x = lower_ci, xend = upper_ci, y = model, yend = model), linewidth = 0.5) +
  facet_wrap(~ sample_size, ncol = 3, labeller = labeller(sample_size = function(x) paste0("Sample Size: ", x))) +
  scale_y_discrete(labels = c(
    "yhat_lr" = "Baseline", 
    "yhat_adv" = "SLR Adversary",
    "yhat_adv_corr" = "Correlational adversary", 
    "yhat_lr_biascorrect" = "Bias Correction",
    "imputation" = "Multiple Imputation",
    "forest" = "Forest",
    "yhat_adv_mg" = "Adversarial MG",
    "ptd" = "PTD",
    "ipd_ppi_plusplus" = "PPI++",
    "ipd_postpi_analytic" = "PostPI Analytic",
    "ipd_postpi_boot" = "PostPI Boot",
    "ipd_ppi" = "PPI",
    "ipd_pspa" = "PSPA",
    "adv_tuned" = "Adversarial Tuned"
  )) +
  labs(
    x = "Estimate with 95% Confidence Interval",
    y = ""
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 16),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    strip.text = element_text(size = 14, face = "bold")
  ) +
  annotate("text", x = ground_truth_mean, y = 0.5, label = "Ground Truth", 
           color = "red", hjust = -0.2, vjust = -0.5, size = 4)

model_ci_plot

ggsave(filename = "../figures/model_estimates_samplesize_ci_cat.png", width = 12, height = 12)


# Create a summary dataframe with mean and confidence intervals for each model across multiple sample sizes
summary_df <- results_df %>% 
  filter(sample_size %in% c(600, 1800, 3000, 5000, 7000, 10000),
         model %in% c(
            # "tree_cover", 
            # "yhat_lr", 
            "yhat_adv", 
            "yhat_adv_corr", 
            # "yhat_lr_biascorrect", 
            # "imputation", 
            # "ground_truth", 
            # "yhat_adv_mg", 
                      # "ptd", 
                    #   "ipd_ppi_plusplus",
                    #   "ipd_postpi_analytic",
                      # "ipd_postpi_boot",
                      # "ipd_ppi",
                    #   "ipd_pspa",
                      
                      # "yhat_lr", (already included above)
                      "adv_tuned")) %>%
  group_by(sample_size, model) %>%
  summarise(
    mean_estimate = mean(estimate, na.rm = TRUE),
    lower_ci = mean_estimate - 1.96 * sd(estimate, na.rm = TRUE) / sqrt(n()),
    upper_ci = mean_estimate + 1.96 * sd(estimate, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Create a mapping for model labels
model_labels <- c(
  "yhat_lr" = "Baseline", 
  "yhat_adv" = "SLR Adversary",
  "yhat_adv_corr" = "Correlational adversary", 
  "adv_tuned" = "Adversarial Tuned"
  # "yhat_adv_mg" = "Adversarial MG"
)

# Create the plot with sample size on x-axis and estimates on y-axis
compare_adversaries_plot <- ggplot(summary_df, aes(x = sample_size, y = mean_estimate, color = model)) +
  geom_hline(yintercept = 0, color = "red", linetype = "dotted", linewidth = 0.8) +
  # geom_line(linewidth = 1) +
  geom_point(size = 3, position = position_dodge(width = 300)) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 200, position = position_dodge(width = 300)) +
  scale_color_brewer(palette = "Set1", labels = model_labels) +
  scale_fill_brewer(palette = "Set1", labels = model_labels) +
  geom_hline(yintercept = ground_truth_mean, color = "red", linetype = "dashed", linewidth = 0.8) +
  labs(
    x = "Sample Size",
    y = "Estimate with 95% Confidence Interval",
    color = "Model",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 16),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(face = "bold")
  ) +
  annotate("text", x = max(summary_df$sample_size), y = ground_truth_mean, 
           label = "Ground Truth", color = "red", hjust = 1, vjust = -0.5, size = 4)

compare_adversaries_plot

ggsave(
  filename = "../figures/compare_adversaries_plot.png",
  plot = compare_adversaries_plot,
  width = 10,
  height = 7,
  dpi = 300
)
