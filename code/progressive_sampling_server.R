library(tidyverse); library(doParallel); library(boot); #library(tictoc)
library(doParallel)

# Determine the number of cores based on SLURM environment variable
num_slurm_cpus <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
if (is.na(num_slurm_cpus) || num_slurm_cpus < 1) {
  cl <- makeCluster(1) # Default to 1 core if not in SLURM or var is invalid
} else {
  cl <- makeCluster(num_slurm_cpus)
}

# Hyperparameters
n_points <- 20000
num_runs <- 5  # Number of runs for each size
sizes <- c(seq(10000,3000,by = -1000),seq(2400, 300, by = -300))
print(c(n_points, num_runs, sizes))
# setwd("code/analysis/progressive_sampling/")
source("../code/modeling/debiaser.R")

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Assign the first argument to 'output_file'
output_file <- args[1]


## Bootstrap bias and se of bias ----
bootstrap_estimate_se <- function(data, nu_var) {
  # Define the statistic function
  ratio_stat <- function(data, indices) {
    # Allows boot to select sample
    d <- data[indices, ]
    # Compute the covariance and variance
    cov_nu_D <- cov(d[[nu_var]], d$D)
    var_D <- var(d$D)
    # Return the ratio
    return(cov_nu_D / var_D)
  }

  # Apply the bootstrapping method with R = 1000 bootstrap replications
  results <- boot(data = data, statistic = ratio_stat, R = 1000)

  # Estimate is the original statistic computed with all data
  estimate <- ratio_stat(data, 1:nrow(data))

  # Standard error from bootstrapping
  se <- sd(results$t)

  # Return the estimate and bootstrapped standard error
  return(list(estimate = estimate, std_err = se))
}


# Get Bastin, drop slope
# get_data <- function(n_points = 20000){
#     # Get Bastin, drop slope
#     df <- read_csv("../../../data/processed/w_africa_l7_model_prepped.csv") %>% rename(old_slope = slope)


#     # Take a random sample
#     rs_vals <- slice_sample(df, n= n_points, replace = F) 

#     # Simulate slope
#     slope <- rpois(n_points,lambda = 1)

#     # Make treatment dependent on slope (are you near a road)
#     D <- rbinom(n_points, 1, pmax(1 - slope/4, 0))

#     # get a data frame of the unique slope values, ordered
#     tc_rank<-data.frame(
#     tree_cover = as.numeric(unique(names(table(rs_vals$tree_cover)))), 
#     rank = 1:length(as.numeric(unique(names(table(rs_vals$tree_cover))))))

#     # Get the rank of the tree cover of each point
#     dat <- data.frame(slope, D, rs_vals) %>% left_join(
#     tc_rank
#     )

#     # If slope is higher, add the slope to the rank, get rid of remote sensing variables
#     # Now if you have a higher slope, we kept your old tree cover, but made a new variable
#     # to help us choose places which higher tree cover to get RS data from
#     no_rs <- dat %>% select(slope, D, tree_cover, rank) %>%
#     mutate(new_rank = rank + slope) %>%
#     mutate(new_rank = ifelse(new_rank > 14, 14, new_rank)) %>%
#     rename(old_tree_cover = tree_cover) %>% 
#     left_join(tc_rank %>% rename(new_rank = rank))%>% arrange(tree_cover)

#     (tbl<-table(no_rs$tree_cover))

#     new_df <- map2_df(names(tbl), tbl, ~{
#     df %>% 
#         filter(tree_cover == .x) %>%
#         slice_sample(n = .y, replace = T)
#     }) %>% arrange(tree_cover) %>% 
#     select(B1_median:EVI_75perc) %>% bind_cols(no_rs) %>% 
#     rename(apparent_tree_cover = tree_cover,
#             tree_cover = old_tree_cover)
    
#     return(new_df)}

new_df <- read_rds("../data/power_analysis/biased_data.RDS")
   print("read data")
ssize <- nrow(new_df)
new_df <- new_df %>% sample_n(size = ssize, replace = FALSE)

registerDoParallel(cl)
#tic()
results_df <- foreach(size = sizes, .packages = c("keras3", "tensorflow", "boot", "tidyverse", "doParallel", "mice", "PTDBoot", "ipd"), .combine = 'rbind') %dopar% {
  # Initialize a data frame to store the results of each run
  run_results_df <- data.frame()

  for (run in 1:num_runs) {
    #randomly sort the order of new_df
    sample_data <- new_df[1:size, ]
    test_data <- new_df[(size + 1):nrow(new_df), ]
    sample_data <- sample_data[sample(1:nrow(sample_data), nrow(sample_data)), ]  # Shuffle rows
    sample_data$fold <- 3
    sample_data[1:round(nrow(sample_data) / 3), "fold"] <- 1
    sample_data[(round(nrow(sample_data) / 3) + 1):round(2 * nrow(sample_data) / 3), "fold"] <- 2
    table(sample_data$fold)

    ### Only train on satellite data information to predict forest 
    (train_vars <- sort(grep("B[1-8]_|NDVI_|NDBI_|EVI_", names(new_df), value = TRUE)))

    treatment <- model.matrix(~D, new_df)

    sample_data$yhat_lr <- train_all_folds(x = sample_data[, c("fold", "tree_cover", "D", train_vars)],
                                          y_name =  "tree_cover", treat_name = "D", y_class = "continuous",
                                          debias_wt = 0, epochs = 50, model = "deep_nn", verbose = F)



  sample_data <- sample_data %>% mutate(nu = tree_cover - yhat_lr)

  bootstrap_results <- foreach(nu_var=c("nu"), .combine='c', .packages=c('boot', 'dplyr', 'magrittr')) %dopar% {
    lapply(size, function(size_boot) {
      # Sample the data according to the current sample size
      boot_sample_data <- sample_data %>% slice_sample(n = size_boot)
      # Apply the bootstrap_estimate_se function to the sampled data and the current nu_var
      bootstrap_estimate_se(boot_sample_data, nu_var)
    })
  }

  # Combine the results into a data frame
  bootstrap_results_df <- do.call(rbind, lapply(bootstrap_results, function(x) {
    # Convert the list to a data frame
    data.frame(t(sapply(x, function(y) unlist(y))))
  }))
  names(bootstrap_results_df) <- c("bias_estimate", "bias_std_error")

  bootstrap_results_df$nu_var <- c("nu")
  bootstrap_results_df$model <- c("yhat_lr")
  print(summary(lm(yhat_lr ~ D, data = sample_data)))
  print(summary(lm(tree_cover ~ D, data = sample_data)))
  print(bootstrap_results_df)

  sample_data <- new_df
  sample_data$fold <- 0
  sample_data[(size+1):nrow(sample_data), "fold"] <- 1
  table(sample_data$fold)

  sample_data$yhat_lr <- unlist(rev(adb_model(x = sample_data[, c("fold", "tree_cover", "D", train_vars)],
                                        y_name =  "tree_cover", treat_name = "D", y_class = "continuous",
                                        debias_wt = 0, epochs = 50, model = "deep_nn",
                                        f = 1, predicttrain = TRUE, verbose = F)))

  sample_data$yhat_adv <- c(sample_data$tree_cover[1:size],adb_model(x = sample_data[, c("fold", "tree_cover", "D", train_vars)],
                                          y_name =  "tree_cover", treat_name = "D", y_class = "continuous",
                                          debias_wt = 20, epochs = 100, model = "deep_nn",
                                          f = 1, predicttrain = FALSE, verbose = F))

  sample_data$yhat_adv_corr <- c(sample_data$tree_cover[1:size],adb_model(x = sample_data[, c("fold", "tree_cover", "D", train_vars)],
                                              y_name =  "tree_cover", treat_name = "D",y_class = "continuous",
                                              debias_wt = 1, debias_method = "corr", epochs = 100, model = "deep_nn",
                                              f = 1, predicttrain = FALSE, verbose = F))

  sample_data$yhat_adv_mg <- c(sample_data$tree_cover[1:size],adb_model(x = sample_data[, c("fold", "tree_cover", "D", train_vars)],
                                              y_name =  "tree_cover", treat_name = "D",y_class = "continuous",
                                              debias_wt = 1, debias_method = "mg", epochs = 100, model = "deep_nn",
                                              f = 1, predicttrain = FALSE, verbose = F))



    models_to_run <- c(
      "tree_cover", "yhat_lr", "yhat_adv", "yhat_adv_corr", "yhat_adv_mg",
      "imputation", "ground_truth", "ptd",
      "ipd_postpi_analytic", "ipd_ppi", "ipd_ppi_plusplus", "ipd_pspa", "ipd_postpi_boot"
    )
    temp_results_dat <- tibble::tibble(
      model = character(),
      sample_size = numeric(),
      estimate = numeric(),
      std_error = numeric(),
      bias_estimate = numeric(),
      bias_std_error = numeric(),
      rsq_preds = numeric()
    )

    # Define common data splits based on 'fold' column
    # fold == 0 is labeled data (size rows)
    # fold == 1 is unlabeled data (remaining rows, where yhat_lr serves as proxy)
    GoodDat_samp <- sample_data %>% dplyr::filter(fold == 0)
    UnlabeledDat_samp <- sample_data %>% dplyr::filter(fold == 1)

    n_ipd_bootstrap_samples <- 50 # Number of bootstrap samples for IPD postpi_boot

    for (model_name in models_to_run) {
      # Initialize results for current model iteration
      current_estimate <- NA_real_
      current_std_error <- NA_real_
      current_bias_estimate <- NA_real_
      current_bias_std_error <- NA_real_
      current_rsq <- NA_real_

      if (model_name == "dsl") {
        cat(paste0("Skipping model: ", model_name, " as per instruction.\n"))
        next
      }

      if (model_name == "tree_cover") {
        model_summary <- summary(lm(tree_cover ~ D, data = GoodDat_samp)) # Use labeled data
        current_rsq <- model_summary$r.squared
        coef_summary <- coef(model_summary)
        if ("D" %in% rownames(coef_summary)) {
          current_estimate <- coef_summary["D", "Estimate"]
          current_std_error <- coef_summary["D", "Std. Error"]
        }
      } else if (model_name == "imputation") {
        imputation_df <- sample_data # Full sample_data for imputation setup
        imputation_df$tree_cover[imputation_df$fold == 1] <- NA # Set tree_cover to NA for "unlabeled" part
        imputation_df_for_mice <- imputation_df %>% dplyr::select(tree_cover, D, yhat_lr) # yhat_lr can be auxiliary

        # Ensure D is not a factor for mice, or handle appropriately if it is
        if(is.factor(imputation_df_for_mice$D)) {
            imputation_df_for_mice$D <- as.numeric(as.character(imputation_df_for_mice$D))
        } else {
            imputation_df_for_mice$D <- as.numeric(imputation_df_for_mice$D)
        }

        imp <- mice::mice(imputation_df_for_mice, printFlag = FALSE)
        fit <- with(imp, lm(tree_cover ~ D))
        est_pooled <- mice::pool(fit)
        
        pooled_summary <- summary(est_pooled)
        term_D_row <- pooled_summary[pooled_summary$term == "D", ]
        if (nrow(term_D_row) > 0) {
            current_estimate <- term_D_row$estimate
            current_std_error <- term_D_row$std.error
        }
        
        # R-squared for imputation: often not straightforward from pooled object.
        # This uses first imputed dataset for R-squared calculation for simplicity.
        completed_data_for_rsq <- mice::complete(imp, 1)
        current_rsq <- summary(lm(tree_cover ~ D, data = completed_data_for_rsq))$r.squared

      } else if (model_name == "ground_truth") {
        model_summary <- summary(lm(tree_cover ~ D, data = GoodDat_samp)) # Use labeled data
        coef_summary <- coef(model_summary)
        if ("D" %in% rownames(coef_summary)) {
          current_estimate <- coef_summary["D", "Estimate"]
          current_std_error <- coef_summary["D", "Std. Error"]
        }
        current_rsq <- model_summary$r.squared
      } else if (model_name == "ptd") {
        # PTD data prep
        PTD_GoodDat_completeSamp <- GoodDat_samp
        PTD_PredictionDat_completeSamp <- GoodDat_samp %>% dplyr::mutate(tree_cover = yhat_lr)
        PTD_PredictionDat_incompleteSamp <- UnlabeledDat_samp %>%
          dplyr::select(-tree_cover) %>% # Remove original tree_cover from unlabeled part
          dplyr::rename(tree_cover = yhat_lr) # Use yhat_lr as the outcome proxy

        alphaSig <- 0.05
        PTD_ests_CIs <- PTD_bootstrap.glm(
          true_data_completeSamp = PTD_GoodDat_completeSamp,
          predicted_data_completeSamp = PTD_PredictionDat_completeSamp,
          predicted_data_incompleteSamp = PTD_PredictionDat_incompleteSamp,
          regFormula.glm = "tree_cover ~ D",
          GLM_type = "linear",
          alpha = alphaSig,
          TuningScheme = "Diagonal",
          speedup = TRUE
        )
        
        current_estimate <- PTD_ests_CIs$PTD_estimate[2] # Assuming D is the second coefficient
        CIs <- PTD_ests_CIs$PTD_Boot_CIs[2, ]
        current_std_error <- (CIs[2] - CIs[1]) / (2 * qnorm(1 - alphaSig / 2))
        current_rsq <- NA_real_ # PTD doesn't typically produce an R-squared for prediction quality
      
      } else if (startsWith(model_name, "ipd_")) {
        ipd_method_name <- sub("ipd_", "", model_name)
        
        labeled_data_for_ipd <- GoodDat_samp %>%
          dplyr::mutate(
            outcome_ipd = tree_cover,
            prediction_ipd = yhat_lr,
            set_label_ipd = "labeled"
          ) %>%
          dplyr::select(outcome_ipd, prediction_ipd, D, set_label_ipd)

        unlabeled_data_for_ipd <- UnlabeledDat_samp %>%
          dplyr::mutate(
            outcome_ipd = NA_real_,
            prediction_ipd = yhat_lr,
            set_label_ipd = "unlabeled"
          ) %>%
          dplyr::select(outcome_ipd, prediction_ipd, D, set_label_ipd)
        
        ipd_combined_data <- dplyr::bind_rows(labeled_data_for_ipd, unlabeled_data_for_ipd)

        if (is.factor(ipd_combined_data$D)) { ipd_combined_data$D <- as.numeric(as.character(ipd_combined_data$D)) } 
        else { ipd_combined_data$D <- as.numeric(ipd_combined_data$D) }
        ipd_combined_data$prediction_ipd <- as.numeric(ipd_combined_data$prediction_ipd)
        ipd_combined_data$outcome_ipd <- as.numeric(ipd_combined_data$outcome_ipd)
        
        ipd_combined_data <- as.data.frame(ipd_combined_data)
        ipd_formula <- outcome_ipd - prediction_ipd ~ D
        
        ipd_args <- list(
          formula = ipd_formula,
          data = ipd_combined_data,
          method = ipd_method_name,
          model = "ols",
          label = "set_label_ipd"
        )
        
        if (ipd_method_name == "postpi_boot") {
          ipd_args$nboot <- n_ipd_bootstrap_samples
        }
        
        fit_ipd <- tryCatch({
          do.call(ipd::ipd, ipd_args)
        }, error = function(e) {
          cat(paste0("Error running IPD method ", ipd_method_name, ": ", e$message, "\n"))
          return(NULL)
        })
        
        if (!is.null(fit_ipd)) {
          coef_summary_ipd <- summary(fit_ipd)$coefficients
          if ("D" %in% rownames(coef_summary_ipd)) {
            current_estimate <- coef_summary_ipd["D", "Estimate"]
            # Attempt to get standard error, checking common column names
            if ("Std.Error" %in% colnames(coef_summary_ipd)) {
              current_std_error <- coef_summary_ipd["D", "Std.Error"]
            } else if ("Std. Error" %in% colnames(coef_summary_ipd)) {
              current_std_error <- coef_summary_ipd["D", "Std. Error"]
            } else {
              cat(paste0("Warning: Standard error column ('Std.Error' or 'Std. Error') not found in summary for IPD method '", ipd_method_name, "' for coefficient 'D'. Standard error will be NA.\n"))
              # current_std_error remains NA_real_ as initialized
            }
          }
        }
        current_rsq <- NA_real_

      } else if (model_name == "yhat_adv_mg") {
        model_summary <- summary(lm(as.formula(paste0(model_name, " ~ D")), data = sample_data)) # Full sample_data
        coef_summary <- coef(model_summary)
         if ("D" %in% rownames(coef_summary)) {
            current_estimate <- coef_summary["D", "Estimate"]
            current_std_error <- coef_summary["D", "Std. Error"]
        }
        current_rsq <- summary(lm(as.formula(paste0(model_name, " ~ tree_cover")), data = sample_data))$r.squared
      } else { # Covers yhat_lr, yhat_adv, yhat_adv_corr
        model_summary <- summary(lm(as.formula(paste0(model_name, " ~ D")), data = sample_data)) # Full sample_data
        coef_summary <- coef(model_summary)
        if ("D" %in% rownames(coef_summary)) {
            current_estimate <- coef_summary["D", "Estimate"]
            current_std_error <- coef_summary["D", "Std. Error"]
        }
        
        if (model_name == "yhat_lr") {
          current_bias_estimate <- bootstrap_results_df$bias_estimate
          current_bias_std_error <- bootstrap_results_df$bias_std_error
        } else {
          current_bias_estimate <- NA_real_
          current_bias_std_error <- NA_real_
        }
        current_rsq <- summary(lm(as.formula(paste0(model_name, " ~ tree_cover")), data = sample_data))$r.squared
      }
      
      temp_results_dat <- temp_results_dat %>%
        tibble::add_row(
          model = model_name,
          sample_size = size, # 'size' is from the outer loop
          estimate = current_estimate,
          std_error = current_std_error,
          bias_estimate = current_bias_estimate,
          bias_std_error = current_bias_std_error,
          rsq_preds = current_rsq
        )
    }
  
  run_results_df <- rbind(run_results_df, temp_results_dat)
  } 
  print(paste0("size = ", size))
  run_results_df
}
stopCluster(cl)
#toc()


results_df <- results_df %>% 
  filter(model == "yhat_lr") %>%
  mutate(estimate = estimate+bias_estimate,
  model = "yhat_lr_biascorrect") %>%
  bind_rows(results_df)

  # results_df <- results_df %>%
  #   group_by(sample_size) %>%
  #   mutate(run = row_number()) %>%
  #   ungroup()

  # ggplot(results_df, aes(x = sample_size, color = model)) +
  #   geom_point(aes(y = estimate), shape = 1, position = position_dodge(100)) +
  #   geom_smooth(aes(y = estimate), method = "loess", se = TRUE, color = "black", linetype = "dashed") + # Add lowess line with confidence band
  #   geom_line(aes(y = estimate, group = run), alpha = 0.3) + # Add a line for each run
  #   labs(x = "Sample Size", y = "Inverted Estimate / Bias Estimate", title = "Inverted Estimate and Bias Estimate with Standard Errors for Each Model Across All Sample Sizes") +
  #   facet_wrap(~model)

write_rds(results_df, file = paste0("../data/progressive_sampling/", output_file))