suppressPackageStartupMessages({
  library(tidyverse)
  library(keras3)
  library(tensorflow)
})

DEBIAS_WT <- 1
SAMPLE_SIZE <- 10000L
EPOCHS_LR <- 50L
EPOCHS_ADV <- 100L

code_dir <- local({
  marker <- "debiaser.R"
  e <- Sys.getenv("ALPHA_CODE_DIR", unset = "")
  if (nzchar(e) && file.exists(file.path(e, marker))) return(e)
  args_all <- commandArgs(trailingOnly = FALSE)
  f <- args_all[startsWith(args_all, "--file=")]
  if (length(f)) {
    d <- dirname(sub("^--file=", "", f[1]))
    if (file.exists(file.path(d, marker))) return(d)
  }
  gw <- getwd()
  if (file.exists(file.path(gw, marker))) return(gw)
  stop("Cannot locate code directory containing ", marker)
})

source(file.path(code_dir, "debiaser.R"))

load_roads_df <- function(project_root) {
  csv_path <- file.path(project_root, "data", "w_africa_l7_model_prepped.csv")
  if (!file.exists(csv_path)) stop("Missing roads data at ", csv_path)

  df <- read.csv(csv_path)
  df <- df[!is.na(df$distance), ]
  df <- df[df$distance < 32000, ]
  df$distance <- df$distance / 1000
  df$forest <- if_else(df$land_use == "forest", 1, 0)
  df$D <- log1p(df$distance)
  as_tibble(df)
}

run_once <- function(new_df, size = SAMPLE_SIZE, debias_wt = DEBIAS_WT,
                     epochs_lr = EPOCHS_LR, epochs_adv = EPOCHS_ADV,
                     shuffle_seed = NULL) {
  if (!is.null(shuffle_seed)) set.seed(shuffle_seed)
  ssize <- nrow(new_df)
  new_df <- new_df %>% sample_n(size = ssize, replace = FALSE)

  sample_data <- new_df[seq_len(size), ]
  sample_data <- sample_data[sample(nrow(sample_data)), ]
  sample_data$fold <- 3L
  sample_data[seq_len(round(nrow(sample_data) / 3)), "fold"] <- 1L
  sample_data[(round(nrow(sample_data) / 3) + 1):round(2 * nrow(sample_data) / 3), "fold"] <- 2L

  train_vars <- sort(grep("B[1-8]_|NDVI_|NDBI_|EVI_", names(new_df), value = TRUE))
  sample_data$yhat_lr <- train_all_folds(
    x = sample_data[, c("fold", "forest", "D", train_vars)],
    y_name = "forest", treat_name = "D", y_class = "categorical",
    debias_wt = 0, epochs = epochs_lr, model = "deep_nn", verbose = FALSE
  )

  sample_data <- new_df
  sample_data$fold <- 0L
  sample_data[(size + 1):nrow(sample_data), "fold"] <- 1L

  sample_data$yhat_lr <- unlist(rev(adb_model(
    x = sample_data[, c("fold", "forest", "D", train_vars)],
    y_name = "forest", treat_name = "D", y_class = "categorical",
    debias_wt = 0, epochs = epochs_lr, model = "deep_nn",
    f = 1, predicttrain = TRUE, verbose = FALSE
  )))

  sample_data$yhat_adv_corr <- c(
    sample_data$forest[seq_len(size)],
    adb_model(
      x = sample_data[, c("fold", "forest", "D", train_vars)],
      y_name = "forest", treat_name = "D", y_class = "categorical",
      debias_wt = debias_wt, debias_method = "corr",
      epochs = epochs_adv, model = "deep_nn",
      f = 1, predicttrain = FALSE, verbose = FALSE
    )
  )

  good <- sample_data %>% filter(fold == 0)
  unlabeled <- sample_data %>% filter(fold == 1)
  coef_d <- function(fit) {
    cc <- coef(summary(fit))
    if (!("D" %in% rownames(cc))) stop("Treatment coefficient D not found.")
    unname(cc["D", "Estimate"])
  }
  est_forest <- coef_d(lm(forest ~ D, data = good))
  est_adv <- coef_d(lm(yhat_adv_corr ~ D, data = sample_data))
  mse_adv_corr <- mean((unlabeled$forest - unlabeled$yhat_adv_corr)^2, na.rm = TRUE)

  list(
    debias_wt = debias_wt,
    estimate_forest = est_forest,
    estimate_adv_corr = est_adv,
    mse_measurement_model = mse_adv_corr,
    sample_size = size
  )
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  debias_wt <- DEBIAS_WT
  shuffle_seed <- as.integer(sample.int(1e9, 1L))
  out_path <- NA_character_

  if (length(args) >= 1) debias_wt <- as.numeric(args[[1]])
  if (length(args) >= 2) shuffle_seed <- as.integer(args[[2]])
  if (length(args) >= 3) out_path <- args[[3]]

  fast <- Sys.getenv("ALPHA_SENSITIVITY_FAST", "0") == "1"
  elr <- if (fast) 2L else EPOCHS_LR
  eadv <- if (fast) 2L else EPOCHS_ADV

  project_root <- normalizePath(file.path(code_dir, ".."))
  out_dir <- Sys.getenv(
    "ALPHA_SENSITIVITY_ROADS_OUTDIR",
    unset = file.path(project_root, "data", "alpha_sensitivity_results_roads")
  )
  new_df <- load_roads_df(project_root)

  res <- run_once(new_df, debias_wt = debias_wt, epochs_lr = elr, epochs_adv = eadv, shuffle_seed = shuffle_seed)
  res$shuffle_seed <- shuffle_seed

  if (length(args) < 3L) {
    out_path <- file.path(out_dir, "last_run.rds")
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  }
  if (!is.na(out_path) && nzchar(out_path)) {
    saveRDS(res, out_path)
    message("Saved ", out_path)
  }
  message(sprintf(
    "roads debias_wt = %s | estimate_forest = %.6f | estimate_adv_corr = %.6f | mse = %.6f | seed = %s",
    res$debias_wt, res$estimate_forest, res$estimate_adv_corr, res$mse_measurement_model, shuffle_seed
  ))
  invisible(res)
}

if (identical(Sys.getenv("ALPHA_SENSITIVITY_SKIP_MAIN"), "")) main()
