# One run: progressive_sampling_server-style pipeline at a fixed labeled size (10k),
# yhat_adv_corr with debias_method = "corr", plus coef(D) needed for adv_tuned (see process_server_data.R).
#
# Uses code/debiaser.R (keras3 + TensorFlow).
#
# Adjust debias penalty here (overridden by optional CLI first argument):
DEBIAS_WT <- 1
## If non-NULL, single-run RDS also stores estimate_tuned = (1-l)*tree + l*adv_corr (e.g. l from batch).
SINGLE_RUN_ADV_TUNED_LAMBDA <- NULL
SAMPLE_SIZE <- 10000L
N_POINTS <- 20000L
EPOCHS_LR <- 50L
EPOCHS_ADV <- 100L

## R may mangle --file= paths that contain spaces (e.g. Dropbox); fall back to getwd() or ALPHA_CODE_DIR.
code_dir <- local({
  marker <- "debiaser.R"
  e <- Sys.getenv("ALPHA_CODE_DIR", unset = "")
  if (nzchar(e) && file.exists(file.path(e, marker))) {
    return(e)
  }
  args_all <- commandArgs(trailingOnly = FALSE)
  f <- args_all[startsWith(args_all, "--file=")]
  if (length(f)) {
    d <- dirname(sub("^--file=", "", f[1]))
    if (file.exists(file.path(d, marker))) {
      return(d)
    }
  }
  gw <- getwd()
  if (file.exists(file.path(gw, marker))) {
    return(gw)
  }
  stop("Cannot locate code directory containing ", marker, ". Set env ALPHA_CODE_DIR or `cd` to supplement_aistats/code before Rscript.")
})

source(file.path(code_dir, "debiaser.R"))


build_biased_dataset <- function(csv_path, n_points = 20000L, seed = 731L) {
  set.seed(seed)
  df <- read_csv(csv_path, show_col_types = FALSE) %>% rename(old_slope = slope)
  rs_vals <- slice_sample(df, n = n_points, replace = FALSE)
  slope <- rpois(n_points, lambda = 1)
  D <- rbinom(n_points, 1, pmax(1 - slope / 4, 0))
  tc_rank <- tibble(
    tree_cover = as.numeric(unique(names(table(rs_vals$tree_cover)))),
    rank = seq_along(unique(names(table(rs_vals$tree_cover))))
  )
  dat <- tibble(slope, D) %>% bind_cols(rs_vals) %>% left_join(tc_rank, by = "tree_cover")
  no_rs <- dat %>%
    select(slope, D, tree_cover, rank) %>%
    mutate(new_rank = pmin(rank + slope, 14)) %>%
    rename(old_tree_cover = tree_cover) %>%
    left_join(tc_rank %>% rename(new_rank = rank), by = "new_rank") %>%
    arrange(tree_cover)
  tbl <- table(no_rs$tree_cover)
  map2_df(names(tbl), as.integer(tbl), function(tc, n_draw) {
    df %>%
      filter(tree_cover == !!as.numeric(tc)) %>%
      slice_sample(n = n_draw, replace = TRUE)
  }) %>%
    arrange(tree_cover) %>%
    select(B1_median:EVI_75perc) %>%
    bind_cols(no_rs) %>%
    rename(
      apparent_tree_cover = tree_cover,
      tree_cover = old_tree_cover
    )
}


load_new_df <- function(project_root) {
  rds_path <- file.path(project_root, "data/power_analysis/biased_data.RDS")
  if (file.exists(rds_path)) {
    return(read_rds(rds_path))
  }
  csv_path <- file.path(project_root, "data/w_africa_l7_model_prepped.csv")
  if (!file.exists(csv_path)) {
    stop("Need data/power_analysis/biased_data.RDS or data/w_africa_l7_model_prepped.csv under ", project_root)
  }
  build_biased_dataset(csv_path, n_points = N_POINTS)
}


#' One replication; matches progressive_sampling_server.R folding / training for yhat_adv_corr.
run_once <- function(
    new_df,
    size = SAMPLE_SIZE,
    debias_wt = DEBIAS_WT,
    epochs_lr = EPOCHS_LR,
    epochs_adv = EPOCHS_ADV,
    shuffle_seed = NULL) {
  if (!is.null(shuffle_seed)) {
    set.seed(shuffle_seed)
  }
  ssize <- nrow(new_df)
  new_df <- new_df %>% sample_n(size = ssize, replace = FALSE)

  sample_data <- new_df[seq_len(size), ]
  sample_data <- sample_data[sample(nrow(sample_data)), ]
  sample_data$fold <- 3L
  sample_data[seq_len(round(nrow(sample_data) / 3)), "fold"] <- 1L
  sample_data[(round(nrow(sample_data) / 3) + 1):round(2 * nrow(sample_data) / 3), "fold"] <- 2L

  train_vars <- sort(grep("B[1-8]_|NDVI_|NDBI_|EVI_", names(new_df), value = TRUE))

  sample_data$yhat_lr <- train_all_folds(
    x = sample_data[, c("fold", "tree_cover", "D", train_vars)],
    y_name = "tree_cover",
    treat_name = "D",
    y_class = "continuous",
    debias_wt = 0,
    epochs = epochs_lr,
    model = "deep_nn",
    verbose = FALSE
  )

  sample_data <- new_df
  sample_data$fold <- 0L
  sample_data[(size + 1):nrow(sample_data), "fold"] <- 1L

  sample_data$yhat_lr <- unlist(rev(adb_model(
    x = sample_data[, c("fold", "tree_cover", "D", train_vars)],
    y_name = "tree_cover",
    treat_name = "D",
    y_class = "continuous",
    debias_wt = 0,
    epochs = epochs_lr,
    model = "deep_nn",
    f = 1,
    predicttrain = TRUE,
    verbose = FALSE
  )))

  sample_data$yhat_adv_corr <- c(
    sample_data$tree_cover[seq_len(size)],
    adb_model(
      x = sample_data[, c("fold", "tree_cover", "D", train_vars)],
      y_name = "tree_cover",
      treat_name = "D",
      y_class = "continuous",
      debias_wt = debias_wt,
      debias_method = "corr",
      epochs = epochs_adv,
      model = "deep_nn",
      f = 1,
      predicttrain = FALSE,
      verbose = FALSE
    )
  )

  good <- sample_data %>% filter(fold == 0)
  unlabeled <- sample_data %>% filter(fold == 1)
  coef_d <- function(fit) {
    cc <- coef(summary(fit))
    if (!("D" %in% rownames(cc))) stop("Treatment coefficient D not found in model.")
    unname(cc["D", "Estimate"])
  }
  est_tree <- coef_d(lm(tree_cover ~ D, data = good))
  est_adv <- coef_d(lm(yhat_adv_corr ~ D, data = sample_data))
  mse_adv_corr <- mean((unlabeled$tree_cover - unlabeled$yhat_adv_corr)^2, na.rm = TRUE)

  list(
    debias_wt = debias_wt,
    estimate_tree = est_tree,
    estimate_adv_corr = est_adv,
    mse_measurement_model = mse_adv_corr,
    sample_size = size
  )
}


adv_tuned_lambda <- function(estimate_tree, estimate_adv) {
  d <- estimate_tree - estimate_adv
  v <- stats::var(d)
  if (!is.finite(v) || v < 1e-12) {
    return(NA_real_)
  }
  stats::cov(estimate_tree, d) / v
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
  new_df <- load_new_df(project_root)
  out_dir <- Sys.getenv(
    "ALPHA_SENSITIVITY_OUTDIR",
    unset = file.path(project_root, "data", "alpha_sensitivity_results")
  )

  res <- run_once(
    new_df,
    debias_wt = debias_wt,
    epochs_lr = elr,
    epochs_adv = eadv,
    shuffle_seed = shuffle_seed
  )
  res$shuffle_seed <- shuffle_seed

  lam_single <- SINGLE_RUN_ADV_TUNED_LAMBDA
  if (is.null(lam_single) || !is.finite(lam_single)) {
    res$estimate_tuned <- NA_real_
    res$lambda_used <- NA_real_
  } else {
    res$lambda_used <- lam_single
    res$estimate_tuned <- (1 - lam_single) * res$estimate_tree + lam_single * res$estimate_adv_corr
  }

  message(sprintf(
    "debias_wt = %s | estimate_tree = %.6f | estimate_adv_corr = %.6f | mse = %.6f | seed = %s",
    res$debias_wt, res$estimate_tree, res$estimate_adv_corr, res$mse_measurement_model, shuffle_seed
  ))

  if (length(args) < 3L) {
    out_path <- file.path(out_dir, "last_run.rds")
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  }
  if (!is.na(out_path) && nzchar(out_path)) {
    saveRDS(res, out_path)
    message("Saved ", out_path)
  }
  invisible(res)
}


if (identical(Sys.getenv("ALPHA_SENSITIVITY_SKIP_MAIN"), "")) {
  main()
}
