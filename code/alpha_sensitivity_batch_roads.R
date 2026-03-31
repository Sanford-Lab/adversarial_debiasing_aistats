suppressPackageStartupMessages(library(tidyverse))

n_replicates <- as.integer(Sys.getenv("ALPHA_SENSITIVITY_N_REP", "50"))
debias_weights <- {
  w_env <- Sys.getenv("ALPHA_SENSITIVITY_WEIGHTS", unset = "")
  if (nzchar(w_env)) as.numeric(strsplit(w_env, ",", fixed = TRUE)[[1]]) else c(0.5, 1, 2, 5)
}

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

roads_script <- file.path(code_dir, "alpha_sensitivity_roads.R")

adv_tuned_lambda <- function(estimate_truth, estimate_adv) {
  d <- estimate_truth - estimate_adv
  v <- stats::var(d)
  if (!is.finite(v) || v < 1e-12) return(NA_real_)
  stats::cov(estimate_truth, d) / v
}

run_weight_batch <- function(w, n_rep, script_path, rscript) {
  tmp_files <- character(n_rep)
  on.exit(unlink(tmp_files[file.exists(tmp_files)]), add = TRUE)
  est_truth <- numeric(n_rep)
  est_adv <- numeric(n_rep)
  mse_measurement <- numeric(n_rep)

  for (i in seq_len(n_rep)) {
    tmp_files[[i]] <- tempfile(fileext = ".rds")
    log_file <- tempfile(fileext = ".log")
    on.exit(unlink(log_file, force = TRUE), add = TRUE)
    args <- c(script_path, as.character(w), as.character(i), tmp_files[[i]])
    status <- suppressWarnings(system2(rscript, args, stdout = log_file, stderr = log_file))
    if (!file.exists(tmp_files[[i]])) {
      stop(
        "alpha_sensitivity_roads.R did not write output for weight ", w, " rep ", i,
        " (system2 exit ", paste(status, collapse = " "), ").\n\n",
        "Command:\n  ", rscript, " ", paste(shQuote(args), collapse = " "), "\n\n",
        "Last log lines:\n",
        paste(utils::tail(readLines(log_file, warn = FALSE), 40), collapse = "\n")
      )
    }
    r <- readRDS(tmp_files[[i]])
    est_truth[[i]] <- r$estimate_forest
    est_adv[[i]] <- r$estimate_adv_corr
    mse_measurement[[i]] <- r$mse_measurement_model
  }

  lam <- adv_tuned_lambda(est_truth, est_adv)
  tuned <- (1 - lam) * est_truth + lam * est_adv
  n_ok <- sum(is.finite(tuned))
  se_est <- if (n_ok > 1L) stats::sd(tuned, na.rm = TRUE) / sqrt(n_ok) else NA_real_
  n_mse <- sum(is.finite(mse_measurement))
  mse_mean <- mean(mse_measurement, na.rm = TRUE)
  mse_se <- if (n_mse > 1L) stats::sd(mse_measurement, na.rm = TRUE) / sqrt(n_mse) else NA_real_

  tibble(
    debias_wt = w,
    lambda = lam,
    mean_estimate = mean(tuned, na.rm = TRUE),
    se = se_est,
    ci_lower = mean_estimate - stats::qnorm(0.975) * se_est,
    ci_upper = mean_estimate + stats::qnorm(0.975) * se_est,
    mse_mean = mse_mean,
    mse_ci_lower = mse_mean - stats::qnorm(0.975) * mse_se,
    mse_ci_upper = mse_mean + stats::qnorm(0.975) * mse_se,
    n = n_rep
  )
}

main <- function() {
  if (!file.exists(roads_script)) stop("Missing ", roads_script)
  rscript <- Sys.getenv("ALPHA_RSCRIPT", unset = file.path(R.home("bin"), "Rscript"))
  rows <- map_dfr(debias_weights, run_weight_batch, n_rep = n_replicates, script_path = roads_script, rscript = rscript)
  print(rows, n = Inf)

  out_dir <- Sys.getenv(
    "ALPHA_SENSITIVITY_ROADS_OUTDIR",
    unset = file.path(dirname(code_dir), "data", "alpha_sensitivity_results_roads")
  )
  out_csv <- file.path(out_dir, "summary_by_weight.csv")
  dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
  write_csv(rows, out_csv)
  message("Wrote ", out_csv)
  invisible(rows)
}

main()
