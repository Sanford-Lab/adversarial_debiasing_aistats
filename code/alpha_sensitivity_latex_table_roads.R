suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

out_dir <- Sys.getenv(
  "ALPHA_SENSITIVITY_ROADS_OUTDIR",
  unset = file.path("data", "alpha_sensitivity_results_roads")
)
in_csv <- file.path(out_dir, "summary_by_weight.csv")
out_tex <- file.path(out_dir, "summary_by_weight_table.tex")

if (!file.exists(in_csv)) stop("Missing input file: ", in_csv)
dat <- read_csv(in_csv, show_col_types = FALSE)
if (!all(c("debias_wt", "n") %in% names(dat))) stop("Input CSV missing required columns.")

fmt <- function(x, digits = 4) ifelse(is.na(x), "--", formatC(x, format = "f", digits = digits))

dat_fmt <- dat %>%
  mutate(
    debias_wt = fmt(debias_wt, 1),
    mean_estimate = fmt(mean_estimate, 4),
    ci_est = ifelse(is.na(ci_lower) | is.na(ci_upper), "--", paste0("[", fmt(ci_lower, 4), ", ", fmt(ci_upper, 4), "]")),
    mse_mean = fmt(mse_mean, 4),
    ci_mse = ifelse(is.na(mse_ci_lower) | is.na(mse_ci_upper), "--", paste0("[", fmt(mse_ci_lower, 4), ", ", fmt(mse_ci_upper, 4), "]")),
    n = as.integer(n)
  ) %>%
  select(debias_wt, mean_estimate, ci_est, mse_mean, ci_mse, n)

weights_txt <- paste(dat_fmt$debias_wt, collapse = ", ")
n_rep <- unique(dat_fmt$n)
n_txt <- if (length(n_rep) == 1) as.character(n_rep) else "possibly varying"

caption_txt <- paste0(
  "Alpha-sensitivity results for the tuned adversarial measurement model (yhat\\_adv\\_corr) on roads data ",
  "at labeled sample size 10,000. For each debiasing weight (", weights_txt, "), ",
  n_txt, " independent replications were run with randomized data order/folds; ",
  "within each weight, the tuning parameter $\\\\lambda$ was estimated from replication-level ",
  "covariance/variance and used to form tuned estimates. The table reports the mean tuned estimate ",
  "with a 95\\% confidence interval and the mean measurement-model MSE (unlabeled portion) with a 95\\% confidence interval."
)

header <- c(
  "\\begin{table}[!htbp]",
  "\\centering",
  paste0("\\caption{", caption_txt, "}"),
  "\\label{tab:alpha_sensitivity_roads_results}",
  "\\begin{tabular}{lccccc}",
  "\\toprule",
  "Debias Weight & Mean Estimate & 95\\% CI (Estimate) & Mean MSE & 95\\% CI (MSE) & $n$ \\\\",
  "\\midrule"
)

rows <- apply(dat_fmt, 1, function(r) {
  paste0(
    r[["debias_wt"]], " & ",
    r[["mean_estimate"]], " & ",
    r[["ci_est"]], " & ",
    r[["mse_mean"]], " & ",
    r[["ci_mse"]], " & ",
    r[["n"]], " \\\\"
  )
})

footer <- c("\\bottomrule", "\\end{tabular}", "\\end{table}")
dir.create(dirname(out_tex), recursive = TRUE, showWarnings = FALSE)
writeLines(c(header, rows, footer), out_tex)
message("Wrote LaTeX table: ", out_tex)
