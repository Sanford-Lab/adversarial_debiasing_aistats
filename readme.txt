This folder contains supplemental material for "Adversarial Debiasing for Parameter Recovery", including code used for the experiments.

Paper (OpenReview): https://openreview.net/forum?id=LtLmVk2CAx&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Daistats.org%2FAISTATS%2F2026%2FConference%2FAuthors%23your-submissions)

Code files in `code/`:
- `code/debiaser.R`: adversarial debiasing model definitions and training helpers used by simulation scripts.
- `code/progressive_sampling_server.R`: synthetic/continuous-treatment progressive sampling simulation.
- `code/progressive_sampling_server_roads_cat.R`: roads/categorical-treatment progressive sampling simulation.
- `code/process_server_data.R`: post-processing and figures for outputs from `progressive_sampling_server.R`.
- `code/process_server_data_roads_cat.R`: post-processing and figures for outputs from `progressive_sampling_server_roads_cat.R`.
- `code/alpha_sensitivity.R`: one-run alpha/debias-weight sensitivity evaluation for `yhat_adv_corr` at sample size 10,000.
- `code/alpha_sensitivity_batch.R`: repeated alpha sensitivity runs over multiple weights with summary table output.
- `code/alpha_sensitivity_roads.R`: one-run alpha/debias-weight sensitivity evaluation for the roads/categorical-treatment setting.
- `code/alpha_sensitivity_batch_roads.R`: repeated roads alpha sensitivity runs over multiple weights with summary table output.
- `code/alpha_sensitivity_latex_table.R`: creates LaTeX-ready summary tables from non-roads alpha sensitivity batch outputs.
- `code/alpha_sensitivity_latex_table_roads.R`: creates LaTeX-ready summary tables from roads alpha sensitivity batch outputs.
- `code/all_jobs.sh`: Slurm submission script for cluster runs.
- `code/run_r_script.sh`: helper shell wrapper used by the cluster workflow.

Environment setup:
- `tf_r_env.yml` is the original environment export used for cluster/Linux workflows.
- `tf_r_env_mac.yml` is a macOS-friendly environment file aligned with current `keras3` + TensorFlow usage in this supplement.

On macOS, create/use the environment with:

```
conda env create -f tf_r_env_mac.yml
conda activate tf_r_env_mac
```

Then install/update the R packages in that environment:

```
Rscript -e 'install.packages(c("tensorflow","keras3"), repos="https://cloud.r-project.org", dependencies=NA)'
```

Run figure-processing scripts:

```
cd code
Rscript process_server_data_roads_cat.R
Rscript process_server_data.R
```

Run alpha sensitivity scripts:

```
cd code
# single run (optional args: debias_wt, seed, output_rds)
Rscript alpha_sensitivity.R

# batch run (defaults: n=50, weights=0.5,1,2,5)
Rscript alpha_sensitivity_batch.R
```

Optional environment variables for batch runs:
- `ALPHA_SENSITIVITY_N_REP` (example: `50`)
- `ALPHA_SENSITIVITY_WEIGHTS` (example: `"0.5,1,2,5"`)
- `ALPHA_SENSITIVITY_FAST=1` for quick smoke tests

HPC usage:

```
cd code
sbatch all_jobs.sh
```

As written, `all_jobs.sh` runs the roads/categorical treatment configuration. To switch datasets, edit the corresponding command lines in `all_jobs.sh`.
