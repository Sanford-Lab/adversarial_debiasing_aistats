#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8G

# Your module loading and environment setup here
module load miniconda
source activate tf_r_env

Rscript progressive_sampling_server_roads_cat.R "$1"

# Check if Rscript ran successfully
if [ $? -eq 0 ]; then
    touch "complete_flag_$2.txt"
fi

# # Your module loading and environment setup here
# module load miniconda
# source activate tf_r_env

# Rscript progressive_sampling_server.R "$1"

# # Check if Rscript ran successfully
# if [ $? -eq 0 ]; then
#     touch "complete_flag_$2.txt"
# fi
