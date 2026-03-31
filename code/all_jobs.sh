#!/bin/bash

for i in {1..20}
do
   sbatch --partition=day --job-name="R_prog_roads_cat_$i" run_r_script.sh "output_$i.RDS" $i
done

# for i in {1..20}
# do
#    sbatch --partition=day --job-name="R_prog_$i" run_r_script.sh "output_$i.RDS" $i
# done