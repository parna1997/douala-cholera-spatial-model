#!/bin/bash
# Submit the simulation, aggregation, analysis and validation jobs with Slurm dependencies.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/scratch/$USER/cholera_project}"
cd "$PROJECT_ROOT"
mkdir -p logs
submit() { local out; out=$(sbatch --parsable "$@"); echo "${out%%;*}"; }
j00=$(submit slurm/00_collect_environment.sh)
j01=$(submit --dependency=afterok:"$j00" slurm/01_precompute_douala.sh)
j02=$(submit --dependency=afterok:"$j01" slurm/02_run_douala_array.sh)
j03=$(submit --dependency=afterok:"$j02" slurm/03_aggregate_douala.sh)
j04=$(submit --dependency=afterok:"$j03" slurm/04_postprocess_douala.sh)
j05=$(submit --dependency=afterok:"$j03" slurm/05_overture_validation_douala.sh)
j06=$(submit --dependency=afterok:"$j04":"$j05" slurm/06_package_results.sh)
printf 'env %s\nprecompute %s\narray %s\naggregate %s\npostprocess %s\noverture %s\nfinalise %s\n' "$j00" "$j01" "$j02" "$j03" "$j04" "$j05" "$j06"
echo 'Monitor: squeue -u "$USER"; sacct -j JOBID --format=JobID,State,ExitCode'
