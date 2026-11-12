#!/bin/bash
#SBATCH --job-name=douala_aggregate
#SBATCH --partition=nodes
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/scratch/$USER/cholera_project}"
cd "$PROJECT_ROOT"
mkdir -p logs
module load miniforge/24.7.1
conda activate airenotebook
export PYTHONUNBUFFERED=1
export AIRE_USE_NETWORKIT=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export RUN_SOBOL=0
export RUN_PARAMETER_SCREENING=0
export RUN_MOBILITY_COMPARATOR_MAIN=0
export RUN_COOCCURRENCE_ENSEMBLE=1
export COOCCURRENCE_N_SEEDS=50
export COOCCURRENCE_SEED_START=42
export COOCCURRENCE_RESUME=1
# Strictly validate all 200 scenario checkpoints before aggregating anything.
python code/check_outputs.py --project-root "$PROJECT_ROOT" --city_slug douala --run_id 3000 --seed-start 42 --n-seeds 50 --checkpoint-version 7 --write-report
# Aggregate all checkpoints and generate model figures.
python code/ensemble_runner.py \
  --model-source code/douala_model.py \
  --city_slug douala --run_id 3000 --mode aggregate --model-plots
# Validate the aggregate CSV set as well.
python code/check_outputs.py --project-root "$PROJECT_ROOT" --city_slug douala --run_id 3000 --seed-start 42 --n-seeds 50 --checkpoint-version 7 --require-aggregate-csvs --write-report
echo "[aggregate] SUCCESS"
