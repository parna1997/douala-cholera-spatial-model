#!/bin/bash
#SBATCH --job-name=douala_50
#SBATCH --partition=nodes
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-49%10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# One independent stochastic seed per array task.
# 50 seeds = 42..91; at most 10 tasks/nodes execute concurrently.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/scratch/$USER/cholera_project}"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/douala
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
SEED_START=42
SEED=$((SEED_START + SLURM_ARRAY_TASK_ID))
export RUN_COOCCURRENCE_ENSEMBLE=1
export COOCCURRENCE_N_SEEDS=1
export COOCCURRENCE_SEED_START=$SEED
export COOCCURRENCE_RESUME=1
export RUN_PAPER_PLOTS=0
export RUN_DIAGNOSTIC_PLOTS=0
export KEEP_LEGACY_PLOTS=1
echo "[SLURM] task=${SLURM_ARRAY_TASK_ID}; stochastic_seed=${SEED}; run_id=3000"
python code/ensemble_runner.py \
  --model-source code/douala_model.py \
  --city_slug douala --run_id 3000 --mode worker
echo "[worker] SUCCESS seed=${SEED}"
