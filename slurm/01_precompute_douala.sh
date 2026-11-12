#!/bin/bash
#SBATCH --job-name=douala_precompute
#SBATCH --partition=nodes
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
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
python code/ensemble_runner.py \
  --model-source code/douala_model.py \
  --city_slug douala --run_id 3000 --mode precompute
echo "[precompute] SUCCESS"
