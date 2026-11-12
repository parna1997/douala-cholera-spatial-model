#!/bin/bash
#SBATCH --job-name=douala_post
#SBATCH --partition=nodes
#SBATCH --time=12:00:00
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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
python code/postprocess.py \
  --project-root "$PROJECT_ROOT" --city_slug douala --run_id 3000 \
  --seed-start 42 --n-seeds 50 --bootstrap-resamples 10000 --permutations 10000
python code/check_figures.py --project-root "$PROJECT_ROOT" --city_slug douala --run_id 3000 --min-width-px 3000
echo "[postprocess] SUCCESS"
