#!/bin/bash
#SBATCH --job-name=douala_env
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/scratch/$USER/cholera_project}"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/douala/run_3000/environment
module load miniforge/24.7.1
conda activate airenotebook
export PYTHONUNBUFFERED=1
ENVOUT="outputs/douala/run_3000/environment"
python --version 2>&1 | tee "$ENVOUT/python_version.txt"
conda env export --no-builds > "$ENVOUT/conda_environment.yml"
conda list --explicit > "$ENVOUT/conda_explicit.txt"
conda list > "$ENVOUT/conda_list.txt"
(slurm --version || srun --version || true) > "$ENVOUT/slurm_version.txt" 2>&1
python code/preflight.py --project-root "$PROJECT_ROOT" --city_slug douala | tee "$ENVOUT/preflight_stdout.txt"
echo "[environment] SUCCESS"
