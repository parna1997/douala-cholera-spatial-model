#!/bin/bash
#SBATCH --job-name=douala_finalise
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Validate outputs and create a results-only archive; exclude per-seed checkpoints.
# Include environment metadata and validated aggregate outputs.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/scratch/$USER/cholera_project}"
cd "$PROJECT_ROOT"
module load miniforge/24.7.1
conda activate airenotebook
RUN_DIR="$PROJECT_ROOT/outputs/douala/run_3000"
BASE="$RUN_DIR/cooccurrence_ensemble"
ENV_DIR="$RUN_DIR/environment"
ARCHIVE="$RUN_DIR/douala_results_no_checkpoints.tgz"

python code/check_outputs.py \
  --project-root "$PROJECT_ROOT" --city_slug douala --run_id 3000 \
  --seed-start 42 --n-seeds 50 --checkpoint-version 7 \
  --require-aggregate-csvs --write-report
python code/check_figures.py \
  --project-root "$PROJECT_ROOT" --city_slug douala --run_id 3000 --min-width-px 3000
for filename in \
  "$BASE/statistics/primary_endpoint_comparisons.csv" \
  "$BASE/statistics/tap_contamination_headline_mean_sd.csv" \
  "$BASE/overture_validation/baseline_validation_metrics.csv" \
  "$ENV_DIR/python_version.txt" \
  "$ENV_DIR/conda_environment.yml" \
  "$ENV_DIR/conda_explicit.txt" \
  "$ENV_DIR/conda_list.txt" \
  "$ENV_DIR/slurm_version.txt" \
  "$ENV_DIR/preflight_stdout.txt"; do
  test -s "$filename" || { echo "MISSING: $filename" >&2; exit 1; }
done

mapfile -t AGGREGATE_CSVS < <(
  find "$BASE" -maxdepth 1 -type f \
    \( -name 'douala_*_mean_sd.csv' -o -name 'douala_*_per_seed.csv' \) \
    -printf '%f\n' | sort
)
test "${#AGGREGATE_CSVS[@]}" -gt 0
RESULT_DIRS=(statistics figures overture_validation validation)
for d in "${RESULT_DIRS[@]}"; do test -d "$BASE/$d"; done
for d in paper_figures final_paper_main; do
  if [[ -d "$BASE/$d" ]]; then RESULT_DIRS+=("$d"); fi
done

# Archive generated model and external-validation results only.
rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" \
  -C "$RUN_DIR" environment \
  -C "$BASE" "${RESULT_DIRS[@]}" "${AGGREGATE_CSVS[@]}"
gzip -t "$ARCHIVE"
tar -tzf "$ARCHIVE" > "$RUN_DIR/douala_archive_contents.txt"
echo "[finalise] archive=$ARCHIVE"
ls -lh "$ARCHIVE"
echo '[finalise] SUCCESS'
