#!/bin/bash
#SBATCH --job-name=douala_overture
#SBATCH --partition=nodes
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# External Overture spatial plausibility assessment.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/scratch/$USER/cholera_project}"
cd "$PROJECT_ROOT"
mkdir -p logs
module load miniforge/24.7.1
conda activate airenotebook
export PYTHONUNBUFFERED=1
# Default: automatic latest release from Overture's authoritative STAC catalogue.
# An explicit release can be set with: sbatch --export=ALL,OVERTURE_RELEASE=2026-10-21.0 ...
ARGS=(--project-root "$PROJECT_ROOT" --city_slug douala --run_id 3000 --release "${OVERTURE_RELEASE:-latest}")
if [[ -n "${OVERTURE_PLACES_FILE:-}" ]]; then ARGS+=(--places-file "$OVERTURE_PLACES_FILE"); fi
if [[ -n "${OVERTURE_INFRA_FILE:-}" ]]; then ARGS+=(--infrastructure-file "$OVERTURE_INFRA_FILE"); fi
if [[ "${OVERTURE_SAVE_FEATURE_DATA:-0}" == "1" ]]; then ARGS+=(--save-feature-data); fi
python code/validate_overture_baseline.py "${ARGS[@]}"
BASE=outputs/douala/run_3000/cooccurrence_ensemble/overture_validation
for result in baseline_validation_metrics.csv baseline_validation_hub_shares.csv baseline_validation_metadata.json baseline_validation_summary.txt; do
  test -s "$BASE/$result" || { echo "Missing expected Overture result: $BASE/$result" >&2; exit 1; }
done
python - <<'PY'
from pathlib import Path
import json
f=Path('outputs/douala/run_3000/cooccurrence_ensemble/overture_validation/baseline_validation_metadata.json')
m=json.loads(f.read_text())
print('[Overture] DATA RELEASE:',m['overture_release'])
print('[Overture] SCHEMA:',m['overture_schema_version'])
print('[Overture] reference release:',m['publication_reference_release'])
print('[Overture] same reference release:',m['is_publication_reference_release'])
PY
echo '[Overture validation] SUCCESS'
