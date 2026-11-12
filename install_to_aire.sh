#!/bin/bash
# Install model source and Slurm scripts into the specified project directory.
# Refuse installation while Douala jobs are active.
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/scratch/$USER/cholera_project}"
if command -v squeue >/dev/null 2>&1; then
  jobs="$(squeue -h -u "$USER" -o '%j' | grep -E '^(douala_|douala_overture)' || true)"
  [[ -z "$jobs" ]] || { echo 'STOP: Douala Slurm jobs are active; do not replace source files mid-run.'; echo "$jobs"; exit 1; }
fi
[[ -d "$PROJECT_ROOT/inputs/douala/data" ]] || { echo "STOP: existing Douala inputs not found: $PROJECT_ROOT/inputs/douala/data"; exit 1; }
[[ -f "$PROJECT_ROOT/configs/cities.csv" ]] || { echo 'STOP: existing configs/cities.csv missing'; exit 1; }
[[ -f "$PROJECT_ROOT/inputs/douala/data/pop.tif" && -f "$PROJECT_ROOT/inputs/douala/data/pop.csv" ]] || { echo 'STOP: WorldPop pop.tif/pop.csv missing'; exit 1; }
BACKUP_DIR="$PROJECT_ROOT/source_backups/douala_before_install_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR/code" "$BACKUP_DIR/slurm" "$PROJECT_ROOT/code" "$PROJECT_ROOT/slurm" "$PROJECT_ROOT/logs"
for item in "$PACKAGE_DIR"/code/* "$PACKAGE_DIR"/slurm/*; do
  relative="${item#"$PACKAGE_DIR"/}"
  if [[ -e "$PROJECT_ROOT/$relative" ]]; then
    cp -p "$PROJECT_ROOT/$relative" "$BACKUP_DIR/$relative"
  fi
  cp -p "$item" "$PROJECT_ROOT/$relative"
done
if [[ -e "$PROJECT_ROOT/submit_douala_pipeline.sh" ]]; then
  cp -p "$PROJECT_ROOT/submit_douala_pipeline.sh" "$BACKUP_DIR/submit_douala_pipeline.sh"
fi
cp -p "$PACKAGE_DIR/submit_douala_pipeline.sh" "$PROJECT_ROOT/submit_douala_pipeline.sh"
echo "Model code and HPC scripts installed to: $PROJECT_ROOT"
echo "Source backup directory: $BACKUP_DIR"
echo 'Inputs, caches, outputs and checkpoints were not modified.'
