#!/bin/bash
#SBATCH --job-name=s4-depth-controls
#SBATCH --array=0-20%3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/depth-controls-%A_%a.out
#SBATCH --error=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/depth-controls-%A_%a.err

set -euo pipefail
ROOT=/share/home/mitan/Cuhksz4090/project
cd "$ROOT"
eval "$(conda shell.bash hook)"
conda activate data-proc

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=17

MANIFEST_DIR=outputs/stage4_revision/manifests/deadline_compute_scope
TASKS="$MANIFEST_DIR/selected_depth_controls.tsv"
DECISION="$MANIFEST_DIR/worker_benchmark.json"
[[ -f "$TASKS" && -f "$DECISION" ]] || {
  echo "Resolved manifests and worker benchmark are required" >&2
  exit 2
}
python3 scripts/validate_stage4_compute_scope.py \
  --config configs/stage4_revision.yaml \
  --manifest-dir "$MANIFEST_DIR"

IFS=$'\t' read -r MODEL VARIANT INPUT MIDDLE FINAL < <(
  awk -F $'\t' -v task="$SLURM_ARRAY_TASK_ID" \
    'NR > 1 && $1 == task {print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6}' "$TASKS"
)
MODEL=${MODEL%$'\r'}
VARIANT=${VARIANT%$'\r'}
INPUT=${INPUT%$'\r'}
MIDDLE=${MIDDLE%$'\r'}
FINAL=${FINAL%$'\r'}
[[ -n "$MODEL" && -n "$VARIANT" && -n "$FINAL" ]] || {
  echo "No unique control task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
}
WORKERS="$(
  python3 - "$DECISION" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads(Path(sys.argv[1]).read_text())["selected_layer_workers"])
PY
)"

/usr/bin/time -v python3 scripts/run_stage4_revision.py \
  --config configs/stage4_revision.yaml \
  fit "$MODEL" "$VARIANT" \
  --layers "$INPUT" "$MIDDLE" "$FINAL" \
  --layer-workers "$WORKERS"
