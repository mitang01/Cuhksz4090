#!/bin/bash
#SBATCH --job-name=s4-fast-primary
#SBATCH --array=0-7%3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/fast-primary-%A_%a.out
#SBATCH --error=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/fast-primary-%A_%a.err

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
TASKS="$MANIFEST_DIR/primary_fast_refits.tsv"
DECISION="$MANIFEST_DIR/worker_benchmark.json"
[[ -f "$TASKS" && -f "$DECISION" ]] || {
  echo "Resolved manifests and worker benchmark are required" >&2
  exit 2
}

IFS=$'\t' read -r MODEL LAYER_LIST < <(
  awk -F $'\t' -v task="$SLURM_ARRAY_TASK_ID" \
    'NR > 1 && $1 == task {print $2 "\t" $3}' "$TASKS"
)
[[ -n "$MODEL" && -n "$LAYER_LIST" ]] || {
  echo "No unique primary task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
}
IFS=',' read -r -a LAYERS <<< "$LAYER_LIST"
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
  fast-refit "$MODEL" \
  --layers "${LAYERS[@]}" \
  --layer-workers "$WORKERS"
