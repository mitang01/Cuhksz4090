#!/bin/bash
#SBATCH --job-name=stage4-nulls
#SBATCH --array=0-899%3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/nulls-%A_%a.out
#SBATCH --error=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/nulls-%A_%a.err

set -euo pipefail

# Site-specific partition, account, and wall time must be supplied as sbatch
# arguments. Shell variables cannot be interpolated in #SBATCH directives.
ROOT=/share/home/mitan/Cuhksz4090/project
cd "$ROOT"
eval "$(conda shell.bash hook)"
conda activate data-proc

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONHASHSEED=17

MODELS_TSV=configs/stage4_models.tsv
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this file as an array job}"
if (( TASK_ID < 0 || TASK_ID >= 900 )); then
  echo "Array task must be in [0, 900)" >&2
  exit 2
fi

MODEL_ORDER=$((TASK_ID / 100))
NULL_INDEX=$((TASK_ID % 100))
MODEL="$(
  awk -F $'\t' -v order="$MODEL_ORDER" \
    'NR > 1 && $2 == order {print $3}' "$MODELS_TSV"
)"
if [[ -z "$MODEL" ]]; then
  echo "No unique model at order $MODEL_ORDER in $MODELS_TSV" >&2
  exit 2
fi
if [[ "$(awk -F $'\t' -v order="$MODEL_ORDER" \
  'NR > 1 && $2 == order {count++} END {print count + 0}' "$MODELS_TSV")" -ne 1 ]]; then
  echo "Model order $MODEL_ORDER is duplicated in $MODELS_TSV" >&2
  exit 2
fi

# Omitting --layer deliberately processes all discovered layers in this one
# model/null-index task. Completed integrity-valid layer units are resumed.
/usr/bin/time -v python scripts/run_stage4_revision.py \
  --config configs/stage4_revision.yaml \
  null "$MODEL" "$NULL_INDEX" \
  --full-null-count 100

python - "$MODEL" "$NULL_INDEX" <<'PY'
import json
import sys

print(json.dumps({
    "state": "complete",
    "kind": "stage4_full_null",
    "model": sys.argv[1],
    "null_index": int(sys.argv[2]),
    "all_layers": True,
    "cpu_only": True,
}, sort_keys=True))
PY
