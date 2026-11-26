#!/bin/bash
# Submit with --array, --cpus-per-task, --mem, --time, --output, and --error.

set -euo pipefail

ROOT="${PROJECT_ROOT:-/share/home/mitan/Cuhksz4090/project}"
CONFIG="${REVISED_TRF_CONFIG:-configs/revised_trf.yaml}"
MANIFEST="${REVISED_TRF_MANIFEST_DIR:-outputs/stage4_revision/revised_trf/manifests}/null_units.tsv"
CONDA_ENV="${CONDA_ENV:-data-proc}"
cd "$ROOT"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONHASHSEED=17

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script with an array range}"
IFS=$'\t' read -r task_id model layer variant outer_mode shift < <(
  awk -F $'\t' -v task="$TASK_ID" 'NR > 1 && $1 == task {print; exit}' "$MANIFEST"
)
if [[ -z "${model:-}" || "$task_id" != "$TASK_ID" || "$variant" != "null" ]]; then
  echo "No valid null manifest row for task $TASK_ID" >&2
  exit 2
fi

/usr/bin/time -v python3 scripts/run_revised_controls.py \
  --config "$CONFIG" null "$model" "$shift" --layer "$layer"
