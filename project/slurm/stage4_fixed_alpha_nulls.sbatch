#!/bin/bash
#SBATCH --job-name=s4-fixed-nulls
#SBATCH --array=0-79%3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/fixed-nulls-%A_%a.out
#SBATCH --error=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/fixed-nulls-%A_%a.err

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
TASKS="$MANIFEST_DIR/structured_nulls.tsv"
[[ -f "$TASKS" ]] || {
  echo "Resolved null manifest is required" >&2
  exit 2
}
python3 scripts/validate_stage4_compute_scope.py \
  --config configs/stage4_revision.yaml \
  --manifest-dir "$MANIFEST_DIR"

IFS=$'\t' read -r MODEL SHIFT LAYER < <(
  awk -F $'\t' -v task="$SLURM_ARRAY_TASK_ID" \
    'NR > 1 && $1 == task {print $2 "\t" $3 "\t" $4}' "$TASKS"
)
[[ -n "$MODEL" && -n "$SHIFT" && -n "$LAYER" ]] || {
  echo "No unique null task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
}

# One atomic MODEL × SHIFT × LAYER unit. Alphas come from the matching
# observed grouped fold and inner CV is intentionally not repeated.
/usr/bin/time -v python3 scripts/run_stage4_revision.py \
  --config configs/stage4_revision.yaml \
  fixed-alpha-null "$MODEL" "$SHIFT" \
  --layers "$LAYER" \
  --layer-workers 1
