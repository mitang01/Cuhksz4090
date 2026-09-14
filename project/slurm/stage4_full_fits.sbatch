#!/bin/bash
#SBATCH --job-name=stage4-fits
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/full-fits-%A_%a.out
#SBATCH --error=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/full-fits-%A_%a.err

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

run_timed() {
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "$@"
  else
    "$@"
  fi
}

MODELS_TSV=configs/stage4_models.tsv
GROUP="${SLURM_ARRAY_TASK_ID:?submit this file as an array job}"

python - "$MODELS_TSV" <<'PY'
import csv
import sys

expected = [
    ("0", "0", "hubert_base", "facebook/hubert-base-ls960"),
    ("0", "1", "hubert_large_refactor_rerun", "facebook/hubert-large-ls960-ft"),
    ("0", "2", "wav2vec2_base", "facebook/wav2vec2-base-960h"),
    ("1", "3", "wav2vec2_large", "facebook/wav2vec2-large-960h-lv60-self"),
    ("1", "4", "wavlm_base_plus", "microsoft/wavlm-base-plus"),
    ("1", "5", "wavlm_large", "microsoft/wavlm-large"),
    ("2", "6", "data2vec_audio_base", "facebook/data2vec-audio-base-960h"),
    ("2", "7", "xls_r_300m", "facebook/wav2vec2-xls-r-300m"),
    ("2", "8", "whisper_medium_encoder", "openai/whisper-medium"),
]
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    reader = csv.reader(stream, delimiter="\t")
    header = next(reader, None)
    rows = [tuple(row) for row in reader]
if header != ["group", "order", "model", "checkpoint"] or rows != expected:
    raise SystemExit("configs/stage4_models.tsv is not the fixed ordered nine-model manifest")
PY

mapfile -t MODELS < <(
  awk -F $'\t' -v group="$GROUP" 'NR > 1 && $1 == group {print $3}' "$MODELS_TSV"
)
if [[ "${#MODELS[@]}" -ne 3 ]]; then
  echo "Expected exactly three models in group $GROUP" >&2
  exit 2
fi

for model in "${MODELS[@]}"; do
  for variant in original rich capacity zero_lag; do
    run_timed python scripts/run_stage4_revision.py \
      --config configs/stage4_revision.yaml \
      fit "$model" "$variant"
  done
done

python - "$GROUP" "${MODELS[@]}" <<'PY'
import json
import sys

print(json.dumps({
    "state": "complete",
    "kind": "stage4_full_fits",
    "group": int(sys.argv[1]),
    "models": sys.argv[2:],
    "variants": ["original", "rich", "capacity", "zero_lag"],
    "all_layers": True,
    "cpu_only": True,
}, sort_keys=True))
PY
