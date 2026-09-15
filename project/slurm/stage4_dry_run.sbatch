#!/bin/bash
#SBATCH --job-name=stage4-dry
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/dry-%j.out
#SBATCH --error=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/dry-%j.err

set -euo pipefail
START_SECONDS=$SECONDS

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

MODEL=hubert_base
NULL_COUNT=20
LOG_DIR=outputs/stage4_revision/logs
STDOUT_PATH="$LOG_DIR/dry-${SLURM_JOB_ID}.out"
STDERR_PATH="$LOG_DIR/dry-${SLURM_JOB_ID}.err"
TEST_REPORT="$LOG_DIR/dry-tests-${SLURM_JOB_ID}.xml"
EXIT_REPORT="$LOG_DIR/dry-exit-${SLURM_JOB_ID}.json"
CURRENT_COMPONENT=initialization
mkdir -p "$LOG_DIR"

write_exit_report() {
  local exit_status=$?
  python - "$EXIT_REPORT" "$exit_status" "$CURRENT_COMPONENT" \
    "$((SECONDS - START_SECONDS))" <<'PY'
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = {
    "state": "complete" if int(sys.argv[2]) == 0 else "failed",
    "exit_status": int(sys.argv[2]),
    "last_component": sys.argv[3],
    "elapsed_seconds": int(sys.argv[4]),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}
trap write_exit_report EXIT

run_component() {
  CURRENT_COMPONENT=$1
  shift
  echo "STAGE4_COMPONENT_START $CURRENT_COMPONENT $(date -u +%FT%TZ)"
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "$@"
  else
    "$@"
  fi
  echo "STAGE4_COMPONENT_END $CURRENT_COMPONENT $(date -u +%FT%TZ)"
}

run_component unit_tests python -m pytest -q \
  --junitxml="$TEST_REPORT" \
  tests/test_stage4_audit.py \
  tests/test_stage4_encoding.py \
  tests/test_stage4_features.py \
  tests/test_stage4_figures.py \
  tests/test_stage4_nulls.py \
  tests/test_stage4_qc.py \
  tests/test_stage4_runner.py \
  tests/test_stage4_statistics.py

run_component input_audit python scripts/audit_stage4_inputs.py \
  --manifest outputs/manifest.csv \
  --features outputs/features \
  --model-root outputs \
  --output outputs/stage4_revision/audit/input_audit.json

run_component provenance_qc python scripts/run_stage4_qc.py \
  --manifest outputs/manifest.csv \
  --model-root outputs \
  --output-dir outputs/stage4_revision/provenance_qc

python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path(
    "outputs/stage4_revision/provenance_qc/provenance_qc.json"
).read_text(encoding="utf-8"))
if report["status"] == "FAIL":
    raise SystemExit(f"Stage 4 QC failed: {report['reasons']}")
PY

# Gate 1: one minimal-alpha synthetic fit plus one real HuBERT Base
# layer/recording I/O, lag, null-shift, schema, and atomic-write check.
# Nested recording-level CV is deliberately not fit on the single recording.
run_component functional_smoke python scripts/run_stage4_revision.py \
  --config configs/stage4_revision.yaml \
  functional-smoke --model "$MODEL"

SMOKE_STATUS="$(
  python - <<'PY'
from pathlib import Path

paths = list(Path(
    "outputs/stage4_revision/functional_smoke/hubert_base"
).glob("*/*/status.json"))
if not paths:
    raise SystemExit("Functional smoke status was not published")
print(max(paths, key=lambda path: path.stat().st_mtime_ns))
PY
)"

# Gate 2 begins only after Gate 1 has published a complete status.
run_component rich_feature_extraction python scripts/extract_stage4_rich_features.py \
  --config configs/features_stage4_rich.yaml \
  --manifest outputs/manifest.csv \
  --validation-report outputs/validation_report.json \
  --output outputs/stage4_revision/features_rich

for variant in original rich capacity zero_lag; do
  run_component "pilot_fit_${variant}" python scripts/run_stage4_revision.py \
    --config configs/stage4_revision.yaml \
    fit "$MODEL" "$variant"
done

for null_index in $(seq 0 $((NULL_COUNT - 1))); do
  run_component "pilot_null_${null_index}" python scripts/run_stage4_revision.py \
    --config configs/stage4_revision.yaml \
    null "$MODEL" "$null_index" \
    --dry-null-count "$NULL_COUNT"
done

run_component pilot_integrity python scripts/validate_stage4_pilot.py \
  --config configs/stage4_revision.yaml \
  --model "$MODEL" \
  --null-count "$NULL_COUNT" \
  --job-id "$SLURM_JOB_ID" \
  --smoke-status "$SMOKE_STATUS" \
  --stdout "$STDOUT_PATH" \
  --stderr "$STDERR_PATH" \
  --test-report "$TEST_REPORT"

CURRENT_COMPONENT=complete
