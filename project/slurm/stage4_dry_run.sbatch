#!/bin/bash
#SBATCH --job-name=stage4-dry
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

run_timed() {
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "$@"
  else
    "$@"
  fi
}

run_timed python -m pytest -q \
  tests/test_stage4_audit.py \
  tests/test_stage4_encoding.py \
  tests/test_stage4_features.py \
  tests/test_stage4_figures.py \
  tests/test_stage4_nulls.py \
  tests/test_stage4_qc.py \
  tests/test_stage4_runner.py \
  tests/test_stage4_statistics.py

run_timed python scripts/audit_stage4_inputs.py \
  --manifest outputs/manifest.csv \
  --features outputs/features \
  --model-root outputs \
  --output outputs/stage4_revision/audit/input_audit.json

run_timed python scripts/extract_stage4_rich_features.py \
  --config configs/features_stage4_rich.yaml \
  --manifest outputs/manifest.csv \
  --validation-report outputs/validation_report.json \
  --output outputs/stage4_revision/features_rich

run_timed python scripts/run_stage4_qc.py \
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

run_timed python scripts/run_stage4_revision.py \
  --config configs/stage4_revision.yaml \
  synthetic-test

MODEL=hubert_base
LAYER="$(
  python - <<'PY'
from speech_strf.stage4_runner import discover_layers

print(discover_layers("outputs/hubert_base/activations.h5")[0])
PY
)"

# Inspect one real manifest recording and one real HuBERT Base canonical layer
# before any fit. This is an integrity smoke, not a synthetic fixture.
run_timed python - "$MODEL" "$LAYER" <<'PY'
import json
import sys

import h5py
import numpy as np
import pandas as pd

model, layer = sys.argv[1:]
manifest = pd.read_csv("outputs/manifest.csv")
recording_id = str(manifest.iloc[0]["recording_id"])
with np.load(f"outputs/features/{recording_id}.npz", allow_pickle=False) as features:
    feature_times = np.asarray(features["times"], dtype=np.float64)
    if not np.isfinite(features["matrix"]).all():
        raise SystemExit("real-recording feature matrix contains non-finite values")
with h5py.File(f"outputs/{model}/activations.h5", "r") as store:
    group = store[recording_id]
    if not bool(group.attrs.get("complete", False)):
        raise SystemExit("real-recording activation group is incomplete")
    values = np.asarray(group["canonical"][layer])
    activation_times = np.asarray(group["canonical_timestamps"], dtype=np.float64)
if values.ndim != 2 or values.shape[0] != feature_times.size:
    raise SystemExit("real recording-layer frame shape does not match features")
if not np.isfinite(values).all():
    raise SystemExit("real recording-layer activations contain non-finite values")
if not np.allclose(activation_times, feature_times, rtol=0.0, atol=1e-8):
    raise SystemExit("real recording-layer timestamps do not match features")
print(json.dumps({
    "state": "passed",
    "kind": "real_recording_layer_smoke",
    "model": model,
    "recording_id": recording_id,
    "layer": layer,
    "frames": int(values.shape[0]),
    "dimensions": int(values.shape[1]),
}, sort_keys=True))
PY

for variant in original rich capacity zero_lag; do
  run_timed python scripts/run_stage4_revision.py \
    --config configs/stage4_revision.yaml \
    fit "$MODEL" "$variant"
done

for null_index in $(seq 0 19); do
  run_timed python scripts/run_stage4_revision.py \
    --config configs/stage4_revision.yaml \
    null "$MODEL" "$null_index" \
    --dry-null-count 20
done

python - "$MODEL" "$LAYER" "$((SECONDS - START_SECONDS))" \
  "outputs/stage4_revision/logs/dry-acceptance-${SLURM_JOB_ID}.json" <<'PY'
import json
from pathlib import Path
import sys

payload = {
    "state": "complete",
    "kind": "stage4_dry_run",
    "model": sys.argv[1],
    "smoke_layer": sys.argv[2],
    "all_model_layers": True,
    "variants": ["original", "rich", "capacity", "zero_lag"],
    "dry_null_indexes_completed": list(range(20)),
    "dry_null_ensemble_size": 20,
    "cpu_only": True,
    "elapsed_seconds": int(sys.argv[3]),
    "exit_status": 0,
    "peak_memory_evidence": "see /usr/bin/time -v entries and Slurm MaxRSS",
}
Path(sys.argv[4]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
