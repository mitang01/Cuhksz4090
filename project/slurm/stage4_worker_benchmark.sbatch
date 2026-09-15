#!/bin/bash
#SBATCH --job-name=s4-worker-bench
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/worker-benchmark-%j.out
#SBATCH --error=/share/home/mitan/Cuhksz4090/project/outputs/stage4_revision/logs/worker-benchmark-%j.err

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
DECISION="$MANIFEST_DIR/worker_benchmark.json"
python3 scripts/resolve_stage4_compute_scope.py \
  --config configs/stage4_revision.yaml \
  --output "$MANIFEST_DIR"

mapfile -t BENCHMARK < <(
  python3 - "$MANIFEST_DIR/resolved_model_layers.json" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text())["benchmark"]
print(value["model"])
print("\n".join(value["layers"]))
print(value["model_layer_count"])
PY
)
MODEL="${BENCHMARK[0]}"
LAYERS=("${BENCHMARK[1]}" "${BENCHMARK[2]}")
MODEL_LAYER_COUNT="${BENCHMARK[3]}"

SEQUENTIAL_START=$SECONDS
python3 scripts/run_stage4_revision.py \
  --config configs/stage4_revision.yaml \
  fast-refit "$MODEL" \
  --layers "${LAYERS[@]}" \
  --layer-workers 1 \
  --output-namespace "worker_benchmark_seq_${SLURM_JOB_ID}"
SEQUENTIAL_SECONDS=$((SECONDS - SEQUENTIAL_START))

PARALLEL_START=$SECONDS
set +e
python3 scripts/run_stage4_revision.py \
  --config configs/stage4_revision.yaml \
  fast-refit "$MODEL" \
  --layers "${LAYERS[@]}" \
  --layer-workers 2 \
  --output-namespace "worker_benchmark_parallel_${SLURM_JOB_ID}"
PARALLEL_STATUS=$?
set -e
PARALLEL_SECONDS=$((SECONDS - PARALLEL_START))

python3 - "$DECISION" "$MODEL" "${LAYERS[0]}" "${LAYERS[1]}" "$MODEL_LAYER_COUNT" \
  "$SEQUENTIAL_SECONDS" "$PARALLEL_SECONDS" "$PARALLEL_STATUS" <<'PY'
import json
import math
import os
from pathlib import Path
import statistics
import sys

path = Path(sys.argv[1])
if path.exists():
    raise SystemExit(f"Refusing to overwrite benchmark decision {path}")
layer_count = int(sys.argv[5])
sequential = int(sys.argv[6])
parallel = int(sys.argv[7])
parallel_status = int(sys.argv[8])
workers = 2 if parallel_status == 0 and parallel < sequential else 1
per_layer_one = sequential / 2
primary_seconds = (
    math.ceil(layer_count / 2) * parallel
    if workers == 2
    else layer_count * per_layer_one
)
null_seconds = per_layer_one * (4 / 6)
# Obtain complete primary counts from the all-layer task manifest.
scope = json.loads((path.parent / "resolved_model_layers.json").read_text())
pilot_times = [
    float(value["fit_runtime_seconds"])
    for value in scope["measured_hubert_base_original_resources"]
    if value.get("fit_runtime_seconds") is not None
]
pilot_median = statistics.median(pilot_times) if pilot_times else None
control_seconds = (
    None
    if pilot_median is None
    else (
        math.ceil(3 / 2) * pilot_median
        if workers == 2
        else 3 * pilot_median
    )
)
with (path.parent / "primary_fast_refits.tsv").open(encoding="utf-8") as stream:
    next(stream)
    primary_counts = {
        fields[1]: int(fields[3])
        for fields in (line.rstrip("\n").split("\t") for line in stream)
    }
primary_estimates = {
    model: (
        math.ceil(count / 2) * parallel
        if workers == 2 else count * per_layer_one
    )
    for model, count in primary_counts.items()
}
payload = {
    "state": "complete",
    "model": sys.argv[2],
    "layers": sys.argv[3:5],
    "model_layer_count": layer_count,
    "sequential_1_worker_8_threads_seconds": sequential,
    "parallel_2_workers_8_threads_seconds": parallel,
    "parallel_exit_status": parallel_status,
    "selected_layer_workers": workers,
    "fallback_reason": (
        None if workers == 2 else
        "parallel_failed_or_was_not_faster_than_one_worker"
    ),
    "measured_runtime_estimates": {
        "benchmark_model_all_layer_seconds": primary_seconds,
        "primary_seconds_by_model_from_layer_count": primary_estimates,
        "selected_depth_3_layer_task_seconds": control_seconds,
        "selected_depth_estimate_is_conservative": (
            "scaled from the completed HuBERT Base original median per-layer "
            "runtime, which includes both LORO and grouped fits"
        ),
        "one_fixed_alpha_null_unit_seconds": null_seconds,
        "basis": (
            "primary/null estimates use the measured fixed-alpha two-layer "
            "benchmark; control estimate uses measured completed-pilot runtime"
        ),
    },
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
