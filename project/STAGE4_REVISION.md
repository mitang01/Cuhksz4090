# Stage 4 revision: CPU Slurm runbook

## Fixed scope and limitations

Run from the remote repository root:

```bash
cd /share/home/mitan/Cuhksz4090/project
```

The Stage 4 model set is exactly the nine ordered rows in
`configs/stage4_models.tsv`. The first fit array dimension has three groups,
each containing three models; `--array=0-2%3` therefore permits at most three
concurrent fit tasks. The null array is independently resumable and maps
`0-899` deterministically to nine models by 100 null indexes, with at most
three tasks concurrent. Each null task processes all discovered layers.

All three job files are CPU-only: they request 16 CPUs and 64 GB, request no
GPU resource, clear `CUDA_VISIBLE_DEVICES`, and set `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to 16.

The Cloud workspace used to prepare this runbook does not contain the real
audio, TextGrids, feature archives, or nine activation stores mounted at the
remote paths. Consequently, no real-input acceptance result is reported here.
The commands below must run against the real inputs on the remote cluster
before any scientific result is accepted.

The predictor family name remains **phonetic**. It means framewise indicators
for labeled phone intervals, using the deterministic global union of nonempty
labels from tiers named `phone`, `phones`, `phoneme`, or `phonemes` across the
manifest. Do not rename this family to `phone` or `phoneme`.

## Prespecified estimands and null hypotheses

The primary unit is one held-out recording. For each model, layer, recording,
and family, the effect is full-model R² minus that family's reduced-model R².
Layers are then averaged arithmetically within each model × recording × family
cell before inference. The primary estimand is the equal-recording mean across
the 12 benchmark recordings; duration weighting is secondary. The exact
two-sided sign-flip null is sign symmetry of the 12 paired recording effects
around zero. Its scope is consistency across this fixed benchmark, not a
population claim about narratives. Five-family Benjamini–Hochberg correction
is performed separately within each model.

The matched-null estimand is the recording-level observed conditional ΔR²
minus the mean conditional ΔR² over 100 seeded, within-recording circular
shifts. Inference again occurs only after layer averaging and never treats
layers or checkpoints as independent replicates. The fixed layer aggregation
rule is not selected from the results.

## Synchronize code without touching results

The author selected a direct Git update on the cluster rather than `rsync`.
Run this exact equivalent from the cluster:

```bash
git -C /share/home/mitan/Cuhksz4090 fetch origin \
  cursor/stage4-recording-inference-cc5a
git -C /share/home/mitan/Cuhksz4090 switch \
  cursor/stage4-recording-inference-cc5a
git -C /share/home/mitan/Cuhksz4090 pull --ff-only origin \
  cursor/stage4-recording-inference-cc5a
```

Git does not track the existing ignored `project/outputs/` products, so these
commands do not delete or overwrite completed activation or fit results.

The locked governance inputs are intentionally not copied or edited by this
branch. Before the dry run, both existing author-controlled files must be
available at the paths resolved from the project root:

```bash
cd /share/home/mitan/Cuhksz4090/project
test -f ../paper1_pipeline_20260914/stage3_review/revision_roadmap.json
test -f ../paper1_pipeline_20260914/stage4_revision/claim_surface_manifest.json
```

If either check fails, place the author's existing file at that exact path
before analysis. Do not create a substitute; the input audit fails closed.

## Site settings and log directory

Slurm reads `#SBATCH` lines before starting the shell, so shell variables do
not interpolate in directives. Keep the resource defaults in the files and
override editable site values with `sbatch` arguments:

```bash
cd /share/home/mitan/Cuhksz4090/project
mkdir -p outputs/stage4_revision/logs

PARTITION=EDIT_ME
ACCOUNT=EDIT_ME
DRY_WALL=04:00:00
FIT_WALL=24:00:00
NULL_WALL=24:00:00
SITE_ARGS=(--partition="$PARTITION" --account="$ACCOUNT")
```

If the cluster does not require an account, define
`SITE_ARGS=(--partition="$PARTITION")`. Command-line `--time` overrides the
literal `#SBATCH --time` value. Standard output and error are written beneath
`outputs/stage4_revision/logs`.

## Exact preflight and dry run

The dry job uses only the `hubert_base` activation checkpoint. In order, it
runs the eight Stage 4 test files, the fail-closed nine-model input audit, rich
feature extraction, provenance/annotation QC, the deterministic synthetic
test, one real manifest-recording/canonical-layer integrity smoke, all four fit
variants for every HuBERT Base layer, and all 20 nonfinal dry null shifts for
every HuBERT Base layer.

The corresponding direct commands, useful for diagnosing a failed dry job,
are:

```bash
pytest -q \
  tests/test_stage4_audit.py \
  tests/test_stage4_encoding.py \
  tests/test_stage4_features.py \
  tests/test_stage4_figures.py \
  tests/test_stage4_nulls.py \
  tests/test_stage4_qc.py \
  tests/test_stage4_runner.py \
  tests/test_stage4_statistics.py

python scripts/audit_stage4_inputs.py \
  --manifest outputs/manifest.csv \
  --features outputs/features \
  --model-root outputs \
  --output outputs/stage4_revision/audit/input_audit.json

python scripts/extract_stage4_rich_features.py \
  --config configs/features_stage4_rich.yaml \
  --manifest outputs/manifest.csv \
  --validation-report outputs/validation_report.json \
  --output outputs/stage4_revision/features_rich

python scripts/run_stage4_qc.py \
  --manifest outputs/manifest.csv \
  --model-root outputs \
  --output-dir outputs/stage4_revision/provenance_qc

python scripts/run_stage4_revision.py \
  --config configs/stage4_revision.yaml synthetic-test
```

`slurm/stage4_dry_run.sbatch` contains the exact real recording-layer smoke
and dry fit/null commands because its discovered layer is intentionally not
hard-coded. Submit it with:

```bash
DRY_JOB=$(sbatch --parsable "${SITE_ARGS[@]}" --time="$DRY_WALL" \
  slurm/stage4_dry_run.sbatch)
echo "$DRY_JOB"
sacct -j "$DRY_JOB" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS
```

Do not submit full fits or production nulls unless the dry job is `COMPLETED`
with exit code `0:0`, its final JSON status says `"state": "complete"`, and all
of these checks succeed:

```bash
python - <<'PY'
import json
from pathlib import Path

audit = json.loads(Path(
    "outputs/stage4_revision/audit/input_audit.json"
).read_text(encoding="utf-8"))
synthetic = json.loads(Path(
    "outputs/stage4_revision/synthetic_test/result.json"
).read_text(encoding="utf-8"))
qc = json.loads(Path(
    "outputs/stage4_revision/provenance_qc/provenance_qc.json"
).read_text(encoding="utf-8"))
assert audit["complete"] is True, audit["errors"]
assert synthetic["passed"] is True, synthetic
assert qc["status"] != "FAIL", qc["reasons"]
print({"audit": "complete", "synthetic": "passed", "qc": qc["status"]})
PY
```

Review every QC warning rather than treating `WARN` as automatic approval.

## Full fits and production nulls

Only after the preceding dry-run success criteria are met, submit:

```bash
FIT_JOB=$(sbatch --parsable "${SITE_ARGS[@]}" --time="$FIT_WALL" \
  slurm/stage4_full_fits.sbatch)
NULL_JOB=$(sbatch --parsable "${SITE_ARGS[@]}" --time="$NULL_WALL" \
  slurm/stage4_nulls.sbatch)
printf 'FIT_JOB=%s\nNULL_JOB=%s\n' "$FIT_JOB" "$NULL_JOB"
```

The fit tasks run `original`, `rich`, `capacity`, and `zero_lag` in that fixed
order for every discovered layer. The null command always passes
`--full-null-count 100`; dry nulls are stored separately and are never eligible
for final inference. Rerunning either array is safe: a unit resumes only when
its status, identity, source/config hashes, artifact hashes, and artifact
schemas validate. Missing or invalid units are recomputed, while an existing
invalid unit is preserved with a corruption diagnostic.

Inspect array status:

```bash
sacct -j "$FIT_JOB","$NULL_JOB" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS
squeue -j "$FIT_JOB","$NULL_JOB"
```

If individual tasks fail, resubmit only those array indexes after diagnosing
the corresponding `.err` file. Do not summarize partial production nulls.

After every fit and null array task is `COMPLETED` with exit code `0:0`, verify
the integrity-valid unit counts and then summarize:

```bash
python - <<'PY'
from pathlib import Path

from speech_strf.stage4_audit import STAGE4_MODEL_DIRECTORIES
from speech_strf.stage4_runner import VARIANTS, discover_layers, validate_unit

root = Path("outputs/stage4_revision")
fit_valid = fit_expected = null_valid = null_expected = 0
per_model = {}
for model in STAGE4_MODEL_DIRECTORIES:
    layers = discover_layers(Path("outputs") / model / "activations.h5")
    per_model[model] = len(layers)
    for variant in VARIANTS:
        for layer in layers:
            fit_expected += 1
            fit_valid += validate_unit(
                root / {
                    "original": "fits_original_recording_level",
                    "rich": "fits_rich_acoustic",
                    "capacity": "fits_capacity_matched",
                    "zero_lag": "zero_lag",
                }[variant] / model / layer
            )[0]
    for layer in layers:
        for index in range(100):
            null_expected += 1
            null_valid += validate_unit(
                root / "null_controls" / "full" / model / layer / f"null_{index:03d}"
            )[0]
summary = {
    "layers_per_model": per_model,
    "fit_units": {"valid": fit_valid, "expected": fit_expected},
    "full_null_units": {"valid": null_valid, "expected": null_expected},
}
print(summary)
if fit_valid != fit_expected or null_valid != null_expected:
    raise SystemExit("Stage 4 production units are incomplete or fail integrity")
PY

python scripts/run_stage4_revision.py \
  --config configs/stage4_revision.yaml summarize
python scripts/make_stage4_figures.py \
  --source-dir outputs/stage4_revision/figures/source_tables \
  --output-dir outputs/stage4_revision/figures
```

The expected fit count is four variants times the sum of the nine discovered
layer counts. The expected production-null count is 100 times that layer sum.
Final inference must use only `outputs/stage4_revision/null_controls/full`;
never mix in `null_controls/dry_nonfinal`.

## Integrity expectations

- `outputs/stage4_revision/audit/input_audit.json` has `complete: true`, exactly
  nine model directories, matching saved identities, complete HDF5 groups,
  finite arrays, and canonical timestamps matching original feature times
  within `1e-8` seconds.
- Every rich feature `.npz` has a matching `.sha256` sidecar and matching
  provenance hashes for its config, manifest, validation report, audio,
  alignment, extractor, and global phone-category union.
- Every fit/null leaf has `status.json` with `schema_version: 1`,
  `state: complete`, the correct model/layer/variant/null identity, matching
  config and immutable-source hashes, and SHA-256/schema-valid
  `scores.csv`, `split_reports.json`, `pca_reports.json`, and
  `capacity_reports.json`. Observed-fit units also require `predictions.npz`;
  null units intentionally omit predictions to prevent unbounded redundant
  storage.
- Full null completeness is exactly indexes `000` through `099` for every
  model-layer pair. Dry null artifacts remain nonfinal and separate.
- Summary and figure generation happen only after complete integrity checks;
  source CSVs, not rendered figures, are the auditable numerical record.

## Download list

Download these review artifacts after successful completion:

```text
outputs/stage4_revision/audit/input_audit.json
outputs/stage4_revision/provenance_qc/provenance_qc.json
outputs/stage4_revision/provenance_qc/provenance_qc.csv
outputs/stage4_revision/provenance_qc/recording_annotation_qc.csv
outputs/stage4_revision/provenance_qc/temporal_context.csv
outputs/stage4_revision/provenance_qc/annotation_totals.csv
outputs/stage4_revision/synthetic_test/result.json
outputs/stage4_revision/model_comparison/
outputs/stage4_revision/pca_coverage/
outputs/stage4_revision/figures/source_tables/
outputs/stage4_revision/figures/
outputs/stage4_revision/logs/
```

Retain the full remote `fits_*`, `zero_lag/`, `null_controls/full/`, and
rich-feature archives as primary reproducibility outputs. They are usually too
large for a routine review download; if archived, preserve directory structure
and SHA-256 sidecars/status files together.
