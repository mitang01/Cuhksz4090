# HuBERT-first speech STRF pipeline

This phase provides a leakage-safe, configuration-driven pipeline for asking how
acoustic, prosodic, phonetic, onset, and word-level predictors explain held-out
HuBERT activations. It is methodologically inspired by Zhang et al., *Neuron*
(2026), DOI `10.1016/j.neuron.2025.10.011`; it is an independent implementation,
not an exact replication.

## Setup and smoke test

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[test]'
pytest
python3 scripts/smoke_test.py --config configs/smoke_test.yaml
```

The smoke command uses generated audio-like fixtures and makes a nested
group-CV results table plus SVG/PDF diagnostics without downloading a model.

## Real-data sequence

```bash
python3 scripts/build_manifest.py --config configs/data.yaml
python3 scripts/extract_features.py --config configs/features.yaml
# The completed legacy HuBERT extraction/fit/figures are now a locked reference.
python3 scripts/validate_model_registry.py
```

Extraction refuses to run unless the validation report is valid. Models are
local-cache-only unless `--allow-download` is explicit. Audio remains read-only;
do not commit or upload audio, annotations, transcripts, or activations.

The default 4090D extraction configuration uses CUDA FP16 with 20-second core
windows and one second of context on both sides. Activation frames are stitched
by their convolutional receptive-field centers; overlap frames are retained
exactly once. Completed recording groups are skipped on rerun. Use
`--overwrite` only to intentionally recompute them.

## Multi-model benchmark

`configs/models.yaml` is the model registry. Every nonreference model writes to
`outputs/<model-key>/`; the existing legacy `outputs/activations.h5`, fit
results, and figures are never modified. All adapters preserve native
timestamps and linearly interpolate audio-model states onto the same 50 Hz
audio-relative grid used by HuBERT. Downstream fitting uses only canonical-grid
states and the existing feature files, lags, grouped folds, metrics, reduced
models, and figure functions.

Validate registry structure and installed Transformers classes:

```bash
python3 scripts/validate_model_registry.py
```

A complete `hf download MODEL_ID --local-dir DIRECTORY` snapshot already
contains model weights, `config.json`, and the processor/tokenizer metadata.
No separate calculation creates these files. Transfer the entire directory;
manually copying only `pytorch_model.bin` or `model.safetensors` is insufficient.
BERT needs its tokenizer files and Whisper needs `preprocessor_config.json`.
Before submitting an offline job, verify that the exact checkpoint directory
exists on the compute node and directly contains `config.json`:

```bash
test -d "$CHECKPOINT" && test -f "$CHECKPOINT/config.json"
```

If this check fails, inspect the directory name and whether the download was
copied into an additional nested subdirectory. Absolute paths that do not exist
are rejected before Transformers can misinterpret them as Hugging Face repo IDs.

Run one model after staging its checkpoint locally. Pass the same local
checkpoint identity to every stage:

```bash
MODEL=wavlm_base_plus
CHECKPOINT=/share/home/mitan/models/wavlm-base-plus
python3 scripts/run_model_pipeline.py --model "$MODEL" --checkpoint "$CHECKPOINT" --stage extract
python3 scripts/run_model_pipeline.py --model "$MODEL" --checkpoint "$CHECKPOINT" --stage fit
python3 scripts/run_model_pipeline.py --model "$MODEL" --checkpoint "$CHECKPOINT" --stage figures
python3 scripts/compare_to_hubert_reference.py --model wavlm_base_plus
```

The same command pattern applies to these enabled keys:

```text
hubert_base
wav2vec2_base
wav2vec2_large
wavlm_base_plus
wavlm_large
data2vec_audio_base
xls_r_300m
whisper_medium_encoder
bert_base_uncased
```

For example:

```bash
python3 scripts/run_model_pipeline.py --model bert_base_uncased --stage all
python3 scripts/compare_to_hubert_reference.py --model bert_base_uncased
```

`mms_1b_all` is registered but disabled pending license review. The resolved
public W2V-BERT identifier is `facebook/w2v-bert-2.0`, pinned in the registry,
but `w2v_bert_2` remains disabled until its native frontend timing is validated
with the actual checkpoint.

To add a model, add one registry entry. Reuse `generic_speech` for waveform
encoders with a Hugging Face hidden-state model and convolutional timing. Add a
new adapter only for a genuinely different input/timing contract.

### BERT baseline

BERT uses labeled forced-aligned word intervals. A fast tokenizer retains
word-to-wordpiece ownership; wordpieces are mean-pooled per layer. Context is
split at punctuation or a configured 0.5-second gap. Each word vector is held
constant over its `[start, end)` interval on the canonical grid, with zeros
outside aligned words. Coverage and unknown-token fraction are recorded.
BERT-derived values are targets only and are explicitly forbidden from the
predictor feature set. Figures label BERT as a text-only baseline. Because the
checkpoint is English uncased while the stimuli are Mandarin, its tokenizer
coverage requires careful interpretation.

### Comparability and HuBERT regression

Every completed fit writes `comparability_contract.json` and
`run_metadata.json`. Comparison checks stimulus IDs, canonical grid, feature
order/configuration, lag grid, held-out groups, reduced models, metrics, and
result schema. `comparability_report.json` is machine-readable.

Freeze a contract beside the completed legacy HuBERT outputs once:

```bash
python3 scripts/freeze_hubert_reference.py
```

For a non-destructive HuBERT refactor rerun, provide a separate output:

```bash
export HUBERT=/share/home/mitan/models/hubert-large-ls960-ft/hubert-large-ls960-ft
python3 scripts/run_model_pipeline.py \
  --model hubert_large_reference \
  --checkpoint "$HUBERT" \
  --output outputs/hubert_large_refactor_rerun \
  --stage extract
python3 scripts/run_model_pipeline.py \
  --model hubert_large_reference \
  --checkpoint "$HUBERT" \
  --output outputs/hubert_large_refactor_rerun \
  --stage fit
python3 scripts/run_model_pipeline.py \
  --model hubert_large_reference \
  --checkpoint "$HUBERT" \
  --output outputs/hubert_large_refactor_rerun \
  --stage figures
python3 scripts/compare_to_hubert_reference.py \
  --model hubert_large_reference \
  --candidate-root outputs/hubert_large_refactor_rerun \
  --run-hubert-regression
```

`--stage fit` consumes `activations.h5` produced by `--stage extract` in the
same output directory; it does not load or extract the checkpoint itself. In a
Slurm script using `set -u`, define or export `HUBERT` before the first
`"$HUBERT"` expansion.

Regression tolerances are `2e-3` absolute for FP16 activations, `1e-6` seconds
for timestamps, and `1e-4` for downstream summaries.

### Cross-model predictability summary

After each candidate has passed `compare_to_hubert_reference.py`, aggregate any
completed set without rerunning a model:

```bash
python3 scripts/summarize_model_comparison.py \
  --model hubert_base \
  --model wavlm_base_plus \
  --model whisper_medium_encoder
```

The command writes `outputs/model_comparison/model_comparison_summary.csv`,
layer and feature-family detail tables, paired fold differences from HuBERT
Large, SVG/PDF predictability figures, and source hashes. It refuses candidates
whose comparability report did not pass. “Higher predictability” means the fixed
feature set explains more variance in that representation; it is not a general
model-quality or causal claim.

Input validation treats a configured overhang of at most 30 ms as a warning only
when the interval label is empty. Original TextGrid times remain unchanged.
Labeled intervals, larger overruns, missing files, and malformed timing remain
errors that block extraction.

## Revised lexical predictors

Extract contextual Mandarin word vectors once, then attach the same sparse
lexical events to both existing predictor designs:

```bash
python3 scripts/extract_lexical_embeddings.py \
  --manifest outputs/manifest.csv \
  --output outputs/stage4_revision/lexical_embeddings

python3 scripts/build_revised_predictors.py \
  --manifest outputs/manifest.csv \
  --original-features outputs/features \
  --rich-features outputs/stage4_revision/features_rich \
  --lexical-root outputs/stage4_revision/lexical_embeddings \
  --activation-store outputs/hubert_base/activations.h5 \
  --output outputs/stage4_revision/revised_predictors
```

The extractor resolves `hfl/chinese-roberta-wwm-ext` to an immutable Hugging
Face commit and records that commit in every archive. Revised archives remove
`recording_onset` by name. They keep the three word-timing columns dense and
store lexical vectors plus onset frame indexes separately, so each training
fold can reduce lexical dimensionality before additive onset rasterization and
lag expansion. Existing feature, activation, and lexical files are read-only;
reruns refuse to overwrite outputs unless `--overwrite` is explicit.

### Revised TRF fits and controls

`configs/revised_trf.yaml` fixes the nine checkpoint identities, five grouped
outer folds, four grouped inner folds, ridge grid, five lags, target/lexical
PCA limits, control scopes, null count, and inference families. Resolve the
actual HDF5 layers before submission:

```bash
python3 scripts/fit_revised_trf.py manifest
python3 scripts/fit_revised_trf.py local-smoke
```

First run and inspect the three required real-data benchmarks. The null uses
the completed middle-layer primary unit's folds, target coordinates, and
penalties:

```bash
MIDDLE=$(python3 - <<'PY'
import json
data = json.load(open(
    "outputs/stage4_revision/revised_trf/manifests/resolved_layers.json",
    encoding="utf-8",
))
print(data["hubert_base"]["selected_depths"]["middle"])
PY
)

/usr/bin/time -v python3 scripts/fit_revised_trf.py \
  fit hubert_base "$MIDDLE" 2>&1 | tee revised-primary-smoke.log
/usr/bin/time -v python3 scripts/run_revised_controls.py \
  fit rich hubert_base "$MIDDLE" 2>&1 | tee revised-rich-smoke.log
/usr/bin/time -v python3 scripts/run_revised_controls.py \
  null hubert_base 0 --layer "$MIDDLE" 2>&1 | tee revised-null-smoke.log
```

Review each unit's `status.json` for `wall_seconds` and `peak_rss_bytes`.
After those pass, submit the resumable arrays. Resource and site selection stay
on the `sbatch` command line:

```bash
mkdir -p outputs/stage4_revision/revised_trf/logs
PRIMARY_N=$(($(wc -l < outputs/stage4_revision/revised_trf/manifests/primary_units.tsv)-1))
CONTROL_N=$(($(wc -l < outputs/stage4_revision/revised_trf/manifests/control_units.tsv)-1))
NULL_N=$(($(wc -l < outputs/stage4_revision/revised_trf/manifests/null_units.tsv)-1))

PRIMARY_JOB=$(sbatch --parsable --partition=cpu \
  --array="0-$((PRIMARY_N-1))%3" --cpus-per-task=16 --mem=64G --time=24:00:00 \
  --output="outputs/stage4_revision/revised_trf/logs/primary-%A_%a.out" \
  --error="outputs/stage4_revision/revised_trf/logs/primary-%A_%a.err" \
  slurm/revised_primary_fits.sbatch)

CONTROL_JOB=$(sbatch --parsable --partition=cpu \
  --array="0-$((CONTROL_N-1))%3" --cpus-per-task=16 --mem=64G --time=24:00:00 \
  --output="outputs/stage4_revision/revised_trf/logs/control-%A_%a.out" \
  --error="outputs/stage4_revision/revised_trf/logs/control-%A_%a.err" \
  slurm/revised_selected_depth_controls.sbatch)

NULL_JOB=$(sbatch --parsable --partition=cpu \
  --dependency="afterok:${PRIMARY_JOB}" \
  --array="0-$((NULL_N-1))%3" --cpus-per-task=16 --mem=64G --time=24:00:00 \
  --output="outputs/stage4_revision/revised_trf/logs/null-%A_%a.out" \
  --error="outputs/stage4_revision/revised_trf/logs/null-%A_%a.err" \
  slurm/revised_word_shift_nulls.sbatch)
```

The control array may run concurrently with the primary array. The null array
must wait for primary completion because every null unit validates and reuses
its matched observed fit. Summarize only after all arrays finish:

```bash
python3 scripts/run_revised_controls.py summarize
```

### Revised result figures

Plotting-schema fit units include fold-specific predictions, target-PCA signs,
ridge coefficients, predictor scaling, and lag mappings. This schema change
invalidates earlier revised fit units safely; rerunning the primary array
preserves old directories with an `.invalid.<timestamp>` suffix and publishes
complete replacements.

Generate layer profiles after all grouped primary units complete:

```bash
python3 scripts/plot_layer_profiles.py
```

Generate a held-out PC1 trajectory for a prespecified recording and excerpt.
`best-lexical` uses inner-validation lexical conditional effect in that
recording's outer fold and never inspects its held-out trajectory:

```bash
python3 scripts/plot_predicted_activations.py \
  --model hubert_base \
  --recording-id story01 \
  --start-seconds 10 \
  --end-seconds 30 \
  --selection best-lexical
```

Generate the benchmark five-point kernel for the same displayed target
component and held-out fold:

```bash
python3 scripts/plot_trf_kernels.py \
  --model hubert_base \
  --layer layer_06_transformer \
  --recording-id story01 \
  --predictor broadband_envelope
```

Every command writes PDF and PNG figures, exact plotted-series CSV files, and
JSON metadata containing selection rules, score definitions, source hashes,
PCA sign conventions, and reconstruction details. The kernel command does not
silently interpolate or refit a denser lag grid.

## Method choices

Paper-compatible choices explicitly represented in configuration are the 50 Hz
analysis grid, HuBERT layer 0 plus all returned transformer hidden states, and a
separate zero-lag analysis. Runtime inspection—not a hard-coded layer count—
records representation count, dimensions, observed frame rate, model config,
requested revision, and resolved commit hash.

Project extensions are configurable temporal lags, nested story-grouped CV,
training-fold-only normalization and PCA, per-unit or PCA targets, and
independently tuned full-versus-reduced ridge models. `ΔR²` is named
**conditional unique contribution** and is not interpreted causally.

Optional phoneme-sequence statistics, word frequency, and lexical surprisal
remain disabled. Contextual lexical embeddings are available from the
forced-aligned Mandarin word intervals through the separate revised-predictor
namespace. Default temporal lags and ridge hyperparameters are project choices,
not claims about the paper.

The retained initial pool contains 12 recordings. `story17` and `story18` are
explicitly excluded in `configs/data.yaml` because they are nonsemantic control
conditions, not stimulus stories for this encoding analysis.

## Current scaling blockers

The configured `/share/workspace3/...` and `/share/home/mitan/...` mounts are
not visible in the Cloud Agent used to create this scaffold. Consequently,
`outputs/validation_report.json` records missing real inputs, and no real model
details are asserted. Before scaling, confirm TextGrid tier names and label
conventions, story grouping for segmented files, approved model-cache location,
compute/device policy, and the Mandarin lexical resources (if optional features
are desired).

