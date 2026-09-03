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
# Run only after reviewing the storage estimate and obtaining download approval:
python3 scripts/extract_activations.py --config configs/models.yaml \
  --confirm-download --recording-id szdt1
# If the one-recording check succeeds, run all; szdt1 is safely skipped:
python3 scripts/extract_activations.py --config configs/models.yaml
python3 scripts/fit_strf.py --config configs/analysis.yaml --layer layer_00_input
python3 scripts/make_figures.py --config configs/analysis.yaml
```

Extraction refuses to run unless the validation report is valid. The checkpoint
command first tries the local Hugging Face cache and otherwise requires the
explicit `--confirm-download` switch. Audio remains read-only; outputs contain
paths and derived data only. Do not commit or upload audio, annotations,
transcripts, or activations.

The default 4090D extraction configuration uses CUDA FP16 with 20-second core
windows and one second of context on both sides. Activation frames are stitched
by their convolutional receptive-field centers; overlap frames are retained
exactly once. Completed recording groups are skipped on rerun. Use
`--overwrite` only to intentionally recompute them.

Input validation treats a configured overhang of at most 30 ms as a warning only
when the interval label is empty. Original TextGrid times remain unchanged.
Labeled intervals, larger overruns, missing files, and malformed timing remain
errors that block extraction.

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

Optional phoneme-sequence statistics, word frequency, lexical surprisal, and
contextual embeddings are disabled and unavailable until suitable Mandarin
transcripts/resources are supplied. Default temporal lags and ridge
hyperparameters are project choices, not claims about the paper.

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

