# Python directed-connectivity pilot

`run_directed_conect.py` is the Python-only alternative to the SPM DCM
pipeline. It preserves the same:

- four BrainVision input recordings,
- trigger-1 trial extraction,
- all-onset and correct-response-2 ERPs,
- occipital/temporal ERP plots and statistics,
- preprocessing and proxy-montage warnings,
- pre-speech picture window,
- F1–F3 network hypotheses.

It replaces SPM DCM with source-space state-space spectral Granger causality
and time-reversed Granger causality from MNE-Connectivity.

## Scientific terminology

This analysis is **not DCM**. It cannot produce:

- DCM connection parameters,
- variational free energy,
- Bayesian model evidence,
- winning-model-family probabilities.

Instead, it estimates predictive directed dependence between source time
courses. F1–F3 are reported only as descriptive averages over prespecified
edge sets. Do not call any edge set a “winning DCM family.”

## Installation

Only Python packages are required:

```bash
cd /share/workspace2/tangmi/eeg_huashan
python3 -m pip install -r requirements.txt
```

On its first run, MNE downloads approximately 435 MB of open `fsaverage`
FreeSurfer template files. No MATLAB or SPM installation is needed. Set
`SUBJECTS_DIR` or pass `--subjects-dir` if the template is already present.

## Run

```bash
cd /share/workspace2/tangmi/eeg_huashan
python3 run_directed_conect.py
```

Regenerate existing output:

```bash
python3 run_directed_conect.py --overwrite
```

For an initial faster smoke run:

```bash
python3 run_directed_conect.py \
  --connectivity-bootstraps 20 \
  --overwrite
```

Use substantially more than 20 bootstrap replicates for reported results.
The default is 200; 1,000 or more is preferable for stable interval tails
when compute resources allow.

## Source model

The script uses:

- the matched MNE `GSN-HydroCel-128/129` or `standard_1005` proxy montage,
- MNE’s open `fsaverage` MRI, BEM and cortical source space,
- dSPM inverse estimates,
- left-hemisphere FreeSurfer `aparc` regions:
  - OT: lateral occipital + fusiform,
  - pMTG: middle temporal + banks STS,
  - ATL: temporal pole,
  - IFG: pars opercularis + triangularis + orbitalis.

These template choices are exploratory. They do not represent individual
head anatomy, cap placement, lesions, or postoperative anatomy.

## Connectivity statistic

For each planned direction, the script calculates:

```text
netTRGC = (GC X→Y − GC Y→X) − (time-reversed GC X→Y − time-reversed GC Y→X)
```

A positive value supports the listed direction relative to its reverse.
Time-reversal correction helps diagnose source mixing but does not eliminate
volume-conduction or inverse-leakage bias.

Picture data use only frequency bands with at least five cycles in the
available pre-speech window. Rest is segmented and analyzed separately.
Epoch bootstrapping provides participant-level 95% descriptive intervals.

## Outputs

Results are written to:

```text
directed_connectivity_results/<participant>/
```

ERP outputs remain at participant level. Directed outputs are under:

```text
directed/picture_all/
directed/picture_correct/
directed/rest/
```

Each directed folder contains:

- `directed_edges.csv`
- `hypothesis_edge_set_scores.csv`
- one network/interval plot per retained frequency band
- source ROI time courses
- the fsaverage inverse operator
- analysis metadata and warnings

`picture_correct` is the primary analysis. `picture_all` is a sensitivity
analysis containing behaviorally heterogeneous premature/missed trials.

## Interpretation limits

The two patients’ unknown-timepoint recordings support pipeline validation
and participant-level exploration only. Bootstrap trials are not independent
participants. Do not make population, longitudinal recovery, lesion-aware,
or postsurgical-reorganization claims from these results.
