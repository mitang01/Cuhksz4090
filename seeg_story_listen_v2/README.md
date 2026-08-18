# Story-listening sEEG preprocessing

`preprocess_seeg.py` implements the requested preprocessing and
speech-responsiveness analysis in Python:

1. Remove 50, 100, 150, 200, and 250 Hz line components (when below Nyquist) with
   MNE's regression/multitaper `spectrum_fit` method.
2. Group numbered contacts by electrode-shaft prefix and apply a
   Laplacian-style reference. Interior contacts are referenced to the mean of
   their two immediate neighbors; shaft endpoints are referenced to their one
   immediate neighbor.
3. Extract amplitude in delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta
   (13–30 Hz), gamma (30–70 Hz), and high gamma (70–150 Hz). Each range is
   divided into a 1/7-octave filterbank, Hilbert magnitudes are averaged, and
   the result is resampled to 128 Hz.
4. Z-score each continuous band using samples pooled across the pre-track
   baselines (default: -1.0 to -0.05 seconds).
5. Read non-empty interval onsets from the first TextGrid `IntervalTier`,
   excluding `story18`, and create -1 to +2 second token epochs.
6. Define speech-responsive electrodes independently in every band: compare
   each token's 50–200 ms mean with its -200 to -50 ms mean using a one-sided
   paired Wilcoxon signed-rank test, then apply Benjamini-Hochberg FDR across
   contacts separately for every recording and band at q=0.01.

The script deliberately uses the requested spelling `prepocessed` in EDF
filenames.

## Install on the cluster

```bash
python3 -m venv ~/.venvs/seeg-story
source ~/.venvs/seeg-story/bin/activate
python3 -m pip install -r seeg_story_listen_v2/requirements.txt
```

MNE's EDF export uses `edfio`, included in the requirements.

## Validate metadata first

Run the metadata-only pass before filtering:

```bash
python3 seeg_story_listen_v2/preprocess_seeg.py --dry-run
```

Defaults match the cluster layout:

- input: `/share/workspace3/ieeg/seeg/story_listen_v2`
- output: `/share/home/mitan/seeg_story_listen_v2`
- trigger map: `<input>/event_stimuli.csv`
- WAV files: `<input>/stimuli_wav`
- TextGrids: `<input>/stimuli_textgrid`

The event reader accepts either:

- one row per track with `stimulus`, `onset`, and `offset` columns; or
- one row per trigger with `time` and `trigger` columns. Repeated
  trigger-to-stimulus mappings are paired as onset then offset unless an
  onset/offset phase is explicit.

Headerless subject event files are also detected when rows have the cluster
format `trigger_label,time,auxiliary_value`, for example
`onset_1,16.743,0`. The first row is retained as data. Subject event files are
matched by the exact numeric subject identifier and same-directory files are
preferred, so digits in parent directory names such as `story_listen_v2` do
not affect matching. Files that wrap each complete row in quotes, for example
`"onset_1, 98.4135, 0"`, are expanded automatically.

Common alternative column names are detected. For nonstandard event tables,
use `--event-time-column NAME` and `--event-label-column NAME`.
The supplied `event_stimuli.csv` headers `trigger_label` and
`stimuli_filename` are recognized; surrounding header whitespace is ignored.
Its `silence` column is applied separately to every mapped onset/off trigger:

```text
corrected onset = onset trigger time - onset silence
corrected offset = offset trigger time + offset silence
```

Thus an onset trigger at 110 seconds with `silence=2` becomes 108 seconds,
while an offset trigger at 110 seconds with `silence=2` becomes 112 seconds.
Corrected times drive audio-duration QC, track baselines, TextGrid token
alignment, and speech-responsive electrode testing. Both raw trigger times and
corrections are retained in `audio_trigger_duration_qc.csv`.

Inspect every generated `audio_trigger_duration_qc.csv`. A row fails the
default check when the onset-to-offset interval differs from WAV duration by
more than 0.1 seconds. Change this reporting threshold with
`--duration-tolerance`.

## Run preprocessing

```bash
python3 seeg_story_listen_v2/preprocess_seeg.py
```

To restart and replace existing outputs:

```bash
python3 seeg_story_listen_v2/preprocess_seeg.py --overwrite
```

To process selected bands:

```bash
python3 seeg_story_listen_v2/preprocess_seeg.py \
  --bands delta theta alpha beta gamma high_gamma
```

The input sampling rate must exceed twice the upper edge of each requested
band. In particular, high gamma requires a sampling rate above 300 Hz.

## Outputs

Input subdirectories are preserved under the output root to prevent recordings
with identical names from overwriting each other.

For each source EDF and frequency band:

- `<original>_prepocessed_<band>.edf`: continuous, referenced, 128 Hz,
  baseline-z-scored Hilbert amplitude for every valid sEEG contact.
- `<original>_responsive_<band>.edf`: the same continuous time-domain data,
  restricted to contacts selected independently for that frequency band.
  Responsive channel names can therefore differ between bands.
- `<original>_qc/<band>_speech_responsiveness.csv`: effect, raw p-value,
  FDR-adjusted p-value, and selection decision for every contact in that band.
- `<original>_qc/<band>_mean_token_erp.npz`: time axis and mean token-locked
  z-scored response for every contact.
- `<original>_qc/<band>_responsive_token_epochs.npz`: individual token epochs
  for responsive contacts.

Additional QC files record audio/trigger duration agreement, TextGrid token
counts, unmapped/unpaired event warnings, dropped channel names, reference
details, parameters, and source paths. EDF cannot represent zero channels, so
when no contact passes FDR the script writes
`<band>_no_responsive_channels.txt` instead of an invalid empty responsive EDF.
`<band>_speech_response_diagnostics.json` summarizes token count, minimum raw
and adjusted p-values, maximum effect, and the final number selected.
`<band>_top_candidates.csv` lists the 20 strongest positive candidates,
including channels that did not pass q=0.01. These per-band files distinguish
a strict statistical non-result from event-alignment or token-count problems.

The derived signals are dimensionless z-scores, but EDF has no standardized
z-score physical unit. The files therefore use the explicit convention
**1 z-unit = 1 µV**. With MNE, recover the numeric z-score values using
`raw.get_data(units="uV")`. This convention is also recorded in every
`processing_metadata.json`.

## Important checks

- Channel labels must end in a contact number, such as `LA1`, `LA2`, or
  `LA-01`. Labels without this structure are listed as dropped. Override
  non-sEEG label exclusion with `--exclude-channel-regex`.
- Missing TextGrids for non-`story18` tracks stop full preprocessing rather
  than silently reducing the speech-token set.
- Existing results made with the former 60 Hz defaults must be regenerated
  with `--overwrite`; otherwise the script intentionally refuses to replace
  them.
- EDF outputs are written through temporary files and reopened to verify
  channel order and sample count before being finalized.
- Keep the raw EDFs as the archival data. Hilbert-amplitude EDFs are derived
  features, not conventional band-passed voltage traces.

## Plot full-duration band overviews

`plot_band_edfs.py` uses Matplotlib's non-interactive `Agg` backend and creates
one PNG for each subject/recording. Available bands are placed side by side in
canonical order, with every channel shown over the complete recording.
Min/max time-bin reduction keeps long recordings practical while retaining
brief extrema.

```bash
python3 seeg_story_listen_v2/plot_band_edfs.py
```

By default it discovers
`/share/home/mitan/seeg_story_listen_v2/**/*_prepocessed_<band>.edf` and writes
PNG files below `/share/home/mitan/seeg_story_listen_v2/plots`, preserving
subject subdirectories. The display range is **±0.005 mV** (±5 µV) around each
channel baseline and traces are clipped at that range so artifacts do not
flatten the remaining channels. For the derived EDF convention of
1 z-unit = 1 µV, this displays approximately ±5 z-units.

Useful options:

```bash
python3 seeg_story_listen_v2/plot_band_edfs.py \
  --scale-mv 0.005 \
  --max-time-bins 1500 \
  --dpi 150 \
  --overwrite
```

Use `--kind responsive` to plot responsive-band EDFs instead. Because
responsive contacts are selected independently by band, those panels can have
different channel sets. Use `--bands delta theta high_gamma` to limit the
included panels.

## Fit L2 ridge-regularized STRFs

`run_strf.py` fits five stimulus-to-neural encoding models:

1. mel
2. mel + syllable onset
3. mel + syllable onset + boundary strength
4. mel + syllable onset + prosodic structure depth
5. all four feature families

The neural response defaults to the preprocessed high-gamma amplitude. Contacts
are retained when `fdr_p_value < 0.05` and the speech-response effect is
positive. Log-mel power is extracted from each WAV, `syl_onset` impulses come
from non-empty intervals in the first TextGrid `IntervalTier`, and prosody comes
from `<story>.prosodic_word_depth.tsv`:

- `end` supplies the boundary time;
- `boundary_strength_after` supplies `boundary_strength`;
- `prosodic_word_depth` supplies `struc_depth`.

The estimator uses `reg_type="ridge"`, applying ordinary L2 regularization to
all lagged coefficients without treating neighboring feature columns as a
feature-space adjacency. Regularization strength is selected by nested,
stimulus-grouped cross-validation. No neighboring samples from the same
stimulus are randomly split between training and testing.

Contact selection uses the responsiveness CSV generated from the complete
recording, as requested. Predictive accuracy is therefore conditional on this
preselected responsive-contact population; it is not an unbiased estimate for
all implanted contacts.

Before a full run, validate file discovery and annotation schemas:

```bash
python3 seeg_story_listen_v2/run_strf.py --validate-only
```

Run the complete analysis with 1,000 held-out permutations:

```bash
python3 seeg_story_listen_v2/run_strf.py --overwrite
```

Use `--max-recordings 1 --n-permutations 20` for a computational pilot. The
cluster paths supplied to the preprocessing pipeline are also the STRF
defaults; every path can be overridden on the command line.

Each track is divided into independent fixed-duration epochs before fitting so
that time delays never cross stimulus boundaries. The last incomplete epoch is
excluded and its retained duration is visible in `alignment_qc.csv`.

### STRF outputs

The output root defaults to `/share/home/mitan/seeg_story_listen_v2/strf` and
contains:

- `recording_manifest.csv`: EDF, event, audio, TextGrid, prosody, and corrected
  neural timing paths;
- `analysis_config.json`: complete parameters, models, and permutation-test
  definition;
- `alignment_qc.csv` and `aligned_data/**/*.npz`: alignment diagnostics and
  analysis-ready arrays;
- `recordings/*/cv_folds.csv`: reproducible outer stimulus folds;
- `model_metrics.csv`: held-out R², correlation, and MSE per model, fold, and
  contact;
- `stimulus_model_metrics.csv`: the corresponding held-out metrics per
  stimulus, including an auxiliary training-mean `M0_null` baseline used only
  to test the mel contribution;
- `alpha_selection.csv`: inner-CV regularization results;
- `model_comparisons.csv`: nested-model delta R² with outer-fold-blocked
  sign-flip permutation p-values and BH-FDR values;
- `feature_contributions.csv`: held-out full-versus-reduced delta R² for each
  feature. Stimulus deltas are averaged within each outer fold, significance
  uses 1,000 paired sign-flip permutations across those fold blocks, and
  p-values receive BH-FDR correction across contacts and features;
- `predictions_outer_fold_*.npz` and `model_coefficients.npz`: held-out
  predictions and fold-specific filters;
- `figures/*.png`: non-interactive coefficient, predictive-accuracy,
  nested-comparison, and conditional-feature-contribution plots.

Because `struc_depth` is estimated from boundary strength, those predictors can
share substantial variance. For boundary strength and structure depth,
feature-contribution tests compare the full model against a model lacking the
feature of interest. These estimate unique predictive information conditional
on the other predictors; they do not make the two annotations statistically
independent.

## Fit one group STRF across subjects and electrodes

`run_strf_group.py` uses the same alignment, five models, L2 ridge estimator,
nested story-level cross-validation, permutation tests, and figure generation
as `run_strf.py`. Instead of fitting each recording and contact separately, it:

1. selects contacts with `fdr_p_value < 0.05`;
2. excludes every channel whose stripped, case-insensitive label starts with
   `MISC`;
3. requires the same stimulus set in every included recording and aligns all
   remaining subject-electrode responses by stimulus;
4. truncates copies of a stimulus to their shortest common duration; and
5. computes an equal-electrode-weight group mean at every sample.

The resulting STRF describes the response averaged over all included electrodes
and subjects. It is a pooled descriptive model, not a subject-level random
effects model. Because the preprocessed responses are baseline z-scores, each
electrode enters the average on a comparable scale.

For the group analysis, the FDR threshold is applied without an additional
effect-direction filter, so significant negative and positive contacts are both
included. Copies of a stimulus may differ in duration by at most 0.1 seconds
(`--group-duration-tolerance`) before the run fails rather than silently
truncating an anomalous recording.

Validate and run it with:

```bash
python3 seeg_story_listen_v2/run_strf_group.py --validate-only
python3 seeg_story_listen_v2/run_strf_group.py --overwrite
```

The default output is `/share/home/mitan/seeg_story_listen_v2/strf_group`.
`recording_manifest.csv`, `analysis_config.json`, `alignment_qc.csv`, and the
standard model outputs are retained. Additional group audit files are:

- `recording_inclusion.csv`: included/skipped recording counts;
- `excluded_channels.csv`: all excluded `MISC*` labels;
- `group_membership.csv`: every recording, electrode, and stimulus entering
  the mean;
- `group_aggregation.csv`: subject/electrode counts and common duration for
  each stimulus;
- `aligned_group_data/*.npz`: the final group feature and response arrays.

Model outputs and figures use the same names and formats as the individual
pipeline under `recordings/GROUP/`, including `GROUP_*_coefficients.png`,
`GROUP_model_accuracy.png`, `GROUP_model_comparisons.png`,
`GROUP_feature_contributions.png`, and held-out prediction plots.

Group figures include uncertainty wherever the plotted quantity is a bar or
line: model-accuracy bars show SEM, model-comparison and feature-contribution
bars show 95% confidence intervals across stimuli, event-feature coefficient
lines show 95% confidence bands across outer folds, and prediction lines show
a residual-based 95% interval. The group feature-contribution figure omits
`mel` and displays only `syl_onset`, `boundary_strength`, and `struc_depth`;
the mel STRF remains available in every coefficient figure.
