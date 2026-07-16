# EEG full-duration plots

This folder is intended to be synchronized with:

```text
/share/workspace2/tangmi/eeg_huashan
```

`plot_raw_eeg.py` finds the recordings in `20260630` and `20260702` and writes
one static PNG beside each recording. Each image shows the entire recording,
all non-trigger channels, and a 500 µV scale bar. Long traces are reduced with
a min/max envelope, which preserves sharp extrema better than ordinary
subsampling.

Brain Products recordings are loaded from their `.vhdr` files. Keep each
`.vhdr`, `.vmrk`, and `.eeg` triplet together with the original file names.
EDF, BDF, EEGLAB SET, and FIF files are also supported.

## Run on the remote Linux machine

```bash
cd /share/workspace2/tangmi/eeg_huashan
python3 -m pip install -r requirements.txt
python3 plot_raw_eeg.py
```

Expected outputs are named:

```text
20260630/<recording>_raw_full_duration.png
20260702/<recording>_raw_full_duration.png
```

The script checks for recording names containing `BCI`, `picNaming`, `rest`,
and `semantic` in each folder. Missing names produce warnings. Existing plots
are left untouched; pass `--overwrite` to regenerate them.

Useful options:

```bash
# Explicitly select the data root and regenerate plots
python3 plot_raw_eeg.py \
  --root /share/workspace2/tangmi/eeg_huashan \
  --scale-uv 500 \
  --overwrite

# Plot one dated folder only
python3 plot_raw_eeg.py --dates 20260630
```

Matplotlib uses its non-interactive `Agg` backend, so the command works in a
terminal without a display server and produces image files only.
