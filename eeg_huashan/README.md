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

## Convert Nihon Kohden recordings to EDF+

`convert_nihon_kohden_to_edf.py` replaces the Brainstorm conversion step. It
accepts either one Nihon Kohden `.EEG` file or a directory, recursively converts
all recordings to EDF+, and reopens each result to verify its channel names,
sample count, sampling frequency, and annotation count.

Keep the same-stem sidecars beside each `.EEG` file:

- `.21E` contains channel/electrode labels.
- `.PNT` contains recording metadata and measurement date.
- `.LOG` contains annotations/events.

On case-sensitive Linux filesystems, keep the extensions uppercase as generated
by the NK system. The script can still convert without a sidecar, but warns
which metadata will be missing.

```bash
cd /share/workspace2/tangmi/eeg_huashan
python3 -m pip install -r requirements.txt

# Convert one recording beside the source file
python3 convert_nihon_kohden_to_edf.py /data/patient01/recording.EEG

# Convert a directory tree into a separate output tree
python3 convert_nihon_kohden_to_edf.py /data/nk \
  --output-dir /data/edf
```

Existing EDF files are not changed unless `--overwrite` is passed. If text in
the annotations is garbled, retry with the encoding used by the recorder, for
example `--encoding cp932` for Japanese or `--encoding cp936` for simplified
Chinese. EDF is a 16-bit interchange format and cannot retain every proprietary
NK field, so keep the original NK files as the archival copy. EDF output may
also contain patient identifiers from the source and should be handled as
sensitive data.
