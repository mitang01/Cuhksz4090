#!/usr/bin/env python3
"""Run the two-participant picture-naming ERP and SPM12 DCM pilot.

Python/MNE performs auditable preprocessing, trial classification, sensor ERP
statistics, and EEGLAB export. Actual DCM inversion is delegated to SPM12 in
MATLAB; this script never substitutes generic connectivity for DCM.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dcm_pilot_results"
DEFAULT_DATASETS = (
    {
        "participant": "Pp1_WWL",
        "task": "picture",
        "vhdr": (
            "/share/workspace2/tangmi/eeg_huashan/20260630/"
            "20260630_picNaming_001_WWL.vhdr"
        ),
    },
    {
        "participant": "Pp1_WWL",
        "task": "rest",
        "vhdr": (
            "/share/workspace2/tangmi/eeg_huashan/20260630/"
            "20260630_rest_001_WWL.Vhdr"
        ),
    },
    {
        "participant": "Pp2_JYP",
        "task": "picture",
        "vhdr": (
            "/share/workspace2/tangmi/eeg_huashan/20260702/"
            "20260702_picNaming_001_JYP.vhdr"
        ),
    },
    {
        "participant": "Pp2_JYP",
        "task": "rest",
        "vhdr": (
            "/share/workspace2/tangmi/eeg_huashan/20260702/"
            "20260702_rest_001_JYP.vhdr"
        ),
    },
)

ROI_ANCHORS = {
    "occipital": ("O1", "Oz", "O2"),
    "temporal": ("T7", "TP7", "P7", "T8", "TP8", "P8"),
}
ERP_WINDOWS = {
    "early_visual_80_180ms": (0.080, 0.180),
    "visual_lexical_180_300ms": (0.180, 0.300),
    "lexicosemantic_300_500ms": (0.300, 0.500),
    "late_preparation_500_800ms": (0.500, 0.800),
}
ONSET_PREFIXES = ("Stimulus/",)
TERMINAL_PREFIXES = ("Stimulus/", "Response/")


@dataclass
class Trial:
    """One picture-onset trial and its first terminal response."""

    trial_index: int
    onset_sample: int
    onset_seconds: float
    outcome: str
    terminal_description: str | None = None
    terminal_sample: int | None = None
    terminal_seconds: float | None = None
    response_latency_seconds: float | None = None
    picture_offset_seen: bool = False
    picture_offset_sample: int | None = None

    @property
    def is_correct(self) -> bool:
        return self.outcome == "response_2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process two picture-naming/rest EEG datasets and run actual "
            "SPM12 DCM-ERP/DCM-CSD models."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Analysis output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--spm-path",
        type=Path,
        default=None,
        help="SPM12 directory; alternatively set SPM12_PATH.",
    )
    parser.add_argument(
        "--matlab-command",
        default="matlab",
        help="MATLAB executable name or full path.",
    )
    parser.add_argument(
        "--skip-dcm",
        action="store_true",
        help="Run ERP/preprocessing/export only; do not invoke MATLAB/SPM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--permutations", type=int, default=4096)
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=30.0)
    parser.add_argument("--tmin", type=float, default=-0.2)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument(
        "--reject-uv",
        type=float,
        default=500.0,
        help="Epoch peak-to-peak rejection threshold in µV.",
    )
    parser.add_argument(
        "--flat-uv",
        type=float,
        default=0.5,
        help="Epoch flat-channel peak-to-peak threshold in µV.",
    )
    parser.add_argument(
        "--rest-epoch-seconds",
        type=float,
        default=2.0,
        help="Length of fixed rest segments exported for DCM-CSD.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON dataset manifest. Entries require participant, "
            "task ('picture'/'rest'), and vhdr."
        ),
    )
    return parser.parse_args(argv)


def load_datasets(manifest: Path | None) -> list[dict[str, str]]:
    if manifest is None:
        datasets = [dict(item) for item in DEFAULT_DATASETS]
    else:
        with manifest.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        datasets = payload["datasets"] if isinstance(payload, dict) else payload

    required = {"participant", "task", "vhdr"}
    for dataset in datasets:
        missing = required - set(dataset)
        if missing:
            raise ValueError(f"Dataset entry is missing: {sorted(missing)}")
        if dataset["task"] not in {"picture", "rest"}:
            raise ValueError(f"Unsupported task: {dataset['task']}")
    return datasets


def marker_code(description: str, allowed_prefixes: tuple[str, ...]) -> int | None:
    """Extract an integer trigger code only from approved marker classes."""
    if not description.startswith(allowed_prefixes):
        return None
    match = re.search(r"(\d+)\s*$", description)
    return int(match.group(1)) if match else None


def annotation_events(raw: mne.io.BaseRaw) -> list[tuple[int, str]]:
    """Return annotation sample/description pairs in acquisition order."""
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    descriptions = {value: key for key, value in event_id.items()}
    return [(int(row[0]), descriptions[int(row[2])]) for row in events]


def classify_trials(
    events: Sequence[tuple[int, str]], sfreq: float, first_samp: int = 0
) -> list[Trial]:
    """Pair trigger 1 with the first following 2/4 or the next trigger 1."""
    trials: list[Trial] = []
    current: Trial | None = None

    def seconds(sample: int) -> float:
        return (sample - first_samp) / sfreq

    for sample, description in events:
        onset_code = marker_code(description, ONSET_PREFIXES)
        terminal_code = marker_code(description, TERMINAL_PREFIXES)

        if onset_code == 1:
            if current is not None:
                current.outcome = "missing_before_next_onset"
                trials.append(current)
            current = Trial(
                trial_index=len(trials) + 1,
                onset_sample=sample,
                onset_seconds=seconds(sample),
                outcome="open",
            )
            continue

        if current is None:
            continue

        if terminal_code == 3 and not current.picture_offset_seen:
            current.picture_offset_seen = True
            current.picture_offset_sample = sample
            continue

        if terminal_code in {2, 4}:
            current.outcome = f"response_{terminal_code}"
            current.terminal_description = description
            current.terminal_sample = sample
            current.terminal_seconds = seconds(sample)
            current.response_latency_seconds = (
                current.terminal_seconds - current.onset_seconds
            )
            trials.append(current)
            current = None

    if current is not None:
        current.outcome = "missing_at_recording_end"
        trials.append(current)
    return trials


def normalize_egi_names(raw: mne.io.BaseRaw) -> dict[str, str]:
    """Normalize unambiguous EGI labels such as E001 to E1."""
    rename: dict[str, str] = {}
    existing = set(raw.ch_names)
    for name in raw.ch_names:
        match = re.fullmatch(r"E0*(\d+)", name, flags=re.IGNORECASE)
        if not match:
            continue
        normalized = f"E{int(match.group(1))}"
        if normalized != name and normalized not in existing:
            rename[name] = normalized
    if rename:
        raw.rename_channels(rename)
    return rename


def set_proxy_montage(raw: mne.io.BaseRaw) -> dict[str, Any]:
    """Attach a documented template montage; never map channels by order."""
    rename = normalize_egi_names(raw)
    eeg_names = [
        name
        for name, kind in zip(raw.ch_names, raw.get_channel_types())
        if kind == "eeg"
    ]
    egi_count = sum(bool(re.fullmatch(r"E\d+", name)) for name in eeg_names)

    if egi_count >= 100:
        montage_name = (
            "GSN-HydroCel-129"
            if "E129" in eeg_names
            else "GSN-HydroCel-128"
        )
    else:
        montage_name = "standard_1005"

    montage = mne.channels.make_standard_montage(montage_name)
    montage_names_lower = {name.casefold() for name in montage.ch_names}
    matched = [
        name for name in eeg_names if name.casefold() in montage_names_lower
    ]
    unmatched = [name for name in eeg_names if name not in matched]
    if len(matched) < 16:
        raise RuntimeError(
            f"Only {len(matched)} EEG labels match {montage_name}; refusing "
            "to assign coordinates by channel order. Supply digitized montage data."
        )

    raw.set_montage(montage, match_case=False, on_missing="warn")
    report = {
        "montage": montage_name,
        "status": "PROXY — not individually digitized",
        "renamed_channels": rename,
        "matched_eeg_channels": matched,
        "unmatched_eeg_channels": unmatched,
        "warning": (
            "Template sensor coordinates do not represent this participant's "
            "actual cap placement and limit source/DCM anatomical precision."
        ),
    }
    logging.warning(report["warning"])
    return report


def preprocess_raw(
    vhdr: Path, l_freq: float, h_freq: float
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    montage_report = set_proxy_montage(raw)
    raw.filter(l_freq, h_freq, picks="eeg", phase="zero", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw, montage_report


def raw_audit(
    raw: mne.io.BaseRaw, vhdr: Path, montage_report: dict[str, Any]
) -> dict[str, Any]:
    annotation_counts: dict[str, int] = {}
    for description in raw.annotations.description:
        annotation_counts[str(description)] = (
            annotation_counts.get(str(description), 0) + 1
        )
    return {
        "source": str(vhdr),
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "duration_seconds": float(raw.times[-1]),
        "n_channels": len(raw.ch_names),
        "channel_names": raw.ch_names,
        "channel_types": raw.get_channel_types(),
        "bads": raw.info["bads"],
        "annotation_counts": annotation_counts,
        "montage": montage_report,
        "preprocessing_warning": (
            "No automatic ICA was applied because ocular/cardiac component "
            "rejection requires reviewed EOG/ECG labels and human inspection."
        ),
    }


def epoch_picture(
    raw: mne.io.BaseRaw,
    trials: Sequence[Trial],
    *,
    correct_only: bool,
    tmin: float,
    tmax: float,
    reject_uv: float,
    flat_uv: float,
) -> mne.Epochs:
    selected = [
        trial
        for trial in trials
        if (trial.is_correct if correct_only else True)
    ]
    if not selected:
        variant = "correct-response" if correct_only else "all-onset"
        raise RuntimeError(f"No {variant} trials were found")
    events = np.array(
        [[trial.onset_sample, 0, 1] for trial in selected], dtype=int
    )
    return mne.Epochs(
        raw,
        events,
        event_id={"picture_onset": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0.0),
        preload=True,
        picks="eeg",
        reject={"eeg": reject_uv * 1e-6},
        flat={"eeg": flat_uv * 1e-6},
        reject_by_annotation=True,
        on_missing="raise",
        verbose="ERROR",
    )


def epoch_rest(
    raw: mne.io.BaseRaw,
    duration: float,
    reject_uv: float,
    flat_uv: float,
) -> mne.Epochs:
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=0.0,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    epochs.drop_bad(
        reject={"eeg": reject_uv * 1e-6},
        flat={"eeg": flat_uv * 1e-6},
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise RuntimeError("No clean fixed-length rest epochs remain")
    return epochs


def select_roi_channels(
    epochs: mne.Epochs, nearest_per_anchor: int = 2
) -> dict[str, list[str]]:
    """Select proxy-montage sensors nearest preregistered 10-10 anchors."""
    montage = epochs.get_montage()
    if montage is None:
        raise RuntimeError("No montage is attached")
    positions = montage.get_positions()["ch_pos"]
    standard = mne.channels.make_standard_montage("standard_1005")
    anchors = standard.get_positions()["ch_pos"]
    available = {
        name: np.asarray(positions[name], dtype=float)
        for name in epochs.ch_names
        if name in positions and np.isfinite(positions[name]).all()
    }
    if len(available) < 16:
        raise RuntimeError("Too few positioned EEG channels for ROI selection")

    result: dict[str, list[str]] = {}
    for roi, roi_anchors in ROI_ANCHORS.items():
        selected: list[str] = []
        for anchor in roi_anchors:
            target = np.asarray(anchors[anchor], dtype=float)
            nearest = sorted(
                available,
                key=lambda channel: float(
                    np.linalg.norm(available[channel] - target)
                ),
            )[:nearest_per_anchor]
            selected.extend(nearest)
        result[roi] = list(dict.fromkeys(selected))
    return result


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values."""
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = (total - rank) * float(p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = min(1.0, running)
    return adjusted.tolist()


def temporal_cluster_statistics(
    epochs: mne.Epochs,
    roi_channels: dict[str, list[str]],
    *,
    permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Within-recording sign-flip tests across trials, corrected across ROIs."""
    times = epochs.times
    test_mask = (times >= 0.0) & (times <= 0.8)
    rows: list[dict[str, Any]] = []
    significant_masks = {
        roi: np.zeros(times.size, dtype=bool) for roi in roi_channels
    }

    for roi, channels in roi_channels.items():
        trial_waveforms = (
            epochs.get_data(picks=channels).mean(axis=1)[:, test_mask] * 1e6
        )
        if trial_waveforms.shape[0] < 4:
            logging.warning(
                "%s has only %d epochs; skipping permutation inference",
                roi,
                trial_waveforms.shape[0],
            )
            continue
        threshold = float(
            stats.t.ppf(0.975, df=trial_waveforms.shape[0] - 1)
        )
        n_permutations: int | str = (
            "all" if trial_waveforms.shape[0] <= 12 else permutations
        )
        t_obs, clusters, p_values, _ = (
            mne.stats.permutation_cluster_1samp_test(
                trial_waveforms,
                threshold=threshold,
                n_permutations=n_permutations,
                tail=0,
                adjacency=None,
                out_type="mask",
                seed=seed,
                verbose=False,
            )
        )
        test_indices = np.flatnonzero(test_mask)
        for cluster_index, (cluster, p_value) in enumerate(
            zip(clusters, p_values), start=1
        ):
            cluster_mask = np.asarray(
                cluster[0] if isinstance(cluster, tuple) else cluster,
                dtype=bool,
            )
            indices = test_indices[cluster_mask]
            rows.append(
                {
                    "roi": roi,
                    "cluster": cluster_index,
                    "start_seconds": float(times[indices[0]]),
                    "end_seconds": float(times[indices[-1]]),
                    "cluster_mass": float(
                        np.abs(t_obs[cluster_mask]).sum()
                    ),
                    "p_uncorrected": float(p_value),
                    "_indices": indices,
                }
            )

    adjusted = holm_adjust([row["p_uncorrected"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_holm_across_roi_clusters"] = p_adjusted
        row["significant_0.05"] = p_adjusted < 0.05
        if row["significant_0.05"]:
            significant_masks[row["roi"]][row["_indices"]] = True
    return rows, significant_masks


def descriptive_erp_statistics(
    epochs: mne.Epochs, roi_channels: dict[str, list[str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for roi, channels in roi_channels.items():
        waveform = (
            epochs.get_data(picks=channels).mean(axis=(0, 1)) * 1e6
        )
        for window_name, (start, end) in ERP_WINDOWS.items():
            mask = (epochs.times >= start) & (epochs.times < end)
            segment = waveform[mask]
            segment_times = epochs.times[mask]
            peak_index = int(np.argmax(np.abs(segment)))
            rows.append(
                {
                    "roi": roi,
                    "window": window_name,
                    "n_retained_epochs": len(epochs),
                    "n_channels": len(channels),
                    "mean_amplitude_uv": float(segment.mean()),
                    "peak_amplitude_uv": float(segment[peak_index]),
                    "peak_latency_seconds": float(
                        segment_times[peak_index]
                    ),
                }
            )
    return rows


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(int), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return list(zip(starts, stops))


def plot_erps(
    epochs: mne.Epochs,
    roi_channels: dict[str, list[str]],
    significant_masks: dict[str, np.ndarray],
    output: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(
        len(roi_channels), 1, figsize=(11, 7), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    times_ms = epochs.times * 1e3

    for axis, (roi, channels) in zip(axes, roi_channels.items()):
        trial_waveforms = (
            epochs.get_data(picks=channels).mean(axis=1) * 1e6
        )
        mean = trial_waveforms.mean(axis=0)
        if len(epochs) > 1:
            sem = stats.sem(trial_waveforms, axis=0)
            interval = stats.t.ppf(0.975, len(epochs) - 1) * sem
        else:
            interval = np.full_like(mean, np.nan)
        axis.plot(times_ms, mean, color="black", linewidth=1.2)
        axis.fill_between(
            times_ms,
            mean - interval,
            mean + interval,
            color="0.5",
            alpha=0.25,
            label="95% trial-level CI",
        )
        axis.axvline(0, color="tab:blue", linestyle="--", linewidth=0.8)
        axis.axhline(0, color="0.7", linewidth=0.6)
        for start, stop in contiguous_regions(significant_masks[roi]):
            axis.axvspan(
                times_ms[start],
                times_ms[stop],
                color="tab:red",
                alpha=0.18,
                label="cluster p(Holm)<0.05",
            )
        axis.set_title(f"{roi.capitalize()} ROI ({', '.join(channels)})")
        axis.set_ylabel("Amplitude (µV)")
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time from picture onset (ms)")
    figure.suptitle(f"{title}\nN={len(epochs)} retained trials")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("No results\n", encoding="utf-8")
        return
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)


def export_epochs(epochs: mne.Epochs, output_base: Path) -> Path:
    """Save MNE FIF plus EEGLAB SET for SPM conversion."""
    fif_path = output_base.with_suffix(".fif")
    set_path = output_base.with_suffix(".set")
    epochs.save(fif_path, overwrite=True, verbose="ERROR")
    epochs.export(set_path, fmt="eeglab", overwrite=True, verbose="ERROR")
    return set_path


def write_trial_table(path: Path, participant: str, trials: Sequence[Trial]) -> None:
    rows = []
    for trial in trials:
        row = asdict(trial)
        row.update(
            {
                "participant": participant,
                "included_all": True,
                "included_correct_response_2": trial.is_correct,
            }
        )
        rows.append(row)
    write_csv(path, rows)


def process_picture(
    dataset: dict[str, str], args: argparse.Namespace, output_dir: Path
) -> list[dict[str, str]]:
    participant = dataset["participant"]
    subject_dir = output_dir / participant
    subject_dir.mkdir(parents=True, exist_ok=True)
    vhdr = Path(dataset["vhdr"])
    raw, montage_report = preprocess_raw(vhdr, args.l_freq, args.h_freq)
    audit = raw_audit(raw, vhdr, montage_report)
    (subject_dir / "picture_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    trials = classify_trials(
        annotation_events(raw), raw.info["sfreq"], raw.first_samp
    )
    write_trial_table(subject_dir / "picture_trials.csv", participant, trials)

    analyses: list[dict[str, str]] = []
    for variant, correct_only in (("picture_all", False), ("picture_correct", True)):
        epochs = epoch_picture(
            raw,
            trials,
            correct_only=correct_only,
            tmin=args.tmin,
            tmax=args.tmax,
            reject_uv=args.reject_uv,
            flat_uv=args.flat_uv,
        )
        roi_channels = select_roi_channels(epochs)
        (subject_dir / f"{variant}_roi_channels.json").write_text(
            json.dumps(roi_channels, indent=2), encoding="utf-8"
        )
        cluster_rows, significant_masks = temporal_cluster_statistics(
            epochs,
            roi_channels,
            permutations=args.permutations,
            seed=args.seed,
        )
        descriptive_rows = descriptive_erp_statistics(epochs, roi_channels)
        write_csv(subject_dir / f"{variant}_erp_clusters.csv", cluster_rows)
        write_csv(
            subject_dir / f"{variant}_erp_descriptive_statistics.csv",
            descriptive_rows,
        )
        plot_erps(
            epochs,
            roi_channels,
            significant_masks,
            subject_dir / f"{variant}_erp.png",
            f"{participant}: {variant.replace('_', ' ')}",
        )
        set_path = export_epochs(
            epochs, subject_dir / f"{variant}_clean_epochs"
        )
        analyses.append(
            {
                "participant": participant,
                "analysis": variant,
                "kind": "ERP",
                "set_file": str(set_path.resolve()),
            }
        )
    raw.close()
    return analyses


def process_rest(
    dataset: dict[str, str], args: argparse.Namespace, output_dir: Path
) -> dict[str, str]:
    participant = dataset["participant"]
    subject_dir = output_dir / participant
    subject_dir.mkdir(parents=True, exist_ok=True)
    vhdr = Path(dataset["vhdr"])
    raw, montage_report = preprocess_raw(vhdr, 1.0, 45.0)
    audit = raw_audit(raw, vhdr, montage_report)
    (subject_dir / "rest_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    epochs = epoch_rest(
        raw,
        args.rest_epoch_seconds,
        args.reject_uv,
        args.flat_uv,
    )
    set_path = export_epochs(epochs, subject_dir / "rest_clean_segments")
    raw.close()
    return {
        "participant": participant,
        "analysis": "rest",
        "kind": "CSD",
        "set_file": str(set_path.resolve()),
    }


def write_method_notes(output_dir: Path, args: argparse.Namespace) -> None:
    notes = f"""DCM PILOT METHOD AND LIMITATION NOTES

Sensor template
---------------
The script selects MNE's GSN-HydroCel-128/129 template when EGI E-numbered
channels match; otherwise it tries standard_1005 by channel label. This is a
proxy montage, not an individual digitization. No channel is assigned a
position by order. See each *_audit.json for actual matching and warnings.

MRI/head template
-----------------
SPM12's canonical MNI template head model is used with equivalent-current
dipoles (ECD). Source priors are approximate MNI coordinates:
  left OT     [-42, -64, -12]
  left pMTG   [-56, -46,   2]
  left ATL    [-50,   8, -28]
  left IFG    [-48,  26,  10]
Template anatomy and cap coordinates limit anatomical precision. Results must
be described as hypothesis-constrained pilot effective connectivity, not
individual source localization or evidence of post-surgical reorganization.

ERP statistics
--------------
ERPs are baseline corrected over {args.tmin:.3f} to 0 s. Temporal cluster
sign-flip tests use trials as observations within each recording, with Holm
correction across all clusters in the two sensor ROIs for each ERP variant.
These are within-participant/recording-level tests. Trials are not independent
participants, so p-values cannot support population inference from N=2.

Artifact handling
-----------------
The deterministic pilot applies filtering, average reference, annotation
rejection, and fixed amplitude/flatness epoch criteria. ICA is intentionally
not automated because EOG/ECG identities and components require review.
Inspect audits and raw data before interpreting output.

DCM distinction
---------------
Picture analyses use SPM12 DCM for evoked responses (ERP). Rest uses a separate
SPM12 cross-spectral-density DCM (CSD), because unstructured rest is not a
condition or baseline in ERP-DCM. The architecture families are compared
separately within picture_all, picture_correct, and rest. No direct
picNaming-vs-rest ERP-DCM contrast is claimed.

Model families
--------------
F1: OT -> pMTG, OT -> ATL, pMTG -> IFG, ATL -> IFG
F2: F1 + IFG -> pMTG and IFG -> ATL feedback
F3: F1 + direct OT -> IFG route

Each pilot family contains one prespecified model, so "family probability" is
numerically a model posterior probability. A larger full-study model space
should contain multiple nuisance variants per family.

Software
--------
MNE-Python: {mne.__version__}
SPM12 revision and MATLAB version are written by run_spm_dcm.m at execution.
"""
    (output_dir / "METHOD_NOTES.txt").write_text(notes, encoding="utf-8")


def invoke_spm(
    manifest: Path, spm_path: Path | None, matlab_command: str
) -> None:
    if spm_path is None:
        env_value = os.environ.get("SPM12_PATH")
        spm_path = Path(env_value) if env_value else None
    if spm_path is None or not spm_path.is_dir():
        raise RuntimeError(
            "SPM12 was not found. Pass --spm-path /path/to/spm12 or set "
            "SPM12_PATH. ERP outputs are complete; DCM was not run."
        )
    executable = shutil.which(matlab_command)
    if executable is None:
        raise RuntimeError(
            f"MATLAB executable not found: {matlab_command}. "
            "ERP outputs are complete; DCM was not run."
        )

    def matlab_quote(path: Path) -> str:
        return str(path.resolve()).replace("'", "''")

    command = (
        f"addpath('{matlab_quote(spm_path)}'); "
        f"addpath('{matlab_quote(SCRIPT_DIR / 'spm')}'); "
        f"run_spm_dcm('{matlab_quote(manifest)}');"
    )
    subprocess.run([executable, "-batch", command], check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tmin >= 0 or args.tmax <= 0 or args.tmin >= args.tmax:
        raise ValueError("Epoch must span zero with tmin < 0 < tmax")
    if (
        not math.isfinite(args.reject_uv)
        or args.reject_uv <= 0
        or args.flat_uv < 0
    ):
        raise ValueError("Artifact thresholds must be finite and non-negative")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} exists; pass --overwrite to replace it"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "pipeline.log"),
        ],
    )

    datasets = load_datasets(args.manifest)
    for dataset in datasets:
        if not Path(dataset["vhdr"]).is_file():
            raise FileNotFoundError(dataset["vhdr"])

    analyses: list[dict[str, str]] = []
    for dataset in datasets:
        logging.info(
            "Processing %s %s", dataset["participant"], dataset["task"]
        )
        if dataset["task"] == "picture":
            analyses.extend(
                process_picture(dataset, args, args.output_dir)
            )
        else:
            analyses.append(process_rest(dataset, args, args.output_dir))

    write_method_notes(args.output_dir, args)
    dcm_manifest = args.output_dir / "spm_dcm_manifest.json"
    dcm_manifest.write_text(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "analyses": analyses,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.skip_dcm:
        invoke_spm(dcm_manifest, args.spm_path, args.matlab_command)
    else:
        logging.warning(
            "DCM skipped by request. EEGLAB files and SPM manifest are ready."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # Keep a concise command-line failure message.
        print(f"ERROR: {error}", file=sys.stderr)
        raise
