#!/usr/bin/env python3
"""Create static, full-duration plots of Brain Products EEG recordings."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np


DEFAULT_DATES = ("20260630", "20260702")
EXPECTED_RECORDINGS = ("BCI", "picNaming", "rest", "semantic")
SUPPORTED_EXTENSIONS = (".vhdr", ".edf", ".bdf", ".set", ".fif")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot every EEG recording over its full duration and save a PNG "
            "beside the source data."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the dated data folders (default: script directory).",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=list(DEFAULT_DATES),
        help="Names of dated folders to process.",
    )
    parser.add_argument(
        "--scale-uv",
        type=float,
        default=500.0,
        help="Amplitude represented above and below each channel baseline.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=6000,
        help="Maximum horizontal bins per channel; min/max values are preserved.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--keep-dc",
        action="store_true",
        help="Keep each channel's DC offset instead of centering at its median.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace plots that already exist.",
    )
    return parser.parse_args(argv)


def find_recordings(folder: Path) -> list[Path]:
    """Return readable EEG headers/files, without treating .eeg as standalone."""
    recordings: list[Path] = []
    for extension in SUPPORTED_EXTENSIONS:
        recordings.extend(folder.glob(f"*{extension}"))
        recordings.extend(folder.glob(f"*{extension.upper()}"))
    return sorted(set(recordings), key=lambda path: path.name.casefold())


def check_expected_recordings(folder: Path, recordings: Sequence[Path]) -> list[str]:
    """Report expected condition names that do not occur in a recording stem."""
    stems = [path.stem.casefold() for path in recordings]
    return [
        condition
        for condition in EXPECTED_RECORDINGS
        if not any(condition.casefold() in stem for stem in stems)
    ]


def read_raw(path: Path) -> mne.io.BaseRaw:
    """Read one supported recording without preloading the complete data."""
    readers = {
        ".vhdr": mne.io.read_raw_brainvision,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
    }
    return readers[path.suffix.casefold()](path, preload=False, verbose="ERROR")


def minmax_reduce(
    times: np.ndarray, values: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a trace while preserving each time bin's extrema."""
    if values.size <= max_points * 2:
        return times, values

    edges = np.linspace(0, values.size, max_points + 1, dtype=np.int64)
    reduced_times = np.empty(max_points * 2, dtype=np.float64)
    reduced_values = np.empty(max_points * 2, dtype=np.float64)

    for bin_index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = values[start:stop]
        minimum = int(np.argmin(chunk))
        maximum = int(np.argmax(chunk))
        first, second = sorted((minimum, maximum))
        output_index = bin_index * 2
        reduced_times[output_index : output_index + 2] = times[
            start + np.array((first, second))
        ]
        reduced_values[output_index : output_index + 2] = chunk[[first, second]]

    return reduced_times, reduced_values


def data_channel_indices(raw: mne.io.BaseRaw) -> list[int]:
    """Select signal channels while omitting trigger channels."""
    return [
        index
        for index, channel_type in enumerate(raw.get_channel_types())
        if channel_type != "stim"
    ]


def plot_recording(
    source: Path,
    output: Path,
    *,
    scale_uv: float,
    max_points: int,
    dpi: int,
    remove_dc: bool,
) -> None:
    """Render all channels of one recording to a non-interactive PNG."""
    raw = read_raw(source)
    picks = data_channel_indices(raw)
    if not picks:
        raise ValueError(f"No data channels found in {source}")

    duration = (raw.n_times - 1) / raw.info["sfreq"]
    row_spacing = scale_uv * 2.0
    figure_height = max(6.0, min(36.0, 1.0 + len(picks) * 0.32))
    figure, axis = plt.subplots(figsize=(24, figure_height), constrained_layout=True)

    for row, channel_index in enumerate(picks):
        values_uv = raw.get_data(picks=[channel_index])[0] * 1e6
        if remove_dc:
            values_uv = values_uv - np.median(values_uv)
        times = np.arange(values_uv.size, dtype=np.float64) / raw.info["sfreq"]
        times, values_uv = minmax_reduce(times, values_uv, max_points)
        baseline = (len(picks) - row - 1) * row_spacing
        axis.plot(times, values_uv + baseline, color="black", linewidth=0.35)

    baselines = np.arange(len(picks) - 1, -1, -1, dtype=float) * row_spacing
    axis.set_yticks(baselines, [raw.ch_names[index] for index in picks], fontsize=6)
    axis.set_xlim(0.0, max(duration, 1.0 / raw.info["sfreq"]))
    axis.set_ylim(-scale_uv, baselines[0] + scale_uv)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Channels")
    axis.set_title(
        f"{source.stem} — full duration {duration:.1f} s, scale ±{scale_uv:g} µV"
    )
    axis.grid(axis="x", color="0.88", linewidth=0.5)

    scale_x = duration * 0.985
    axis.plot(
        [scale_x, scale_x],
        [baselines[0] - scale_uv, baselines[0]],
        color="tab:red",
        linewidth=2.0,
        clip_on=False,
    )
    axis.text(
        scale_x,
        baselines[0] - scale_uv / 2,
        f" {scale_uv:g} µV",
        color="tab:red",
        fontsize=8,
        va="center",
    )
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    raw.close()


def process_folder(
    folder: Path,
    *,
    scale_uv: float,
    max_points: int,
    dpi: int,
    remove_dc: bool,
    overwrite: bool,
) -> int:
    if not folder.is_dir():
        print(f"ERROR: data folder does not exist: {folder}", file=sys.stderr)
        return 1

    recordings = find_recordings(folder)
    if not recordings:
        print(f"ERROR: no supported EEG recordings found in {folder}", file=sys.stderr)
        return 1

    missing = check_expected_recordings(folder, recordings)
    if missing:
        print(
            f"WARNING: {folder} has no recording matching: {', '.join(missing)}",
            file=sys.stderr,
        )

    failures = 0
    for source in recordings:
        output = source.with_name(f"{source.stem}_raw_full_duration.png")
        if output.exists() and not overwrite:
            print(f"Skipping existing plot: {output}")
            continue
        try:
            print(f"Plotting {source} -> {output}")
            plot_recording(
                source,
                output,
                scale_uv=scale_uv,
                max_points=max_points,
                dpi=dpi,
                remove_dc=remove_dc,
            )
        except Exception as error:  # Continue so one damaged file does not stop the batch.
            failures += 1
            print(f"ERROR: failed to plot {source}: {error}", file=sys.stderr)
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.scale_uv) or args.scale_uv <= 0:
        print("ERROR: --scale-uv must be a positive finite number", file=sys.stderr)
        return 2
    if args.max_points < 2 or args.dpi < 1:
        print("ERROR: --max-points must be at least 2 and --dpi must be positive", file=sys.stderr)
        return 2

    failures = 0
    for date in args.dates:
        failures += process_folder(
            args.root / date,
            scale_uv=args.scale_uv,
            max_points=args.max_points,
            dpi=args.dpi,
            remove_dc=not args.keep_dc,
            overwrite=args.overwrite,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
