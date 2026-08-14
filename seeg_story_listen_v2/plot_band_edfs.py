#!/usr/bin/env python3
"""Render full-duration, multi-band EDF overview PNGs without a display."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np


DEFAULT_INPUT = Path("/share/home/mitan/seeg_story_listen_v2")
CANONICAL_BANDS = ("delta", "theta", "alpha", "beta", "gamma", "high_gamma")
BAND_PATTERN = "|".join(sorted(CANONICAL_BANDS, key=len, reverse=True))
EDF_PATTERN = re.compile(
    rf"^(?P<base>.+)_(?P<kind>prepocessed|responsive)_(?P<band>{BAND_PATTERN})$",
    re.IGNORECASE,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one non-interactive full-duration PNG per subject/recording, "
            "with all available frequency-band EDFs arranged as panels."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="PNG destination (default: <input-dir>/plots)",
    )
    parser.add_argument(
        "--kind",
        choices=("prepocessed", "responsive"),
        default="prepocessed",
        help="Plot all-contact processed EDFs or responsive-contact EDFs",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        choices=CANONICAL_BANDS,
        default=list(CANONICAL_BANDS),
        help="Bands to include, in canonical order when available",
    )
    parser.add_argument(
        "--scale-mv",
        type=float,
        default=0.2,
        help="Displayed amplitude above and below each channel baseline in mV",
    )
    parser.add_argument(
        "--max-time-bins",
        type=int,
        default=1500,
        help="Maximum min/max time bins per trace",
    )
    parser.add_argument(
        "--channel-chunk-size",
        type=int,
        default=16,
        help="Channels read from EDF at one time",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def discover_groups(
    input_dir: Path, kind: str, bands: Sequence[str]
) -> dict[tuple[Path, str], dict[str, Path]]:
    """Group matching EDFs by parent directory and original recording stem."""
    requested = set(bands)
    groups: dict[tuple[Path, str], dict[str, Path]] = defaultdict(dict)
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.casefold() != ".edf":
            continue
        match = EDF_PATTERN.fullmatch(path.stem)
        if not match or match.group("kind").casefold() != kind.casefold():
            continue
        band = match.group("band").casefold()
        if band not in requested:
            continue
        key = (path.parent, match.group("base"))
        if band in groups[key]:
            raise ValueError(
                f"duplicate {band} EDFs for {match.group('base')} in {path.parent}"
            )
        groups[key][band] = path
    return dict(groups)


def minmax_reduce(
    times: np.ndarray, values: np.ndarray, max_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a trace while retaining each time bin's minimum and maximum."""
    if values.size <= max_bins * 2:
        return times, values
    edges = np.linspace(0, values.size, max_bins + 1, dtype=np.int64)
    reduced_times = np.empty(max_bins * 2, dtype=np.float64)
    reduced_values = np.empty(max_bins * 2, dtype=np.float64)
    for bin_index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = values[start:stop]
        first, second = sorted((int(np.argmin(chunk)), int(np.argmax(chunk))))
        output_index = bin_index * 2
        source_indices = start + np.asarray((first, second))
        reduced_times[output_index : output_index + 2] = times[source_indices]
        reduced_values[output_index : output_index + 2] = chunk[[first, second]]
    return reduced_times, reduced_values


def plot_band(
    axis: plt.Axes,
    path: Path,
    band: str,
    *,
    scale_volts: float,
    max_time_bins: int,
    channel_chunk_size: int,
) -> tuple[int, float]:
    """Plot one EDF as vertically offset, full-duration channel traces."""
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    try:
        channel_count = len(raw.ch_names)
        if not channel_count:
            raise ValueError(f"EDF has no channels: {path}")
        sfreq = float(raw.info["sfreq"])
        duration = (raw.n_times - 1) / sfreq
        times = np.arange(raw.n_times, dtype=np.float64) / sfreq
        spacing = 2 * scale_volts
        baselines = np.arange(channel_count - 1, -1, -1, dtype=float) * spacing

        for first in range(0, channel_count, channel_chunk_size):
            stop = min(first + channel_chunk_size, channel_count)
            data = raw.get_data(start=0, stop=raw.n_times, picks=range(first, stop))
            for local_index, values in enumerate(data):
                channel_index = first + local_index
                values = values - np.median(values)
                # A fixed display scale is most useful when extreme artifacts
                # cannot make every other trace visually flat.
                values = np.clip(values, -scale_volts, scale_volts)
                plot_times, plot_values = minmax_reduce(
                    times, values, max_time_bins
                )
                axis.plot(
                    plot_times,
                    plot_values + baselines[channel_index],
                    color="black",
                    linewidth=0.25,
                    rasterized=True,
                )

        axis.set_xlim(0, max(duration, 1 / sfreq))
        axis.set_ylim(-scale_volts, baselines[0] + scale_volts)
        axis.set_yticks(baselines)
        axis.set_yticklabels(raw.ch_names, fontsize=4)
        axis.set_xlabel("Time (s)")
        axis.set_title(
            f"{band.replace('_', ' ').title()}\n"
            f"{channel_count} channels, {sfreq:g} Hz, {duration:.1f} s",
            fontsize=9,
        )
        axis.grid(axis="x", color="0.88", linewidth=0.4)
        return channel_count, duration
    finally:
        raw.close()


def output_path(
    parent: Path,
    base: str,
    input_dir: Path,
    output_dir: Path,
    kind: str,
) -> Path:
    relative_parent = parent.relative_to(input_dir)
    return output_dir / relative_parent / f"{base}_{kind}_all_bands_full_length.png"


def render_group(
    parent: Path,
    base: str,
    files: dict[str, Path],
    destination: Path,
    *,
    kind: str,
    bands: Sequence[str],
    scale_mv: float,
    max_time_bins: int,
    channel_chunk_size: int,
    dpi: int,
    overwrite: bool,
) -> str:
    if destination.exists() and not overwrite:
        return f"SKIP {destination} (already exists)"
    available = [band for band in bands if band in files]
    if not available:
        raise ValueError(f"no requested bands found for {base}")

    channel_counts: list[int] = []
    for band in available:
        raw = mne.io.read_raw_edf(files[band], preload=False, verbose="ERROR")
        try:
            channel_counts.append(len(raw.ch_names))
        finally:
            raw.close()
    figure_height = max(8.0, min(40.0, 2.0 + max(channel_counts) * 0.16))
    figure_width = max(12.0, 5.0 * len(available))
    figure, axes = plt.subplots(
        1,
        len(available),
        figsize=(figure_width, figure_height),
        squeeze=False,
        constrained_layout=True,
    )
    try:
        scale_volts = scale_mv * 1e-3
        summaries = []
        for axis, band in zip(axes[0], available):
            count, duration = plot_band(
                axis,
                files[band],
                band,
                scale_volts=scale_volts,
                max_time_bins=max_time_bins,
                channel_chunk_size=channel_chunk_size,
            )
            summaries.append(f"{band}:{count}ch/{duration:.1f}s")
        figure.suptitle(
            f"{base} — {kind} bands, full duration, display scale ±{scale_mv:g} mV "
            "(traces clipped at scale)",
            fontsize=12,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, facecolor="white")
        return f"OK   {destination} ({', '.join(summaries)})"
    finally:
        plt.close(figure)


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {args.input_dir}")
    if (
        not math.isfinite(args.scale_mv)
        or args.scale_mv <= 0
        or args.max_time_bins < 2
        or args.channel_chunk_size < 1
        or args.dpi < 1
    ):
        raise ValueError(
            "--scale-mv must be positive and finite; --max-time-bins must be at "
            "least 2; --channel-chunk-size and --dpi must be positive"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else args.input_dir / "plots"
    )
    try:
        validate_args(args)
        groups = discover_groups(args.input_dir, args.kind, args.bands)
        if not groups:
            raise FileNotFoundError(
                f"no {args.kind} frequency-band EDFs found below {args.input_dir}"
            )
        failures = 0
        for (parent, base), files in sorted(
            groups.items(), key=lambda item: (str(item[0][0]), item[0][1])
        ):
            destination = output_path(
                parent, base, args.input_dir, args.output_dir, args.kind
            )
            try:
                print(
                    render_group(
                        parent,
                        base,
                        files,
                        destination,
                        kind=args.kind,
                        bands=CANONICAL_BANDS,
                        scale_mv=args.scale_mv,
                        max_time_bins=args.max_time_bins,
                        channel_chunk_size=args.channel_chunk_size,
                        dpi=args.dpi,
                        overwrite=args.overwrite,
                    )
                )
            except Exception as error:
                failures += 1
                print(f"ERROR {base}: {error}", file=sys.stderr)
        print(f"Finished: {len(groups) - failures} succeeded, {failures} failed")
        return 1 if failures else 0
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
