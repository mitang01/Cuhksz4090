#!/usr/bin/env python3
"""Convert Nihon Kohden recordings to EDF+ and save static raw-signal plots."""

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


SIDECAR_EXTENSIONS = (".21E", ".PNT", ".LOG")
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one Nihon Kohden .EEG file, or every .EEG file below a "
            "directory, to EDF+. Keep matching .21E, .PNT, and .LOG files "
            "beside each source so labels, metadata, and events are imported."
        )
    )
    parser.add_argument("input", type=Path, help="Nihon Kohden .EEG file or directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "EDF destination. By default each EDF is written beside its source. "
            "Directory input preserves its subdirectory structure."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=SCRIPT_DIR,
        help=(
            "Full-duration raw PNG destination "
            "(default: the directory containing this script)"
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="NK sidecar text encoding (for example utf-8, cp932, or cp936)",
    )
    parser.add_argument(
        "--scale-uv",
        type=float,
        default=500.0,
        help="Amplitude above and below each channel baseline in the raw plot",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=6000,
        help="Maximum horizontal min/max bins per plotted channel",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Raw plot resolution")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing EDF and PNG files"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not reopen each output to verify its structure and signal scale",
    )
    return parser.parse_args(argv)


def find_recordings(input_path: Path) -> list[Path]:
    """Find NK .EEG files, case-insensitively."""
    if input_path.is_file():
        if input_path.suffix.casefold() != ".eeg":
            raise ValueError(f"input file must have an .EEG extension: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input does not exist: {input_path}")
    return sorted(
        (
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".eeg"
        ),
        key=lambda path: str(path).casefold(),
    )


def output_path(source: Path, input_path: Path, destination: Path | None) -> Path:
    """Choose an EDF path, preserving relative folders for directory input."""
    if destination is None:
        return source.with_suffix(".edf")
    if input_path.is_dir():
        return destination / source.relative_to(input_path).with_suffix(".edf")
    return destination / source.with_suffix(".edf").name


def plot_path(source: Path, input_path: Path, destination: Path) -> Path:
    """Choose a raw plot path, preserving relative folders."""
    name = f"{source.stem}_raw_full_duration.png"
    if input_path.is_dir():
        return destination / source.relative_to(input_path).parent / name
    return destination / name


def missing_sidecars(source: Path) -> list[str]:
    """Return absent optional NK sidecars."""
    return [
        extension
        for extension in SIDECAR_EXTENSIONS
        if not source.with_suffix(extension).is_file()
    ]


def validate_edf(source_raw: mne.io.BaseRaw, destination: Path) -> None:
    """Reopen an EDF and verify structure plus sampled signal fidelity."""
    exported = mne.io.read_raw_edf(destination, preload=False, verbose="ERROR")
    try:
        problems: list[str] = []
        if exported.ch_names != source_raw.ch_names:
            problems.append("channel names/order differ")
        if exported.n_times != source_raw.n_times:
            problems.append(
                f"sample count differs ({source_raw.n_times} -> {exported.n_times})"
            )
        if exported.info["sfreq"] != source_raw.info["sfreq"]:
            problems.append(
                "sampling frequency differs "
                f"({source_raw.info['sfreq']} -> {exported.info['sfreq']})"
            )
        if len(exported.annotations) != len(source_raw.annotations):
            problems.append(
                "annotation count differs "
                f"({len(source_raw.annotations)} -> {len(exported.annotations)})"
            )
        if not problems:
            window_size = min(1000, source_raw.n_times)
            starts = {
                0,
                max(0, source_raw.n_times // 2 - window_size // 2),
                source_raw.n_times - window_size,
            }
            for start in starts:
                stop = start + window_size
                expected = source_raw.get_data(start=start, stop=stop)
                observed = exported.get_data(start=start, stop=stop)
                # EDF is 16-bit, so small quantization error is expected.
                tolerance = np.maximum(
                    np.max(np.abs(expected), axis=1) / 30_000, 1e-10
                )
                error = np.max(np.abs(expected - observed), axis=1)
                bad = np.flatnonzero(error > tolerance)
                if bad.size:
                    channel = int(bad[0])
                    problems.append(
                        f"signal values differ for {source_raw.ch_names[channel]} "
                        f"(maximum error {error[channel]:.6g})"
                    )
                    break
        if problems:
            raise RuntimeError("; ".join(problems))
    finally:
        exported.close()


def mark_misc_channels_as_volts(destination: Path, channel_indices: Sequence[int]) -> None:
    """Correct MNE's EDF unit label for voltage-valued NK misc channels."""
    if not channel_indices:
        return
    with destination.open("r+b") as edf:
        edf.seek(252)
        try:
            signal_count = int(edf.read(4).decode("ascii").strip())
        except ValueError as error:
            raise RuntimeError("exported EDF has an invalid signal count") from error
        if max(channel_indices) >= signal_count:
            raise RuntimeError("exported EDF has fewer signals than expected")

        # EDF stores each header field for all signals contiguously. Physical
        # dimensions follow the 16-byte labels and 80-byte transducer fields.
        dimensions_offset = 256 + signal_count * (16 + 80)
        for index in channel_indices:
            edf.seek(dimensions_offset + index * 8)
            edf.write(b"V       ")


def minmax_reduce(
    times: np.ndarray, values: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a trace while retaining each time bin's extrema."""
    if values.size <= max_points * 2:
        return times, values
    edges = np.linspace(0, values.size, max_points + 1, dtype=np.int64)
    reduced_times = np.empty(max_points * 2, dtype=np.float64)
    reduced_values = np.empty(max_points * 2, dtype=np.float64)
    for bin_index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = values[start:stop]
        first, second = sorted((int(np.argmin(chunk)), int(np.argmax(chunk))))
        index = bin_index * 2
        reduced_times[index : index + 2] = times[
            start + np.array((first, second))
        ]
        reduced_values[index : index + 2] = chunk[[first, second]]
    return reduced_times, reduced_values


def render_raw_plot(
    raw: mne.io.BaseRaw,
    source: Path,
    destination: Path,
    *,
    scale_uv: float,
    max_points: int,
    dpi: int,
) -> None:
    """Render a static full-duration plot of the NK signal channels."""
    channel_types = raw.get_channel_types()
    picks = [
        index
        for index, channel_type in enumerate(channel_types)
        if channel_type not in {"misc", "stim"}
    ]
    if not picks:
        raise RuntimeError("no plottable signal channels found")

    duration = (raw.n_times - 1) / raw.info["sfreq"]
    row_spacing = scale_uv * 2
    figure_height = max(6.0, min(40.0, 1.0 + len(picks) * 0.3))
    figure, axis = plt.subplots(figsize=(24, figure_height), constrained_layout=True)
    times = np.arange(raw.n_times, dtype=np.float64) / raw.info["sfreq"]
    for row, channel_index in enumerate(picks):
        values_uv = raw.get_data(picks=[channel_index])[0] * 1e6
        values_uv -= np.median(values_uv)
        plot_times, plot_values = minmax_reduce(times, values_uv, max_points)
        baseline = (len(picks) - row - 1) * row_spacing
        axis.plot(plot_times, plot_values + baseline, color="black", linewidth=0.35)

    baselines = np.arange(len(picks) - 1, -1, -1, dtype=float) * row_spacing
    axis.set_yticks(baselines, [raw.ch_names[index] for index in picks], fontsize=6)
    axis.set_xlim(0, max(duration, 1 / raw.info["sfreq"]))
    axis.set_ylim(-scale_uv, baselines[0] + scale_uv)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Channels")
    axis.set_title(
        f"{source.stem} — raw full duration {duration:.1f} s, scale ±{scale_uv:g} µV"
    )
    axis.grid(axis="x", color="0.88", linewidth=0.5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, facecolor="white")
    plt.close(figure)


def process_recording(
    source: Path,
    destination: Path,
    plot_destination: Path | None,
    *,
    encoding: str,
    overwrite: bool,
    validate: bool,
    scale_uv: float,
    max_points: int,
    dpi: int,
) -> str:
    """Convert and optionally plot one recording."""
    make_edf = overwrite or not destination.exists()
    make_plot = plot_destination is not None and (
        overwrite or not plot_destination.exists()
    )
    if not make_edf and not make_plot:
        return f"SKIP {source} (outputs already exist)"

    missing = missing_sidecars(source)
    if missing:
        print(
            f"WARNING: {source} has no {', '.join(missing)} sidecar(s); "
            "some labels, metadata, or events may be absent",
            file=sys.stderr,
        )

    raw = mne.io.read_raw_nihon(
        source, preload=True, encoding=encoding, verbose="ERROR"
    )
    temporary = destination.with_name(f".{destination.name}.partial.edf")
    completed: list[str] = []
    try:
        if make_edf:
            misc_indices = [
                index
                for index, kind in enumerate(raw.get_channel_types())
                if kind == "misc"
            ]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.unlink(missing_ok=True)
            mne.export.export_raw(
                temporary,
                raw,
                fmt="edf",
                physical_range="channelwise",
                overwrite=True,
                verbose="ERROR",
            )
            # MNE labels misc data as microvolts without converting its numeric
            # values from volts. Correct the header to prevent a 1e6 scale error.
            mark_misc_channels_as_volts(temporary, misc_indices)
            if validate:
                validate_edf(raw, temporary)
            temporary.replace(destination)
            completed.append(str(destination))
        if make_plot and plot_destination is not None:
            render_raw_plot(
                raw,
                source,
                plot_destination,
                scale_uv=scale_uv,
                max_points=max_points,
                dpi=dpi,
            )
            completed.append(str(plot_destination))
        duration = raw.n_times / raw.info["sfreq"]
        return (
            f"OK   {source} -> {', '.join(completed)} "
            f"({len(raw.ch_names)} channels, {duration:.1f} s, "
            f"{len(raw.annotations)} annotations)"
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        raw.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        not math.isfinite(args.scale_uv)
        or args.scale_uv <= 0
        or args.max_points < 2
        or args.dpi < 1
    ):
        print(
            "ERROR: --scale-uv must be positive and finite, --max-points must "
            "be at least 2, and --dpi must be positive",
            file=sys.stderr,
        )
        return 2

    input_path = args.input.expanduser().resolve()
    destination = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None else None
    )
    plot_directory = (
        args.plot_dir.expanduser().resolve() if args.plot_dir is not None else None
    )
    try:
        recordings = find_recordings(input_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if not recordings:
        print(f"ERROR: no .EEG files found below {input_path}", file=sys.stderr)
        return 2

    failures = 0
    for source in recordings:
        edf_destination = output_path(source, input_path, destination)
        plot_destination = (
            plot_path(source, input_path, plot_directory)
            if plot_directory is not None
            else None
        )
        try:
            print(
                process_recording(
                    source,
                    edf_destination,
                    plot_destination,
                    encoding=args.encoding,
                    overwrite=args.overwrite,
                    validate=not args.skip_validation,
                    scale_uv=args.scale_uv,
                    max_points=args.max_points,
                    dpi=args.dpi,
                )
            )
        except Exception as error:
            failures += 1
            print(f"ERROR: failed to process {source}: {error}", file=sys.stderr)

    print(f"Finished: {len(recordings) - failures} succeeded, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
