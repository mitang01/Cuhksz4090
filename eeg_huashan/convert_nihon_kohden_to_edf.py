#!/usr/bin/env python3
"""Convert Nihon Kohden EEG recordings to EDF+ with MNE-Python."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import mne
import numpy as np


SIDECAR_EXTENSIONS = (".21E", ".PNT", ".LOG")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one Nihon Kohden .EEG file, or every .EEG file below a "
            "directory, to EDF+. Keep the matching .21E, .PNT, and .LOG files "
            "beside each source file so channel labels, metadata, and events "
            "can be imported."
        )
    )
    parser.add_argument("input", type=Path, help="Nihon Kohden .EEG file or directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Destination directory. By default each EDF is written beside its "
            "source. Directory input preserves its subdirectory structure."
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding used by the NK sidecars (for example utf-8 or cp932)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing EDF files"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not reopen each output to verify its structure",
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


def convert_recording(
    source: Path,
    destination: Path,
    *,
    encoding: str,
    overwrite: bool,
    validate: bool,
) -> str:
    """Convert one recording and return a short summary."""
    if destination.exists() and not overwrite:
        return f"SKIP {destination} (already exists)"

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
    try:
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
        # MNE currently labels misc data as microvolts without converting its
        # numeric values from volts. Correct the EDF header to prevent a 1e6
        # scale error when NK marker channels are read back.
        mark_misc_channels_as_volts(temporary, misc_indices)
        if validate:
            validate_edf(raw, temporary)
        temporary.replace(destination)
        duration = raw.n_times / raw.info["sfreq"]
        return (
            f"OK   {source} -> {destination} "
            f"({len(raw.ch_names)} channels, {duration:.1f} s, "
            f"{len(raw.annotations)} annotations)"
        )
    except Exception:
        # Never leave an apparently valid result after export validation fails.
        temporary.unlink(missing_ok=True)
        raise
    finally:
        raw.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    destination = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None else None
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
        output = output_path(source, input_path, destination)
        try:
            print(
                convert_recording(
                    source,
                    output,
                    encoding=args.encoding,
                    overwrite=args.overwrite,
                    validate=not args.skip_validation,
                )
            )
        except Exception as error:
            failures += 1
            print(f"ERROR: failed to convert {source}: {error}", file=sys.stderr)

    converted = len(recordings) - failures
    print(f"Finished: {converted} succeeded, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
