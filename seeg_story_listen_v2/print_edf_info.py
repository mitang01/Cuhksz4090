#!/usr/bin/env python3
"""Print MNE Raw and Info summaries for every EDF under a directory tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, TextIO

import mne


DEFAULT_INPUT = Path("/share/workspace3/ieeg/seeg/story_listen_v2")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find EDF files and print the same metadata that "
            "print(raw) and print(raw.info) show in an interactive MNE session: "
            "sampling frequency, filters, channels, participant fields, and "
            "other non-empty Info entries."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory to search recursively (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional text file for the printed report (default: stdout)",
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help=(
            "Also include filenames containing _prepocessed_ or _responsive_. "
            "Those are excluded by default so the report focuses on converted "
            "raw EDFs."
        ),
    )
    return parser.parse_args(argv)


def find_edfs(root: Path, *, include_processed: bool) -> list[Path]:
    """Return EDF paths under root, sorted for stable reports."""
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() != ".edf":
            continue
        stem = path.stem
        if not include_processed and (
            "_prepocessed_" in stem or "_responsive_" in stem
        ):
            continue
        paths.append(path)
    return sorted(paths, key=lambda path: str(path).casefold())


def write_recording_report(source: Path, stream: TextIO) -> None:
    """Load one EDF without preloading samples and print its Raw/Info text."""
    raw = mne.io.read_raw_edf(source, preload=False, verbose="ERROR")
    try:
        stream.write(f"{'=' * 80}\n")
        stream.write(f"file: {source}\n")
        stream.write(f"{'-' * 80}\n")
        stream.write("print(raw):\n")
        stream.write(f"{raw}\n")
        stream.write(f"{'-' * 80}\n")
        stream.write("print(raw.info):\n")
        stream.write(f"{raw.info}\n")
        stream.write("\n")
    finally:
        raw.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    recordings = find_edfs(input_dir, include_processed=args.include_processed)
    if not recordings:
        print(f"ERROR: no EDF files found below {input_dir}", file=sys.stderr)
        return 2

    output_path = (
        args.output.expanduser().resolve() if args.output is not None else None
    )
    stream: TextIO
    close_stream = False
    if output_path is None:
        stream = sys.stdout
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stream = output_path.open("w", encoding="utf-8")
        close_stream = True

    failures = 0
    try:
        stream.write(f"EDF metadata report for {input_dir}\n")
        stream.write(f"files discovered: {len(recordings)}\n\n")
        for source in recordings:
            try:
                write_recording_report(source, stream)
                print(f"OK   {source}", file=sys.stderr)
            except Exception as error:
                failures += 1
                print(f"ERROR: failed to read {source}: {error}", file=sys.stderr)
                stream.write(f"{'=' * 80}\n")
                stream.write(f"file: {source}\n")
                stream.write(f"ERROR: {error}\n\n")
        stream.write(
            f"Finished: {len(recordings) - failures} succeeded, {failures} failed\n"
        )
    finally:
        if close_stream:
            stream.close()

    if output_path is not None:
        print(f"Wrote report to {output_path}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
