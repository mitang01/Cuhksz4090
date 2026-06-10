#!/usr/bin/env python3
"""Move specified folders from source paths to destination paths."""

from pathlib import Path
import shutil
import sys


MOVE_TASKS = [
    (
        Path("/share/workspace2/tangmi/sub4_story_listen"),
        Path("/share/workspace3/ieeg/micro/story_listen_v1"),
    ),
    (
        Path("/share/workspace2/tangmi/bistable_sub4"),
        Path("/share/workspace3/ieeg/micro/word_boun_perce_v1"),
    ),
    (
        Path("/share/workspace2/tangm/20260120-20260123"),
        Path("/share/workspace3/ieeg/micro"),
    ),
    (
        Path("/share/workspace2/tangmi/szu_raw_edf"),
        Path("/share/workspace3/ieeg/seeg/story_listen_v1/sub-001"),
    ),
    (
        Path("/share/workspace2/tangmi/szu_raw_nk"),
        Path("/share/workspace3/ieeg/seeg/story_listen_v1/sub-001"),
    ),
]


def move_folder(source: Path, destination: Path) -> None:
    """Move source folder to destination, creating parent directories if needed."""
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def main() -> int:
    for source, destination in MOVE_TASKS:
        try:
            print(f"Moving: {source} -> {destination}")
            move_folder(source, destination)
            print("  Done.")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  Failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
