#!/usr/bin/env python3
"""Bistable word-perception analysis for Combinato single units.

This follows the bistable test in Zhang, Leonard, Bhaya-Grossman, Gwilliams,
and Chang, Neuron (2026), https://doi.org/10.1016/j.neuron.2025.10.011.
The paper compared high-gamma on ECoG. Here the same comparisons are run on
sorted microwire units.

Each CSV row is one cycle of a repeating ambiguous sound. The row names the
delivered stimulus and the word the listener reported. Cycles are locked to
the stimulus TTL in the merged Intan file, then aligned to the perceived
word onset. For a two-syllable loop, the percept that starts on the second
syllable is shifted by half the cycle. One-syllable percepts stay on the TTL.

Pipeline figures, written in this order:

  01_percept_reports.png    which word was reported on each cycle
  02_alignment.png          TTL time, perceived onset, and the shift used
  03_evoked_response.png    firing rate aligned to the perceived onset
  04_percept_ranksum.png    rank-sum z between the two percepts
  05_region_summary.png     evoked and percept-selective units by region

Example:

  python spike_sorting_script/combinato/word_boun/bistable_unit_pipeline.py \\
    --subject sub-007 \\
    --log-dir /path/to/the/four/csv/files
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata

RESULT_ROOT = Path("/share/home/mitan/combinato_result/word_boun")
DATA_ROOT = Path("/share/workspace3/ieeg/micro/word_boun_perce_v2")
REGIONS = {
    "sub-007": [
        (17, 32, "ATL"),
        (33, 48, "VmPFC"),
        (65, 80, "HG"),
        (97, 112, "Amygdala"),
    ],
    "sub-008": [
        (17, 32, "ATL"),
        (33, 48, "Amygdala"),
        (49, 64, "HG"),
        (97, 112, "VmPFC"),
    ],
    "sub-009": [(1, 16, "Amygdala"), (17, 32, "VmPFC"), (33, 48, "HG")],
    "sub-010": [(17, 32, "Amygdala"), (33, 48, "VmPFC")],
    "sub-011": [(17, 32, "HG"), (33, 48, "ATL")],
    "sub-012": [
        (1, 16, "ATL"),
        (17, 32, "HG"),
        (33, 48, "Amygdala"),
        (49, 64, "VmPFC"),
    ],
}
REGION_ORDER = ("ATL", "HG", "VmPFC", "Amygdala")
STIMULUS_COLUMNS = (
    "stimulus",
    "stimulus_file",
    "sound",
    "sound_file",
    "audio",
    "wav",
    "filename",
    "word",
    "item",
    "token",
)
PERCEPT_COLUMNS = (
    "percept",
    "perception",
    "perceived",
    "perceived_word",
    "response",
    "mapped_response",
    "choice",
    "button",
    "hand",
)
SYLLABLE_COLUMNS = ("syllables", "n_syllables", "syllable_count")
SHIFT_COLUMNS = ("shift_ms", "boundary_shift_ms")
ONSET_COLUMNS = ("onset_s", "onset_sec", "time_s", "anchor_s")
ONSET_MS_COLUMNS = ("onset_ms", "time_ms")
EXCLUDED_PERCEPTS = {
    "",
    "none",
    "nan",
    "na",
    "other",
    "others",
    "neither",
    "both",
    "ambiguous",
    "nonword",
    "non-word",
}
TYPE_NAMES = {-1: "artifact", 0: "unassigned", 1: "mu", 2: "su"}
TYPE_CODES = {"artifact": -1, "unassigned": 0, "mu": 1, "su": 2}
PERCEPT_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666")
Z_THRESHOLD = 1.959963984540054


@dataclass
class Trial:
    block: str
    csv_path: str
    row: int
    stimulus: str
    percept: str
    syllables: int | None
    anchor_s: float
    shift_s: float
    align_s: float
    shift_rule: str
    included: bool


@dataclass
class Unit:
    unit_id: str
    channel: str
    channel_number: int
    region: str
    group_id: int
    unit_type: str
    spikes_s: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Zhang et al. bistable word analysis on Combinato units.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "CSV columns, case-insensitive:\n"
            "  stimulus              delivered sound (required)\n"
            "  percept or response   reported word (required)\n"
            "  syllables             1 or 2. Two-syllable cycles can take a half-cycle shift.\n"
            "  shift_ms              explicit perceived-onset shift for that row\n"
            "  onset_s               use this recording time instead of the TTL\n"
            "  left_word, right_word map a left/right button onto word names\n"
            "\n"
            "A two-syllable shift is applied only when syllables is 2, the stimulus is\n"
            "listed in --two-syllable, or --shift names that percept. The percept whose\n"
            "name matches the stimulus file keeps shift 0. The other moves by half the\n"
            "median cycle, which assumes the two syllables have similar duration.\n"
        ),
    )
    parser.add_argument("--subject", default="sub-007")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--merged", type=Path, default=None, help="Merged subject .mat with board_dig_in_data.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory containing the stimulus-order CSVs.")
    parser.add_argument("--csv", action="append", type=Path, default=None, help="One stimulus-order CSV. Repeatable.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sort-label", default="sort_pos_simple")
    parser.add_argument("--sign", choices=("pos", "neg"), default="pos")
    parser.add_argument("--sample-rate", type=float, default=30000.0)
    parser.add_argument("--trigger-channel", type=int, default=None, help="1-based digital channel. Default: best count match.")
    parser.add_argument("--stimulus-column", default=None)
    parser.add_argument("--percept-column", default=None)
    parser.add_argument("--two-syllable", action="append", default=None, help="Stimulus name whose other percept starts half a cycle later.")
    parser.add_argument("--shift", action="append", default=None, help="percept=milliseconds, for example boater=240")
    parser.add_argument("--types", default="mu,su", help="Combinato group types to keep. Automatic clustering labels units mu.")
    parser.add_argument("--min-spikes", type=int, default=50)
    parser.add_argument("--min-trials", type=int, default=8)
    parser.add_argument("--t-before", type=float, default=0.2)
    parser.add_argument("--t-after", type=float, default=0.6)
    parser.add_argument("--bin-ms", type=float, default=10.0)
    parser.add_argument("--sigma-ms", type=float, default=25.0)
    parser.add_argument("--evoked-z", type=float, default=3.0)
    parser.add_argument("--cluster-alpha", type=float, default=0.05)
    parser.add_argument("--min-cluster-bins", type=int, default=2)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-switches", action="store_true", help="Drop the first cycle after a percept change.")
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def stem(value: str) -> str:
    text = str(value).strip().lower()
    text = Path(text).stem if text else ""
    return re.sub(r"[\s]+", "", text)


def natural_key(path: Path) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path)))


def decode_item(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        return "".join(decode_item(part) for part in value.ravel())
    return str(value).strip()


def find_column(columns: Sequence[str], requested: str | None, aliases: Sequence[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    if requested:
        found = lookup.get(requested.lower())
        require(found is not None, f"Column {requested!r} is not in {list(columns)}")
        return found
    for name in aliases:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text.splitlines() else ""
    delimiter = "\t" if first.count("\t") > first.count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    require(bool(rows), f"{path} has no data rows")
    columns = list(rows[0].keys())
    clean = []
    for row in rows:
        item = {str(key): ("" if value is None else str(value).strip()) for key, value in row.items()}
        if any(item.values()):
            clean.append(item)
    require(bool(clean), f"{path} has no non-empty rows")
    return columns, clean


def parse_shift_args(items: Sequence[str] | None) -> dict[str, float]:
    shifts: dict[str, float] = {}
    for item in items or []:
        require("=" in item, f"--shift must look like percept=milliseconds, got {item!r}")
        name, raw = item.split("=", 1)
        shifts[stem(name)] = float(raw)
    return shifts


def parse_types(text: str) -> set[int]:
    codes = set()
    for part in text.split(","):
        name = part.strip().lower()
        require(name in TYPE_CODES, f"Unknown unit type {part!r}. Use mu and su.")
        codes.add(TYPE_CODES[name])
    return codes


def csv_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.csv or [])
    if args.log_dir is not None:
        require(args.log_dir.is_dir(), f"CSV directory not found: {args.log_dir}")
        paths.extend(path for path in args.log_dir.glob("*.csv") if path.is_file())
    unique = sorted({path.resolve() for path in paths}, key=natural_key)
    require(bool(unique), "Pass --log-dir or repeat --csv with the stimulus-order files.")
    return unique


def optional_float(row: dict[str, str], column: str | None) -> float | None:
    if column is None:
        return None
    text = row.get(column, "").strip()
    if not text:
        return None
    return float(text)


def map_percept(row: dict[str, str], raw: str, columns: Sequence[str]) -> str:
    label = stem(raw)
    lookup = {column.lower(): column for column in columns}
    if label in {"left", "l", "right", "r"} and "left_word" in lookup and "right_word" in lookup:
        column = lookup["left_word"] if label in {"left", "l"} else lookup["right_word"]
        mapped = stem(row.get(column, ""))
        if mapped:
            return mapped
    if label in {"l", "left"}:
        return "left"
    if label in {"r", "right"}:
        return "right"
    if label in {"both", "either"}:
        return "both"
    if label in {"neither", "none", "other"}:
        return "neither" if label == "neither" else label
    return stem(raw)


def load_blocks(paths: Sequence[Path], args: argparse.Namespace) -> list[dict]:
    blocks = []
    for path in paths:
        columns, rows = read_table(path)
        stimulus_column = find_column(columns, args.stimulus_column, STIMULUS_COLUMNS)
        percept_column = find_column(columns, args.percept_column, PERCEPT_COLUMNS)
        require(
            stimulus_column is not None and percept_column is not None,
            f"{path.name} needs a stimulus column and a percept/response column. Columns: {columns}",
        )
        syllable_column = find_column(columns, None, SYLLABLE_COLUMNS)
        shift_column = find_column(columns, None, SHIFT_COLUMNS)
        onset_column = find_column(columns, None, ONSET_COLUMNS)
        onset_ms_column = find_column(columns, None, ONSET_MS_COLUMNS)
        parsed = []
        for row_number, row in enumerate(rows, start=1):
            stimulus = stem(row.get(stimulus_column, ""))
            if not stimulus:
                continue
            syllables = None
            if syllable_column is not None and row.get(syllable_column, "").strip():
                syllables = int(float(row[syllable_column]))
            onset = optional_float(row, onset_column)
            if onset is None and onset_ms_column is not None:
                onset_ms = optional_float(row, onset_ms_column)
                onset = None if onset_ms is None else onset_ms / 1000.0
            parsed.append(
                {
                    "row": row_number,
                    "stimulus": stimulus,
                    "percept": map_percept(row, row.get(percept_column, ""), columns),
                    "syllables": syllables,
                    "shift_ms": optional_float(row, shift_column),
                    "onset_s": onset,
                }
            )
        require(bool(parsed), f"{path} has no stimulus names")
        blocks.append({"path": path, "name": path.stem, "rows": parsed, "columns": columns})
    return blocks


def dataset_time_axis(shape: tuple[int, ...]) -> int:
    require(len(shape) == 2, f"board_dig_in_data must be 2D, got shape {shape}")
    if shape[0] < shape[1]:
        return 1
    return 0


def read_sample_rate(handle: h5py.File, default: float) -> float:
    if "board_dig_in_sample_rate" in handle:
        return float(np.asarray(handle["board_dig_in_sample_rate"]).ravel()[0])
    try:
        return float(np.asarray(handle["frequency_parameters"]["board_dig_in_sample_rate"]).ravel()[0])
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def rising_edges(dataset, channel: int, time_axis: int, block: int = 1_000_000) -> np.ndarray:
    n_time = int(dataset.shape[time_axis])
    edges: list[np.ndarray] = []
    prev_high = False
    pos = 0
    while pos < n_time:
        take = min(block, n_time - pos)
        if time_axis == 0:
            piece = np.asarray(dataset[pos : pos + take, channel]).ravel()
        else:
            piece = np.asarray(dataset[channel, pos : pos + take]).ravel()
        high = piece > 0
        if high.size and high[0] and not prev_high and pos > 0:
            edges.append(np.array([pos], dtype=np.int64))
        if high.size > 1:
            rises = np.flatnonzero((~high[:-1]) & high[1:]) + 1
            if rises.size:
                edges.append(rises.astype(np.int64) + pos)
        prev_high = bool(high[-1]) if high.size else prev_high
        pos += take
    if not edges:
        return np.array([], dtype=np.int64)
    return np.concatenate(edges)


def edge_times_s(indices: np.ndarray, sample_rate: float, timestamps: np.ndarray | None) -> np.ndarray:
    if timestamps is not None and timestamps.size:
        return timestamps[indices]
    return indices.astype(np.float64) / sample_rate


def choose_trigger_channel(
    counts: Sequence[int],
    expected: int,
    forced: int | None,
) -> int:
    if forced is not None:
        require(1 <= forced <= len(counts), f"Trigger channel {forced} is outside 1..{len(counts)}")
        return forced - 1
    scored = []
    for index, count in enumerate(counts):
        if count == 0:
            continue
        scored.append((abs(count - expected) / max(expected, 1), index))
    require(bool(scored), "No rising edges were found on the digital inputs.")
    return min(scored)[1]


def split_edges(times: np.ndarray, n_blocks: int, row_counts: Sequence[int]) -> tuple[list[np.ndarray], str]:
    if times.size == 0:
        return [np.array([], dtype=np.float64) for _ in range(n_blocks)], "no edges"
    if n_blocks == 1:
        return [times], "single file"
    if times.size == sum(row_counts):
        parts = []
        start = 0
        for count in row_counts:
            parts.append(times[start : start + count])
            start += count
        return parts, "row count"
    intervals = np.diff(times)
    positive = intervals[intervals > 0]
    median = float(np.median(positive)) if positive.size else 0.0
    gap_at = np.flatnonzero(intervals > max(3.0 * median, 2.0))
    groups = []
    start = 0
    for gap in gap_at:
        groups.append(times[start : gap + 1])
        start = gap + 1
    groups.append(times[start:])
    if len(groups) == n_blocks:
        return groups, "pause between files"
    # Fall back to the row counts even when a few edges are missing or extra.
    parts = []
    start = 0
    for count in row_counts:
        parts.append(times[start : start + count])
        start += count
    return parts, "file order"


def cycle_seconds(anchors: Sequence[float]) -> float:
    values = np.asarray(list(anchors), dtype=np.float64)
    if values.size < 3:
        return 0.0
    intervals = np.diff(np.sort(values))
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        return 0.0
    median = float(np.median(intervals))
    typical = intervals[intervals < max(3.0 * median, 2.0)]
    if typical.size == 0:
        return median
    return float(np.median(typical))


def stimulus_parts(stimulus: str) -> list[str]:
    return [part for part in re.split(r"[_\-+]+", stem(stimulus)) if part]


def shift_for_pair(
    stimulus: str,
    percepts: tuple[str, str],
    cycle_s: float,
    syllables: int | None,
    two_syllable: set[str],
    explicit_ms: dict[str, float],
) -> tuple[dict[str, float], str]:
    shifts = {percept: explicit_ms[stem(percept)] / 1000.0 for percept in percepts if stem(percept) in explicit_ms}
    use_half = syllables == 2 or stem(stimulus) in two_syllable
    if not use_half:
        for percept in percepts:
            shifts.setdefault(percept, 0.0)
        return shifts, "onset"
    parts = stimulus_parts(stimulus)
    prefix = parts[0] if parts else stem(stimulus)
    matched = [percept for percept in percepts if stem(percept) == prefix or stem(percept) == stem(stimulus)]
    if len(matched) != 1 or cycle_s <= 0:
        for percept in percepts:
            shifts.setdefault(percept, 0.0)
        rule = "onset" if cycle_s <= 0 else "onset-unmatched"
        return shifts, rule
    for percept in percepts:
        if percept in shifts:
            continue
        shifts[percept] = 0.0 if percept in matched else cycle_s / 2.0
    return shifts, "half-cycle"


def build_trials(blocks: list[dict], trigger_parts: list[np.ndarray] | None, args: argparse.Namespace) -> tuple[list[Trial], list[str]]:
    warnings: list[str] = []
    explicit = parse_shift_args(args.shift)
    two_syllable = {stem(name) for name in (args.two_syllable or [])}
    draft: list[dict] = []
    for block, edges in zip(blocks, trigger_parts or [None] * len(blocks)):
        rows = block["rows"]
        onsets = [row["onset_s"] for row in rows]
        if all(onset is not None for onset in onsets):
            anchors = [float(onset) for onset in onsets]
            source = "onset column"
        else:
            require(edges is not None, f"{block['name']} has no onset column and no TTL file was loaded")
            anchors = [float(value) for value in edges[: len(rows)]]
            source = "TTL"
            if len(edges) != len(rows):
                warnings.append(
                    f"{block['name']}: {len(edges)} TTL edges for {len(rows)} rows; using {len(anchors)}"
                )
        if len(anchors) < len(rows):
            warnings.append(f"{block['name']}: dropped {len(rows) - len(anchors)} rows with no onset")
        for row, anchor in zip(rows, anchors):
            draft.append({**row, "block": block["name"], "csv": str(block["path"]), "anchor_s": anchor, "source": source})

    by_stimulus: dict[str, list[dict]] = {}
    for item in draft:
        by_stimulus.setdefault(item["stimulus"], []).append(item)

    trials: list[Trial] = []
    for stimulus, items in by_stimulus.items():
        labels = [item["percept"] for item in items]
        pair = primary_percepts(labels)
        syllable_values = [item["syllables"] for item in items if item["syllables"] is not None]
        syllables = int(Counter(syllable_values).most_common(1)[0][0]) if syllable_values else None
        if stem(stimulus) in two_syllable:
            syllables = 2 if syllables is None else syllables
        cycle_s = cycle_seconds([item["anchor_s"] for item in items])
        if pair is None:
            pair_shifts, rule = {}, "no-contrast"
            warnings.append(f"{stimulus}: fewer than two reported words, so there is no percept contrast")
        else:
            pair_shifts, rule = shift_for_pair(stimulus, pair, cycle_s, syllables, two_syllable, explicit)
            if rule == "onset-unmatched":
                warnings.append(
                    f"{stimulus}: neither percept matches the stimulus name, so both stay on the TTL. "
                    "Pass --shift if the second word starts later."
                )
        previous = None
        for item in items:
            percept = item["percept"]
            is_switch = previous is not None and percept not in EXCLUDED_PERCEPTS and percept != previous
            if percept not in EXCLUDED_PERCEPTS:
                previous = percept
            if item["shift_ms"] is not None:
                shift_s = float(item["shift_ms"]) / 1000.0
                item_rule = "shift_ms"
            else:
                shift_s = pair_shifts.get(percept, 0.0)
                item_rule = rule
            included = percept not in EXCLUDED_PERCEPTS and not (args.exclude_switches and is_switch)
            trials.append(
                Trial(
                    block=item["block"],
                    csv_path=item["csv"],
                    row=item["row"],
                    stimulus=stimulus,
                    percept=percept,
                    syllables=item["syllables"] if item["syllables"] is not None else syllables,
                    anchor_s=item["anchor_s"],
                    shift_s=shift_s,
                    align_s=item["anchor_s"] + shift_s,
                    shift_rule=item_rule,
                    included=included,
                )
            )
    return trials, warnings


def primary_percepts(labels: Sequence[str]) -> tuple[str, str] | None:
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    for index, label in enumerate(labels):
        if stem(label) in EXCLUDED_PERCEPTS:
            continue
        counts[label] += 1
        first.setdefault(label, index)
    if len(counts) < 2:
        return None
    ranked = sorted(counts, key=lambda label: (-counts[label], first[label]))
    return ranked[0], ranked[1]


def load_region_map(path: Path | None, subject: str) -> dict[int, str]:
    regions = {number: name for start, stop, name in REGIONS.get(subject, []) for number in range(start, stop + 1)}
    if path is None or not path.is_file():
        return regions
    with h5py.File(path, "r") as handle:
        if "channel_number" not in handle or "brain_region" not in handle:
            return regions
        numbers = [int(value) for value in np.asarray(handle["channel_number"]).ravel()]
        names = [decode_item(value) for value in np.asarray(handle["brain_region"]).ravel()]
    for number, name in zip(numbers, names):
        if name:
            regions[number] = name
    return regions


def load_triggers(path: Path, n_rows: int, row_counts: Sequence[int], args: argparse.Namespace) -> tuple[list[np.ndarray], dict]:
    require(path.is_file(), f"Merged recording not found: {path}")
    with h5py.File(path, "r") as handle:
        require("board_dig_in_data" in handle, f"{path} has no board_dig_in_data")
        digital = handle["board_dig_in_data"]
        time_axis = dataset_time_axis(tuple(int(value) for value in digital.shape))
        n_channels = int(digital.shape[1 - time_axis])
        sample_rate = read_sample_rate(handle, args.sample_rate)
        timestamps = None
        if "t_dig" in handle:
            timestamps = np.asarray(handle["t_dig"], dtype=np.float64).ravel()
            if timestamps.size != digital.shape[time_axis]:
                timestamps = None
        counts = []
        cached = []
        for channel in range(n_channels):
            indices = rising_edges(digital, channel, time_axis)
            cached.append(indices)
            counts.append(int(indices.size))
        chosen = choose_trigger_channel(counts, n_rows, args.trigger_channel)
        times = edge_times_s(cached[chosen], sample_rate, timestamps)
        n_time = int(digital.shape[time_axis])
    parts, how = split_edges(times, len(row_counts), row_counts)
    info = {
        "trigger_channel_1based": chosen + 1,
        "trigger_counts": counts,
        "trigger_split": how,
        "sample_rate_hz": sample_rate,
        "n_samples": n_time,
        "duration_s": n_time / sample_rate,
    }
    return parts, info


def times_to_seconds(times: np.ndarray, duration_s: float, sample_rate: float) -> tuple[np.ndarray, str]:
    if times.size == 0:
        return times.astype(np.float64), "empty"
    peak = float(np.nanmax(times))
    candidates = [("seconds", peak, 1.0), ("milliseconds", peak / 1000.0, 1000.0)]
    if sample_rate > 0:
        candidates.append(("samples", peak / sample_rate, sample_rate))
    inside = [item for item in candidates if 0 <= item[1] <= duration_s * 1.05 + 1.0]
    if not inside:
        return times / 1000.0, "milliseconds"
    inside.sort(key=lambda item: item[1], reverse=True)
    if len(inside) > 1 and abs(inside[0][1] - inside[1][1]) <= 0.05 * max(inside[0][1], 1.0):
        preferred = [item for item in inside[:2] if item[0] == "milliseconds"]
        chosen = preferred[0] if preferred else inside[0]
    else:
        chosen = inside[0]
    return times / chosen[2], chosen[0]


def load_units(subject_dir: Path, args: argparse.Namespace, regions: dict[int, str], duration_s: float) -> tuple[list[Unit], list[str]]:
    notes: list[str] = []
    clocks: set[str] = set()
    accepted = parse_types(args.types)
    units: list[Unit] = []
    if not subject_dir.is_dir():
        raise SystemExit(f"Combinato subject folder not found: {subject_dir}")
    for channel_dir in sorted(subject_dir.glob("A-*"), key=natural_key):
        if not channel_dir.is_dir():
            continue
        match = re.fullmatch(r"A-(\d+)", channel_dir.name)
        if match is None:
            continue
        number = int(match.group(1))
        data_path = channel_dir / f"data_{channel_dir.name}.h5"
        sort_path = channel_dir / args.sort_label / "sort_cat.h5"
        if not data_path.is_file() or not sort_path.is_file():
            notes.append(f"{channel_dir.name}: no {args.sort_label}/sort_cat.h5")
            continue
        with h5py.File(sort_path, "r") as sort_file, h5py.File(data_path, "r") as data_file:
            sign = decode_item(sort_file.attrs.get("sign", args.sign)) or args.sign
            if sign != args.sign:
                notes.append(f"{channel_dir.name}: sorting sign is {sign}, skipped")
                continue
            node = f"{sign}/times"
            if node not in data_file:
                notes.append(f"{channel_dir.name}: no {node}")
                continue
            raw_times = np.asarray(data_file[node], dtype=np.float64).ravel()
            spike_s, time_unit = times_to_seconds(raw_times, duration_s, args.sample_rate)
            clocks.add(time_unit)
            index = np.asarray(sort_file["index"]).ravel().astype(np.int64)
            classes = np.asarray(sort_file["classes"]).ravel()
            groups = np.asarray(sort_file["groups"])
            types = np.asarray(sort_file["types"])
        if groups.ndim == 2 and groups.shape[0] == 2 and groups.shape[1] != 2:
            groups = groups.T
        if types.ndim == 2 and types.shape[0] == 2 and types.shape[1] != 2:
            types = types.T
        if index.size != classes.size:
            notes.append(f"{channel_dir.name}: index and classes have different lengths")
            continue
        valid = (index >= 0) & (index < spike_s.size)
        if not np.all(valid):
            notes.append(f"{channel_dir.name}: {int(np.count_nonzero(~valid))} spike indexes were outside the extracted times")
            index = index[valid]
            classes = classes[valid]
        type_by_group = {int(row[0]): int(row[1]) for row in types}
        for group_id, type_code in sorted(type_by_group.items()):
            if type_code not in accepted or group_id <= 0:
                continue
            class_ids = groups[groups[:, 1] == group_id, 0]
            class_ids = class_ids[class_ids != 0]
            if class_ids.size == 0:
                continue
            chosen = np.isin(classes, class_ids)
            spikes = np.sort(spike_s[index[chosen]])
            if spikes.size < args.min_spikes:
                notes.append(f"{channel_dir.name} group {group_id}: {spikes.size} spikes, below {args.min_spikes}")
                continue
            units.append(
                Unit(
                    unit_id=f"{channel_dir.name}-g{group_id}",
                    channel=channel_dir.name,
                    channel_number=number,
                    region=regions.get(number, "unknown"),
                    group_id=group_id,
                    unit_type=TYPE_NAMES.get(type_code, str(type_code)),
                    spikes_s=spikes,
                )
            )
    if clocks:
        notes.append("spike clock: " + ", ".join(sorted(clocks)))
    return units, notes


def analysis_window(trials: Sequence[Trial], args: argparse.Namespace) -> tuple[float, float, np.ndarray]:
    anchors = [trial.anchor_s for trial in trials]
    cycle_s = cycle_seconds(anchors)
    t_before = args.t_before
    t_after = args.t_after
    if cycle_s > 0:
        t_after = min(t_after, 0.85 * cycle_s)
        t_before = min(t_before, 0.45 * cycle_s)
    require(t_before > 0 and t_after > 0, "The analysis window collapsed. Check the TTL spacing.")
    bin_s = args.bin_ms / 1000.0
    edges = np.arange(-t_before, t_after + bin_s * 0.5, bin_s)
    return t_before, t_after, edges


def trial_rates(spikes_s: np.ndarray, aligns_s: np.ndarray, edges: np.ndarray, sigma_s: float) -> np.ndarray:
    bin_s = float(edges[1] - edges[0])
    n_bins = edges.size - 1
    rates = np.zeros((aligns_s.size, n_bins), dtype=np.float64)
    spikes = np.asarray(spikes_s, dtype=np.float64)
    for index, onset in enumerate(aligns_s):
        left = int(np.searchsorted(spikes, onset + edges[0], side="left"))
        right = int(np.searchsorted(spikes, onset + edges[-1], side="right"))
        relative = spikes[left:right] - onset
        rates[index] = np.histogram(relative, bins=edges)[0] / bin_s
    sigma_bins = sigma_s / bin_s
    if sigma_bins > 0 and rates.size:
        rates = gaussian_filter1d(rates, sigma=sigma_bins, axis=1, mode="constant", truncate=4.0)
    return rates


def evoked_score(rates: np.ndarray) -> float:
    """Peak of the trial-averaged rate, in SD of that average across time.

    Zhang et al. kept electrodes whose high-gamma waveform left its own
    within-trial mean by more than 3 SD. The same screen is applied to the
    average firing-rate waveform, so cycles with no spikes do not have to
    carry a defined within-cycle SD.
    """
    if rates.size == 0:
        return 0.0
    average = rates.mean(axis=0)
    std = float(average.std())
    if std == 0.0:
        return 0.0
    return float(np.max(np.abs(average - average.mean())) / std)


def ranksum_z(rates: np.ndarray, group_a: np.ndarray) -> np.ndarray:
    n1 = int(np.count_nonzero(group_a))
    n2 = int(rates.shape[0] - n1)
    if n1 == 0 or n2 == 0:
        return np.full(rates.shape[1], np.nan)
    ranks = rankdata(rates, axis=0, method="average")
    expected = n1 * (n1 + n2 + 1) / 2.0
    variance = n1 * n2 * (n1 + n2 + 1) / 12.0
    if variance <= 0:
        return np.zeros(rates.shape[1])
    return (ranks[group_a].sum(axis=0) - expected) / np.sqrt(variance)


def cluster_runs(z: np.ndarray, min_bins: int) -> list[tuple[int, int, float]]:
    values = np.nan_to_num(z, nan=0.0)
    runs = []
    for sign in (1.0, -1.0):
        active = sign * values >= Z_THRESHOLD
        index = 0
        while index < active.size:
            if not active[index]:
                index += 1
                continue
            stop = index + 1
            while stop < active.size and active[stop]:
                stop += 1
            if stop - index >= min_bins:
                runs.append((index, stop, float(values[index:stop].sum())))
            index = stop
    return runs


def max_cluster_mass(z: np.ndarray, min_bins: int) -> float:
    masses = [abs(mass) for _, _, mass in cluster_runs(z, min_bins)]
    return max(masses) if masses else 0.0


def percept_test(
    rates: np.ndarray,
    labels: Sequence[str],
    pair: tuple[str, str],
    min_bins: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    group_a = np.asarray([label == pair[0] for label in labels])
    observed = ranksum_z(rates, group_a)
    observed_mass = max_cluster_mass(observed, min_bins)
    null = np.zeros(n_permutations, dtype=np.float64)
    base = group_a.copy()
    for index in range(n_permutations):
        shuffled = rng.permutation(base)
        null[index] = max_cluster_mass(ranksum_z(rates, shuffled), min_bins)
    runs = []
    for start, stop, mass in cluster_runs(observed, min_bins):
        p_value = (1.0 + float(np.count_nonzero(null >= abs(mass)))) / (n_permutations + 1.0)
        runs.append({"start": start, "stop": stop, "mass": mass, "p": p_value})
    best = min(runs, key=lambda item: item["p"]) if runs else None
    return {
        "z": observed,
        "mass": observed_mass,
        "p": None if best is None else best["p"],
        "runs": runs,
        "null_95": float(np.quantile(null, 0.95)) if n_permutations else np.nan,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def color_for(label: str, order: Sequence[str]) -> str:
    if label in EXCLUDED_PERCEPTS:
        return "#BDBDBD"
    if label in order:
        return PERCEPT_COLORS[order.index(label) % len(PERCEPT_COLORS)]
    return "#BDBDBD"


def finish(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_percepts(trials: Sequence[Trial], path: Path, dpi: int) -> None:
    stimuli = list(dict.fromkeys(trial.stimulus for trial in trials))
    fig, axes = plt.subplots(len(stimuli), 1, figsize=(10, max(2.4, 2.2 * len(stimuli))), squeeze=False, layout="constrained")
    for axis, stimulus in zip(axes[:, 0], stimuli):
        chosen = [trial for trial in trials if trial.stimulus == stimulus]
        labels = [trial.percept for trial in chosen]
        pair = primary_percepts(labels)
        order = list(pair) if pair else []
        image = np.zeros((1, len(chosen), 3))
        for index, label in enumerate(labels):
            hex_color = color_for(label, order)
            image[0, index] = matplotlib.colors.to_rgb(hex_color)
        axis.imshow(image, aspect="auto", interpolation="nearest")
        axis.set_yticks([])
        counts = Counter(labels)
        switches = 0
        previous = None
        for label in labels:
            if label in EXCLUDED_PERCEPTS:
                continue
            if previous is not None and label != previous:
                switches += 1
            previous = label
        summary = ", ".join(f"{name} {count}" for name, count in counts.most_common(4))
        axis.set_title(f"{stimulus}: {summary}; switches {switches}/{len(chosen)}", loc="left", fontsize=10)
        axis.set_xlabel("Cycle")
        if order:
            axis.legend(
                handles=[Patch(facecolor=color_for(name, order), edgecolor="none", label=name) for name in order],
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                fontsize=8,
            )
    fig.suptitle("Step 1. Reported word on each cycle", fontsize=13)
    finish(fig, path, dpi)


def plot_alignment(trials: Sequence[Trial], trigger_info: dict, path: Path, dpi: int) -> None:
    stimuli = list(dict.fromkeys(trial.stimulus for trial in trials))
    fig, axes = plt.subplots(len(stimuli), 1, figsize=(10, max(2.6, 2.4 * len(stimuli))), squeeze=False, layout="constrained")
    for axis, stimulus in zip(axes[:, 0], stimuli):
        chosen = [trial for trial in trials if trial.stimulus == stimulus][:40]
        for index, trial in enumerate(chosen):
            axis.plot([trial.anchor_s, trial.anchor_s], [index - 0.35, index + 0.35], color="#4D4D4D", lw=1)
            axis.scatter(trial.align_s, index, s=18, color=color_for(trial.percept, primary_percepts([item.percept for item in chosen]) or ()), zorder=3)
        rules = Counter(trial.shift_rule for trial in trials if trial.stimulus == stimulus)
        shifts = sorted({round(trial.shift_s * 1000.0, 1) for trial in trials if trial.stimulus == stimulus})
        axis.set_title(
            f"{stimulus}: alignment {', '.join(rules)} ; shifts ms {shifts}",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("First cycles")
        axis.set_xlabel("Recording time (s)")
    channel = trigger_info.get("trigger_channel_1based", "onset column")
    split = trigger_info.get("trigger_split", "")
    fig.suptitle(f"Step 2. Perceived onset. Digital channel {channel}. Split: {split}", fontsize=13)
    finish(fig, path, dpi)


def mean_sem(rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rates.size == 0:
        return np.array([]), np.array([])
    mean = rates.mean(axis=0)
    if rates.shape[0] == 1:
        return mean, np.zeros_like(mean)
    sem = rates.std(axis=0, ddof=1) / np.sqrt(rates.shape[0])
    return mean, sem


def plot_evoked(results: list[dict], edges: np.ndarray, path: Path, dpi: int) -> None:
    stimuli = list(dict.fromkeys(item["stimulus"] for item in results))
    regions = [region for region in REGION_ORDER if any(item["region"] == region for item in results)]
    extra = sorted({item["region"] for item in results} - set(regions))
    regions.extend(extra)
    if not stimuli or not regions:
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.text(0.5, 0.5, "No units were loaded", ha="center", va="center")
        axis.axis("off")
        fig.suptitle("Step 3. Evoked response")
        finish(fig, path, dpi)
        return
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, axes = plt.subplots(
        len(stimuli),
        len(regions),
        figsize=(3.3 * len(regions), 2.5 * len(stimuli)),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )
    for row, stimulus in enumerate(stimuli):
        for col, region in enumerate(regions):
            axis = axes[row, col]
            chosen = [item for item in results if item["stimulus"] == stimulus and item["region"] == region and item["pair"]]
            evoked = [item for item in chosen if item["evoked"]]
            shown = evoked or chosen
            if not shown:
                axis.set_title(f"{region}\n{stimulus}\nno units", fontsize=9)
                axis.axvline(0, color="#BDBDBD", lw=0.8)
                continue
            for percept, color in zip(shown[0]["pair"], PERCEPT_COLORS):
                stack = []
                for item in shown:
                    mask = np.asarray(item["labels"]) == percept
                    if np.any(mask):
                        stack.append(item["rates"][mask].mean(axis=0))
                if not stack:
                    continue
                mean, sem = mean_sem(np.vstack(stack))
                axis.plot(centers, mean, color=color, lw=1.4, label=percept)
                axis.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)
            axis.axvline(0, color="#BDBDBD", lw=0.8)
            axis.set_title(f"{region} | {stimulus}\n{len(evoked)}/{len(chosen)} evoked", fontsize=9)
            if row == len(stimuli) - 1:
                axis.set_xlabel("Perceived onset (s)")
            if col == 0:
                axis.set_ylabel("Spikes/s")
            if row == 0 and col == 0:
                axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Step 3. Firing rate at the perceived word onset", fontsize=13)
    finish(fig, path, dpi)


def plot_ranksum(results: list[dict], edges: np.ndarray, alpha: float, path: Path, dpi: int) -> None:
    stimuli = list(dict.fromkeys(item["stimulus"] for item in results if item["tested"]))
    if not stimuli:
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.text(0.5, 0.5, "No stimulus had enough trials for a percept contrast", ha="center", va="center")
        axis.axis("off")
        fig.suptitle("Step 4. Percept contrast")
        finish(fig, path, dpi)
        return
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, axes = plt.subplots(len(stimuli), 1, figsize=(10, max(3.2, 3.0 * len(stimuli))), squeeze=False, layout="constrained")
    for axis, stimulus in zip(axes[:, 0], stimuli):
        chosen = [item for item in results if item["stimulus"] == stimulus and item["tested"]]
        significant = [item for item in chosen if item["significant"]]
        drawn = significant or chosen
        for item in drawn:
            axis.plot(centers, item["z"], color="#BDBDBD" if significant else "#D0D0D0", lw=0.8)
        if drawn:
            mean = np.nanmean(np.vstack([item["z"] for item in drawn]), axis=0)
            axis.plot(centers, mean, color="black", lw=1.8)
        pair = drawn[0]["pair"] if drawn else ("", "")
        axis.axhline(Z_THRESHOLD, color="#0072B2", lw=0.6, ls="--")
        axis.axhline(-Z_THRESHOLD, color="#0072B2", lw=0.6, ls="--")
        axis.axvline(0, color="#BDBDBD", lw=0.8)
        axis.set_ylabel(f"z, {pair[0]} > {pair[1]}")
        axis.set_title(
            f"{stimulus}: {len(significant)}/{len(chosen)} units with a cluster at p < {alpha:g}",
            loc="left",
            fontsize=10,
        )
        axis.set_xlabel("Perceived onset (s)")
    fig.suptitle("Step 4. Rank-sum difference between the two reported words", fontsize=13)
    finish(fig, path, dpi)


def plot_summary(results: list[dict], path: Path, dpi: int) -> None:
    regions = [region for region in REGION_ORDER if any(item["region"] == region for item in results)]
    extra = sorted({item["region"] for item in results} - set(regions))
    regions.extend(extra)
    units = {}
    for item in results:
        units.setdefault(item["unit_id"], {"region": item["region"], "evoked": False, "significant": False})
        units[item["unit_id"]]["evoked"] = units[item["unit_id"]]["evoked"] or bool(item["evoked"])
        units[item["unit_id"]]["significant"] = units[item["unit_id"]]["significant"] or bool(item["significant"])
    tested = []
    evoked = []
    selective = []
    for region in regions:
        chosen = [unit for unit in units.values() if unit["region"] == region]
        tested.append(len(chosen))
        evoked.append(sum(unit["evoked"] for unit in chosen))
        selective.append(sum(unit["significant"] for unit in chosen))
    fig, axis = plt.subplots(figsize=(max(6, 1.4 * max(len(regions), 1)), 4), layout="constrained")
    if not regions:
        axis.text(0.5, 0.5, "No units", ha="center")
        axis.axis("off")
    else:
        x = np.arange(len(regions))
        width = 0.25
        axis.bar(x - width, tested, width, label="Units", color="#4D4D4D")
        axis.bar(x, evoked, width, label="Evoked", color="#0072B2")
        axis.bar(x + width, selective, width, label="Percept effect", color="#D55E00")
        axis.set_xticks(x, regions)
        axis.set_ylabel("Units")
        axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Step 5. Evoked responses and percept-selective units", fontsize=13)
    finish(fig, path, dpi)


def trials_for(trials: Sequence[Trial], stimulus: str, included_only: bool = True) -> list[Trial]:
    return [trial for trial in trials if trial.stimulus == stimulus and (trial.included or not included_only)]


def analyze(units: Sequence[Unit], trials: Sequence[Trial], edges: np.ndarray, args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    sigma_s = args.sigma_ms / 1000.0
    stimuli = list(dict.fromkeys(trial.stimulus for trial in trials))
    results = []
    for unit in units:
        for stimulus in stimuli:
            chosen = trials_for(trials, stimulus)
            pair = primary_percepts([trial.percept for trial in chosen])
            labels = [trial.percept for trial in chosen]
            aligns = np.asarray([trial.align_s for trial in chosen], dtype=np.float64)
            rates = trial_rates(unit.spikes_s, aligns, edges, sigma_s) if aligns.size else np.zeros((0, edges.size - 1))
            score = evoked_score(rates)
            evoked = bool(score > args.evoked_z)
            record = {
                "unit_id": unit.unit_id,
                "region": unit.region,
                "channel": unit.channel,
                "group": unit.group_id,
                "unit_type": unit.unit_type,
                "stimulus": stimulus,
                "pair": pair,
                "labels": labels,
                "rates": rates,
                "evoked": evoked,
                "evoked_z": score,
                "tested": False,
                "significant": False,
                "z": np.full(edges.size - 1, np.nan),
                "p": np.nan,
                "mass": 0.0,
                "n_a": 0,
                "n_b": 0,
            }
            if pair is not None and evoked:
                contrast_labels = [label for label in labels if label in pair]
                contrast_rates = rates[np.asarray([label in pair for label in labels])]
                n_a = contrast_labels.count(pair[0])
                n_b = contrast_labels.count(pair[1])
                record["n_a"] = n_a
                record["n_b"] = n_b
                if n_a >= args.min_trials and n_b >= args.min_trials:
                    outcome = percept_test(
                        contrast_rates,
                        contrast_labels,
                        pair,
                        args.min_cluster_bins,
                        args.n_permutations,
                        rng,
                    )
                    record["tested"] = True
                    record["z"] = outcome["z"]
                    record["p"] = outcome["p"] if outcome["p"] is not None else 1.0
                    record["mass"] = outcome["mass"]
                    record["significant"] = bool(outcome["p"] is not None and outcome["p"] < args.cluster_alpha)
            results.append(record)
    return results


def result_rows(results: Sequence[dict]) -> list[dict]:
    rows = []
    for item in results:
        pair = item["pair"] or ("", "")
        rows.append(
            {
                "unit_id": item["unit_id"],
                "region": item["region"],
                "channel": item["channel"],
                "group": item["group"],
                "unit_type": item["unit_type"],
                "stimulus": item["stimulus"],
                "percept_a": pair[0],
                "percept_b": pair[1],
                "n_a": item["n_a"],
                "n_b": item["n_b"],
                "evoked": int(item["evoked"]),
                "evoked_z": f"{item['evoked_z']:.4f}",
                "tested": int(item["tested"]),
                "cluster_mass": f"{item['mass']:.4f}",
                "cluster_p": "" if item["p"] != item["p"] else f"{float(item['p']):.6f}",
                "significant": int(item["significant"]),
                "positive_z": f"{pair[0]} > {pair[1]}" if pair[0] else "",
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    require(args.min_spikes >= 1, "--min-spikes must be at least 1")
    require(args.min_trials >= 2, "--min-trials must be at least 2")
    require(args.n_permutations >= 19, "--n-permutations must be at least 19")
    require(args.bin_ms > 0 and args.sigma_ms >= 0, "Bin width must be positive")
    paths = csv_paths(args)
    merged = args.merged or (DATA_ROOT / args.subject / f"{args.subject}.mat")
    output = args.output_dir or (args.result_root / "bistable" / args.subject)
    output.mkdir(parents=True, exist_ok=True)

    blocks = load_blocks(paths, args)
    print(f"[CSV] {len(blocks)} files, {sum(len(block['rows']) for block in blocks)} cycles")
    for block in blocks:
        print(f"  {block['path']}")
    row_counts = [len(block["rows"]) for block in blocks]
    needs_ttl = any(any(row["onset_s"] is None for row in block["rows"]) for block in blocks)
    trigger_info: dict = {"trigger_split": "onset column", "duration_s": 0.0, "sample_rate_hz": args.sample_rate}
    parts = None
    if needs_ttl:
        parts, trigger_info = load_triggers(merged, sum(row_counts), row_counts, args)
        print(
            f"[TTL] channel {trigger_info['trigger_channel_1based']} "
            f"counts {trigger_info['trigger_counts']} split by {trigger_info['trigger_split']}"
        )
    elif merged.is_file():
        with h5py.File(merged, "r") as handle:
            if "board_dig_in_data" in handle:
                digital = handle["board_dig_in_data"]
                axis = dataset_time_axis(tuple(int(value) for value in digital.shape))
                rate = read_sample_rate(handle, args.sample_rate)
                trigger_info["duration_s"] = int(digital.shape[axis]) / rate
                trigger_info["sample_rate_hz"] = rate
    trials, warnings = build_trials(blocks, parts, args)
    require(bool(trials), "No cycles were built from the CSVs")
    duration_s = float(trigger_info.get("duration_s") or 0.0)
    if duration_s <= 0:
        duration_s = max(trial.align_s for trial in trials) + 1.0
    regions = load_region_map(merged if merged.is_file() else None, args.subject)
    units, unit_notes = load_units(args.result_root / args.subject, args, regions, duration_s)
    warnings.extend(unit_notes)
    print(f"[UNITS] {len(units)} groups kept from {args.result_root / args.subject}")
    t_before, t_after, edges = analysis_window(trials, args)
    print(f"[WINDOW] {-t_before:.3f} to {t_after:.3f} s, bin {args.bin_ms:g} ms")
    results = analyze(units, trials, edges, args) if units else []

    plot_percepts(trials, output / "01_percept_reports.png", args.dpi)
    plot_alignment(trials, trigger_info, output / "02_alignment.png", args.dpi)
    plot_evoked(results, edges, output / "03_evoked_response.png", args.dpi)
    plot_ranksum(results, edges, args.cluster_alpha, output / "04_percept_ranksum.png", args.dpi)
    plot_summary(results, output / "05_region_summary.png", args.dpi)

    write_csv(
        output / "trials.csv",
        [
            {
                "block": trial.block,
                "csv": trial.csv_path,
                "row": trial.row,
                "stimulus": trial.stimulus,
                "percept": trial.percept,
                "syllables": "" if trial.syllables is None else trial.syllables,
                "anchor_s": f"{trial.anchor_s:.6f}",
                "shift_s": f"{trial.shift_s:.6f}",
                "align_s": f"{trial.align_s:.6f}",
                "shift_rule": trial.shift_rule,
                "included": int(trial.included),
            }
            for trial in trials
        ],
    )
    write_csv(
        output / "units.csv",
        [
            {
                "unit_id": unit.unit_id,
                "region": unit.region,
                "channel": unit.channel,
                "channel_number": unit.channel_number,
                "group": unit.group_id,
                "unit_type": unit.unit_type,
                "n_spikes": int(unit.spikes_s.size),
                "mean_rate_hz": f"{unit.spikes_s.size / duration_s:.4f}" if duration_s else "",
            }
            for unit in units
        ],
    )
    write_csv(output / "tests.csv", result_rows(results))
    n_evoked = len({item["unit_id"] for item in results if item["evoked"]})
    n_significant = len({item["unit_id"] for item in results if item["significant"]})
    lines = [
        f"subject {args.subject}",
        f"cycles {len(trials)}",
        f"units {len(units)}",
        f"evoked units {n_evoked}",
        f"percept-selective units {n_significant}",
        f"window_s {-t_before:.4f} {t_after:.4f}",
        f"csv_files {len(blocks)}",
        f"trigger {trigger_info}",
        "Positive rank-sum z means the first reported word has the higher firing rate.",
        "Combinato's automatic groups are labeled mu until someone marks them su. Both are included by default.",
        "Cluster p is not corrected again across units, matching the electrode-wise test in the paper.",
    ]
    lines.extend(f"warning: {note}" for note in warnings)
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"[DONE] {output}")
    print(f"[RESULT] units {len(units)}, evoked {n_evoked}, percept-selective {n_significant}")
    for note in warnings:
        print(f"[NOTE] {note}")
    return {
        "output": output,
        "trials": trials,
        "units": units,
        "results": results,
        "warnings": warnings,
        "window": (t_before, t_after, edges),
    }


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
