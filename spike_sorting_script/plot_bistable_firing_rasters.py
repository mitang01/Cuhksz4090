#!/usr/bin/env python3
"""Create trigger-aligned population firing-rate and raster figures.

The experiment sends one TTL pulse for each odd-numbered stimulus (the first
sound in a pair).  The onset of the following, even-numbered sound is
reconstructed as::

    second_onset = pair_TTL + duration(first_sound)

The script discovers matching Bombcell SortingAnalyzers, CSV logs, and trigger
sources for sub4/sub5/sub6.  It pools trial-averaged single-unit PSTHs either
over all four regions or within one region and writes exactly 15 non-interactive
PNG figures: syllable, word (every second sound), and sub4 session03 response
switches, each for all regions and ATL/HG/VMPFC/Amygdala.
Only Bombcell units labeled good are included.

Firing rates use 10-ms bins and a 50-ms Gaussian kernel by default.  The SEM
is across units.  Raster rows are unit-event observations; a deterministic
reservoir sample keeps population rasters readable without changing the rate
calculation.  Each figure also shows a baseline-normalized unit heatmap sorted
by post-onset peak latency. Significance is assessed after onset with a two-sided one-sample
t-test across units on firing-rate change from the -500-to-0-ms baseline,
followed by max-cluster sign-flip correction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t as t_distribution

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SORTING_ROOT = Path("/share/home/mitan/spike_sorting/mountainsort4")
DEFAULT_LOG_ROOT = Path("/share/workspace3/ieeg/micro/word_boun_perce_v1")
DEFAULT_RAW_ROOTS = [
    Path("/share/workspace3/ieeg/micro/word_boun_perce_v1"),
    Path("/share/workspace2/tangmi/bistable_sub4"),
    Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5"),
    Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_1"),
    Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_2"),
    Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_3"),
]
REGIONS = ("ATL", "HG", "VMPFC", "Amygdala")
ANALYSES = ("syllable", "word", "switch")
SCOPE_NAMES = ("all_regions", *REGIONS)
DEFAULT_ACCEPTED_LABELS = ("good",)
STIMULUS_DURATIONS_MS = {
    "bai": 480.77,
    "bi": 545.40,
    "cai": 368.42,
    "can": 424.23,
    "feng": 549.25,
    "gong": 376.42,
    "hai": 523.40,
    "hua": 433.31,
    "men": 460.73,
    "mi": 575.56,
    "ren": 438.15,
    "se": 312.60,
    "shang": 424.90,
    "shua": 449.98,
    "suo": 501.81,
    "ta": 299.90,
    "xue": 570.21,
    "ya": 524.40,
    "zhuo": 308.06,
}


@dataclass(frozen=True)
class SessionMatch:
    subject: str
    key: str
    sorting_dir: Path
    log_path: Path
    trigger_path: Path
    region_bombcell_paths: dict[str, Path]


@dataclass
class EventGroups:
    syllable: dict[str, np.ndarray]
    word: dict[str, np.ndarray]
    switch: dict[str, np.ndarray]
    stimulus_column: str
    condition_column: str | None
    response_column: str | None
    trigger_count: int
    expected_trigger_count: int
    duration_assumption: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class RateRasterAccumulator:
    edges: np.ndarray
    sigma_bins: float
    max_raster_rows: int
    category: str
    sum_rate: np.ndarray = field(init=False)
    sum_rate_sq: np.ndarray = field(init=False)
    n_units: int = 0
    n_events: int = 0
    _raster_heap: list[tuple[int, int, np.ndarray]] = field(default_factory=list)
    _unit_rates: list[np.ndarray] = field(default_factory=list)
    _serial: int = 0

    def __post_init__(self) -> None:
        n_bins = self.edges.size - 1
        self.sum_rate = np.zeros(n_bins, dtype=np.float64)
        self.sum_rate_sq = np.zeros(n_bins, dtype=np.float64)

    def add_unit(
        self,
        spikes_s: np.ndarray,
        events_s: np.ndarray,
        observation_prefix: str,
    ) -> None:
        """Add one unit, averaging its PSTH over all supplied events."""
        if events_s.size == 0:
            return
        per_event_rates: list[np.ndarray] = []
        bin_s = float(np.diff(self.edges)[0])
        n_bins = self.edges.size - 1
        pad_bins = max(1, int(math.ceil(4.0 * self.sigma_bins)))
        extended_edges = self.edges[0] + np.arange(
            -pad_bins, n_bins + pad_bins + 1, dtype=np.float64
        ) * bin_s
        spikes_s = np.sort(np.asarray(spikes_s, dtype=np.float64).ravel())
        for event_index, onset in enumerate(events_s):
            left = int(np.searchsorted(spikes_s, onset + extended_edges[0], side="left"))
            right = int(np.searchsorted(spikes_s, onset + extended_edges[-1], side="right"))
            relative = spikes_s[left:right] - onset
            counts = np.histogram(relative, bins=extended_edges)[0].astype(np.float64)
            extended_rate = gaussian_filter1d(
                counts / bin_s,
                sigma=self.sigma_bins,
                mode="constant",
                truncate=4.0,
            )
            rate = extended_rate[pad_bins : pad_bins + n_bins]
            per_event_rates.append(rate)
            displayed_spikes = relative[
                (relative >= self.edges[0]) & (relative <= self.edges[-1])
            ]
            self._offer_raster_row(
                displayed_spikes,
                f"{observation_prefix}|{self.category}|{event_index}",
            )
        unit_rate = np.mean(np.vstack(per_event_rates), axis=0)
        self._unit_rates.append(unit_rate)
        self.sum_rate += unit_rate
        self.sum_rate_sq += unit_rate * unit_rate
        self.n_units += 1
        self.n_events += int(events_s.size)

    def _offer_raster_row(self, relative: np.ndarray, key: str) -> None:
        if self.max_raster_rows <= 0:
            return
        # Keep rows with the smallest stable hash values.  A max-heap is
        # represented using negative priorities.
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        priority = int.from_bytes(digest, "big", signed=False)
        entry = (-priority, self._serial, np.asarray(relative, dtype=np.float32))
        self._serial += 1
        if len(self._raster_heap) < self.max_raster_rows:
            heapq.heappush(self._raster_heap, entry)
        elif priority < -self._raster_heap[0][0]:
            heapq.heapreplace(self._raster_heap, entry)

    def mean_sem(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_units == 0:
            empty = np.full_like(self.sum_rate, np.nan)
            return empty, empty.copy()
        mean = self.sum_rate / self.n_units
        if self.n_units == 1:
            return mean, np.zeros_like(mean)
        variance = (self.sum_rate_sq - self.n_units * mean * mean) / (self.n_units - 1)
        sem = np.sqrt(np.maximum(variance, 0.0) / self.n_units)
        return mean, sem

    def raster_rows(self) -> list[np.ndarray]:
        ordered = sorted(self._raster_heap, key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ordered]

    def unit_rates(self) -> np.ndarray:
        if not self._unit_rates:
            return np.empty((0, self.edges.size - 1), dtype=np.float64)
        return np.vstack(self._unit_rates)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot population firing rates and rasters for the bistable-word experiment."
    )
    parser.add_argument("--sorting-root", type=Path, default=DEFAULT_SORTING_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--raw-root",
        action="append",
        type=Path,
        default=None,
        help="Raw-data search root; repeatable. Defaults to the known sub4/sub5/sub6 roots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SORTING_ROOT / "bistable_firing_rate_rasters",
    )
    parser.add_argument("--stimulus-column", default=None)
    parser.add_argument("--condition-column", default=None)
    parser.add_argument("--response-column", default="mapped_response")
    parser.add_argument(
        "--accepted-label",
        action="append",
        default=None,
        help=(
            "Accepted Bombcell label; repeatable "
            "(default: good only; MUA and noise are excluded)."
        ),
    )
    parser.add_argument("--trigger-channel", type=int, default=None, help="Force 1-based TTL channel.")
    parser.add_argument("--sample-rate", type=float, default=30000.0)
    parser.add_argument(
        "--t-before",
        type=float,
        default=0.5,
        help="Seconds before sound onset (paper-equivalent fixation period: 0.5).",
    )
    parser.add_argument(
        "--t-after",
        type=float,
        default=1.35,
        help="Seconds after sound onset (paper-equivalent 1.85-s total window: 1.35).",
    )
    parser.add_argument(
        "--baseline-start",
        type=float,
        default=-0.5,
        help="Start of the pre-onset baseline window in seconds (default: -0.5).",
    )
    parser.add_argument(
        "--baseline-end",
        type=float,
        default=0.0,
        help="End of the pre-onset baseline window in seconds (default: 0).",
    )
    parser.add_argument("--bin-ms", type=float, default=10.0)
    parser.add_argument(
        "--gaussian-sigma-ms",
        type=float,
        default=50.0,
        help="Gaussian smoothing sigma in milliseconds (default: 50).",
    )
    parser.add_argument(
        "--ttest-point-alpha",
        type=float,
        default=0.01,
        help="Pointwise p threshold used to form temporal t-test clusters (default: 0.01).",
    )
    parser.add_argument(
        "--cluster-alpha",
        type=float,
        default=0.01,
        help="Permutation cluster significance threshold (paper: 0.01).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=100,
        help="Number of random unit-level sign flips for cluster correction (default: 100).",
    )
    parser.add_argument("--max-raster-rows", type=int, default=600)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--strict-trigger-count",
        action="store_true",
        help="Skip sessions unless TTL count exactly equals ceil(CSV rows/2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover and validate matching inputs; do not load analyzers or plot.",
    )
    return parser.parse_args(argv)


def normalize_name(value: Any) -> str:
    return str(value).strip().lower()


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path)))


def extract_subject(text: str) -> str | None:
    match = re.search(r"sub[-_]?0*([456])(?:\D|$)", text, flags=re.IGNORECASE)
    return f"sub{match.group(1)}" if match else None


def extract_session_key(text: str, subject: str) -> str | None:
    if subject == "sub4":
        match = re.search(r"session[_-]?0*(\d+)", text, flags=re.IGNORECASE)
        return f"session{int(match.group(1)):02d}" if match else None
    match = re.search(r"temp_(\d{6}_\d{6})", text, flags=re.IGNORECASE)
    return f"temp_{match.group(1)}".lower() if match else None


def bombcell_path_for_region(session_dir: Path, region: str) -> Path | None:
    candidates = [
        session_dir / region / "auto_curation" / "bombcell",
        session_dir / region.lower() / "auto_curation" / "bombcell",
    ]
    for path in candidates:
        if (path / "analyzer_curated").is_dir():
            return path
    return None


def index_logs(log_root: Path) -> dict[tuple[str, str], Path]:
    indexed: dict[tuple[str, str], Path] = {}
    for path in sorted(log_root.rglob("*.csv"), key=natural_sort_key):
        subject = extract_subject(str(path))
        if subject is None:
            continue
        key = extract_session_key(path.name, subject)
        if key is not None:
            indexed.setdefault((subject, key), path)
    return indexed


def is_trigger_mat(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".mat":
        return False
    try:
        if h5py.is_hdf5(path):
            with h5py.File(path, "r") as handle:
                return "board_dig_in_data" in handle
        data = loadmat(path, variable_names=["board_dig_in_data"])
        return "board_dig_in_data" in data
    except (OSError, ValueError, NotImplementedError):
        return False


def index_trigger_sources(raw_roots: Sequence[Path]) -> dict[tuple[str, str], list[Path]]:
    indexed: dict[tuple[str, str], list[Path]] = defaultdict(list)
    seen: set[Path] = set()
    for root in raw_roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates: Iterable[Path] = [root]
        else:
            mat_candidates: Iterable[Path] = root.rglob("*.mat")
            if "sub4" in str(root).lower() or root == DEFAULT_LOG_ROOT:
                legacy_candidates: Iterable[Path] = root.rglob("bistable_sub4_session*")
                candidates = chain(mat_candidates, legacy_candidates)
            else:
                candidates = mat_candidates
        for path in candidates:
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            subject = extract_subject(str(path))
            if subject is None:
                continue
            key = extract_session_key(str(path), subject)
            if key is None:
                continue
            if path.suffix.lower() == ".mat":
                if is_trigger_mat(path):
                    indexed[(subject, key)].append(path)
            elif re.search(r"bistable_sub4_session\d+$", path.name, re.IGNORECASE):
                # Legacy sub4 merged files may contain TTLs in unassigned tail channels.
                indexed[(subject, key)].append(path)
    return indexed


def choose_trigger_source(paths: Sequence[Path], key: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"No trigger source candidates for {key}")

    def rank(path: Path) -> tuple[int, int, int, str]:
        is_mat = 0 if path.suffix.lower() == ".mat" else 1
        exact_stem = 0 if path.stem.lower() == key else 1
        repeated_key = str(path).lower().count(key)
        return is_mat, exact_stem, -repeated_key, str(path)

    return sorted(paths, key=rank)[0]


def discover_sessions(
    sorting_root: Path,
    log_root: Path,
    raw_roots: Sequence[Path],
) -> tuple[list[SessionMatch], list[dict[str, str]]]:
    logs = index_logs(log_root)
    trigger_sources = index_trigger_sources(raw_roots)
    matches: list[SessionMatch] = []
    discovery_rows: list[dict[str, str]] = []
    for session_dir in sorted(sorting_root.glob("sorting_results*"), key=natural_sort_key):
        if not session_dir.is_dir():
            continue
        subject = extract_subject(session_dir.name)
        key = extract_session_key(session_dir.name, subject) if subject else None
        if subject is None or key is None:
            continue
        region_paths = {
            region: path
            for region in REGIONS
            if (path := bombcell_path_for_region(session_dir, region)) is not None
        }
        if not region_paths:
            continue
        log_path = logs.get((subject, key))
        raw_candidates = trigger_sources.get((subject, key), [])
        status = "matched"
        detail = ""
        if log_path is None:
            status, detail = "skipped", "no matching experiment CSV"
        elif not raw_candidates:
            status, detail = "skipped", "no matching trigger source"
        discovery_rows.append(
            {
                "subject": subject,
                "session_key": key,
                "sorting_dir": str(session_dir),
                "log_path": str(log_path or ""),
                "trigger_candidates": ";".join(str(p) for p in raw_candidates),
                "regions": ";".join(region_paths),
                "status": status,
                "detail": detail,
            }
        )
        if status == "matched":
            matches.append(
                SessionMatch(
                    subject=subject,
                    key=key,
                    sorting_dir=session_dir,
                    log_path=log_path,
                    trigger_path=choose_trigger_source(raw_candidates, key),
                    region_bombcell_paths=region_paths,
                )
            )
    return matches, discovery_rows


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = [str(x) for x in (reader.fieldnames or [])]
        rows = [{str(k): ("" if v is None else str(v)) for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"Experiment log has no data rows: {path}")
    return rows, columns


def resolve_column(
    requested: str | None,
    columns: Sequence[str],
    preferred_names: Sequence[str],
    required: bool,
) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    if requested:
        resolved = lower_map.get(requested.lower())
        if resolved is None and required:
            raise KeyError(f"Requested CSV column '{requested}' not found; columns={list(columns)}")
        return resolved
    for name in preferred_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    if required:
        raise KeyError(f"Could not infer required CSV column; columns={list(columns)}")
    return None


def resolve_stimulus_column(
    requested: str | None,
    rows: Sequence[dict[str, str]],
    columns: Sequence[str],
) -> str:
    resolved = resolve_column(
        requested,
        columns,
        ("stimulus", "stimulus_file", "sound", "sound_file", "audio", "wav", "filename"),
        required=False,
    )
    if resolved:
        return resolved
    scored: list[tuple[int, str]] = []
    duration_names = set(STIMULUS_DURATIONS_MS)
    for column in columns:
        values = [Path(row.get(column, "")).stem.lower() for row in rows]
        score = sum(value in duration_names for value in values)
        scored.append((score, column))
    best_score, best_column = max(scored, default=(0, ""))
    if best_score == 0:
        raise KeyError(
            "Could not infer stimulus filename column from the duration table; "
            f"use --stimulus-column. Columns={list(columns)}"
        )
    return best_column


def resolve_condition_column(
    requested: str | None,
    rows: Sequence[dict[str, str]],
    columns: Sequence[str],
) -> str | None:
    resolved = resolve_column(
        requested,
        columns,
        ("condition", "trial_type", "stimulus_type", "pair_type", "bistable"),
        required=False,
    )
    if resolved:
        return resolved
    for column in columns:
        values = " ".join(normalize_name(row.get(column, "")) for row in rows)
        if "bistable" in values:
            return column
    return None


def condition_label(value: str) -> str:
    normalized = normalize_name(value).replace("_", " ").replace("-", " ")
    if not normalized:
        return "all stimuli"
    if "nonbistable" in normalized or "non bistable" in normalized:
        return "nonbistable"
    if "bistable" in normalized:
        return "bistable"
    return str(value).strip()


def stimulus_stem(value: str) -> str:
    return Path(str(value).strip()).stem.lower()


def load_mat_digital(path: Path) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Return digital data as samples x channels, optional timestamps, sample rate."""
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            digital = np.asarray(handle["board_dig_in_data"])
            timestamps = None
            if "t_dig" in handle:
                timestamps = np.asarray(handle["t_dig"], dtype=np.float64).ravel()
            sample_rate = 30000.0
            try:
                sample_rate = float(
                    np.asarray(
                        handle["frequency_parameters"]["board_dig_in_sample_rate"]
                    ).ravel()[0]
                )
            except (KeyError, IndexError, TypeError):
                pass
    else:
        data = loadmat(path)
        digital = np.asarray(data["board_dig_in_data"])
        timestamps = (
            np.asarray(data["t_dig"], dtype=np.float64).ravel() if "t_dig" in data else None
        )
        sample_rate = 30000.0
        frequency = data.get("frequency_parameters")
        if frequency is not None:
            try:
                sample_rate = float(frequency["board_dig_in_sample_rate"][0, 0][0, 0])
            except (IndexError, TypeError, ValueError):
                pass
    if digital.ndim == 1:
        digital = digital[:, None]
    if digital.ndim != 2:
        raise ValueError(f"Unexpected board_dig_in_data shape {digital.shape} in {path}")
    if timestamps is not None:
        if digital.shape[0] != timestamps.size and digital.shape[1] == timestamps.size:
            digital = digital.T
    elif digital.shape[0] < digital.shape[1]:
        digital = digital.T
    return digital, timestamps, sample_rate


def rising_indices(signal: np.ndarray) -> np.ndarray:
    binary = np.asarray(signal).ravel() > 0
    if binary.size < 2:
        return np.array([], dtype=np.int64)
    return np.flatnonzero((~binary[:-1]) & binary[1:]) + 1


def select_ttl_channel(
    times_per_channel: Sequence[np.ndarray],
    expected_count: int,
    forced_channel_1based: int | None,
) -> int:
    if forced_channel_1based is not None:
        index = forced_channel_1based - 1
        if index < 0 or index >= len(times_per_channel):
            raise IndexError(
                f"TTL channel {forced_channel_1based} outside 1..{len(times_per_channel)}"
            )
        return index
    scores: list[tuple[float, int]] = []
    for index, times in enumerate(times_per_channel):
        count = int(times.size)
        if count == 0:
            continue
        count_error = abs(count - expected_count) / max(expected_count, 1)
        if count > 2:
            intervals = np.diff(times)
            regularity = float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 10.0
        else:
            regularity = 10.0
        scores.append((count_error + 0.02 * regularity, index))
    if not scores:
        raise RuntimeError("No rising TTL edges found on any candidate channel")
    return min(scores)[1]


def extract_triggers(
    path: Path,
    expected_count: int,
    sample_rate: float,
    forced_channel_1based: int | None,
) -> tuple[np.ndarray, int, list[int]]:
    channel_number_offset = 0
    forced_candidate_channel = forced_channel_1based
    if path.suffix.lower() == ".mat":
        digital, timestamps, mat_sample_rate = load_mat_digital(path)
        rate = mat_sample_rate if mat_sample_rate > 0 else sample_rate
        per_channel: list[np.ndarray] = []
        for channel in range(digital.shape[1]):
            indices = rising_indices(digital[:, channel])
            if timestamps is not None and timestamps.size == digital.shape[0]:
                per_channel.append(timestamps[indices])
            else:
                per_channel.append(indices.astype(np.float64) / rate)
    else:
        # Legacy sub4 files were sorted as 80-channel int16 streams.  Only the
        # unassigned channels 65-80 are considered TTL candidates.
        channel_number_offset = 64
        if forced_channel_1based is not None and forced_channel_1based > 16:
            forced_candidate_channel = forced_channel_1based - channel_number_offset
        raw = np.memmap(path, dtype=np.int16, mode="r")
        n_channels = 80
        if raw.size % n_channels:
            raise ValueError(f"Legacy binary size is not divisible by 80 channels: {path}")
        data = raw.reshape(-1, n_channels)
        per_channel = []
        for channel in range(64, 80):
            signal = data[:, channel]
            probe = np.asarray(signal[:: max(1, signal.size // 1_000_000)], dtype=np.float64)
            low, high = np.percentile(probe, [1, 99])
            if not np.isfinite(high) or high <= low:
                per_channel.append(np.array([], dtype=np.float64))
                continue
            indices = rising_indices(signal > (low + high) / 2.0)
            per_channel.append(indices.astype(np.float64) / sample_rate)
    channel = select_ttl_channel(per_channel, expected_count, forced_candidate_channel)
    counts = [int(times.size) for times in per_channel]
    return (
        np.asarray(per_channel[channel], dtype=np.float64),
        channel + 1 + channel_number_offset,
        counts,
    )


def pooled_times(times: Sequence[float], label: str) -> dict[str, np.ndarray]:
    values = np.asarray(times, dtype=np.float64)
    return {label: values} if values.size else {}


def build_event_groups(
    session: SessionMatch,
    args: argparse.Namespace,
) -> tuple[EventGroups, dict[str, Any]]:
    rows, columns = read_csv_rows(session.log_path)
    stimulus_column = resolve_stimulus_column(args.stimulus_column, rows, columns)
    condition_column = resolve_condition_column(args.condition_column, rows, columns)
    response_column = resolve_column(
        args.response_column,
        columns,
        ("mapped_response",),
        required=False,
    )
    expected_count = int(math.ceil(len(rows) / 2))
    triggers, trigger_channel, channel_counts = extract_triggers(
        session.trigger_path,
        expected_count=expected_count,
        sample_rate=float(args.sample_rate),
        forced_channel_1based=args.trigger_channel,
    )
    warnings: list[str] = []
    if (
        session.subject == "sub4"
        and session.key == "session03"
        and response_column is None
    ):
        warnings.append(
            "mapped_response column was not found; response-switch figures will contain no events"
        )
    if triggers.size != expected_count:
        message = f"TTL count {triggers.size} != expected ceil(rows/2)={expected_count}"
        if args.strict_trigger_count:
            raise ValueError(message)
        warnings.append(message)
    usable_pairs = min(int(triggers.size), expected_count)
    if usable_pairs == 0:
        raise ValueError("No usable TTL/stimulus pairs")
    usable_rows = min(len(rows), 2 * usable_pairs)
    rows = rows[:usable_rows]
    triggers = triggers[:usable_pairs]

    all_onsets: list[float] = []
    word_onsets: list[float] = []
    missing_duration_names: set[str] = set()
    for row_index, row in enumerate(rows):
        pair_index = row_index // 2
        if pair_index >= triggers.size:
            break
        if row_index % 2 == 0:
            onset = float(triggers[pair_index])
        else:
            first_name = stimulus_stem(rows[row_index - 1].get(stimulus_column, ""))
            duration_ms = STIMULUS_DURATIONS_MS.get(first_name)
            if duration_ms is None:
                missing_duration_names.add(first_name or "<blank>")
                continue
            onset = float(triggers[pair_index] + duration_ms / 1000.0)
            word_onsets.append(onset)
        all_onsets.append(onset)
    if missing_duration_names:
        raise KeyError(
            "No supplied WAV duration for stimulus name(s): "
            + ", ".join(sorted(missing_duration_names))
        )

    switch_times: list[float] = []
    if session.subject == "sub4" and session.key == "session03" and response_column:
        previous: str | None = None
        for pair_index in range(usable_pairs):
            response = rows[2 * pair_index].get(response_column, "").strip()
            if not response:
                continue
            if previous is not None and normalize_name(response) != normalize_name(previous):
                switch_times.append(float(triggers[pair_index]))
            previous = response

    groups = EventGroups(
        syllable=pooled_times(all_onsets, "all syllable onsets"),
        word=pooled_times(word_onsets, "all word onsets"),
        switch=pooled_times(switch_times, "all response switches"),
        stimulus_column=stimulus_column,
        condition_column=condition_column,
        response_column=response_column,
        trigger_count=int(len(triggers)),
        expected_trigger_count=expected_count,
        duration_assumption=(
            "even onset = preceding odd TTL + supplied duration of the odd WAV; "
            "no extra inter-stimulus gap"
        ),
        warnings=warnings,
    )
    qc = {
        "trigger_channel_1based": trigger_channel,
        "trigger_counts_per_channel": channel_counts,
        "csv_rows": len(rows),
        "usable_pairs": usable_pairs,
    }
    return groups, qc


def read_bombcell_labels(path: Path) -> tuple[dict[str, str], str | None]:
    labels_path = path / "bombcell_labels.csv"
    if not labels_path.is_file():
        return {}, None
    rows, columns = read_csv_rows(labels_path)
    label_column = next(
        (column for column in columns if column.lower() in {"bombcell_label", "label"}),
        None,
    )
    if label_column is None:
        return {}, None
    id_columns = [
        column
        for column in columns
        if column.lower() in {"unit_id", "unit", "index", "unnamed: 0"} or not column.strip()
    ]
    id_column = id_columns[0] if id_columns else columns[0]
    return {
        str(row.get(id_column, "")).strip(): normalize_name(row.get(label_column, ""))
        for row in rows
    }, label_column


def unit_spike_diagnostics(units: dict[str, np.ndarray]) -> dict[str, int]:
    zero_spike_units = sum(spikes.size == 0 for spikes in units.values())
    return {
        "included_units_with_spikes": int(len(units) - zero_spike_units),
        "included_units_with_zero_recorded_spikes": int(zero_spike_units),
        "included_spike_count": int(sum(spikes.size for spikes in units.values())),
    }


def load_bombcell_units(
    bombcell_path: Path,
    accepted_labels: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        import spikeinterface.full as si
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SpikeInterface is required to load Bombcell analyzer objects. "
            "Run this script in the spike-sorting environment."
        ) from exc
    analyzer_path = bombcell_path / "analyzer_curated"
    analyzer = si.load_sorting_analyzer(str(analyzer_path))
    sorting = analyzer.sorting
    labels, label_column = read_bombcell_labels(bombcell_path)
    if label_column is None:
        raise FileNotFoundError(
            f"Bombcell labels are required for good-unit-only plotting: "
            f"{bombcell_path / 'bombcell_labels.csv'}"
        )
    if not labels:
        raise RuntimeError(f"Bombcell label table contains no units: {bombcell_path}")
    frequency = float(sorting.get_sampling_frequency())
    units: dict[str, np.ndarray] = {}
    excluded = 0
    for unit_id in sorting.get_unit_ids():
        unit_key = str(unit_id)
        label = labels.get(unit_key)
        if labels and (label or "") not in accepted_labels:
            excluded += 1
            continue
        samples = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0)
        units[unit_key] = np.asarray(samples, dtype=np.float64) / frequency
    return units, {
        "analyzer_path": str(analyzer_path),
        "label_column": label_column,
        "units_in_analyzer": int(len(sorting.get_unit_ids())),
        "units_included": int(len(units)),
        "units_excluded_by_label": excluded,
        **unit_spike_diagnostics(units),
        "sampling_frequency_hz": frequency,
    }


def make_edges(args: argparse.Namespace) -> np.ndarray:
    bin_s = float(args.bin_ms) / 1000.0
    if bin_s <= 0 or args.t_before <= 0 or args.t_after <= 0:
        raise ValueError("bin-ms, t-before, and t-after must all be positive")
    n_bins = int(round((float(args.t_before) + float(args.t_after)) / bin_s))
    return np.linspace(-float(args.t_before), float(args.t_after), n_bins + 1)


def make_accumulator(
    edges: np.ndarray,
    args: argparse.Namespace,
    category: str,
) -> RateRasterAccumulator:
    return RateRasterAccumulator(
        edges=edges,
        sigma_bins=(float(args.gaussian_sigma_ms) / float(args.bin_ms)),
        max_raster_rows=int(args.max_raster_rows),
        category=category,
    )


def add_session_to_accumulators(
    session: SessionMatch,
    event_groups: EventGroups,
    accumulators: dict[tuple[str, str, str], RateRasterAccumulator],
    edges: np.ndarray,
    args: argparse.Namespace,
    accepted_labels: set[str],
) -> list[dict[str, Any]]:
    unit_qc_rows: list[dict[str, Any]] = []
    groups_by_analysis = {
        "syllable": event_groups.syllable,
        "word": event_groups.word,
        "switch": event_groups.switch,
    }
    for region, bombcell_path in session.region_bombcell_paths.items():
        units, qc = load_bombcell_units(bombcell_path, accepted_labels)
        qc.update(
            {
                "subject": session.subject,
                "session_key": session.key,
                "region": region,
                "bombcell_path": str(bombcell_path),
            }
        )
        unit_qc_rows.append(qc)
        for analysis, categories in groups_by_analysis.items():
            for category, event_times in categories.items():
                for scope in ("all_regions", region):
                    key = (analysis, scope, category)
                    accumulator = accumulators.setdefault(
                        key, make_accumulator(edges, args, category)
                    )
                    for unit_id, spikes in units.items():
                        accumulator.add_unit(
                            spikes_s=spikes,
                            events_s=event_times,
                            observation_prefix=f"{session.subject}|{session.key}|{region}|{unit_id}",
                        )
    return unit_qc_rows


def category_colors(categories: Sequence[str]) -> dict[str, Any]:
    palette = plt.get_cmap("tab10")
    return {category: palette(index % 10) for index, category in enumerate(categories)}


def one_sample_t(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_units = values.shape[0]
    means = np.mean(values, axis=0)
    standard_deviations = np.std(values, axis=0, ddof=1)
    standard_errors = standard_deviations / math.sqrt(n_units)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = means / standard_errors
    t_values = np.where(
        (standard_errors == 0) & (means > 0), np.inf, t_values
    )
    t_values = np.where(
        (standard_errors == 0) & (means < 0), -np.inf, t_values
    )
    t_values = np.where(
        (standard_errors == 0) & (means == 0), 0.0, t_values
    )
    p_values = 2.0 * t_distribution.sf(np.abs(t_values), n_units - 1)
    return t_values, p_values


def temporal_clusters(
    t_values: np.ndarray,
    p_values: np.ndarray,
    point_alpha: float,
) -> list[tuple[int, int, float]]:
    significant = np.isfinite(p_values) & (p_values < point_alpha)
    indices = np.flatnonzero(significant)
    if indices.size == 0:
        return []
    sign_changes = np.sign(t_values[indices][1:]) != np.sign(t_values[indices][:-1])
    breaks = np.flatnonzero((np.diff(indices) > 1) | sign_changes) + 1
    groups = np.split(indices, breaks)
    return [
        (int(group[0]), int(group[-1]), float(np.sum(np.abs(t_values[group]))))
        for group in groups
        if group.size
    ]


def cluster_permutation_ttest(
    category_accumulators: dict[str, RateRasterAccumulator],
    centers: np.ndarray,
    baseline_start_s: float,
    baseline_end_s: float,
    point_alpha: float,
    cluster_alpha: float,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    """One-sample t-test of unit firing-rate change with sign-flip correction."""
    unit_rate_blocks = [
        accumulator.unit_rates() for accumulator in category_accumulators.values()
    ]
    unit_rate_blocks = [block for block in unit_rate_blocks if block.shape[0] > 0]
    if not unit_rate_blocks:
        return {"status": "not_estimable", "reason": "no units with matching events"}
    unit_rates = np.vstack(unit_rate_blocks)
    if unit_rates.shape[0] < 2:
        return {
            "status": "not_estimable",
            "reason": "one-sample t-test requires at least two units",
            "n_units": int(unit_rates.shape[0]),
        }
    baseline_mask = (centers >= baseline_start_s) & (centers < baseline_end_s)
    if not np.any(baseline_mask):
        return {"status": "not_estimable", "reason": "baseline window contains no bins"}
    baseline = np.mean(unit_rates[:, baseline_mask], axis=1, keepdims=True)
    changes = unit_rates - baseline
    t_values, p_values = one_sample_t(changes)
    post_onset_mask = centers >= baseline_end_s
    cluster_p_values = np.where(post_onset_mask, p_values, 1.0)
    observed_clusters = temporal_clusters(t_values, cluster_p_values, point_alpha)

    rng = np.random.default_rng(seed)
    null_maxima = np.zeros(n_permutations, dtype=np.float64)
    for permutation_index in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=(changes.shape[0], 1))
        permuted_t, permuted_p = one_sample_t(changes * signs)
        permuted_cluster_p = np.where(post_onset_mask, permuted_p, 1.0)
        clusters = temporal_clusters(permuted_t, permuted_cluster_p, point_alpha)
        null_maxima[permutation_index] = max(
            (cluster[2] for cluster in clusters), default=0.0
        )

    clusters = []
    for start, end, statistic in observed_clusters:
        permutation_p = (
            1.0 + float(np.count_nonzero(null_maxima >= statistic))
        ) / (n_permutations + 1.0)
        clusters.append(
            {
                "start_index": start,
                "end_index": end,
                "t_abs_sum": statistic,
                "direction": "increase" if np.mean(t_values[start : end + 1]) > 0 else "decrease",
                "permutation_p": permutation_p,
                "significant": bool(permutation_p <= cluster_alpha),
            }
        )
    return {
        "status": "ok",
        "n_units": int(changes.shape[0]),
        "baseline_start_s": baseline_start_s,
        "baseline_end_s": baseline_end_s,
        "point_alpha": point_alpha,
        "cluster_alpha": cluster_alpha,
        "n_permutations": n_permutations,
        "t_values": t_values,
        "point_p_values": p_values,
        "mean_change_hz": np.mean(changes, axis=0),
        "clusters": clusters,
    }


def normalized_unit_heatmap(
    category_accumulators: dict[str, RateRasterAccumulator],
    centers: np.ndarray,
    baseline_start_s: float,
    baseline_end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    blocks = [
        accumulator.unit_rates() for accumulator in category_accumulators.values()
    ]
    blocks = [block for block in blocks if block.shape[0] > 0]
    if not blocks:
        return np.empty((0, centers.size)), np.array([], dtype=np.float64)
    unit_rates = np.vstack(blocks)
    baseline_mask = (centers >= baseline_start_s) & (centers < baseline_end_s)
    if not np.any(baseline_mask):
        raise ValueError("Heatmap baseline window contains no bins")
    baseline = np.mean(unit_rates[:, baseline_mask], axis=1, keepdims=True)
    changes = unit_rates - baseline
    scale = np.max(np.abs(changes), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    normalized = changes / scale
    post_indices = np.flatnonzero(centers >= 0.0)
    if post_indices.size == 0:
        return normalized, np.full(normalized.shape[0], np.nan)
    peak_indices = post_indices[np.argmax(normalized[:, post_indices], axis=1)]
    order = np.argsort(centers[peak_indices], kind="stable")
    return normalized[order], centers[peak_indices][order]


def unit_rate_diagnostics(
    category_accumulators: dict[str, RateRasterAccumulator],
) -> dict[str, int]:
    blocks = [
        accumulator.unit_rates() for accumulator in category_accumulators.values()
    ]
    blocks = [block for block in blocks if block.shape[0] > 0]
    if not blocks:
        return {"included_units": 0, "zero_rate_units": 0}
    unit_rates = np.vstack(blocks)
    return {
        "included_units": int(unit_rates.shape[0]),
        "zero_rate_units": int(np.count_nonzero(np.all(unit_rates == 0.0, axis=1))),
    }


def plot_rate_and_raster(
    output_path: Path,
    analysis: str,
    scope: str,
    category_accumulators: dict[str, RateRasterAccumulator],
    statistics: dict[str, Any],
    edges: np.ndarray,
    dpi: int,
) -> None:
    categories = sorted(category_accumulators)
    colors = category_colors(categories)
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, (raster_ax, heatmap_ax, rate_ax) = plt.subplots(
        3,
        1,
        figsize=(11, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.6, 1.4]},
        constrained_layout=True,
    )
    row_offset = 0
    for category in categories:
        accumulator = category_accumulators[category]
        rows = accumulator.raster_rows()
        for row_index, spikes in enumerate(rows):
            if spikes.size:
                raster_ax.scatter(
                    spikes,
                    np.full(spikes.size, row_offset + row_index),
                    s=2.2,
                    color=colors[category],
                    alpha=0.75,
                    linewidths=0,
                    rasterized=True,
                )
        if rows:
            midpoint = row_offset + (len(rows) - 1) / 2.0
            raster_ax.text(
                edges[-1] + 0.015 * (edges[-1] - edges[0]),
                midpoint,
                category,
                color=colors[category],
                fontsize=8,
                va="center",
                clip_on=False,
            )
            row_offset += len(rows)
            raster_ax.axhline(row_offset - 0.5, color="0.85", linewidth=0.6)
        mean, sem = accumulator.mean_sem()
        if accumulator.n_units:
            label = (
                f"{category} (units={accumulator.n_units}, "
                f"events summed across units={accumulator.n_events})"
            )
            rate_ax.plot(centers, mean, color=colors[category], linewidth=2.0, label=label)
            rate_ax.fill_between(
                centers,
                mean - sem,
                mean + sem,
                color=colors[category],
                alpha=0.2,
                linewidth=0,
            )
    baseline_start = float(statistics.get("baseline_start_s", edges[0]))
    baseline_end = float(statistics.get("baseline_end_s", 0.0))
    heatmap, _peak_latencies = normalized_unit_heatmap(
        category_accumulators,
        centers,
        baseline_start_s=baseline_start,
        baseline_end_s=baseline_end,
    )
    rate_diagnostics = unit_rate_diagnostics(category_accumulators)
    if heatmap.shape[0]:
        image = heatmap_ax.imshow(
            heatmap,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            extent=[edges[0], edges[-1], heatmap.shape[0] - 0.5, -0.5],
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            rasterized=True,
        )
        colorbar = fig.colorbar(image, ax=heatmap_ax, pad=0.01)
        colorbar.set_label("Normalized ΔFR")
        heatmap_ax.set_ylabel(
            f"Good units (n={heatmap.shape[0]})\nsorted by peak latency"
        )
        heatmap_ax.set_yticks([])
        if rate_diagnostics["zero_rate_units"] == rate_diagnostics["included_units"]:
            heatmap_ax.set_title(
                "All included unit PSTHs are 0 Hz: no spikes contributed within "
                "the analyzed event windows",
                fontsize=9,
            )
        else:
            heatmap_ax.set_title(
                "Unit activity: baseline-subtracted and scaled to each unit’s maximum |ΔFR|",
                fontsize=9,
            )
    else:
        heatmap_ax.text(
            0.5,
            0.5,
            "No good units with matching events",
            transform=heatmap_ax.transAxes,
            ha="center",
            va="center",
            color="0.4",
        )
        heatmap_ax.set_ylabel("Good units")
    for axis in (raster_ax, heatmap_ax, rate_ax):
        axis.axvline(0, color="black", linestyle="--", linewidth=1.0)
        axis.axvspan(0, 0.57556, color="0.5", alpha=0.08)
    for axis in (raster_ax, rate_ax):
        axis.grid(axis="x", alpha=0.2)
    pretty_analysis = {
        "syllable": "All stimulus onsets (syllable)",
        "word": "Every second stimulus onset (word)",
        "switch": "Firing rate when word percepts switch (sub4 session03)",
    }[analysis]
    pretty_scope = "all four regions" if scope == "all_regions" else scope
    raster_ax.set_title(f"{pretty_analysis} — {pretty_scope}")
    raster_ax.set_ylabel("Sampled unit-event raster rows")
    raster_ax.set_ylim(max(row_offset - 0.5, 0.5), -0.5)
    if row_offset == 0:
        raster_ax.text(
            0.5,
            0.5,
            "No matching events/units",
            transform=raster_ax.transAxes,
            ha="center",
            va="center",
            color="0.4",
        )
    rate_ax.set_xlabel("Time from onset (s)")
    rate_ax.set_ylabel("Firing rate (Hz)\nmean ± SEM across units")
    if rate_diagnostics["zero_rate_units"]:
        rate_ax.text(
            0.01,
            0.82,
            (
                f"Zero-rate PSTHs: {rate_diagnostics['zero_rate_units']}/"
                f"{rate_diagnostics['included_units']} included units; these units "
                "had no spike contribution in the analyzed event windows."
            ),
            transform=rate_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.35",
        )
    direction_colors = {"increase": "#b2182b", "decrease": "#2166ac"}
    if statistics.get("status") == "ok":
        labeled_directions: set[str] = set()
        for cluster in statistics.get("clusters", []):
            if not cluster["significant"]:
                continue
            start = edges[int(cluster["start_index"])]
            end = edges[int(cluster["end_index"]) + 1]
            direction = cluster["direction"]
            color = direction_colors[direction]
            rate_ax.axvspan(start, end, color=color, alpha=0.08, linewidth=0)
            rate_ax.plot(
                [start, end],
                [0.98] * 2,
                transform=rate_ax.get_xaxis_transform(),
                color=color,
                linewidth=4,
                solid_capstyle="butt",
                label=(
                    f"significant {direction} vs pre-onset baseline"
                    if direction not in labeled_directions
                    else None
                ),
            )
            labeled_directions.add(direction)
        rate_ax.text(
            0.01,
            0.90,
            (
                "Two-sided one-sample t-test on ΔFR from "
                f"{statistics['baseline_start_s'] * 1000:g}–"
                f"{statistics['baseline_end_s'] * 1000:g} ms baseline"
            ),
            transform=rate_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.35",
        )
    else:
        rate_ax.text(
            0.01,
            0.98,
            f"Baseline one-sample t-test not estimable: "
            f"{statistics.get('reason', 'insufficient data')}",
            transform=rate_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.35",
        )
    if categories and any(acc.n_units for acc in category_accumulators.values()):
        rate_ax.legend(frameon=False, fontsize=8, loc="upper right")
    rate_ax.set_xlim(edges[0], edges[-1])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def write_rate_summary(
    path: Path,
    accumulators: dict[tuple[str, str, str], RateRasterAccumulator],
    edges: np.ndarray,
) -> None:
    centers = (edges[:-1] + edges[1:]) / 2.0
    rows: list[dict[str, Any]] = []
    for (analysis, scope, category), accumulator in sorted(accumulators.items()):
        mean, sem = accumulator.mean_sem()
        diagnostics = unit_rate_diagnostics({category: accumulator})
        for index, time in enumerate(centers):
            rows.append(
                {
                    "analysis": analysis,
                    "scope": scope,
                    "category": category,
                    "time_s": f"{time:.6f}",
                    "mean_rate_hz": f"{mean[index]:.8f}",
                    "sem_rate_hz": f"{sem[index]:.8f}",
                    "n_units": accumulator.n_units,
                    "n_units_with_nonzero_event_psth": (
                        diagnostics["included_units"] - diagnostics["zero_rate_units"]
                    ),
                    "n_zero_rate_units": diagnostics["zero_rate_units"],
                    "events_summed_across_units": accumulator.n_events,
                    "raster_rows_plotted": len(accumulator.raster_rows()),
                }
            )
    write_csv(path, rows)


def write_ttest_summaries(
    output_dir: Path,
    statistics_results: dict[tuple[str, str], dict[str, Any]],
    edges: np.ndarray,
) -> None:
    centers = (edges[:-1] + edges[1:]) / 2.0
    timecourse_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for (analysis, scope), result in sorted(statistics_results.items()):
        status = result.get("status", "not_estimable")
        reason = result.get("reason", "")
        clusters = result.get("clusters", [])
        is_estimable = status == "ok"
        cluster_rows.append(
            {
                "analysis": analysis,
                "scope": scope,
                "record_type": "summary",
                "analysis_status": status,
                "reason": reason,
                "n_units": result.get("n_units", ""),
                "candidate_cluster_count": len(clusters) if is_estimable else "",
                "significant_cluster_count": (
                    sum(bool(cluster["significant"]) for cluster in clusters)
                    if is_estimable
                    else ""
                ),
                "cluster_id": "",
                "direction": "",
                "start_s": "",
                "end_s": "",
                "t_abs_sum": "",
                "permutation_p": "",
                "cluster_significant": "",
            }
        )
        if not is_estimable:
            timecourse_rows.append(
                {
                    "analysis": analysis,
                    "scope": scope,
                    "analysis_status": status,
                    "reason": reason,
                    "time_s": "",
                    "mean_baseline_corrected_rate_hz": "",
                    "t_value": "",
                    "point_p": "",
                    "point_significant": "",
                    "in_candidate_cluster": "",
                    "cluster_id": "",
                    "cluster_permutation_p": "",
                    "in_cluster_corrected_significant_cluster": "",
                    "n_units": result.get("n_units", ""),
                    "point_alpha": "",
                    "cluster_alpha": "",
                }
            )
            continue
        cluster_by_index: dict[int, tuple[int, dict[str, Any]]] = {}
        for cluster_id, cluster in enumerate(clusters, start=1):
            for index in range(cluster["start_index"], cluster["end_index"] + 1):
                cluster_by_index[index] = (cluster_id, cluster)
        for index, time in enumerate(centers):
            cluster_membership = cluster_by_index.get(index)
            cluster_id = cluster_membership[0] if cluster_membership else ""
            cluster = cluster_membership[1] if cluster_membership else None
            timecourse_rows.append(
                {
                    "analysis": analysis,
                    "scope": scope,
                    "analysis_status": status,
                    "reason": reason,
                    "time_s": f"{time:.6f}",
                    "mean_baseline_corrected_rate_hz": (
                        f"{result['mean_change_hz'][index]:.8g}"
                    ),
                    "t_value": f"{result['t_values'][index]:.8g}",
                    "point_p": f"{result['point_p_values'][index]:.8g}",
                    "point_significant": bool(
                        np.isfinite(result["point_p_values"][index])
                        and result["point_p_values"][index] < result["point_alpha"]
                    ),
                    "in_candidate_cluster": cluster is not None,
                    "cluster_id": cluster_id,
                    "cluster_permutation_p": (
                        f"{cluster['permutation_p']:.8g}" if cluster else ""
                    ),
                    "in_cluster_corrected_significant_cluster": (
                        bool(cluster["significant"]) if cluster else False
                    ),
                    "n_units": result["n_units"],
                    "point_alpha": result["point_alpha"],
                    "cluster_alpha": result["cluster_alpha"],
                }
            )
        for cluster_id, cluster in enumerate(clusters, start=1):
            cluster_rows.append(
                {
                    "analysis": analysis,
                    "scope": scope,
                    "record_type": "cluster",
                    "analysis_status": status,
                    "reason": reason,
                    "n_units": result["n_units"],
                    "candidate_cluster_count": len(clusters),
                    "significant_cluster_count": sum(
                        bool(candidate["significant"]) for candidate in clusters
                    ),
                    "cluster_id": cluster_id,
                    "direction": cluster["direction"],
                    "start_s": f"{edges[cluster['start_index']]:.6f}",
                    "end_s": f"{edges[cluster['end_index'] + 1]:.6f}",
                    "t_abs_sum": f"{cluster['t_abs_sum']:.8g}",
                    "permutation_p": f"{cluster['permutation_p']:.8g}",
                    "cluster_significant": cluster["significant"],
                }
            )
    write_csv(output_dir / "sliding_one_sample_t_timecourse.csv", timecourse_rows)
    write_csv(output_dir / "cluster_significance.csv", cluster_rows)


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.gaussian_sigma_ms <= 0:
        raise ValueError("--gaussian-sigma-ms must be positive")
    if args.max_raster_rows < 0:
        raise ValueError("--max-raster-rows cannot be negative")
    if args.trigger_channel is not None and args.trigger_channel < 1:
        raise ValueError("--trigger-channel must be at least 1")
    if not 0 < args.ttest_point_alpha < 1:
        raise ValueError("--ttest-point-alpha must be between 0 and 1")
    if not 0 < args.cluster_alpha < 1:
        raise ValueError("--cluster-alpha must be between 0 and 1")
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be at least 1")
    if args.baseline_start >= args.baseline_end:
        raise ValueError("--baseline-start must be earlier than --baseline-end")
    if args.baseline_start < -args.t_before or args.baseline_end > args.t_after:
        raise ValueError("baseline window must lie inside the plotted analysis window")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_roots = args.raw_root if args.raw_root is not None else DEFAULT_RAW_ROOTS
        sessions, discovery_rows = discover_sessions(
            sorting_root=args.sorting_root.expanduser().resolve(),
            log_root=args.log_root.expanduser().resolve(),
            raw_roots=[path.expanduser().resolve() for path in raw_roots],
        )
        write_csv(output_dir / "session_discovery.csv", discovery_rows)
        print(f"[INFO] Matched sessions: {len(sessions)}")
        for session in sessions:
            print(
                f"  {session.subject} {session.key}: {session.log_path.name} | "
                f"{session.trigger_path.name} | regions={','.join(session.region_bombcell_paths)}"
            )
        if not sessions:
            raise RuntimeError(
                "No complete sorting/CSV/trigger matches. See session_discovery.csv for reasons."
            )
        if args.dry_run:
            print(f"[DONE] Discovery report: {output_dir / 'session_discovery.csv'}")
            return 0

        edges = make_edges(args)
        accepted_labels = {
            normalize_name(label)
            for label in (args.accepted_label or list(DEFAULT_ACCEPTED_LABELS))
        }
        accumulators: dict[tuple[str, str, str], RateRasterAccumulator] = {}
        session_qc_rows: list[dict[str, Any]] = []
        unit_qc_rows: list[dict[str, Any]] = []
        for session in sessions:
            print(f"[SESSION] {session.subject} {session.key}")
            try:
                event_groups, trigger_qc = build_event_groups(session, args)
            except Exception as exc:  # noqa: BLE001
                print(f"  [SKIP] event construction failed: {type(exc).__name__}: {exc}")
                session_qc_rows.append(
                    {
                        "subject": session.subject,
                        "session_key": session.key,
                        "status": "skipped",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            row = {
                "subject": session.subject,
                "session_key": session.key,
                "status": "processed",
                "sorting_dir": str(session.sorting_dir),
                "log_path": str(session.log_path),
                "trigger_path": str(session.trigger_path),
                "stimulus_column": event_groups.stimulus_column,
                "condition_column": event_groups.condition_column or "",
                "response_column": event_groups.response_column or "",
                "expected_trigger_count": event_groups.expected_trigger_count,
                "used_trigger_count": event_groups.trigger_count,
                "syllable_events": sum(map(len, event_groups.syllable.values())),
                "word_events": sum(map(len, event_groups.word.values())),
                "switch_events": sum(map(len, event_groups.switch.values())),
                "duration_assumption": event_groups.duration_assumption,
                "warnings": event_groups.warnings,
                **trigger_qc,
            }
            session_qc_rows.append(row)
            try:
                unit_qc_rows.extend(
                    add_session_to_accumulators(
                        session=session,
                        event_groups=event_groups,
                        accumulators=accumulators,
                        edges=edges,
                        args=args,
                        accepted_labels=accepted_labels,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                row["status"] = "analyzer_error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  [SKIP] analyzer loading failed: {row['error']}")

        write_csv(output_dir / "session_qc.csv", session_qc_rows)
        write_csv(output_dir / "unit_inclusion_qc.csv", unit_qc_rows)
        if not accumulators:
            raise RuntimeError("No analyzers contributed units; see session_qc.csv")

        figure_paths: list[Path] = []
        statistics_results: dict[tuple[str, str], dict[str, Any]] = {}
        for analysis in ANALYSES:
            for scope in SCOPE_NAMES:
                category_accumulators = {
                    category: accumulator
                    for (item_analysis, item_scope, category), accumulator in accumulators.items()
                    if item_analysis == analysis and item_scope == scope
                }
                seed_text = f"{analysis}|{scope}|cluster-permutation"
                seed = int.from_bytes(
                    hashlib.blake2b(seed_text.encode(), digest_size=4).digest(), "big"
                )
                statistics = cluster_permutation_ttest(
                    category_accumulators=category_accumulators,
                    centers=(edges[:-1] + edges[1:]) / 2.0,
                    baseline_start_s=float(args.baseline_start),
                    baseline_end_s=float(args.baseline_end),
                    point_alpha=float(args.ttest_point_alpha),
                    cluster_alpha=float(args.cluster_alpha),
                    n_permutations=int(args.n_permutations),
                    seed=seed,
                )
                statistics_results[(analysis, scope)] = statistics
                output_path = output_dir / f"{analysis}_{scope}.png"
                plot_rate_and_raster(
                    output_path=output_path,
                    analysis=analysis,
                    scope=scope,
                    category_accumulators=category_accumulators,
                    statistics=statistics,
                    edges=edges,
                    dpi=int(args.dpi),
                )
                figure_paths.append(output_path)
        write_rate_summary(output_dir / "population_rate_summary.csv", accumulators, edges)
        write_ttest_summaries(output_dir, statistics_results, edges)
        run_info = {
            "figures": [str(path) for path in figure_paths],
            "figure_count": len(figure_paths),
            "accepted_bombcell_labels": sorted(accepted_labels),
            "rate_definition": (
                f"{args.bin_ms:g}-ms spike-count bins, Gaussian sigma="
                f"{args.gaussian_sigma_ms:g} ms; trials averaged within unit, then units averaged"
            ),
            "significance_test": (
                "two-sided one-sample t-test across units on firing-rate change from "
                f"the {args.baseline_start:g} to {args.baseline_end:g} s baseline, at "
                f"{args.bin_ms:g}-ms steps; pointwise cluster-forming p < "
                f"{args.ttest_point_alpha:g}; {args.n_permutations} unit-level sign "
                f"flips; max-cluster summed-|t| alpha={args.cluster_alpha:g}"
            ),
            "raster_definition": (
                "rows are deterministic sampled unit-event observations; raster subsampling "
                "does not affect firing rates"
            ),
        }
        (output_dir / "run_info.json").write_text(
            json.dumps(run_info, indent=2), encoding="utf-8"
        )
        print(f"[DONE] Wrote {len(figure_paths)} figures to {output_dir}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
