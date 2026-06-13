#!/usr/bin/env python3
"""
Align good-unit firing rates to digital trigger timestamps for session outputs.

This script is designed for session folders like:
  /share/home/mitan/spike_sorting/mountainsort4/sorting_results_session2_v2
  /share/home/mitan/spike_sorting/mountainsort4/sorting_results_session3_v2
  /share/home/mitan/spike_sorting/mountainsort4/sorting_results_session5_v2

It follows the same board_dig_in_data extraction logic used in
spike_sorting/plot_board_dig_in_data.py:
  - board_dig_in_data is sampled digital state matrix (0/1)
  - trigger timestamps are rising edges (0 -> 1) mapped through t_dig

Outputs per session:
  - trigger_qc_summary.csv
  - trigger_qc_summary.txt
  - firing_rate_timecourse_summary.csv
  - firing_rate_flattened.csv
  - firing_rate_timecourse.png
  - firing_rate_heatmap_all_triggers.png
  - firing_rate_heatmap_odd_triggers.png

Notes:
  - Good units are defined using the same QC outputs as spikesort_mountainsort4_v2.py.
  - This script never opens interactive plot windows; it only saves figures to disk.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    plt.ioff()
except ModuleNotFoundError:
    matplotlib = None
    plt = None


REFERENCE_OUTPUT_ROOT = Path("/share/home/mitan/spike_sorting/mountainsort4")
REFERENCE_INPUT_BASE = Path("/share/workspace3/ieeg/micro/word_boun_perce_v1/bistable_sub4")

DEFAULT_SESSION_DIRS: Dict[str, Path] = {
    s: REFERENCE_OUTPUT_ROOT / f"sorting_results_{s}_v2"
    for s in ("session2", "session3", "session5")
}

DEFAULT_MAT_SEARCH_ROOTS: Dict[str, List[Path]] = {
    "session2": [
        REFERENCE_INPUT_BASE / "bistable_sub4_session2",
        REFERENCE_INPUT_BASE,
    ],
    "session3": [
        REFERENCE_INPUT_BASE / "bistable_sub4_session3",
        REFERENCE_INPUT_BASE,
    ],
    "session5": [
        REFERENCE_INPUT_BASE / "bistable_sub4_session5",
        REFERENCE_INPUT_BASE,
    ],
}


@dataclass
class TriggerInfo:
    mat_path: Path
    sample_rate_hz: float
    trigger_channel_index_0based: int
    trigger_channel_label: str
    all_trigger_times_s: np.ndarray
    all_counts_per_channel: List[int]


def parse_kv_arg(items: Iterable[str], arg_name: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{arg_name} must use key=value format, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"{arg_name} contains empty key/value: {item}")
        parsed[key] = value
    return parsed


def parse_session_name_from_dir(path: Path) -> str:
    m = re.search(r"(session\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot infer session name from directory name: {path}")
    return m.group(1)


def ensure_runtime_dependencies() -> None:
    missing: List[str] = []
    if h5py is None:
        missing.append("h5py")
    if plt is None:
        missing.append("matplotlib")
    if missing:
        names = ", ".join(missing)
        raise ModuleNotFoundError(
            "Missing required Python package(s): "
            f"{names}. Install before running, e.g. `pip install {names}`"
        )


def decode_channel_names(f: h5py.File) -> List[str]:
    chg = f["board_dig_in_channels"]
    refs = np.array(chg["native_channel_name"], dtype=object)
    names: List[str] = []
    for i in range(refs.shape[0]):
        ref = refs[i, 0]
        ds = f[ref]
        arr = np.array(ds, dtype=np.uint16)
        s = "".join(chr(int(c)) for c in arr.flatten() if c)
        names.append(s or f"ch{i}")
    return names


def orient_dig_data(dig_arr: np.ndarray, t_len: int) -> Tuple[np.ndarray, bool]:
    if dig_arr.ndim == 1:
        return dig_arr[:, None], False
    if dig_arr.shape[0] == t_len:
        return dig_arr, False
    if dig_arr.shape[1] == t_len:
        return dig_arr.T, True
    return dig_arr, False


def event_indices(binary_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = (binary_signal > 0.5).astype(np.int8)
    d = np.diff(s, prepend=s[0])
    rising = np.where(d == 1)[0]
    falling = np.where(d == -1)[0]
    return rising, falling


def is_valid_trigger_mat(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".mat":
        return False
    try:
        with h5py.File(path, "r") as f:
            return "board_dig_in_data" in f and "t_dig" in f
    except OSError:
        return False


def discover_mat_file(session_name: str, session_dir: Path, explicit: Dict[str, Path]) -> Path:
    if session_name in explicit:
        candidate = explicit[session_name]
        if not is_valid_trigger_mat(candidate):
            raise FileNotFoundError(
                f"Provided MAT file for {session_name} is missing or invalid: {candidate}"
            )
        return candidate

    search_roots: List[Path] = [session_dir, session_dir.parent]
    search_roots.extend(DEFAULT_MAT_SEARCH_ROOTS.get(session_name, []))

    session_pattern = f"*{session_name}*.mat"
    patterns = (session_pattern, "Temp*.mat", "*.mat")

    tested: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for p in root.rglob(pattern):
                if p in tested:
                    continue
                tested.append(p)
                if is_valid_trigger_mat(p):
                    return p

    hint = (
        f"Could not auto-discover MAT file for {session_name}. "
        f"Use --mat {session_name}=/path/to/file.mat"
    )
    raise FileNotFoundError(hint)


def choose_trigger_channel(
    dig: np.ndarray, t_full: np.ndarray, labels: List[str], force_channel_0based: int | None = None
) -> Tuple[int, np.ndarray, List[int]]:
    n_ch = dig.shape[1]
    per_ch_rising: List[np.ndarray] = []
    counts: List[int] = []

    for ch in range(n_ch):
        rising_idx, _ = event_indices(dig[:, ch])
        rising_ts = t_full[rising_idx] if rising_idx.size else np.array([], dtype=np.float64)
        per_ch_rising.append(rising_ts)
        counts.append(int(rising_ts.size))

    if force_channel_0based is not None:
        if force_channel_0based < 0 or force_channel_0based >= n_ch:
            raise ValueError(
                f"--trigger-channel out of range: {force_channel_0based} for n_ch={n_ch}"
            )
        return force_channel_0based, per_ch_rising[force_channel_0based], counts

    candidates = [i for i, c in enumerate(counts) if c >= 600]
    if not candidates:
        best_idx = int(np.argmax(np.array(counts)))
        return best_idx, per_ch_rising[best_idx], counts

    # Prefer channels closest to 600 triggers and with more regular intervals.
    best_score = float("inf")
    best_idx = candidates[0]
    for idx in candidates:
        ts = per_ch_rising[idx]
        if ts.size < 2:
            continue
        intervals = np.diff(ts)
        mean_itv = float(np.mean(intervals))
        std_itv = float(np.std(intervals))
        cv = std_itv / mean_itv if mean_itv > 0 else float("inf")
        count_penalty = abs(ts.size - 600) / 600.0
        score = count_penalty + cv
        if score < best_score:
            best_score = score
            best_idx = idx

    return best_idx, per_ch_rising[best_idx], counts


def extract_triggers_from_mat(
    mat_path: Path, force_channel_0based: int | None = None
) -> TriggerInfo:
    with h5py.File(mat_path, "r") as f:
        t_full = np.array(f["t_dig"][:, 0], dtype=np.float64).ravel()
        dig_raw = np.array(f["board_dig_in_data"], dtype=np.float64)
        dig, _ = orient_dig_data(dig_raw, t_full.shape[0])
        sr = float(np.array(f["frequency_parameters"]["board_dig_in_sample_rate"]).ravel()[0])
        try:
            labels = decode_channel_names(f)
        except Exception:
            labels = [f"DIG {i + 1}" for i in range(dig.shape[1])]

    ch_idx, trigger_ts, counts = choose_trigger_channel(dig, t_full, labels, force_channel_0based)
    label = labels[ch_idx] if ch_idx < len(labels) else f"DIG {ch_idx + 1}"
    return TriggerInfo(
        mat_path=mat_path,
        sample_rate_hz=sr,
        trigger_channel_index_0based=ch_idx,
        trigger_channel_label=label,
        all_trigger_times_s=np.asarray(trigger_ts, dtype=np.float64),
        all_counts_per_channel=counts,
    )


def load_good_unit_spike_times(session_dir: Path) -> Dict[str, np.ndarray]:
    pkl_path = session_dir / "good_units_spike_times.pkl"
    if pkl_path.is_file():
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Unexpected pickle format in {pkl_path}; expected dict")
        out: Dict[str, np.ndarray] = {}
        for k, v in data.items():
            out[str(k)] = np.asarray(v, dtype=np.float64).ravel()
        return out

    # Fallback path for runs where good_units_spike_times.pkl may be absent:
    # reconstruct good-unit spikes from all_spike_times.pkl + QC summary CSV.
    all_pkl_path = session_dir / "all_spike_times.pkl"
    if not all_pkl_path.is_file():
        raise FileNotFoundError(
            "Missing both good_units_spike_times.pkl and all_spike_times.pkl in "
            f"{session_dir}"
        )
    with all_pkl_path.open("rb") as f:
        all_data = pickle.load(f)
    if not isinstance(all_data, dict):
        raise TypeError(f"Unexpected pickle format in {all_pkl_path}; expected dict")

    summary_candidates = [
        session_dir / "good_units_summary.csv",
        session_dir / "all_regions_units_summary.csv",
    ]
    summary_path = next((p for p in summary_candidates if p.is_file()), None)
    if summary_path is None:
        raise FileNotFoundError(
            "Need one of good_units_summary.csv or all_regions_units_summary.csv "
            f"to identify good units in {session_dir}"
        )

    good_keys: set[str] = set()
    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "global_key" not in (reader.fieldnames or []):
            raise ValueError(f"{summary_path} missing required column: global_key")
        has_auto_qc = "auto_qc_pass" in (reader.fieldnames or [])
        for row in reader:
            k = (row.get("global_key") or "").strip()
            if not k:
                continue
            if has_auto_qc:
                qc_val = (row.get("auto_qc_pass") or "").strip().lower()
                if qc_val not in {"true", "1", "yes"}:
                    continue
            good_keys.add(k)

    out: Dict[str, np.ndarray] = {}
    for k, v in all_data.items():
        ks = str(k)
        if ks in good_keys:
            out[ks] = np.asarray(v, dtype=np.float64).ravel()
    if not out:
        raise RuntimeError(
            f"Recovered zero good units from fallback inputs in {session_dir}. "
            f"summary={summary_path.name}, all_spikes={all_pkl_path.name}"
        )
    return out


def make_groups(first_600: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "all_triggers": first_600.copy(),
        "odd_triggers": first_600[::2].copy(),  # 1st, 3rd, 5th, ...
    }


def aligned_population_rate(
    unit_spikes: List[np.ndarray],
    triggers_s: np.ndarray,
    t_before_s: float,
    t_after_s: float,
    bin_size_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if triggers_s.size == 0:
        raise ValueError("No triggers available for alignment.")
    if not unit_spikes:
        raise ValueError("No good units available for alignment.")

    edges = np.arange(-t_before_s, t_after_s + bin_size_s * 1.01, bin_size_s, dtype=np.float64)
    if edges.size < 2:
        raise ValueError("Invalid binning setup; check t_before/t_after/bin_size.")
    centers = (edges[:-1] + edges[1:]) / 2.0

    n_trig = triggers_s.size
    n_bins = centers.size
    n_units = len(unit_spikes)
    rates = np.zeros((n_trig, n_bins), dtype=np.float64)

    for i, trig in enumerate(triggers_s):
        total_counts = np.zeros(n_bins, dtype=np.float64)
        lo = trig - t_before_s
        hi = trig + t_after_s
        for spikes in unit_spikes:
            left = np.searchsorted(spikes, lo, side="left")
            right = np.searchsorted(spikes, hi, side="right")
            if right <= left:
                continue
            rel = spikes[left:right] - trig
            hist, _ = np.histogram(rel, bins=edges)
            total_counts += hist.astype(np.float64)
        rates[i, :] = total_counts / (n_units * bin_size_s)

    return centers, edges, rates


def trigger_count_annotation(n: int) -> str:
    if n < 600:
        return f"WARNING: trigger count {n} is below expected minimum (600)."
    if n > 2000:
        return f"WARNING: trigger count {n} is above expected range upper bound (2000)."
    return f"OK: trigger count {n} is within expected range (600-2000)."


def write_trigger_qc_summary(
    output_dir: Path,
    session_name: str,
    trigger_info: TriggerInfo,
    first_600_count: int,
    note: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trigger_qc_summary.csv"
    txt_path = output_dir / "trigger_qc_summary.txt"

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "session",
                "mat_file",
                "trigger_channel_index_1based",
                "trigger_channel_label",
                "sample_rate_hz",
                "total_rising_triggers",
                "selected_equivalent_trigger_count",
                "count_note",
            ]
        )
        w.writerow(
            [
                session_name,
                str(trigger_info.mat_path),
                trigger_info.trigger_channel_index_0based + 1,
                trigger_info.trigger_channel_label,
                f"{trigger_info.sample_rate_hz:.6f}",
                int(trigger_info.all_trigger_times_s.size),
                int(first_600_count),
                note,
            ]
        )

    per_ch_lines = [
        f"  ch{idx + 1}: {count} rising edges"
        for idx, count in enumerate(trigger_info.all_counts_per_channel)
    ]
    text = "\n".join(
        [
            f"Session: {session_name}",
            f"MAT file: {trigger_info.mat_path}",
            f"Selected trigger channel: {trigger_info.trigger_channel_index_0based + 1} "
            f"({trigger_info.trigger_channel_label})",
            f"Sampling rate (board_dig_in): {trigger_info.sample_rate_hz:.6f} Hz",
            f"Total rising triggers on selected channel: {trigger_info.all_trigger_times_s.size}",
            f"Selected equivalent trigger count (first N<=600): {first_600_count}",
            f"Count note: {note}",
            "Per-channel rising-edge counts:",
            *per_ch_lines,
            "",
        ]
    )
    txt_path.write_text(text, encoding="utf-8")


def write_flattened_csv(
    output_dir: Path,
    session_name: str,
    group_rates: Dict[str, np.ndarray],
    group_triggers: Dict[str, np.ndarray],
    centers: np.ndarray,
    edges: np.ndarray,
    n_units: int,
) -> None:
    flat_path = output_dir / "firing_rate_flattened.csv"
    with flat_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "session",
                "group",
                "trigger_rank_within_group",
                "trigger_time_s",
                "time_bin_start_s",
                "time_bin_end_s",
                "time_bin_center_s",
                "population_rate_hz",
                "n_good_units",
            ]
        )
        for group_name, rates in group_rates.items():
            triggers = group_triggers[group_name]
            for trig_idx in range(rates.shape[0]):
                trig_t = float(triggers[trig_idx])
                for b in range(rates.shape[1]):
                    w.writerow(
                        [
                            session_name,
                            group_name,
                            trig_idx + 1,
                            f"{trig_t:.9f}",
                            f"{edges[b]:.9f}",
                            f"{edges[b + 1]:.9f}",
                            f"{centers[b]:.9f}",
                            f"{rates[trig_idx, b]:.9f}",
                            n_units,
                        ]
                    )


def write_timecourse_summary_csv(
    output_dir: Path,
    session_name: str,
    group_rates: Dict[str, np.ndarray],
    centers: np.ndarray,
    n_units: int,
) -> None:
    out = output_dir / "firing_rate_timecourse_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "session",
                "group",
                "time_bin_center_s",
                "mean_population_rate_hz",
                "sem_population_rate_hz",
                "n_triggers",
                "n_good_units",
            ]
        )
        for group_name, rates in group_rates.items():
            mean_rate = rates.mean(axis=0)
            sem_rate = rates.std(axis=0, ddof=1) / math.sqrt(rates.shape[0]) if rates.shape[0] > 1 else np.zeros_like(mean_rate)
            for i, t in enumerate(centers):
                w.writerow(
                    [
                        session_name,
                        group_name,
                        f"{t:.9f}",
                        f"{mean_rate[i]:.9f}",
                        f"{sem_rate[i]:.9f}",
                        rates.shape[0],
                        n_units,
                    ]
                )


def plot_timecourse(
    output_dir: Path,
    session_name: str,
    centers: np.ndarray,
    group_rates: Dict[str, np.ndarray],
) -> None:
    plt.figure(figsize=(10, 5))
    for group_name, rates in group_rates.items():
        mean_rate = rates.mean(axis=0)
        sem_rate = rates.std(axis=0, ddof=1) / math.sqrt(rates.shape[0]) if rates.shape[0] > 1 else np.zeros_like(mean_rate)
        plt.plot(centers, mean_rate, linewidth=2.0, label=f"{group_name} (n={rates.shape[0]} triggers)")
        plt.fill_between(centers, mean_rate - sem_rate, mean_rate + sem_rate, alpha=0.2)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.xlabel("Time relative to trigger (s)")
    plt.ylabel("Population firing rate (Hz)\n(mean across good units)")
    plt.title(f"{session_name}: Trigger-aligned firing rate")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "firing_rate_timecourse.png", dpi=180)
    plt.close()


def plot_heatmap(
    output_dir: Path,
    session_name: str,
    group_name: str,
    centers: np.ndarray,
    rates: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        rates,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[centers[0], centers[-1], 1, rates.shape[0]],
    )
    ax.axvline(0.0, color="w", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.set_xlabel("Time relative to trigger (s)")
    ax.set_ylabel("Trigger rank")
    ax.set_title(f"{session_name}: {group_name} trigger-aligned population rate (Hz)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Hz")
    fig.tight_layout()
    safe_name = group_name.replace(" ", "_")
    fig.savefig(output_dir / f"firing_rate_heatmap_{safe_name}.png", dpi=180)
    plt.close(fig)


def process_one_session(
    session_name: str,
    session_dir: Path,
    mat_path: Path,
    output_subdir: str,
    t_before_s: float,
    t_after_s: float,
    bin_size_s: float,
    trigger_channel_0based: int | None,
) -> None:
    print("=" * 88)
    print(f"[SESSION] {session_name}")
    print(f"  sorting dir: {session_dir}")
    print(f"  MAT file:    {mat_path}")

    output_dir = session_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    spike_dict = load_good_unit_spike_times(session_dir)
    unit_spikes = [sp for sp in spike_dict.values() if sp.size > 0]
    if not unit_spikes:
        raise RuntimeError(f"No non-empty good unit spike trains found in {session_dir}")
    print(f"  good units loaded: {len(unit_spikes)}")

    trigger_info = extract_triggers_from_mat(mat_path, force_channel_0based=trigger_channel_0based)
    n_total = int(trigger_info.all_trigger_times_s.size)
    note = trigger_count_annotation(n_total)
    first_n = min(600, n_total)
    if first_n < 600:
        note = f"{note} Also only {first_n} triggers available for equivalent-trigger groups."

    first_600 = trigger_info.all_trigger_times_s[:first_n]
    groups = make_groups(first_600)

    print(
        "  selected trigger channel: "
        f"{trigger_info.trigger_channel_index_0based + 1} ({trigger_info.trigger_channel_label})"
    )
    print(f"  total selected-channel rising triggers: {n_total}")
    print(f"  equivalent trigger subset size: {first_n}")
    print(f"  note: {note}")

    group_rates: Dict[str, np.ndarray] = {}
    centers_ref: np.ndarray | None = None
    edges_ref: np.ndarray | None = None

    for group_name, trig in groups.items():
        centers, edges, rates = aligned_population_rate(
            unit_spikes=unit_spikes,
            triggers_s=trig,
            t_before_s=t_before_s,
            t_after_s=t_after_s,
            bin_size_s=bin_size_s,
        )
        if centers_ref is None:
            centers_ref = centers
            edges_ref = edges
        group_rates[group_name] = rates
        print(f"  computed rates for {group_name}: n_triggers={trig.size}, n_bins={rates.shape[1]}")

    assert centers_ref is not None and edges_ref is not None

    write_trigger_qc_summary(
        output_dir=output_dir,
        session_name=session_name,
        trigger_info=trigger_info,
        first_600_count=first_n,
        note=note,
    )
    write_flattened_csv(
        output_dir=output_dir,
        session_name=session_name,
        group_rates=group_rates,
        group_triggers=groups,
        centers=centers_ref,
        edges=edges_ref,
        n_units=len(unit_spikes),
    )
    write_timecourse_summary_csv(
        output_dir=output_dir,
        session_name=session_name,
        group_rates=group_rates,
        centers=centers_ref,
        n_units=len(unit_spikes),
    )
    plot_timecourse(output_dir=output_dir, session_name=session_name, centers=centers_ref, group_rates=group_rates)
    plot_heatmap(
        output_dir=output_dir,
        session_name=session_name,
        group_name="all_triggers",
        centers=centers_ref,
        rates=group_rates["all_triggers"],
    )
    plot_heatmap(
        output_dir=output_dir,
        session_name=session_name,
        group_name="odd_triggers",
        centers=centers_ref,
        rates=group_rates["odd_triggers"],
    )
    print(f"  outputs written to: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align good-unit firing rate to digital triggers for session2/3/5."
    )
    parser.add_argument(
        "--session-dir",
        action="append",
        default=[],
        metavar="SESSION=PATH",
        help=(
            "Override session output directory mapping. Repeatable. "
            "Example: --session-dir session2=/share/.../sorting_results_session2_v2"
        ),
    )
    parser.add_argument(
        "--mat",
        action="append",
        default=[],
        metavar="SESSION=PATH",
        help=(
            "Provide MAT file path per session. Repeatable. "
            "Example: --mat session2=/share/.../session2_triggers.mat"
        ),
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=["session2", "session3", "session5"],
        help="Sessions to process (default: session2 session3 session5).",
    )
    parser.add_argument(
        "--output-subdir",
        default="firing_rate_alignment",
        help="Output subfolder created under each session directory.",
    )
    parser.add_argument(
        "--t-before",
        type=float,
        default=0.5,
        help="Seconds before trigger for alignment window (default: 0.5).",
    )
    parser.add_argument(
        "--t-after",
        type=float,
        default=1.0,
        help="Seconds after trigger for alignment window (default: 1.0).",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.02,
        help="Bin size in seconds for firing rate (default: 0.02).",
    )
    parser.add_argument(
        "--trigger-channel",
        type=int,
        default=None,
        help=(
            "Force trigger channel index (1-based) for all sessions. "
            "If omitted, auto-select by count/regularity."
        ),
    )
    args = parser.parse_args()

    try:
        ensure_runtime_dependencies()
    except ModuleNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    try:
        session_dir_overrides = parse_kv_arg(args.session_dir, "--session-dir")
        mat_overrides_raw = parse_kv_arg(args.mat, "--mat")
    except ValueError as e:
        print(f"Argument error: {e}", file=sys.stderr)
        return 2

    session_dirs: Dict[str, Path] = {k: v for k, v in DEFAULT_SESSION_DIRS.items()}
    for k, v in session_dir_overrides.items():
        session_dirs[k] = Path(v).expanduser().resolve()

    mat_overrides: Dict[str, Path] = {
        k: Path(v).expanduser().resolve() for k, v in mat_overrides_raw.items()
    }

    trigger_channel_0based = None
    if args.trigger_channel is not None:
        if args.trigger_channel < 1:
            print("--trigger-channel must be >=1", file=sys.stderr)
            return 2
        trigger_channel_0based = args.trigger_channel - 1

    requested_sessions = list(dict.fromkeys(args.sessions))
    if not requested_sessions:
        print("No sessions requested.", file=sys.stderr)
        return 2

    for s in requested_sessions:
        if s not in session_dirs:
            print(
                f"Session '{s}' has no directory mapping. Use --session-dir {s}=PATH",
                file=sys.stderr,
            )
            return 2

    for session_name in requested_sessions:
        session_dir = session_dirs[session_name]
        if not session_dir.is_dir():
            print(
                f"Session directory not found for {session_name}: {session_dir}\n"
                f"Use --session-dir {session_name}=PATH to override.",
                file=sys.stderr,
            )
            return 2

        mat_path = discover_mat_file(
            session_name=session_name,
            session_dir=session_dir,
            explicit=mat_overrides,
        )

        process_one_session(
            session_name=session_name,
            session_dir=session_dir,
            mat_path=mat_path,
            output_subdir=args.output_subdir,
            t_before_s=float(args.t_before),
            t_after_s=float(args.t_after),
            bin_size_s=float(args.bin_size),
            trigger_channel_0based=trigger_channel_0based,
        )

    print("=" * 88)
    print("[DONE] All requested sessions processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

