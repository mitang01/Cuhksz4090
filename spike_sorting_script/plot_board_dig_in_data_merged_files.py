#!/usr/bin/env python3
"""
Plot trigger channel from merged binary files and save trigger diagnostics.

Reference behavior: spike_sorting_script/plot_board_dig_in_data.py
New behavior:
  - Input files are merged binary files (not .mat).
  - Default inputs are the 9 paths provided by user.
  - Output files per input:
      1) full-length trigger plot PNG
      2) trigger-detail summary TXT

Notes:
  - The script assumes merged data is int16, sample-major, shape
    (n_samples, num_channels), matching usage in spikesort scripts.
  - Trigger channel can be forced with --trigger-channel (1-based);
    otherwise it is auto-selected from per-channel rising-edge patterns.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None


DEFAULT_MERGED_FILES: list[Path] = [
    Path("/share/workspace2/tangmi/bistable_sub5_session1"),
    Path("/share/workspace2/tangmi/bistable_sub5_session2"),
    Path("/share/workspace2/tangmi/bistable_sub5_session3"),
    Path("/share/workspace2/tangmi/bistable_sub5_session4"),
    Path("/share/workspace2/tangmi/bistable_sub5_session5"),
    Path("/share/workspace2/tangmi/bistable_sub5_session6"),
    Path("/share/workspace2/tangmi/bistable_sub6_session1"),
    Path("/share/workspace2/tangmi/bistable_sub6_session3"),
    Path("/share/workspace2/tangmi/bistable_sub6_session6"),
]


def safe_stem(path: Path) -> str:
    name = path.name.strip()
    if not name:
        name = "merged_file"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def edge_indices_from_bool(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(binary, dtype=np.int8)
    d = np.diff(s, prepend=s[0])
    rising = np.where(d == 1)[0]
    falling = np.where(d == -1)[0]
    return rising, falling


def filter_by_min_high_width(
    rising_idx: np.ndarray,
    falling_idx: np.ndarray,
    min_high_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if rising_idx.size == 0 or falling_idx.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    keep_rising: list[int] = []
    keep_falling: list[int] = []
    j = 0
    for r in rising_idx:
        while j < falling_idx.size and falling_idx[j] <= r:
            j += 1
        if j >= falling_idx.size:
            break
        f = int(falling_idx[j])
        if (f - int(r)) >= min_high_samples:
            keep_rising.append(int(r))
            keep_falling.append(f)
    return np.asarray(keep_rising, dtype=np.int64), np.asarray(keep_falling, dtype=np.int64)


def detect_binary_and_edges(
    signal: np.ndarray,
    sampling_rate_hz: float,
    min_pulse_ms: float,
    max_probe_points: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    n = signal.shape[0]
    stride = max(1, n // max_probe_points)
    probe = np.asarray(signal[::stride], dtype=np.float64)
    lo = float(np.percentile(probe, 1.0))
    hi = float(np.percentile(probe, 99.0))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return (
            np.zeros(n, dtype=bool),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            float("nan"),
            lo,
            hi,
        )

    thr = 0.5 * (lo + hi)
    binary = np.asarray(signal > thr, dtype=bool)

    rising_raw, falling_raw = edge_indices_from_bool(binary)
    min_high_samples = max(
        1,
        int(round((float(min_pulse_ms) / 1000.0) * float(sampling_rate_hz))),
    )
    rising, falling = filter_by_min_high_width(
        rising_idx=rising_raw,
        falling_idx=falling_raw,
        min_high_samples=min_high_samples,
    )
    return binary, rising, falling, thr, lo, hi


def choose_trigger_channel(
    rising_ts_per_channel: list[np.ndarray],
    forced_channel_0based: int | None,
) -> int:
    n_ch = len(rising_ts_per_channel)
    if n_ch == 0:
        raise ValueError("No channels available to select trigger channel.")

    if forced_channel_0based is not None:
        if forced_channel_0based < 0 or forced_channel_0based >= n_ch:
            raise ValueError(
                f"--trigger-channel out of range: {forced_channel_0based + 1}, "
                f"valid range is [1, {n_ch}]"
            )
        return forced_channel_0based

    counts = np.asarray([ts.size for ts in rising_ts_per_channel], dtype=np.int64)
    nonzero = np.where(counts > 0)[0]
    if nonzero.size == 0:
        return 0

    # Prefer channels with >= 600 triggers and near-regular intervals.
    # This follows the same practical heuristic used in alignment script.
    candidates = [int(i) for i in nonzero if counts[i] >= 600]
    if not candidates:
        return int(nonzero[np.argmax(counts[nonzero])])

    best_idx = candidates[0]
    best_score = float("inf")
    for idx in candidates:
        ts = rising_ts_per_channel[idx]
        if ts.size < 2:
            score = abs(ts.size - 600) / 600.0 + 10.0
        else:
            itv = np.diff(ts)
            mean_itv = float(np.mean(itv))
            std_itv = float(np.std(itv))
            cv = std_itv / mean_itv if mean_itv > 0 else float("inf")
            count_penalty = abs(ts.size - 600) / 600.0
            score = count_penalty + cv
        if score < best_score:
            best_score = score
            best_idx = idx
    return int(best_idx)


def render_selected_channel_plot(
    out_png: Path,
    file_name: str,
    t_plot: np.ndarray,
    binary_plot: np.ndarray,
    rising_idx: np.ndarray,
    sampling_rate_hz: float,
    n_plotted_samples: int,
    n_total_samples: int,
    plot_step: int,
    duration_s: float,
) -> None:
    shown = float(t_plot[-1] - t_plot[0]) if t_plot.size > 1 else 0.0
    width = float(np.clip(11.0 * max(shown, 1.0) / 60.0, 11.0, 80.0))
    fig, ax = plt.subplots(1, 1, figsize=(width, 3.0), constrained_layout=True)
    ax.plot(t_plot, binary_plot, color="C0", linewidth=0.8, drawstyle="steps-post")

    in_window = rising_idx[rising_idx < n_plotted_samples]
    visible = in_window[::plot_step] if plot_step > 1 else in_window
    if visible.size:
        ax.scatter(
            visible / sampling_rate_hz,
            np.ones_like(visible, dtype=np.float64),
            s=14,
            color="C3",
            marker="v",
            alpha=0.85,
            label="rising edge",
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_ylabel("Trigger state")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"{file_name} — selected trigger channel\n"
        f"shown {shown:.3f} s, full {duration_s:.3f} s, step={plot_step}, "
        f"samples={n_plotted_samples}/{n_total_samples}",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def process_one_file(
    merged_path: Path,
    output_dir: Path,
    num_channels: int,
    sample_rate_hz: float,
    dtype_str: str,
    min_pulse_ms: float,
    seconds: float,
    decimate: int,
    max_plot_samples: int,
    max_preview: int,
    trigger_channel_0based: int | None,
) -> tuple[Path, Path]:
    if not merged_path.is_file():
        raise FileNotFoundError(f"Input merged file not found: {merged_path}")

    try:
        dtype = np.dtype(dtype_str)
    except TypeError as exc:
        raise ValueError(f"Invalid --dtype: {dtype_str}") from exc

    flat = np.memmap(merged_path, dtype=dtype, mode="r")
    if flat.size == 0:
        raise RuntimeError(f"Input file is empty: {merged_path}")
    if flat.size % num_channels != 0:
        raise RuntimeError(
            f"File size ({flat.size} values) is not divisible by num_channels={num_channels} "
            f"for file: {merged_path}"
        )

    n_total = int(flat.size // num_channels)
    data = flat.reshape((n_total, num_channels))
    duration_s = float(n_total / sample_rate_hz)

    if seconds > 0:
        n_plot = min(n_total, max(1, int(seconds * sample_rate_hz)))
    else:
        n_plot = n_total

    step = max(1, int(decimate))
    if max_plot_samples > 0:
        auto_step = int(math.ceil(n_plot / max_plot_samples))
        step = max(step, auto_step, 1)

    plot_indices = np.arange(0, n_plot, step, dtype=np.int64)
    t_plot = plot_indices / sample_rate_hz

    labels = [f"RAW_CH_{i + 1}" for i in range(num_channels)]

    binary_plot_per_ch: list[np.ndarray] = []
    rising_idx_per_ch: list[np.ndarray] = []
    falling_idx_per_ch: list[np.ndarray] = []
    rising_ts_per_ch: list[np.ndarray] = []
    falling_ts_per_ch: list[np.ndarray] = []
    threshold_info_per_ch: list[tuple[float, float, float]] = []

    for ch in range(num_channels):
        signal = data[:, ch]
        binary_full, rising_idx, falling_idx, thr, lo, hi = detect_binary_and_edges(
            signal=signal,
            sampling_rate_hz=sample_rate_hz,
            min_pulse_ms=min_pulse_ms,
        )
        binary_plot_per_ch.append(binary_full[plot_indices].astype(np.float64))
        rising_idx_per_ch.append(rising_idx)
        falling_idx_per_ch.append(falling_idx)
        rising_ts_per_ch.append(rising_idx / sample_rate_hz if rising_idx.size else np.array([], dtype=np.float64))
        falling_ts_per_ch.append(
            falling_idx / sample_rate_hz if falling_idx.size else np.array([], dtype=np.float64)
        )
        threshold_info_per_ch.append((thr, lo, hi))

    trigger_ch = choose_trigger_channel(
        rising_ts_per_channel=rising_ts_per_ch,
        forced_channel_0based=trigger_channel_0based,
    )

    file_tag = safe_stem(merged_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{file_tag}_full_length_trigger_plot.png"
    out_txt = output_dir / f"{file_tag}_trigger_details.txt"

    render_selected_channel_plot(
        out_png=out_png,
        file_name=merged_path.name,
        t_plot=t_plot,
        binary_plot=binary_plot_per_ch[trigger_ch],
        rising_idx=rising_idx_per_ch[trigger_ch],
        sampling_rate_hz=sample_rate_hz,
        n_plotted_samples=n_plot,
        n_total_samples=n_total,
        plot_step=step,
        duration_s=duration_s,
    )

    lines: list[str] = []
    lines.append(f"Input merged file: {merged_path}")
    lines.append(f"dtype: {dtype_str}")
    lines.append(f"Interpreted shape: (samples={n_total}, channels={num_channels})")
    lines.append(f"Sampling rate: {sample_rate_hz:.6f} Hz")
    lines.append(f"Recording duration: {duration_s:.6f} s")
    lines.append(
        f"Plotted window: first {n_plot / sample_rate_hz:.6f} s, effective decimation step: {step}"
    )
    lines.append(f"Min pulse width for trigger detection: {min_pulse_ms:.6f} ms")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  Each channel is converted to binary state by threshold=(1st_percentile + 99th_percentile)/2."
    )
    lines.append(
        "  Trigger timestamps are rising edges (0->1) with minimum high-width filtering."
    )
    lines.append("")
    lines.append(
        f"Selected trigger channel: {trigger_ch + 1} ({labels[trigger_ch]}) "
        f"with {rising_idx_per_ch[trigger_ch].size} rising edges"
    )
    lines.append("")

    for ch in range(num_channels):
        thr, lo, hi = threshold_info_per_ch[ch]
        rising_ts = rising_ts_per_ch[ch]
        falling_ts = falling_ts_per_ch[ch]
        lines.append(
            f"Channel {ch + 1} ({labels[ch]}): "
            f"{rising_ts.size} rising edges, {falling_ts.size} falling edges, "
            f"threshold={thr:.6f}, p1={lo:.6f}, p99={hi:.6f}"
        )
        if rising_ts.size:
            preview = ", ".join(f"{x:.6f}" for x in rising_ts[:max_preview])
            suffix = " ..." if rising_ts.size > max_preview else ""
            lines.append(f"  Rising-edge timestamps (s): {preview}{suffix}")
        else:
            lines.append("  Rising-edge timestamps (s): none")
        if falling_ts.size:
            preview = ", ".join(f"{x:.6f}" for x in falling_ts[:max_preview])
            suffix = " ..." if falling_ts.size > max_preview else ""
            lines.append(f"  Falling-edge timestamps (s): {preview}{suffix}")
        else:
            lines.append("  Falling-edge timestamps (s): none")
        lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    return out_png, out_txt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot and summarize board dig-in style triggers from merged binary files."
        )
    )
    parser.add_argument(
        "merged_files",
        nargs="*",
        type=Path,
        default=DEFAULT_MERGED_FILES,
        help="Merged binary files to process (default: the 9 bistable session files).",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=128,
        help="Number of channels in merged binary file (default: 128).",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=30000.0,
        help="Sampling rate in Hz (default: 30000).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="int16",
        help="Data type of merged binary values (default: int16).",
    )
    parser.add_argument(
        "--min-pulse-ms",
        type=float,
        default=0.5,
        help="Minimum high pulse width in ms for edge validity (default: 0.5).",
    )
    parser.add_argument(
        "--trigger-channel",
        type=int,
        default=None,
        help=(
            "Force trigger channel (1-based). If omitted, channel is auto-selected "
            "from edge-count/regularity."
        ),
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        metavar="T",
        help="Analyze/plot only first T seconds (default: 0 = full recording).",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="After time slicing, use every N-th sample for plotting (default: 1).",
    )
    parser.add_argument(
        "--max-plot-samples",
        type=int,
        default=400_000,
        metavar="N",
        help="Auto-increase decimation to keep plotted samples <= N (default: 400000).",
    )
    parser.add_argument(
        "--max-preview",
        type=int,
        default=12,
        help="Max number of edge timestamps previewed per channel in TXT output.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "board_dig_in_merged_outputs",
        help="Output directory for generated PNG/TXT files.",
    )
    args = parser.parse_args()

    if plt is None:
        print(
            "Error: missing required dependency 'matplotlib'. "
            "Install it first, e.g. `pip install matplotlib`.",
            file=sys.stderr,
        )
        return 2

    if args.num_channels <= 0:
        print("Error: --num-channels must be > 0", file=sys.stderr)
        return 2
    if args.sample_rate <= 0:
        print("Error: --sample-rate must be > 0", file=sys.stderr)
        return 2
    if args.min_pulse_ms <= 0:
        print("Error: --min-pulse-ms must be > 0", file=sys.stderr)
        return 2
    if args.decimate <= 0:
        print("Error: --decimate must be > 0", file=sys.stderr)
        return 2
    if args.max_preview <= 0:
        print("Error: --max-preview must be > 0", file=sys.stderr)
        return 2

    trigger_channel_0based: int | None = None
    if args.trigger_channel is not None:
        if args.trigger_channel < 1:
            print("Error: --trigger-channel must be >= 1", file=sys.stderr)
            return 2
        trigger_channel_0based = args.trigger_channel - 1

    merged_files: list[Path] = [p.expanduser().resolve() for p in args.merged_files]
    if not merged_files:
        print("Error: no input merged files provided", file=sys.stderr)
        return 2

    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    failed = False
    for fp in merged_files:
        try:
            out_png, out_txt = process_one_file(
                merged_path=fp,
                output_dir=out_root,
                num_channels=int(args.num_channels),
                sample_rate_hz=float(args.sample_rate),
                dtype_str=args.dtype,
                min_pulse_ms=float(args.min_pulse_ms),
                seconds=float(args.seconds),
                decimate=int(args.decimate),
                max_plot_samples=int(args.max_plot_samples),
                max_preview=int(args.max_preview),
                trigger_channel_0based=trigger_channel_0based,
            )
            print(f"[OK] {fp}")
            print(f"     plot: {out_png}")
            print(f"     text: {out_txt}")
        except Exception as exc:  # noqa: BLE001
            failed = True
            print(f"[FAIL] {fp}: {exc}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
