#!/usr/bin/env python3
"""
Load Intan/RHD-style MATLAB v7.3 export and plot board_dig_in_data (TTL triggers).

Uses h5py (not scipy.io.loadmat). Default path finds Temp_260120_162205.mat under
this script's directory tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def default_mat_path() -> Path:
    here = Path(__file__).resolve().parent
    found = sorted(here.rglob("Temp_260120_162205.mat"), key=lambda p: len(p.parts))
    if found:
        return found[0]
    return here / "Temp_260120_162205.mat"


def decode_channel_names(f: h5py.File) -> list[str]:
    chg = f["board_dig_in_channels"]
    refs = np.array(chg["native_channel_name"], dtype=object)
    names: list[str] = []
    for i in range(refs.shape[0]):
        ref = refs[i, 0]
        ds = f[ref]
        arr = np.array(ds, dtype=np.uint16)
        s = "".join(chr(int(c)) for c in arr.flatten() if c)
        names.append(s or f"ch{i}")
    return names


def orient_dig_data(dig_arr: np.ndarray, t_len: int) -> tuple[np.ndarray, bool]:
    """
    Return board_dig_in_data in shape (n_samples, n_channels).
    """
    if dig_arr.ndim == 1:
        return dig_arr[:, None], False
    if dig_arr.shape[0] == t_len:
        return dig_arr, False
    if dig_arr.shape[1] == t_len:
        return dig_arr.T, True
    # Fall back to original orientation if no dimension matches t_dig length.
    return dig_arr, False


def event_indices(binary_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (rising_idx, falling_idx) where transitions occur.
    """
    s = (binary_signal > 0.5).astype(np.int8)
    d = np.diff(s, prepend=s[0])
    rising = np.where(d == 1)[0]
    falling = np.where(d == -1)[0]
    return rising, falling


def main() -> int:
    p = argparse.ArgumentParser(description="Plot board_dig_in_data from .mat (v7.3 HDF5).")
    p.add_argument(
        "mat",
        nargs="?",
        type=Path,
        default=None,
        help="Path to .mat file (default: discover Temp_260120_162205.mat near this script)",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        metavar="T",
        help="Plot only the first T seconds (default: 0 = full recording).",
    )
    p.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="After slicing time, use every N-th sample for plotting (default: 1).",
    )
    p.add_argument(
        "--max-plot-samples",
        type=int,
        default=400_000,
        metavar="N",
        help="Auto-increase decimation to keep plotted samples <= N (default: 400000).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, save figure to this path instead of showing interactively.",
    )
    args = p.parse_args()

    mat_path = args.mat.expanduser().resolve() if args.mat else default_mat_path()
    if not mat_path.is_file():
        print(f"Error: file not found: {mat_path}", file=sys.stderr)
        return 1

    with h5py.File(mat_path, "r") as f:
        dig_ds = f["board_dig_in_data"]
        t_ds = f["t_dig"]
        t_full = np.array(t_ds[:, 0], dtype=np.float64).ravel()
        dig_raw = np.array(dig_ds, dtype=np.float64)
        raw_shape = dig_raw.shape
        dig, transposed = orient_dig_data(dig_raw, t_full.shape[0])
        n_total = int(dig.shape[0])
        n_ch = int(dig.shape[1])
        sr_ds = f["frequency_parameters"]["board_dig_in_sample_rate"]
        sr = float(np.array(sr_ds).ravel()[0])

        if args.seconds and args.seconds > 0:
            n = min(n_total, max(1, int(args.seconds * sr)))
        else:
            n = n_total

        step = max(1, int(args.decimate))
        if args.max_plot_samples > 0:
            auto_step = int(np.ceil(n / args.max_plot_samples))
            step = max(step, auto_step, 1)
        sl = slice(0, n, step)

        t = t_full[sl]
        data = dig[sl, :]

        try:
            labels = decode_channel_names(f)
        except Exception:
            labels = [f"DIG {i + 1}" for i in range(n_ch)]

    # Print answers to key questions before plotting.
    duration_from_time = float(t_full[-1] - t_full[0]) if t_full.size > 1 else 0.0
    duration_from_sr = float(n_total / sr) if sr > 0 else float("nan")
    print(f"MAT file: {mat_path}")
    print(f"Sampling rate (board_dig_in): {sr:.6f} Hz")
    print(f"board_dig_in_data raw shape from file: {raw_shape}")
    if transposed:
        print("Interpreted orientation: transposed raw data to (samples, channels)")
    else:
        print("Interpreted orientation: using raw data as (samples, channels)")
    print(f"board_dig_in_data interpreted shape: (samples={n_total}, channels={n_ch})")
    print(
        f"Recording duration: {duration_from_time:.6f} s "
        f"(from t_dig), ~{duration_from_sr:.6f} s (samples/sample_rate)"
    )
    print(f"Plotted window: first {n / sr:.6f} s, effective decimation step: {step}")
    print()
    print("Interpretation:")
    print(
        "  board_dig_in_data is a sampled digital state matrix (0/1), not a direct list of timestamps."
    )
    print(
        "  Event timestamps are obtained from t_dig at transitions, e.g., rising edges 0->1 (trigger onsets)."
    )
    print()

    per_channel_events: list[tuple[np.ndarray, np.ndarray]] = []
    max_preview = 12
    for i in range(n_ch):
        rising_idx, falling_idx = event_indices(dig[:, i])
        per_channel_events.append((rising_idx, falling_idx))
        rising_ts = t_full[rising_idx] if rising_idx.size else np.array([], dtype=np.float64)
        falling_ts = t_full[falling_idx] if falling_idx.size else np.array([], dtype=np.float64)
        label = labels[i] if i < len(labels) else f"DIG {i + 1}"
        print(
            f"Channel {i + 1} ({label}): "
            f"{rising_idx.size} rising edges, {falling_idx.size} falling edges"
        )
        if rising_ts.size:
            preview = ", ".join(f"{x:.6f}" for x in rising_ts[:max_preview])
            suffix = " ..." if rising_ts.size > max_preview else ""
            print(f"  Rising-edge timestamps (s): {preview}{suffix}")
        else:
            print("  Rising-edge timestamps (s): none")
        if falling_ts.size:
            preview = ", ".join(f"{x:.6f}" for x in falling_ts[:max_preview])
            suffix = " ..." if falling_ts.size > max_preview else ""
            print(f"  Falling-edge timestamps (s): {preview}{suffix}")
        else:
            print("  Falling-edge timestamps (s): none")
        print()

    shown = float(t[-1] - t[0]) if t.size > 1 else 0.0
    width = float(np.clip(11.0 * max(shown, 1.0) / 60.0, 11.0, 80.0))
    fig, axes = plt.subplots(
        n_ch,
        1,
        sharex=True,
        figsize=(width, 2.0 * n_ch),
        constrained_layout=True,
    )
    if n_ch == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, data[:, i], color="C0", linewidth=0.8, drawstyle="steps-post")
        rising_idx, _ = per_channel_events[i]
        # Keep only plotted window and decimation so event markers match visible trace.
        in_window = rising_idx[rising_idx < n]
        visible_rising = in_window[::step] if step > 1 else in_window
        if visible_rising.size:
            ax.scatter(
                t_full[visible_rising],
                np.ones_like(visible_rising, dtype=np.float64),
                s=14,
                color="C3",
                marker="v",
                alpha=0.85,
                label="rising edge",
            )
        ax.set_ylabel(labels[i] if i < len(labels) else f"ch{i}", fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        if visible_rising.size:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"board_dig_in_data — {mat_path.name}\n"
        f"shown {shown:.3f} s, full {duration_from_time:.3f} s, step={step}",
        fontsize=10,
    )

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved {args.out}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
