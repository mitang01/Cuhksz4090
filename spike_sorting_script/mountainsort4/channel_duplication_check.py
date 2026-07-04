"""
channel_duplication_check.py
============================
Proof-oriented raw-data check for possible duplicated/repeated channels in
sub5/sub6 128-channel recordings.

For each dataset, this script compares channel pairs with fixed offsets:
  - (c, c+32)
  - (c, c+64)
  - (c, c+96)
when indices are valid.

For every pair it reports:
  - exact_equal_fraction: fraction of samples where values are exactly equal
  - corr: Pearson correlation
  - max_abs_diff: maximum absolute sample difference
  - slope: least-squares slope y ~ slope * x + intercept

This lets you distinguish:
  - exact duplicate (equal fraction ~= 1.0, max diff = 0)
  - scaled/shifted copy (corr ~= 1 but not exactly equal)
  - merely similar / unrelated channels

Usage:
  python channel_duplication_check.py
  python channel_duplication_check.py sub5 sub6
  python channel_duplication_check.py sub5 --window-sec 20 --start middle
"""

import argparse
from pathlib import Path

import numpy as np


SAMPLING_RATE = 30000.0
DTYPE = np.int16

SUBJECT_CONFIGS = {
    "sub5": {"num_channels": 128},
    "sub6": {"num_channels": 128},
}

DATASETS = [
    {"name": "bistable_sub5_session1", "subject": "sub5",
     "path": Path("/share/workspace2/tangmi/bistable_sub5_session1")},
    {"name": "bistable_sub5_session2", "subject": "sub5",
     "path": Path("/share/workspace2/tangmi/bistable_sub5_session2")},
    {"name": "bistable_sub5_session3", "subject": "sub5",
     "path": Path("/share/workspace2/tangmi/bistable_sub5_session3")},
    {"name": "bistable_sub5_session4", "subject": "sub5",
     "path": Path("/share/workspace2/tangmi/bistable_sub5_session4")},
    {"name": "bistable_sub5_session5", "subject": "sub5",
     "path": Path("/share/workspace2/tangmi/bistable_sub5_session5")},
    {"name": "bistable_sub5_session6", "subject": "sub5",
     "path": Path("/share/workspace2/tangmi/bistable_sub5_session6")},
    {"name": "bistable_sub6_session1", "subject": "sub6",
     "path": Path("/share/workspace2/tangmi/bistable_sub6_session1")},
    {"name": "bistable_sub6_session3", "subject": "sub6",
     "path": Path("/share/workspace2/tangmi/bistable_sub6_session3")},
    {"name": "bistable_sub6_session6", "subject": "sub6",
     "path": Path("/share/workspace2/tangmi/bistable_sub6_session6")},
]


def hr(ch="=", n=110):
    print(ch * n)


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_window(raw_path: Path, num_channels: int, start_sec: float, duration_sec: float):
    bytes_per_sample = np.dtype(DTYPE).itemsize
    size = raw_path.stat().st_size
    total_frames = size // (num_channels * bytes_per_sample)
    if total_frames <= 0:
        raise RuntimeError("No frames available with this dtype/num_channels.")
    total_duration_sec = total_frames / SAMPLING_RATE

    start_frame = int(start_sec * SAMPLING_RATE)
    if start_frame >= total_frames:
        start_frame = max(0, total_frames - int(duration_sec * SAMPLING_RATE))
    n_frames = int(duration_sec * SAMPLING_RATE)
    if start_frame + n_frames > total_frames:
        n_frames = total_frames - start_frame
    if n_frames < int(1.0 * SAMPLING_RATE):
        raise RuntimeError("Window too short after bounds check.")

    mm = np.memmap(raw_path, dtype=DTYPE, mode="r")
    start_flat = start_frame * num_channels
    end_flat = (start_frame + n_frames) * num_channels
    arr = np.asarray(mm[start_flat:end_flat]).reshape(n_frames, num_channels)
    return arr, total_duration_sec


def pair_metrics(x: np.ndarray, y: np.ndarray):
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)

    exact_equal_fraction = float(np.mean(x == y))
    diff = y64 - x64
    max_abs_diff = float(np.max(np.abs(diff)))
    mean_abs_diff = float(np.mean(np.abs(diff)))

    x_center = x64 - x64.mean()
    y_center = y64 - y64.mean()
    denom = np.sqrt((x_center * x_center).sum() * (y_center * y_center).sum())
    corr = float((x_center * y_center).sum() / denom) if denom > 0 else np.nan

    var_x = float((x_center * x_center).sum())
    if var_x > 0:
        slope = float((x_center * y_center).sum() / var_x)
    else:
        slope = np.nan
    intercept = float(y64.mean() - slope * x64.mean()) if np.isfinite(slope) else np.nan

    return {
        "equal_frac": exact_equal_fraction,
        "corr": corr,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "slope": slope,
        "intercept": intercept,
    }


def evaluate_offset(arr: np.ndarray, offset: int):
    n_ch = arr.shape[1]
    stats = []
    for c in range(0, n_ch - offset):
        m = pair_metrics(arr[:, c], arr[:, c + offset])
        stats.append((c, c + offset, m))
    return stats


def summarize_offset(stats):
    if not stats:
        return None
    equal_fracs = np.array([s[2]["equal_frac"] for s in stats], dtype=float)
    corrs = np.array([s[2]["corr"] for s in stats], dtype=float)
    max_abs_diffs = np.array([s[2]["max_abs_diff"] for s in stats], dtype=float)
    mean_abs_diffs = np.array([s[2]["mean_abs_diff"] for s in stats], dtype=float)
    slopes = np.array([s[2]["slope"] for s in stats], dtype=float)

    return {
        "n_pairs": len(stats),
        "equal_frac_median": float(np.nanmedian(equal_fracs)),
        "equal_frac_max": float(np.nanmax(equal_fracs)),
        "corr_median": float(np.nanmedian(corrs)),
        "corr_max": float(np.nanmax(corrs)),
        "max_abs_diff_median": float(np.nanmedian(max_abs_diffs)),
        "mean_abs_diff_median": float(np.nanmedian(mean_abs_diffs)),
        "slope_median": float(np.nanmedian(slopes)),
        "n_exact_pairs": int(np.sum(equal_fracs == 1.0)),
        "n_neardup_pairs": int(np.sum((equal_fracs >= 0.9999) | (corrs >= 0.9999))),
    }


def best_pairs(stats, topn=8):
    # Sort by strongest duplication evidence: exact fraction then correlation.
    sorted_stats = sorted(
        stats,
        key=lambda t: (t[2]["equal_frac"], t[2]["corr"]),
        reverse=True,
    )
    return sorted_stats[:topn]


def verdict_for_summary(summary):
    if summary is None:
        return "no_pairs"
    if summary["n_exact_pairs"] > 0:
        return "EXACT_DUPLICATES_PRESENT"
    if summary["equal_frac_max"] >= 0.9999 or summary["corr_max"] >= 0.9999:
        return "NEAR_DUPLICATES_PRESENT"
    if summary["corr_median"] >= 0.95:
        return "STRONGLY_REPETITIVE"
    return "NO_STRONG_DUPLICATION_EVIDENCE"


def print_offset_report(offset, stats):
    summary = summarize_offset(stats)
    verdict = verdict_for_summary(summary)
    print(f"Offset +{offset}: {verdict}")
    if summary is None:
        print("  no valid channel pairs")
        return summary, verdict
    print(
        "  "
        f"pairs={summary['n_pairs']}, exact_pairs={summary['n_exact_pairs']}, "
        f"neardup_pairs={summary['n_neardup_pairs']}"
    )
    print(
        "  "
        f"equal_frac median/max={summary['equal_frac_median']:.6f}/{summary['equal_frac_max']:.6f}, "
        f"corr median/max={summary['corr_median']:.6f}/{summary['corr_max']:.6f}"
    )
    print(
        "  "
        f"mean_abs_diff median={summary['mean_abs_diff_median']:.3f}, "
        f"max_abs_diff median={summary['max_abs_diff_median']:.3f}, "
        f"slope median={summary['slope_median']:.6f}"
    )
    print("  strongest pairs:")
    for c0, c1, m in best_pairs(stats, topn=8):
        print(
            "    "
            f"ch{c0:>3d}-ch{c1:>3d}: equal={m['equal_frac']:.6f}, corr={m['corr']:.6f}, "
            f"mean|diff|={m['mean_abs_diff']:.3f}, max|diff|={m['max_abs_diff']:.1f}, "
            f"slope={m['slope']:.6f}, intercept={m['intercept']:.3f}"
        )
    return summary, verdict


def analyze_dataset(ds, window_sec, start_mode):
    raw = ds["path"]
    if not raw.exists():
        print(f"[MISSING] {raw}")
        return None

    n_ch = SUBJECT_CONFIGS[ds["subject"]]["num_channels"]
    if start_mode == "middle":
        size = raw.stat().st_size
        n_frames = size // (n_ch * np.dtype(DTYPE).itemsize)
        start_sec = max(0.0, (n_frames / SAMPLING_RATE - window_sec) / 2.0)
    else:
        start_sec = 0.0

    arr, duration_sec = load_window(raw, num_channels=n_ch, start_sec=start_sec, duration_sec=window_sec)
    return {
        "dataset": ds["name"],
        "subject": ds["subject"],
        "path": raw,
        "size": raw.stat().st_size,
        "duration_sec": duration_sec,
        "window_sec": arr.shape[0] / SAMPLING_RATE,
        "arr": arr,
    }


def main():
    parser = argparse.ArgumentParser(description="Check exact/near duplicate channels for sub5/sub6 raw files.")
    parser.add_argument("subjects", nargs="*", default=[], help="Optional filter: sub5 sub6")
    parser.add_argument("--window-sec", type=float, default=10.0, help="Window length (seconds), default 10")
    parser.add_argument("--start", choices=("middle", "start"), default="middle", help="Window origin")
    args = parser.parse_args()

    subject_filter = set(args.subjects)
    if subject_filter:
        unknown = sorted(s for s in subject_filter if s not in SUBJECT_CONFIGS)
        if unknown:
            raise ValueError(f"Unknown subjects: {unknown}")

    hr()
    print("Channel duplication checker (proof-oriented)")
    print(f"dtype=int16, fs={SAMPLING_RATE:.0f} Hz, window={args.window_sec:.1f}s from {args.start}")
    if subject_filter:
        print(f"subjects={sorted(subject_filter)}")
    hr()

    summary_rows = []
    for ds in DATASETS:
        if subject_filter and ds["subject"] not in subject_filter:
            continue
        hr("-")
        print(f"Dataset: {ds['name']} ({ds['subject']})")
        print(f"Path   : {ds['path']}")
        try:
            info = analyze_dataset(ds, window_sec=args.window_sec, start_mode=args.start)
            if info is None:
                continue
            print(
                f"Size   : {human_bytes(info['size'])}, total_duration={info['duration_sec']/60:.2f} min, "
                f"window={info['window_sec']:.2f} s"
            )
            arr = info["arr"]
            for off in (32, 64, 96):
                stats = evaluate_offset(arr, off)
                s, verdict = print_offset_report(off, stats)
                if s is not None:
                    summary_rows.append(
                        {
                            "dataset": ds["name"],
                            "subject": ds["subject"],
                            "offset": off,
                            "verdict": verdict,
                            "equal_frac_max": s["equal_frac_max"],
                            "corr_max": s["corr_max"],
                            "n_exact_pairs": s["n_exact_pairs"],
                            "n_neardup_pairs": s["n_neardup_pairs"],
                        }
                    )
        except Exception as exc:
            print(f"ERROR  : {type(exc).__name__}: {exc}")

    hr()
    print("Summary")
    hr()
    print(f"{'dataset':<26} {'subj':<4} {'off':>4} {'verdict':>28} {'eq_max':>10} {'corr_max':>10} {'exact':>7} {'neardup':>8}")
    for r in summary_rows:
        print(
            f"{r['dataset']:<26} {r['subject']:<4} {r['offset']:>4d} "
            f"{r['verdict']:>28} {r['equal_frac_max']:>10.6f} {r['corr_max']:>10.6f} "
            f"{r['n_exact_pairs']:>7d} {r['n_neardup_pairs']:>8d}"
        )
    hr()
    print("Interpretation:")
    print("- EXACT_DUPLICATES_PRESENT => at least one channel pair is bit-identical in the window.")
    print("- NEAR_DUPLICATES_PRESENT  => almost identical or almost perfectly correlated pairs exist.")
    print("- STRONGLY_REPETITIVE      => broad strong correlation pattern, even if not exact clones.")
    hr()


if __name__ == "__main__":
    main()
