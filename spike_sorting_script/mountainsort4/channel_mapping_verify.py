"""
channel_mapping_verify.py
=========================
Quick verifier for the channel-mapping assumption used by
`spikesort_mountainsort4_v2.py`.

Goal:
  Check whether sub5/sub6 really have meaningful neural activity concentrated in
  channels 64:128 (as the sorting script assumes), or whether channels 0:64 also
  look active / even stronger.

What it prints per dataset:
  1) Basic file/decode info (num_channels, duration estimate)
  2) Per-half activity summary for channels [0:half) vs [half:num_channels)
     - median RMS_uV
     - median robust sigma_uV (MAD-based)
     - median threshold crossing rate (events/s using 5 * robust sigma)
  3) Top-N channels by RMS and by crossing rate
  4) A plain verdict:
       - "second_half_dominant" (supports mapping 64:128)
       - "first_half_dominant"  (mapping likely wrong)
       - "both_halves_active"   (mapping ambiguous; needs electrode map check)

Usage:
  python channel_mapping_verify.py
  python channel_mapping_verify.py sub5 sub6
"""

import argparse
from pathlib import Path

import numpy as np


SAMPLING_RATE = 30000.0
GAIN_TO_UV = 0.195
DTYPE = np.int16

SUBJECT_CONFIGS = {
    "sub4": {"num_channels": 80},
    "sub5": {"num_channels": 128},
    "sub6": {"num_channels": 128},
}

DATASETS = [
    {"name": "bistable_sub4_session3", "subject": "sub4",
     "path": Path("/share/workspace3/ieeg/micro/word_boun_perce_v1/bistable_sub4/bistable_sub4_session3")},
    {"name": "bistable_sub4_session5", "subject": "sub4",
     "path": Path("/share/workspace3/ieeg/micro/word_boun_perce_v1/bistable_sub4/bistable_sub4_session5")},
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


def hr(ch="=", n=90):
    print(ch * n)


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def robust_sigma(x):
    """MAD-based sigma estimate; x shape: [samples, channels]."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    return 1.4826 * mad


def crossing_rate(x_uv, sigma_uv, fs, thr_mult=5.0):
    """
    Simple event proxy: count rising threshold crossings of |signal|.
    x_uv shape [samples, channels], sigma_uv shape [channels].
    """
    thr = np.maximum(thr_mult * sigma_uv, 1e-6)
    above = np.abs(x_uv) > thr[None, :]
    rising = above[1:, :] & (~above[:-1, :])
    return rising.sum(axis=0) / (x_uv.shape[0] / fs)


def load_window(raw_path, num_channels, start_sec, duration_sec):
    bytes_per_sample = np.dtype(DTYPE).itemsize
    file_size = raw_path.stat().st_size
    total_frames = file_size // (num_channels * bytes_per_sample)
    total_duration_sec = total_frames / SAMPLING_RATE
    if total_frames <= 0:
        raise RuntimeError("No frames found with given num_channels/dtype.")

    start_frame = int(start_sec * SAMPLING_RATE)
    if start_frame >= total_frames:
        start_frame = max(0, total_frames - int(duration_sec * SAMPLING_RATE))
    n_frames = int(duration_sec * SAMPLING_RATE)
    if start_frame + n_frames > total_frames:
        n_frames = total_frames - start_frame
    if n_frames < int(0.5 * SAMPLING_RATE):
        raise RuntimeError("Recording too short for robust window stats.")

    mm = np.memmap(raw_path, dtype=DTYPE, mode="r")
    flat_start = start_frame * num_channels
    flat_end = (start_frame + n_frames) * num_channels
    seg = np.asarray(mm[flat_start:flat_end]).reshape(n_frames, num_channels).astype(np.float32)
    return seg, total_frames, total_duration_sec


def summarize_half(values, start, end):
    v = values[start:end]
    if v.size == 0:
        return float("nan")
    return float(np.median(v))


def top_channels(metric, topn=10):
    order = np.argsort(metric)[::-1][:topn]
    return [(int(i), float(metric[i])) for i in order]


def verdict_from_halves(second_half_rms, first_half_rms, second_half_rate, first_half_rate):
    rms_ratio = second_half_rms / max(first_half_rms, 1e-9)
    rate_ratio = second_half_rate / max(first_half_rate, 1e-9)

    if rms_ratio >= 1.5 and rate_ratio >= 1.5:
        return "second_half_dominant"
    if rms_ratio <= (1 / 1.5) and rate_ratio <= (1 / 1.5):
        return "first_half_dominant"
    return "both_halves_active"


def analyze_dataset(ds, window_sec, start_mode):
    subject = ds["subject"]
    num_channels = SUBJECT_CONFIGS[subject]["num_channels"]
    raw_path = ds["path"]
    if not raw_path.exists():
        print(f"[MISSING] {raw_path}")
        return None

    if start_mode == "middle":
        file_size = raw_path.stat().st_size
        frames = file_size // (num_channels * np.dtype(DTYPE).itemsize)
        dur = frames / SAMPLING_RATE
        start_sec = max(0.0, (dur - window_sec) / 2.0)
    else:
        start_sec = 0.0

    x_int16, total_frames, total_duration_sec = load_window(
        raw_path=raw_path,
        num_channels=num_channels,
        start_sec=start_sec,
        duration_sec=window_sec,
    )
    x_uv = x_int16 * GAIN_TO_UV

    rms_uv = np.sqrt(np.mean(x_uv * x_uv, axis=0))
    sigma_uv = robust_sigma(x_uv)
    rate_hz = crossing_rate(x_uv=x_uv, sigma_uv=sigma_uv, fs=SAMPLING_RATE, thr_mult=5.0)

    half = num_channels // 2
    first_half_rms = summarize_half(rms_uv, 0, half)
    second_half_rms = summarize_half(rms_uv, half, num_channels)
    first_half_sigma = summarize_half(sigma_uv, 0, half)
    second_half_sigma = summarize_half(sigma_uv, half, num_channels)
    first_half_rate = summarize_half(rate_hz, 0, half)
    second_half_rate = summarize_half(rate_hz, half, num_channels)

    verdict = verdict_from_halves(
        second_half_rms=second_half_rms,
        first_half_rms=first_half_rms,
        second_half_rate=second_half_rate,
        first_half_rate=first_half_rate,
    )

    return {
        "dataset": ds["name"],
        "subject": subject,
        "path": raw_path,
        "num_channels": num_channels,
        "total_frames": total_frames,
        "total_duration_sec": total_duration_sec,
        "window_frames": x_uv.shape[0],
        "window_sec": x_uv.shape[0] / SAMPLING_RATE,
        "first_half_rms": first_half_rms,
        "second_half_rms": second_half_rms,
        "first_half_sigma": first_half_sigma,
        "second_half_sigma": second_half_sigma,
        "first_half_rate": first_half_rate,
        "second_half_rate": second_half_rate,
        "rms_ratio_2nd_over_1st": second_half_rms / max(first_half_rms, 1e-9),
        "rate_ratio_2nd_over_1st": second_half_rate / max(first_half_rate, 1e-9),
        "verdict": verdict,
        "top_rms": top_channels(rms_uv, topn=10),
        "top_rate": top_channels(rate_hz, topn=10),
    }


def print_result(r):
    hr("-")
    print(f"Dataset : {r['dataset']}  (subject={r['subject']}, nch={r['num_channels']})")
    print(f"Path    : {r['path']}")
    print(
        f"Size/Duration estimate: {human_bytes(r['path'].stat().st_size)} "
        f"/ {r['total_duration_sec'] / 60:.2f} min"
    )
    print(f"Window  : {r['window_sec']:.2f} s ({r['window_frames']} frames)")
    print("")
    print("Half-wise medians")
    print(f"  ch[0:{r['num_channels']//2}]   RMS_uV={r['first_half_rms']:.2f}  "
          f"sigma_uV={r['first_half_sigma']:.2f}  rate(>5sigma)={r['first_half_rate']:.2f}/s")
    print(f"  ch[{r['num_channels']//2}:{r['num_channels']}] RMS_uV={r['second_half_rms']:.2f}  "
          f"sigma_uV={r['second_half_sigma']:.2f}  rate(>5sigma)={r['second_half_rate']:.2f}/s")
    print(
        f"  ratios (2nd/1st): RMS={r['rms_ratio_2nd_over_1st']:.3f}, "
        f"rate={r['rate_ratio_2nd_over_1st']:.3f}"
    )
    print(f"Verdict : {r['verdict']}")
    print("")
    print("Top channels by RMS_uV (0-based channel index):")
    print("  " + ", ".join(f"ch{ch}:{val:.1f}" for ch, val in r["top_rms"]))
    print("Top channels by threshold crossing rate (/s):")
    print("  " + ", ".join(f"ch{ch}:{val:.1f}" for ch, val in r["top_rate"]))


def main():
    parser = argparse.ArgumentParser(description="Verify channel-mapping assumptions for sorting.")
    parser.add_argument(
        "subjects",
        nargs="*",
        default=[],
        help="Optional subject filter, e.g. sub5 sub6",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=10.0,
        help="Analysis window length in seconds (default: 10).",
    )
    parser.add_argument(
        "--start",
        choices=("middle", "start"),
        default="middle",
        help="Pick analysis window from middle or start of file (default: middle).",
    )
    args = parser.parse_args()

    subject_filter = set(args.subjects)
    if subject_filter:
        unknown = sorted(s for s in subject_filter if s not in SUBJECT_CONFIGS)
        if unknown:
            raise ValueError(f"Unknown subjects: {unknown}")

    hr()
    print("Channel mapping verifier")
    print(f"Decode assumption: dtype=int16, fs={SAMPLING_RATE:.0f} Hz, gain={GAIN_TO_UV} uV/bit")
    print(f"Window: {args.window_sec}s from {args.start}")
    if subject_filter:
        print(f"Subjects: {sorted(subject_filter)}")
    hr()

    all_results = []
    for ds in DATASETS:
        if subject_filter and ds["subject"] not in subject_filter:
            continue
        try:
            result = analyze_dataset(ds, window_sec=args.window_sec, start_mode=args.start)
            if result is not None:
                print_result(result)
                all_results.append(result)
        except Exception as exc:
            hr("-")
            print(f"Dataset : {ds['name']} ({ds['subject']})")
            print(f"ERROR   : {type(exc).__name__}: {exc}")

    hr()
    print("Summary table")
    hr()
    print(f"{'dataset':<28} {'subj':<4} {'RMS2/1':>8} {'Rate2/1':>9} {'verdict':>22}")
    for r in all_results:
        print(
            f"{r['dataset']:<28} {r['subject']:<4} "
            f"{r['rms_ratio_2nd_over_1st']:>8.3f} {r['rate_ratio_2nd_over_1st']:>9.3f} "
            f"{r['verdict']:>22}"
        )
    hr()
    print("Interpretation")
    print("- If sub5/sub6 show 'first_half_dominant', mapping [64:128] is likely wrong.")
    print("- If sub5/sub6 show 'second_half_dominant', mapping [64:128] is supported.")
    print("- If 'both_halves_active', mapping cannot be proven from amplitude alone;")
    print("  you must validate channel labels against acquisition metadata.")
    hr()


if __name__ == "__main__":
    main()
