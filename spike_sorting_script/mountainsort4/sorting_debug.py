"""
sorting_debug.py
================
Diagnostic script to confirm where the bad sub5/sub6 spike-sorting output
comes from. It does NOT sort anything. It only checks the three hypotheses
raised in the scrutiny of `spikesort_mountainsort4_v2.py`:

  (1) File-size vs. expected bytes:
      Does the raw binary actually contain `num_channels` interleaved int16
      samples at 30 kHz, as `read_binary` assumes? A clean factor mismatch
      (e.g. file is 2x or 0.5x the expected size) reveals a wrong channel
      count or wrong dtype.

  (2) Raw-trace sanity:
      Load a few seconds with the script's assumed parameters and look at the
      int16 value range / std / saturation. A correctly decoded iEE trace has
      a quiet, mostly-small-amplitude distribution with rare large spikes.
      A scrambled trace looks like broadband noise with a wildly different
      amplitude distribution than the known-good sub4 file.

  (3) Cross-subject comparison + alternative-decode probe:
      Print sub4 (known good) vs sub5/sub6 stats side by side, then try a few
      alternative decodings (different num_channels, float32, channel-major
      time_axis=1) and report which one yields a sane duration / amplitude
      distribution for the sub5/sub6 files.

All results are printed to stdout in a readable, sectioned format.

Usage (on the cluster that hosts the /share/... paths):
    python sorting_debug.py
or limit to specific subjects:
    python sorting_debug.py sub5 sub6
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

# ---- Mirror the assumptions from spikesort_mountainsort4_v2.py ----------
SAMPLING_RATE = 30000.0
GAIN_TO_UV = 0.195
ASSUMED_DTYPE = "int16"
ASSUMED_BYTES_PER_SAMPLE = 2  # int16

DATASETS = [
    {"name": "bistable_sub4_session3", "subject": "sub4",
     "path": Path("/share/workspace3/ieeg/micro/word_boun_perce_v1/bistable_sub4/bistable_sub4_session3")},
    {"name": "bistable_sub4_session5", "subject": "sub4",
     "path": Path("/share/workspace3/ieeg/micro/word_boun_perce_v1/bistable_sub4/bistable_sub4_session5")},
    {"name": "bistable_sub5_session1", "subject": "sub5",
     "path": Path("/share/workspace2/tangmi/bistable_sub5_session1")},
    {"name": "bistable_sub5_session2", "subject": "sub5",
     "path": Path("/share/workspace2/tangmi/bistable_sub5_session2")},
    {"name": "bistable_sub6_session1", "subject": "sub6",
     "path": Path("/share/workspace2/tangmi/bistable_sub6_session1")},
    {"name": "bistable_sub6_session3", "subject": "sub6",
     "path": Path("/share/workspace2/tangmi/bistable_sub6_session3")},
]

SUBJECT_NUM_CHANNELS = {"sub4": 80, "sub5": 128, "sub6": 128}

# Alternative decodings to try for the bad subjects.
# Each entry: (label, num_channels, dtype, bytes_per_sample, time_axis)
ALT_DECODES = [
    ("assumed (script default)",  None,    "int16",   2, 0),
    ("num_channels=64 int16",     64,      "int16",   2, 0),
    ("num_channels=80 int16",     80,      "int16",   2, 0),
    ("num_channels=96 int16",     96,      "int16",   2, 0),
    ("num_channels=128 int16",    128,     "int16",   2, 0),
    ("num_channels=256 int16",    256,     "int16",   2, 0),
    ("num_channels=128 float32",  128,     "float32", 4, 0),
    ("num_channels=64 float32",   64,      "float32", 4, 0),
    ("num_channels=128 int16 ch-major(time_axis=1)", 128, "int16", 2, 1),
]


def hr(char="=", n=78):
    print(char * n)


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def resolve_raw_file(recording_path: Path):
    """Return (raw_file_path, is_dir). If a directory, look for a likely .dat."""
    if recording_path.is_file():
        return recording_path, False
    if recording_path.is_dir():
        # Common raw binary names used by Intan/OpenEphys/custom exporters.
        candidates = []
        for name in ("raw.dat", "data.dat", "recording.dat", "continuous.dat",
                     "eeg.dat", "lfp.dat", "signal.dat"):
            p = recording_path / name
            if p.is_file():
                candidates.append(p)
        # Fall back to any .dat / .bin in the folder.
        if not candidates:
            candidates = sorted(
                list(recording_path.glob("*.dat")) + list(recording_path.glob("*.bin"))
            )
        if candidates:
            return candidates[0], True
    return None, recording_path.is_dir()


def file_size_check(ds):
    """Hypothesis (1): does file size match assumed num_channels * int16?"""
    raw, was_dir = resolve_raw_file(ds["path"])
    subj = ds["subject"]
    nch = SUBJECT_NUM_CHANNELS[subj]
    assumed_bytes_per_frame = nch * ASSUMED_BYTES_PER_SAMPLE

    print(f"\nDataset : {ds['name']}  (subject={subj}, assumed num_channels={nch})")
    print(f"Path    : {ds['path']}  ({'dir' if was_dir else 'file'}"
          f"{f' -> resolved to {raw.name}' if was_dir and raw else ''})")
    if raw is None or not raw.exists():
        print("  [MISSING] raw binary not found; skipping.")
        return None

    size = raw.stat().st_size
    print(f"Raw file: {raw}")
    print(f"Size    : {size} bytes ({human_bytes(size)})")

    n_frames_assumed = size / assumed_bytes_per_frame
    dur_assumed_s = n_frames_assumed / SAMPLING_RATE
    dur_assumed_min = dur_assumed_s / 60
    print(f"Under ASSUMED decode ({nch} ch x int16 @ {SAMPLING_RATE:.0f} Hz):")
    print(f"  frames      = {n_frames_assumed:,.0f}")
    print(f"  duration    = {dur_assumed_min:.2f} min  ({dur_assumed_s:,.1f} s)")

    # Try a few alternative channel counts and report which gives a "sane"
    # duration (a few minutes to a few hours). This catches the case where the
    # file was actually written with a different channel count.
    print("  Duration under alternative channel counts (int16, 30 kHz):")
    for alt_nch in (64, 80, 96, 128, 256):
        if alt_nch == nch:
            continue
        alt_frames = size / (alt_nch * ASSUMED_BYTES_PER_SAMPLE)
        alt_min = (alt_frames / SAMPLING_RATE) / 60
        marker = "  <-- plausible" if 1.0 <= alt_min <= 600.0 else ""
        print(f"    {alt_nch:3d} ch -> {alt_min:10.2f} min{marker}")

    # Ratio vs assumed size for a "nice" target duration (helps spot 2x / 0.5x).
    return {
        "raw": raw,
        "size": size,
        "nch_assumed": nch,
        "dur_assumed_min": dur_assumed_min,
    }


def raw_trace_sanity(ds, info, n_seconds=2.0):
    """Hypothesis (2): load n_seconds with assumed params, inspect distribution."""
    if info is None:
        return
    raw = info["raw"]
    nch = info["nch_assumed"]
    frames_to_read = int(n_seconds * SAMPLING_RATE)
    bytes_per_frame = nch * ASSUMED_BYTES_PER_SAMPLE
    total_bytes_needed = frames_to_read * bytes_per_frame
    if total_bytes_needed > info["size"]:
        frames_to_read = info["size"] // bytes_per_frame
        print(f"  [note] file smaller than {n_seconds}s; reading {frames_to_read} frames instead")

    # Read a contiguous chunk from the middle of the file (more representative
    # than the very start, which sometimes has artifacts).
    start_frame = (info["size"] // bytes_per_frame) // 2
    offset_bytes = int(start_frame * bytes_per_frame)
    with open(raw, "rb") as f:
        f.seek(offset_bytes)
        buf = f.read(int(frames_to_read * bytes_per_frame))

    arr = np.frombuffer(buf, dtype="<i2")
    if arr.size == 0:
        print("  [WARN] no bytes read; skipping trace sanity.")
        return
    n_frames = arr.size // nch
    arr = arr[: n_frames * nch].reshape(n_frames, nch)

    # int16 range is [-32768, 32767]. A sane iEE trace usually stays well within
    # this range (no saturation). Report per-channel and global stats.
    global_min = int(arr.min())
    global_max = int(arr.max())
    global_mean = float(arr.mean())
    global_std = float(arr.std())
    global_p99 = float(np.percentile(np.abs(arr), 99))
    sat_hi = int((arr >= 32000).sum())
    sat_lo = int((arr <= -32000).sum())
    sat_frac = (sat_hi + sat_lo) / arr.size

    print(f"Raw trace sanity ({n_frames} frames = {n_frames/SAMPLING_RATE:.2f}s, "
          f"start frame {start_frame:,}):")
    print(f"  int16 min/max       : {global_min} / {global_max}")
    print(f"  int16 mean / std    : {global_mean:.2f} / {global_std:.2f}")
    print(f"  |value| 99th pct    : {global_p99:.1f}  (~{global_p99*GAIN_TO_UV:.1f} uV)")
    print(f"  saturation fraction : {sat_frac*100:.4f}%  "
          f"(hi={sat_hi}, lo={sat_lo})")
    # Per-channel std spread: in a correctly decoded recording, adjacent
    # channels usually have comparable std. In a scrambled/time-axis-wrong
    # recording, std can be near-identical across channels or wildly bimodal.
    per_ch_std = arr.std(axis=0)
    print(f"  per-channel std     : min={per_ch_std.min():.1f} "
          f"med={np.median(per_ch_std):.1f} max={per_ch_std.max():.1f}")

    # A simple "looks like real signal" heuristic: real neural traces have a
    # heavy-tailed but bounded distribution; scrambled byte reshuffles tend to
    # be either near-uniform or have pathological saturation. Flag it.
    flags = []
    if global_std < 50 or global_std > 8000:
        flags.append("std_outside_typical_range")
    if sat_frac > 0.001:
        flags.append("high_saturation")
    if global_min <= -32767 or global_max >= 32767:
        flags.append("hits_int16_rails")
    verdict = "LOOKS_OK" if not flags else "LOOKS_SUSPICIOUS(" + ",".join(flags) + ")"
    print(f"  verdict             : {verdict}")


def alternative_decode_probe(ds, info):
    """Hypothesis (3): try alternative decodings, report duration + std stats."""
    if info is None:
        return
    raw = info["raw"]
    size = info["size"]
    subj = ds["subject"]
    nch_assumed = info["nch_assumed"]

    print(f"\nAlternative-decode probe for {ds['name']} (file size {human_bytes(size)}):")
    print(f"  {'decode':<46}{'dur(min)':>10}{'std':>10}{'99pct':>10}  verdict")
    print(f"  {'-'*46}{'-'*10}{'-'*10}{'-'*10}  {'-'*12}")

    for label, alt_nch, dtype, bps, time_axis in ALT_DECODES:
        if alt_nch is None:
            alt_nch = nch_assumed
        bytes_per_frame = alt_nch * bps
        if bytes_per_frame == 0:
            continue
        n_frames = size / bytes_per_frame
        dur_min = (n_frames / SAMPLING_RATE) / 60
        # Read a small chunk for stats.
        frames_to_read = min(int(2 * SAMPLING_RATE), int(n_frames))
        if frames_to_read <= 0:
            print(f"  {label:<46}{'n/a':>10}{'n/a':>10}{'n/a':>10}  no_data")
            continue
        offset = int((n_frames // 2) * bytes_per_frame)
        if offset + frames_to_read * bytes_per_frame > size:
            offset = max(0, size - frames_to_read * bytes_per_frame)
        try:
            with open(raw, "rb") as f:
                f.seek(offset)
                buf = f.read(int(frames_to_read * bytes_per_frame))
            np_dtype = "<f4" if dtype == "float32" else "<i2"
            flat = np.frombuffer(buf, dtype=np_dtype)
            n_actual = flat.size // alt_nch
            flat = flat[: n_actual * alt_nch]
            if time_axis == 1:
                arr = flat.reshape(alt_nch, n_actual).T  # channel-major
            else:
                arr = flat.reshape(n_actual, alt_nch)
            std = float(arr.std())
            p99 = float(np.percentile(np.abs(arr), 99))
            dur_plausible = 1.0 <= dur_min <= 600.0
            std_plausible = 50.0 <= std <= 8000.0
            if dtype == "float32":
                std_plausible = std_plausible or (0.5 <= std <= 200.0)
            verdict = "plausible" if (dur_plausible and std_plausible) else (
                "dur_ok" if dur_plausible else ("std_ok" if std_plausible else "bad"))
            print(f"  {label:<46}{dur_min:>10.2f}{std:>10.1f}{p99:>10.1f}  {verdict}")
        except Exception as e:
            print(f"  {label:<46}{'err':>10}{'':>10}{'':>10}  {type(e).__name__}: {e}")


def main():
    only_subjects = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    hr()
    print("sorting_debug.py -- raw-binary decode diagnostic")
    print(f"Assumed format for all subjects: dtype={ASSUMED_DTYPE}, "
          f"fs={SAMPLING_RATE:.0f} Hz, gain={GAIN_TO_UV} uV/bit, time_axis=0")
    if only_subjects:
        print(f"Limiting to subjects: {sorted(only_subjects)}")
    hr()

    results = []
    for ds in DATASETS:
        if only_subjects and ds["subject"] not in only_subjects:
            continue

        hr("-")
        print("CHECK 1/3  -- file size vs. assumed decode")
        info = file_size_check(ds)

        hr("-")
        print("CHECK 2/3  -- raw trace sanity under assumed decode")
        raw_trace_sanity(ds, info, n_seconds=2.0)

        hr("-")
        print("CHECK 3/3  -- alternative decode probe")
        alternative_decode_probe(ds, info)

        results.append((ds, info))

    hr()
    print("SUMMARY -- assumed-decode duration per dataset")
    hr()
    print(f"  {'dataset':<28}{'subject':>9}{'nch':>6}{'size':>14}{'dur(min)':>12}  verdict")
    print(f"  {'-'*28}{'-'*9}{'-'*6}{'-'*14}{'-'*12}  {'-'*12}")
    for ds, info in results:
        if info is None:
            print(f"  {ds['name']:<28}{ds['subject']:>9}{'?':>6}{'missing':>14}{'?':>12}  MISSING")
            continue
        dur = info["dur_assumed_min"]
        dur_plausible = 1.0 <= dur <= 600.0
        verdict = "dur_ok" if dur_plausible else "DURATION_OFF"
        print(f"  {ds['name']:<28}{ds['subject']:>9}{info['nch_assumed']:>6}"
              f"{human_bytes(info['size']):>14}{dur:>12.2f}  {verdict}")

    hr()
    print("How to read this output:")
    print("  * If sub4 shows 'dur_ok' + 'LOOKS_OK' but sub5/sub6 show")
    print("    'DURATION_OFF' or 'LOOKS_SUSPICIOUS' under the assumed decode,")
    print("    the bug is the read_binary parameter set (num_channels / dtype /")
    print("    time_axis / fs) being wrong for those files -- exactly the")
    print("    hypothesis from the script scrutiny.")
    print("  * The 'Alternative-decode probe' table shows which decode (channel")
    print("    count, dtype, axis) gives a plausible duration AND a plausible")
    print("    amplitude std for the sub5/sub6 files. The matching row tells you")
    print("    the correct read_binary parameters to use.")
    print("  * A clean 2x / 0.5x size ratio vs sub4 is a strong sign of a wrong")
    print("    channel count or wrong dtype (int16 vs float32).")
    hr()


if __name__ == "__main__":
    main()
