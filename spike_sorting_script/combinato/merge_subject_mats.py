#!/usr/bin/env python3
"""Merge story_listen_v2 Intan .mat chunks into one file per subject.

Output (only when every chunk in that subject matches):
  /share/workspace3/ieeg/micro/story_listen_v2/sub-007/sub-007.mat

Chunks are concatenated in filename order along the time axis.
amplifier_data and, when present, board_dig_in_data are both written.
The merged file is HDF5. Read it with h5py, the same way as the
Intan v7.3 chunks.

If amplifier channel counts, digital channel counts, or channel names
disagree inside one subject, that subject is skipped and the mismatch
is printed. Other subjects are still processed.
"""

from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/share/workspace3/ieeg/micro/story_listen_v2")
SUBJECTS = [
    ("sub-007", "ses-01"),
    ("sub-008", "ses-01"),
    ("sub-010", "ses-01"),
    ("sub-011", "ses-02"),
]
BLOCK = 200000


def channel_names(h5, group_key):
    if group_key not in h5:
        return []
    group = h5[group_key]
    key = None
    if "native_channel_name" in group:
        key = "native_channel_name"
    elif "custom_channel_name" in group:
        key = "custom_channel_name"
    if key is None:
        return []
    names = []
    for ref in np.asarray(group[key], dtype=object).ravel():
        arr = np.asarray(h5[ref]).ravel()
        if arr.dtype.kind in {"U", "S"}:
            names.append("".join(str(x) for x in arr).strip())
        elif np.issubdtype(arr.dtype, np.integer):
            names.append("".join(chr(int(c)) for c in arr if 0 < int(c) < 256).strip())
        else:
            names.append("")
    return names


def time_axis(shape, n_names):
    """Return (n_channels, n_samples, time_axis). time_axis is 0 or 1."""
    if len(shape) != 2:
        raise ValueError(f"expected a 2D array, got shape {shape}")
    if n_names and shape[0] == n_names:
        return n_names, shape[1], 1
    if n_names and shape[1] == n_names:
        return n_names, shape[0], 0
    if shape[0] < shape[1]:
        return shape[0], shape[1], 1
    if shape[1] < shape[0]:
        return shape[1], shape[0], 0
    raise ValueError(f"cannot tell channels from samples in shape {shape}")


def inspect_file(path):
    info = {"path": path}
    if not h5py.is_hdf5(str(path)):
        raise ValueError(f"{path.name} is not an HDF5/v7.3 mat file")
    with h5py.File(path, "r") as h5:
        if "amplifier_data" not in h5:
            raise KeyError(f"{path.name} has no amplifier_data")
        amp_shape = tuple(int(x) for x in h5["amplifier_data"].shape)
        amp_names = channel_names(h5, "amplifier_channels")
        n_amp, n_amp_time, amp_axis = time_axis(amp_shape, len(amp_names))
        info.update(
            amp_shape=amp_shape,
            amp_names=amp_names,
            n_amp=n_amp,
            n_amp_time=n_amp_time,
            amp_time_axis=amp_axis,
        )
        if "board_dig_in_data" not in h5:
            info.update(dig_shape=None, dig_names=[], n_dig=None, n_dig_time=None, dig_time_axis=None)
            return info
        dig_shape = tuple(int(x) for x in h5["board_dig_in_data"].shape)
        dig_names = channel_names(h5, "board_dig_in_channels")
        n_dig, n_dig_time, dig_axis = time_axis(dig_shape, len(dig_names))
        info.update(
            dig_shape=dig_shape,
            dig_names=dig_names,
            n_dig=n_dig,
            n_dig_time=n_dig_time,
            dig_time_axis=dig_axis,
        )
    return info


def usable(names):
    return bool(names) and all(names)


def mismatch_lines(files_info):
    detail = []
    reasons = []
    for info in files_info:
        dig = "none" if info["n_dig"] is None else str(info["n_dig"])
        detail.append(
            f"  {info['path'].name}: amplifier_channels={info['n_amp']} "
            f"amplifier_samples={info['n_amp_time']} "
            f"dig_channels={dig} shape={info['amp_shape']}"
        )

    n_amp = {info["n_amp"] for info in files_info}
    if len(n_amp) != 1:
        reasons.append(f"amplifier channel counts differ: {sorted(n_amp)}")
    if len({info["amp_time_axis"] for info in files_info}) != 1:
        reasons.append("amplifier array orientation differs across files")

    amp_name_lists = [tuple(info["amp_names"]) for info in files_info]
    if any(usable(names) for names in amp_name_lists):
        if not all(usable(names) for names in amp_name_lists) or len(set(amp_name_lists)) != 1:
            reasons.append("amplifier channel names differ or are missing in some files")

    has_dig = [info["n_dig"] is not None for info in files_info]
    if any(has_dig) and not all(has_dig):
        reasons.append("board_dig_in_data is missing from some files")
    if all(has_dig):
        n_dig = {info["n_dig"] for info in files_info}
        if len(n_dig) != 1:
            reasons.append(f"digital channel counts differ: {sorted(n_dig)}")
        if len({info["dig_time_axis"] for info in files_info}) != 1:
            reasons.append("digital array orientation differs across files")
        for info in files_info:
            if info["n_dig_time"] != info["n_amp_time"]:
                reasons.append(
                    f"{info['path'].name}: amplifier samples {info['n_amp_time']} "
                    f"!= digital samples {info['n_dig_time']}"
                )
        dig_name_lists = [tuple(info["dig_names"]) for info in files_info]
        if any(usable(names) for names in dig_name_lists):
            if not all(usable(names) for names in dig_name_lists) or len(set(dig_name_lists)) != 1:
                reasons.append("digital channel names differ or are missing in some files")
    return detail, reasons


def copy_blocks(src_path, key, time_axis, out_ds, start):
    written = start
    with h5py.File(src_path, "r") as src:
        data = src[key]
        n = data.shape[time_axis]
        pos = 0
        while pos < n:
            take = min(BLOCK, n - pos)
            if time_axis == 1:
                out_ds[:, written:written + take] = data[:, pos:pos + take]
            else:
                out_ds[written:written + take, :] = data[pos:pos + take, :]
            pos += take
            written += take
    return written


def write_merged(files_info, out_path):
    first = files_info[0]
    amp_axis = first["amp_time_axis"]
    n_amp = first["n_amp"]
    n_time = sum(info["n_amp_time"] for info in files_info)
    amp_shape = (n_amp, n_time) if amp_axis == 1 else (n_time, n_amp)

    tmp = out_path.with_suffix(".mat.partial")
    if tmp.exists():
        tmp.unlink()
    try:
        with h5py.File(files_info[0]["path"], "r") as src:
            amp_dtype = src["amplifier_data"].dtype
            dig_dtype = src["board_dig_in_data"].dtype if first["n_dig"] is not None else None
        with h5py.File(tmp, "w") as out:
            amp = out.create_dataset("amplifier_data", shape=amp_shape, dtype=amp_dtype)
            written = 0
            for info in files_info:
                written = copy_blocks(info["path"], "amplifier_data", amp_axis, amp, written)
            if first["n_dig"] is not None:
                dig_axis = first["dig_time_axis"]
                n_dig = first["n_dig"]
                dig_shape = (n_dig, n_time) if dig_axis == 1 else (n_time, n_dig)
                dig = out.create_dataset("board_dig_in_data", shape=dig_shape, dtype=dig_dtype)
                written_dig = 0
                for info in files_info:
                    written_dig = copy_blocks(
                        info["path"], "board_dig_in_data", dig_axis, dig, written_dig
                    )
            if usable(first["amp_names"]):
                labels = np.array(first["amp_names"], dtype=h5py.string_dtype(encoding="utf-8"))
                out.create_dataset("amplifier_channel_names", data=labels)
            names = np.array(
                [info["path"].name for info in files_info],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            out.create_dataset("source_files", data=names)
        tmp.replace(out_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return n_amp, n_time


def merge_subject(subject, session, root=ROOT):
    session_dir = root / subject / session
    out_path = root / subject / f"{subject}.mat"
    print(f"\n{subject}")
    if not session_dir.is_dir():
        print(f"[STOP] {subject}: session folder not found: {session_dir}")
        return False
    mats = sorted(session_dir.glob("Temp_*.mat"))
    if not mats:
        print(f"[STOP] {subject}: no Temp_*.mat files in {session_dir}")
        return False
    try:
        files_info = [inspect_file(path) for path in mats]
    except Exception as exc:
        print(f"[STOP] {subject}: could not read chunks: {exc}")
        return False

    detail, reasons = mismatch_lines(files_info)
    for line in detail:
        print(line)
    if reasons:
        print(f"[STOP] {subject}: not merged.")
        for line in reasons:
            print(f"  {line}")
        return False
    if not any(usable(info["amp_names"]) for info in files_info):
        print("  channel names were not found; compared channel counts only")

    n_amp, n_time = write_merged(files_info, out_path)
    print(f"[WROTE] {out_path}")
    print(f"  amplifier_channels={n_amp} samples={n_time} files={len(mats)}")
    return True


def main():
    ok = True
    for subject, session in SUBJECTS:
        if not merge_subject(subject, session):
            ok = False
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
