#!/usr/bin/env python3
"""Merge word_boun_perce_v2 Intan .mat chunks into one file per subject.

Only the microwire channels listed in REGIONS are kept. Channel numbers
are 1-based and match Intan names such as A-017. Output, when the chunks
match and every requested channel is present:

  /share/workspace3/ieeg/micro/word_boun_perce_v2/sub-007/sub-007.mat

Chunks are concatenated in filename order along the time axis.
amplifier_data holds the kept channels. board_dig_in_data is copied in
full when every chunk has it. channel_number and brain_region say which
row is which. The merged file is HDF5. Read it with h5py.

Raw folders:

  /share/workspace3/ieeg/micro/word_boun_perce_v2/sub-007/ses-01
  /share/workspace3/ieeg/micro/word_boun_perce_v2/sub-008/ses-01
  /share/workspace3/ieeg/micro/word_boun_perce_v2/sub-009/ses-01
  /share/workspace3/ieeg/micro/word_boun_perce_v2/sub-010/ses-01
  /share/workspace3/ieeg/micro/word_boun_perce_v2/sub-011/ses-02
  /share/workspace3/ieeg/micro/word_boun_perce_v2/sub-012

sub-008 has one raw file, so nothing is concatenated. A raw export is
rewritten into the merged layout. A file that already has that layout is
moved to sub-008/sub-008.mat. Raw chunks for the other subjects stay where
they are.

sub-012 keeps its raw files directly in the subject folder. There is no
session directory.

Sorting results are not written here. combinato_step1.py writes those
under /share/home/mitan/combinato_result/word_boun.

If amplifier channel counts, digital channel counts, or channel names
disagree inside one subject, or a requested channel is missing, that
subject is skipped and the reason is printed. Other subjects continue.
"""

import shutil
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/share/workspace3/ieeg/micro/word_boun_perce_v2")
# Session folder under the subject. None means the raw files are in the
# subject folder itself.
SUBJECTS = [
    ("sub-007", "ses-01"),
    ("sub-008", "ses-01"),
    ("sub-009", "ses-01"),
    ("sub-010", "ses-01"),
    ("sub-011", "ses-02"),
    ("sub-012", None),
]
# One raw .mat. Do not concatenate. Still write <subject>/<subject>.mat.
SINGLE_FILE_SUBJECTS = {"sub-008"}
# Inclusive 1-based channel numbers.
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
    "sub-009": [
        (1, 16, "Amygdala"),
        (17, 32, "VmPFC"),
        (33, 48, "HG"),
    ],
    "sub-010": [
        (17, 32, "Amygdala"),
        (33, 48, "VmPFC"),
    ],
    "sub-011": [
        (17, 32, "HG"),
        (33, 48, "ATL"),
    ],
    "sub-012": [
        (1, 16, "ATL"),
        (17, 32, "HG"),
        (33, 48, "Amygdala"),
        # Same VmPFC spelling as the other subjects.
        (49, 64, "VmPFC"),
    ],
}
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


def name_number(name):
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def wanted(subject):
    numbers = []
    regions = []
    for start, end, region in REGIONS[subject]:
        for number in range(start, end + 1):
            numbers.append(number)
            regions.append(region)
    return numbers, regions


def channel_indices(info, numbers):
    """Return (indices, how). how is 'name', 'position', or an error string."""
    names = info["amp_names"]
    if usable(names):
        by_number = {}
        for i, name in enumerate(names):
            number = name_number(name)
            if number is None:
                return None, f"could not read a channel number from name {name!r}"
            if number in by_number:
                return None, f"channel number {number} appears twice ({name})"
            by_number[number] = i
        missing = [n for n in numbers if n not in by_number]
        if missing:
            return None, f"requested channels not in file: {missing}"
        return [by_number[n] for n in numbers], "name"
    missing = [n for n in numbers if n < 1 or n > info["n_amp"]]
    if missing:
        return None, f"requested channels outside 1..{info['n_amp']}: {missing}"
    return [n - 1 for n in numbers], "position"


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


def copy_blocks(src_path, key, time_axis, channel_index, out_ds, start):
    written = start
    index = np.asarray(channel_index, dtype=np.int64)
    with h5py.File(src_path, "r") as src:
        data = src[key]
        n = data.shape[time_axis]
        pos = 0
        while pos < n:
            take = min(BLOCK, n - pos)
            if time_axis == 1:
                block = np.asarray(data[:, pos:pos + take])
                out_ds[:, written:written + take] = block[index, :]
            else:
                block = np.asarray(data[pos:pos + take, :])
                out_ds[written:written + take, :] = block[:, index]
            pos += take
            written += take
    return written


def write_merged(files_info, out_path, indices, numbers, regions):
    first = files_info[0]
    amp_axis = first["amp_time_axis"]
    n_keep = len(indices)
    n_time = sum(info["n_amp_time"] for info in files_info)
    amp_shape = (n_keep, n_time) if amp_axis == 1 else (n_time, n_keep)

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
                written = copy_blocks(
                    info["path"], "amplifier_data", amp_axis, indices, amp, written
                )
            if first["n_dig"] is not None:
                dig_axis = first["dig_time_axis"]
                n_dig = first["n_dig"]
                dig_index = list(range(n_dig))
                dig_shape = (n_dig, n_time) if dig_axis == 1 else (n_time, n_dig)
                dig = out.create_dataset("board_dig_in_data", shape=dig_shape, dtype=dig_dtype)
                written_dig = 0
                for info in files_info:
                    written_dig = copy_blocks(
                        info["path"], "board_dig_in_data", dig_axis, dig_index, dig, written_dig
                    )
            text = h5py.string_dtype(encoding="utf-8")
            out.create_dataset("channel_number", data=np.asarray(numbers, dtype=np.int32))
            out.create_dataset("brain_region", data=np.asarray(regions, dtype=text))
            if usable(first["amp_names"]):
                kept_names = [first["amp_names"][i] for i in indices]
                out.create_dataset("amplifier_channel_names", data=np.asarray(kept_names, dtype=text))
            source = [info.get("source_name", info["path"].name) for info in files_info]
            out.create_dataset("source_files", data=np.asarray(source, dtype=text))
        tmp.replace(out_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return n_keep, n_time


def raw_directory(subject, session, root):
    if session is None:
        return root / subject
    return root / subject / session


def output_path(subject, root):
    return root / subject / f"{subject}.mat"


def text_list(dataset):
    values = []
    for item in np.asarray(dataset).ravel():
        if isinstance(item, bytes):
            values.append(item.decode())
        else:
            values.append(str(item))
    return values


def layout_status(path, numbers, regions):
    """Return 'prepared', 'raw', or a reason the file cannot be used."""
    if not path.is_file():
        return "missing"
    if not h5py.is_hdf5(str(path)):
        return "raw"
    with h5py.File(path, "r") as h5:
        if "amplifier_data" not in h5:
            return f"{path.name} has no amplifier_data"
        if "channel_number" not in h5 or "brain_region" not in h5:
            return "raw"
        got_numbers = [int(x) for x in np.asarray(h5["channel_number"]).ravel()]
        got_regions = text_list(h5["brain_region"])
        shape = tuple(int(x) for x in h5["amplifier_data"].shape)
    if got_numbers == list(numbers) and got_regions == list(regions) and len(numbers) in shape:
        return "prepared"
    return (
        f"{path.name} already has channel_number and brain_region, "
        "but they do not match the word_boun channel map"
    )


def as_text(value):
    if isinstance(value, bytes):
        return value.decode().strip()
    if isinstance(value, str):
        return value.strip()
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return ""
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(x) for x in arr).strip()
    if np.issubdtype(arr.dtype, np.integer):
        return "".join(chr(int(c)) for c in arr if 0 < int(c) < 256).strip()
    return ""


def legacy_channel_names(channels):
    names = []
    for obj in np.asarray(channels).ravel():
        name = ""
        if isinstance(obj, np.void) and obj.dtype.names:
            for field in ("native_channel_name", "custom_channel_name"):
                if field in obj.dtype.names:
                    name = as_text(obj[field])
                    if name:
                        break
        else:
            for field in ("native_channel_name", "custom_channel_name"):
                if hasattr(obj, field):
                    name = as_text(getattr(obj, field))
                    if name:
                        break
        names.append(name)
    return names


def write_name_refs(h5, group_name, names):
    group = h5.create_group(group_name)
    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    refs = []
    for i, name in enumerate(names):
        ds = h5.create_dataset(
            f"_names/{group_name}/{i}",
            data=np.array([ord(c) for c in name], dtype=np.uint16),
        )
        refs.append(ds.ref)
    group.create_dataset(
        "native_channel_name",
        data=np.array(refs, dtype=ref_dtype).reshape(len(names), 1),
    )


def convert_legacy_mat(src, dest):
    """Copy a pre-v7.3 .mat into the HDF5 layout inspect_file reads."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError(
            f"{src.name} is not an HDF5/v7.3 mat file, and scipy is not installed"
        ) from exc
    mat = loadmat(str(src), squeeze_me=True, struct_as_record=True)
    if "amplifier_data" not in mat:
        raise KeyError(f"{src.name} has no amplifier_data")
    amp = np.asarray(mat["amplifier_data"])
    if amp.ndim != 2:
        raise ValueError(f"{src.name}: amplifier_data shape {amp.shape} is not 2D")
    names = legacy_channel_names(mat["amplifier_channels"]) if "amplifier_channels" in mat else []
    if not names or len(names) not in amp.shape:
        names = []
    if dest.exists():
        dest.unlink()
    with h5py.File(dest, "w") as h5:
        h5.create_dataset("amplifier_data", data=amp)
        if names:
            write_name_refs(h5, "amplifier_channels", names)
        if "board_dig_in_data" in mat:
            dig = np.asarray(mat["board_dig_in_data"])
            if dig.ndim != 2:
                raise ValueError(f"{src.name}: board_dig_in_data shape {dig.shape} is not 2D")
            h5.create_dataset("board_dig_in_data", data=dig)
            if "board_dig_in_channels" in mat:
                dig_names = legacy_channel_names(mat["board_dig_in_channels"])
                if dig_names and len(dig_names) in dig.shape:
                    write_name_refs(h5, "board_dig_in_channels", dig_names)


def list_raw_mats(directory, out_path, other_names=False):
    """Raw inputs in directory. The merged output file is not an input.

    Chunks are named Temp_*.mat. A single-file subject may instead have one
    differently named .mat.
    """
    temps = sorted(p for p in directory.glob("Temp_*.mat") if p.is_file())
    if temps or not other_names:
        mats = temps
    else:
        mats = sorted(p for p in directory.glob("*.mat") if p.is_file())
    skipped = out_path.resolve() if out_path.exists() else None
    return [p for p in mats if skipped is None or p.resolve() != skipped]


def write_one(subject, src, out_path):
    """Select this subject's channels from one raw file and write out_path."""
    numbers, regions = wanted(subject)
    converted = None
    read_path = src
    try:
        if not h5py.is_hdf5(str(src)):
            converted = out_path.with_name(out_path.name + ".legacy.h5")
            convert_legacy_mat(src, converted)
            read_path = converted
            print(f"  transformed {src.name} from MATLAB into HDF5")
        info = inspect_file(read_path)
        info["source_name"] = src.name
        detail, reasons = mismatch_lines([info])
        for line in detail:
            print(line)
        if reasons:
            print(f"[STOP] {subject}: not written.")
            for line in reasons:
                print(f"  {line}")
            return False
        indices, how = channel_indices(info, numbers)
        if indices is None:
            print(f"[STOP] {subject}: not written.")
            print(f"  {how}")
            return False
        if how == "position":
            print("  channel names were not found; selected channels by 1-based position")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_keep, n_time = write_merged([info], out_path, indices, numbers, regions)
    finally:
        if converted is not None and converted.exists():
            converted.unlink()
    print(f"[WROTE] {out_path}")
    print(f"  single file, no concatenation; source {src.name}")
    print(f"  amplifier_channels={n_keep} samples={n_time}")
    for start, end, region in REGIONS[subject]:
        print(f"  {region}: {start}-{end}")
    return True


def place_single(subject, session, root):
    """Put one raw subject file at the merged-file path."""
    src_dir = raw_directory(subject, session, root)
    out_path = output_path(subject, root)
    numbers, regions = wanted(subject)
    print(f"\n{subject}")
    print("  one raw file: select channels and write the merged-file layout")
    mats = list_raw_mats(src_dir, out_path, other_names=True) if src_dir.is_dir() else []
    if len(mats) > 1:
        print(f"[STOP] {subject}: expected one raw .mat in {src_dir}, found {len(mats)}")
        for path in mats:
            print(f"  {path.name}")
        return False
    if not mats:
        if not src_dir.is_dir() and not out_path.is_file():
            print(f"[STOP] {subject}: raw folder not found: {src_dir}")
            return False
        status = layout_status(out_path, numbers, regions)
        if status == "prepared":
            print(f"[SKIP] {out_path} is already the merged file")
            return True
        if status == "raw":
            print(f"  raw file is already at {out_path}; rewriting the merged layout")
            return write_one(subject, out_path, out_path)
        print(f"[STOP] {subject}: no raw .mat in {src_dir}")
        if status != "missing":
            print(f"  {status}")
        return False

    src = mats[0]
    status = layout_status(src, numbers, regions)
    if status == "prepared":
        if src.resolve() == out_path.resolve():
            print(f"[SKIP] {out_path} is already the merged file")
            return True
        out_path.parent.mkdir(parents=True, exist_ok=True)
        holding = out_path.with_name(out_path.name + ".moving")
        if holding.exists():
            holding.unlink()
        shutil.move(str(src), str(holding))
        if out_path.exists():
            out_path.unlink()
        holding.replace(out_path)
        print(f"[MOVED] {src} -> {out_path}")
        return True
    if status != "raw":
        print(f"[STOP] {subject}: not written.")
        print(f"  {status}")
        return False
    return write_one(subject, src, out_path)


def merge_subject(subject, session, root):
    session_dir = raw_directory(subject, session, root)
    out_path = output_path(subject, root)
    print(f"\n{subject}")
    if not session_dir.is_dir():
        print(f"[STOP] {subject}: raw folder not found: {session_dir}")
        return False
    mats = list_raw_mats(session_dir, out_path)
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

    numbers, regions = wanted(subject)
    indices, how = channel_indices(files_info[0], numbers)
    if indices is None:
        print(f"[STOP] {subject}: not merged.")
        print(f"  {how}")
        return False
    if how == "position":
        print("  channel names were not found; selected channels by 1-based position")

    n_keep, n_time = write_merged(files_info, out_path, indices, numbers, regions)
    print(f"[WROTE] {out_path}")
    print(f"  amplifier_channels={n_keep} samples={n_time} files={len(mats)}")
    for start, end, region in REGIONS[subject]:
        print(f"  {region}: {start}-{end}")
    return True


def main():
    ok = True
    for subject, session in SUBJECTS:
        if subject in SINGLE_FILE_SUBJECTS:
            done = place_single(subject, session, ROOT)
        else:
            done = merge_subject(subject, session, ROOT)
        if not done:
            ok = False
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
