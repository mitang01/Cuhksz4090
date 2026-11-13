#!/usr/bin/env python3
"""Merge the confirmed sub-012 emotion-rating session and report its triggers.

The experimenter identified this session:

  /share/workspace3/ieeg/micro/Temp_260823_144306
    Temp_260823_144306.mat
    Temp_260823_145306.mat
    Temp_260823_150306.mat
    Temp_260823_151306.mat
    Temp_260823_152306.mat

The chunks are concatenated in time order and written to:

  /share/workspace3/ieeg/micro/sub-012.mat

Kept microwires are the sub-012 map from word_boun: 1-16 ATL, 17-32 HG,
33-48 Amygdala, 49-64 VmPFC. amplifier_data holds those channels.
board_dig_in_data is copied in full. The merged file is HDF5.

The script prints a table of rising edges on board_dig_in_data (the Intan
digital input, also called broad_dig_in), with one column per chunk. It
also saves a timeline of those edges, one row per digital channel, time
running left to right as during collection:

  /share/workspace3/ieeg/micro/sub-012_dig_in_edges.png
"""

import re
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

SUBJECT = "sub-012"
OUT = Path("/share/workspace3/ieeg/micro") / f"{SUBJECT}.mat"
FIG = OUT.with_name(f"{SUBJECT}_dig_in_edges.png")
SR = 30000.0
BLOCK = 200000
# Edge indices are only stored up to this many inside the older scanner.
# The timeline keeps its own, higher cap. Counts themselves are complete.
KEEP_EDGES = 128
MAX_PLOT_EDGES = 50000
REGIONS = [
    (1, 16, "ATL"),
    (17, 32, "HG"),
    (33, 48, "Amygdala"),
    (49, 64, "VmPFC"),
]
RAW_DIR = Path("/share/workspace3/ieeg/micro/Temp_260823_144306")
RAW_NAMES = [
    "Temp_260823_144306.mat",
    "Temp_260823_145306.mat",
    "Temp_260823_150306.mat",
    "Temp_260823_151306.mat",
    "Temp_260823_152306.mat",
]
STAMP = re.compile(r"^Temp_(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.mat$")

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
        if not usable(dig_names) and "board_dig_in_channel_names" in h5:
            dig_names = []
            for item in np.asarray(h5["board_dig_in_channel_names"]).ravel():
                if isinstance(item, bytes):
                    dig_names.append(item.decode())
                else:
                    dig_names.append(str(item))
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
                out.create_dataset(
                    "amplifier_channel_names", data=np.asarray(kept_names, dtype=text)
                )
            if usable(first["dig_names"]):
                out.create_dataset(
                    "board_dig_in_channel_names",
                    data=np.asarray(first["dig_names"], dtype=text),
                )
            source = [info.get("source_name", info["path"].name) for info in files_info]
            out.create_dataset("source_files", data=np.asarray(source, dtype=text))
        tmp.replace(out_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return n_keep, n_time


def sample_rate(path):
    try:
        with h5py.File(path, "r") as h5:
            if "frequency_parameters" not in h5:
                return None
            group = h5["frequency_parameters"]
            if "board_dig_in_sample_rate" not in group:
                return None
            value = float(np.asarray(group["board_dig_in_sample_rate"]).ravel()[0])
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def can_scan(files_info):
    if any(info["n_dig"] is None for info in files_info):
        return False
    if len({info["n_dig"] for info in files_info}) != 1:
        return False
    if len({info["dig_time_axis"] for info in files_info}) != 1:
        return False
    if any(info["n_dig_time"] != info["n_amp_time"] for info in files_info):
        return False
    return True


def dig_labels(info):
    names = info["dig_names"]
    labels = []
    for i in range(info["n_dig"]):
        if i < len(names) and str(names[i]).strip():
            labels.append(str(names[i]).strip())
        else:
            labels.append(f"DIG {i + 1}")
    return labels


def load_block(data, axis, pos, take):
    if axis == 1:
        block = np.asarray(data[:, pos:pos + take])
    else:
        block = np.asarray(data[pos:pos + take, :]).T
    return block


def edges_in_block(high, prev, at_start):
    """Rising and falling indices inside one block.

    The first sample of the recording is not a rising edge. That matches
    a diff that prepends the first sample. Later blocks, including the
    first sample of the next file, are compared with the sample before them.
    """
    n_ch, take = high.shape
    rise = [[] for _ in range(n_ch)]
    fall = [[] for _ in range(n_ch)]
    if take == 0:
        return rise, fall, prev
    if not at_start:
        for ch in np.flatnonzero(high[:, 0] & ~prev):
            rise[ch].append(0)
        for ch in np.flatnonzero(prev & ~high[:, 0]):
            fall[ch].append(0)
    if take > 1:
        up = high[:, 1:] & ~high[:, :-1]
        down = high[:, :-1] & ~high[:, 1:]
        for ch in range(n_ch):
            rise[ch].extend((1 + np.flatnonzero(up[ch])).tolist())
            fall[ch].extend((1 + np.flatnonzero(down[ch])).tolist())
    return rise, fall, high[:, -1].copy()


def absorb(store, dropped, counts, rel_indices, base):
    for ch, rel in enumerate(rel_indices):
        n_new = len(rel)
        if n_new == 0:
            continue
        if not dropped[ch]:
            if counts[ch] + n_new > KEEP_EDGES:
                store[ch] = []
                dropped[ch] = True
            else:
                store[ch].extend(base + int(i) for i in rel)
        counts[ch] += n_new


def iter_dig_blocks(files_info):
    """Yield (file_index, pos, block, at_start) with block shaped (channels, samples)."""
    axis = files_info[0]["dig_time_axis"]
    at_start = True
    for file_index, info in enumerate(files_info):
        with h5py.File(info["path"], "r") as h5:
            data = h5["board_dig_in_data"]
            n = info["n_dig_time"]
            pos = 0
            while pos < n:
                take = min(BLOCK, n - pos)
                block = load_block(data, axis, pos, take)
                yield file_index, pos, block, at_start
                at_start = False
                pos += take


def blank_edges(n_ch):
    return {
        "rise": [[] for _ in range(n_ch)],
        "fall": [[] for _ in range(n_ch)],
        "rise_drop": [False] * n_ch,
        "fall_drop": [False] * n_ch,
        "rise_count": [0] * n_ch,
        "fall_count": [0] * n_ch,
        "per_file_rise": [],
    }


def finish_file(state, n_ch, before_rise):
    state["per_file_rise"].append(
        [state["rise_count"][i] - before_rise[i] for i in range(n_ch)]
    )


def scan_level(files_info):
    n_ch = files_info[0]["n_dig"]
    state = blank_edges(n_ch)
    max_value = np.zeros(n_ch, dtype=np.float64)
    uniques = [set() for _ in range(n_ch)]
    many = [False] * n_ch
    prev = None
    file_index = None
    before_rise = None
    offset = 0
    for info_i, pos, block, at_start in iter_dig_blocks(files_info):
        if file_index != info_i:
            if file_index is not None:
                finish_file(state, n_ch, before_rise)
                offset += files_info[file_index]["n_dig_time"]
            file_index = info_i
            before_rise = state["rise_count"][:]
        n_ch_block, take = block.shape
        if n_ch_block != n_ch:
            raise ValueError(f"digital block has {n_ch_block} channels, expected {n_ch}")
        np.maximum(max_value, np.max(block, axis=1), out=max_value)
        for ch in range(n_ch):
            if many[ch]:
                continue
            found = np.unique(block[ch])
            uniques[ch].update(float(x) for x in found.tolist())
            if len(uniques[ch]) > 4:
                many[ch] = True
                uniques[ch].clear()
        high = block > 0.5
        if prev is None:
            prev = np.zeros(n_ch, dtype=bool)
        rise, fall, prev = edges_in_block(high, prev, at_start)
        base = offset + pos
        absorb(state["rise"], state["rise_drop"], state["rise_count"], rise, base)
        absorb(state["fall"], state["fall_drop"], state["fall_count"], fall, base)
    if file_index is not None:
        finish_file(state, n_ch, before_rise)
    state["max_value"] = max_value
    state["uniques"] = uniques
    state["many"] = many
    return state


def two_levels(uniques):
    return len(uniques) <= 2


def channel_kind(state, ch):
    max_value = float(state["max_value"][ch])
    if state["many"][ch] or (max_value > 1 and not two_levels(state["uniques"][ch])):
        if max_value > 65535:
            return "unrecognized"
        if max_value > 1:
            return "bitfield"
    return "binary"


def scan_bits(files_info, channel, n_bits):
    state = blank_edges(n_bits)
    prev = None
    file_index = None
    before_rise = None
    offset = 0
    shifts = (1 << np.arange(n_bits)).astype(np.int64)
    for info_i, pos, block, at_start in iter_dig_blocks(files_info):
        if file_index != info_i:
            if file_index is not None:
                finish_file(state, n_bits, before_rise)
                offset += files_info[file_index]["n_dig_time"]
            file_index = info_i
            before_rise = state["rise_count"][:]
        vals = np.rint(block[channel]).astype(np.int64)
        high = (vals[None, :] & shifts[:, None]) != 0
        if prev is None:
            prev = np.zeros(n_bits, dtype=bool)
        rise, fall, prev = edges_in_block(high, prev, at_start)
        base = offset + pos
        absorb(state["rise"], state["rise_drop"], state["rise_count"], rise, base)
        absorb(state["fall"], state["fall_drop"], state["fall_count"], fall, base)
    if file_index is not None:
        finish_file(state, n_bits, before_rise)
    return state


def kept_edges(state, index, which):
    count = state[f"{which}_count"][index]
    if state[f"{which}_drop"][index] or count > KEEP_EDGES:
        return None
    return state[which][index]


def file_split(per_file, index, names):
    parts = []
    for name, counts in zip(names, per_file):
        parts.append(f"{name}={counts[index]}")
    return ", ".join(parts)


def build_candidates(files_info):
    """Return (table_lines, candidates) from the concatenated digital input."""
    labels = dig_labels(files_info[0])
    names = [info["path"].name for info in files_info]
    level = scan_level(files_info)
    lines = [
        "board_dig_in_data rising edges on the concatenated recording "
        "(digital input, also called broad_dig_in):"
    ]
    candidates = []
    n_ch = files_info[0]["n_dig"]
    for ch in range(n_ch):
        kind = channel_kind(level, ch)
        split = file_split(level["per_file_rise"], ch, names)
        count = level["rise_count"][ch]
        if kind == "binary":
            lines.append(f"  channel {ch + 1} ({labels[ch]}): {count} ({split})")
            candidates.append(
                {
                    "name": f"channel {ch + 1} ({labels[ch]})",
                    "count": count,
                    "rising": kept_edges(level, ch, "rise"),
                    "falling": kept_edges(level, ch, "fall"),
                }
            )
            continue
        if kind == "unrecognized":
            lines.append(
                f"  channel {ch + 1} ({labels[ch]}): {count} level edges ({split}); "
                "values are not a 0/1 line or a 16-bit digital word, so bits were not unpacked"
            )
            candidates.append(
                {
                    "name": f"channel {ch + 1} ({labels[ch]})",
                    "count": count,
                    "rising": kept_edges(level, ch, "rise"),
                    "falling": kept_edges(level, ch, "fall"),
                }
            )
            continue
        max_value = int(level["max_value"][ch])
        n_bits = min(16, max(1, max_value.bit_length()))
        lines.append(
            f"  channel {ch + 1} ({labels[ch]}): packed digital word, "
            f"level edges={count}, max={max_value} ({split})"
        )
        bits = scan_bits(files_info, ch, n_bits)
        any_bit = False
        for bit in range(n_bits):
            bit_count = bits["rise_count"][bit]
            if bit_count == 0:
                continue
            any_bit = True
            bit_split = file_split(bits["per_file_rise"], bit, names)
            bit_name = f"channel {ch + 1} ({labels[ch]}) bit{bit}"
            lines.append(f"    bit{bit}: {bit_count} ({bit_split})")
            candidates.append(
                {
                    "name": bit_name,
                    "count": bit_count,
                    "rising": kept_edges(bits, bit, "rise"),
                    "falling": kept_edges(bits, bit, "fall"),
                }
            )
        if not any_bit:
            lines.append("    no bits were high")
    return lines, candidates


def wanted():
    numbers = []
    regions = []
    for start, end, region in REGIONS:
        for number in range(start, end + 1):
            numbers.append(number)
            regions.append(region)
    return numbers, regions


def file_stamp(path):
    match = STAMP.match(path.name.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(part) for part in match.groups())
    return datetime(2000 + year, month, day, hour, minute, second)


def short_stamp(name):
    match = STAMP.match(name.strip())
    if match is None:
        return name.strip()
    return "".join(match.groups()[3:])


def resolve_listed(directory, names):
    found = []
    missing = []
    for name in names:
        plain = directory / name
        spaced = directory / f" {name}"
        if plain.is_file():
            found.append(plain)
        elif spaced.is_file():
            print(f"  using {spaced.name!r} in place of {name}")
            found.append(spaced)
        else:
            missing.append(name)
    return found, missing


def unlisted_temp(directory, names):
    accepted = set(names)
    accepted.update(f" {name}" for name in names)
    extras = []
    for pattern in ("Temp_*.mat", " Temp_*.mat"):
        for path in sorted(directory.glob(pattern)):
            if path.name not in accepted:
                extras.append(path.name)
    return extras


def order_paths(paths):
    stamped = []
    loose = []
    for path in paths:
        stamp = file_stamp(path)
        if stamp is None:
            print(f"  {path.name} has no Temp timestamp and is appended after the timed chunks")
            loose.append(path)
            continue
        stamped.append((stamp, path))
    stamped.sort(key=lambda item: (item[0], item[1].name))
    return [path for _stamp, path in stamped] + loose


def rate_of(info):
    recorded = info.get("sample_rate")
    if recorded is None or recorded <= 0:
        return SR
    return recorded


def collect_rises(files_info):
    """Rising-edge times in seconds, per-file counts, and channel kinds.

    The first sample of the recording is not a rising edge. The first sample
    of each later chunk is compared with the last sample of the chunk before it.
    """
    n_ch = files_info[0]["n_dig"]
    stored = [[] for _ in range(n_ch)]
    counts = [0] * n_ch
    truncated = [False] * n_ch
    per_file = []
    max_value = np.zeros(n_ch, dtype=np.float64)
    uniques = [set() for _ in range(n_ch)]
    many = [False] * n_ch
    prev = None
    file_index = None
    before = None
    offset_s = 0.0
    for info_i, pos, block, at_start in iter_dig_blocks(files_info):
        if file_index != info_i:
            if file_index is not None:
                per_file.append([counts[ch] - before[ch] for ch in range(n_ch)])
                offset_s += files_info[file_index]["n_dig_time"] / rate_of(files_info[file_index])
            file_index = info_i
            before = counts[:]
        n_ch_block, _take = block.shape
        if n_ch_block != n_ch:
            raise ValueError(f"digital block has {n_ch_block} channels, expected {n_ch}")
        np.maximum(max_value, np.max(block, axis=1), out=max_value)
        for ch in range(n_ch):
            if many[ch]:
                continue
            found = np.unique(block[ch])
            uniques[ch].update(float(value) for value in found.tolist())
            if len(uniques[ch]) > 4:
                many[ch] = True
                uniques[ch].clear()
        high = block > 0.5
        if prev is None:
            prev = np.zeros(n_ch, dtype=bool)
        rise, _fall, prev = edges_in_block(high, prev, at_start)
        rate = rate_of(files_info[info_i])
        for ch, rel in enumerate(rise):
            n_new = len(rel)
            if n_new == 0:
                continue
            room = MAX_PLOT_EDGES - len(stored[ch])
            if room > 0:
                take = rel[:room]
                base = offset_s + pos / rate
                stored[ch].extend(base + (int(index) / rate) for index in take)
            if n_new > room:
                truncated[ch] = True
            counts[ch] += n_new
    if file_index is not None:
        per_file.append([counts[ch] - before[ch] for ch in range(n_ch)])
    state = {"max_value": max_value, "uniques": uniques, "many": many}
    kinds = [channel_kind(state, ch) for ch in range(n_ch)]
    return stored, per_file, counts, truncated, kinds


def summary_lines(labels, per_file, names):
    shorts = [short_stamp(name) for name in names]
    n_ch = len(labels)
    n_files = len(names)
    totals = [
        sum(per_file[file_i][ch] for file_i in range(n_files))
        for ch in range(n_ch)
    ]
    shown = [f"{ch + 1} {labels[ch]}" for ch in range(n_ch)]
    label_w = max([len("channel")] + [len(text) for text in shown])
    col_w = max([5] + [len(text) for text in shorts] + [len(str(total)) for total in totals])
    header = "channel".ljust(label_w)
    for text in shorts:
        header += text.rjust(col_w + 2)
    header += "total".rjust(col_w + 2)
    lines = [
        "board_dig_in rising edges (digital input, also called broad_dig_in):",
        header,
        "-" * len(header),
    ]
    for ch in range(n_ch):
        row = shown[ch].ljust(label_w)
        for file_i in range(n_files):
            row += str(per_file[file_i][ch]).rjust(col_w + 2)
        row += str(totals[ch]).rjust(col_w + 2)
        lines.append(row)
    return lines


def plot_rising_edges(rises, labels, infos, out_path, subject):
    """One row per digital channel. Time runs left to right, as during collection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ch = len(rises)
    bounds = [0.0]
    for info in infos:
        bounds.append(bounds[-1] + info["n_dig_time"] / rate_of(info))
    duration_min = bounds[-1] / 60.0
    if duration_min <= 0:
        duration_min = 1.0
    fig_h = max(4.8, 0.62 * n_ch + 1.6)
    fig, ax = plt.subplots(figsize=(12.5, fig_h), constrained_layout=True)
    for ch in range(n_ch):
        ax.hlines(ch, 0, duration_min, color="0.88", linewidth=0.8, zorder=0)
        if not rises[ch]:
            continue
        minutes = np.asarray(rises[ch], dtype=np.float64) / 60.0
        ax.vlines(
            minutes,
            ch - 0.34,
            ch + 0.34,
            color="#1f4e79",
            linewidth=0.9,
            zorder=2,
        )
    for edge in bounds[1:-1]:
        ax.axvline(edge / 60.0, color="0.55", linestyle="--", linewidth=0.7, zorder=1)
    for file_i, info in enumerate(infos):
        mid = 0.5 * (bounds[file_i] + bounds[file_i + 1]) / 60.0
        ax.annotate(
            short_stamp(info["path"].name),
            xy=(mid, 1.01),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.35",
            annotation_clip=False,
        )
    ax.set_yticks(list(range(n_ch)))
    ax.set_yticklabels([f"{ch + 1}  {labels[ch]}" for ch in range(n_ch)])
    # Channel 1 sits on the top row, in the same order as the summary table.
    ax.set_ylim(n_ch - 0.3, -0.7)
    ax.set_xlim(0, duration_min)
    ax.set_xlabel("Time from the start of the merged recording (minutes)")
    ax.set_ylabel("board_dig_in channel")
    ax.grid(True, axis="x", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        f"{subject}  board_dig_in rising edges\n"
        f"{bounds[-1] / 60.0:.2f} min, {len(infos)} chunks, each tick is one rising edge",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def not_written(subject, out_path, reason):
    print(f"[STOP] {subject}: {reason} {out_path} was not written.")
    if out_path.exists():
        print(f"  existing file left unchanged: {out_path}")


def write_subject(infos, out_path, subject):
    paths = [info["path"] for info in infos]
    if any(path.resolve() == out_path.resolve() for path in paths):
        not_written(subject, out_path, "the output path is one of the raw inputs.")
        return False
    numbers, regions = wanted()
    indices, how = channel_indices(infos[0], numbers)
    if indices is None:
        print(f"[STOP] {subject}: channels were not selected.")
        print(f"  {how}")
        not_written(subject, out_path, "the requested microwires are missing.")
        return False
    if how == "position":
        print("  channel names were not found; selected channels by 1-based position")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_keep, n_time = write_merged(infos, out_path, indices, numbers, regions)
    print(f"[WROTE] {out_path}")
    print(f"  amplifier_channels={n_keep} samples={n_time} files={len(paths)}")
    for start, end, region in REGIONS:
        print(f"  {region}: {start}-{end}")
    return True


def load_infos(paths):
    infos = []
    for path in paths:
        info = inspect_file(path)
        info["sample_rate"] = sample_rate(path)
        infos.append(info)
    return infos


def report_triggers(infos, fig_path, subject):
    if not can_scan(infos):
        print("board_dig_in_data cannot be concatenated across these chunks.")
        return False
    labels = dig_labels(infos[0])
    rises, per_file, _counts, truncated, kinds = collect_rises(infos)
    names = [info["path"].name for info in infos]
    for line in summary_lines(labels, per_file, names):
        print(line)
    for ch, kind in enumerate(kinds):
        if kind != "binary":
            print(
                f"  channel {ch + 1} is {kind}; the table counts crossings of 0.5"
            )
    for ch, flag in enumerate(truncated):
        if flag:
            print(
                f"  channel {ch + 1} has more than {MAX_PLOT_EDGES} rising edges; "
                "the figure shows the first ones and the table count is complete"
            )
    rates = [info["sample_rate"] for info in infos if info["sample_rate"] is not None]
    if rates and all(rate == rates[0] for rate in rates):
        print(f"board_dig_in sample rate: {rates[0]:.6f} Hz")
    seconds = sum(info["n_dig_time"] / rate_of(info) for info in infos)
    print(f"concatenated digital duration: {seconds / 60.0:.2f} min ({seconds:.1f} s)")
    plot_rising_edges(rises, labels, infos, fig_path, subject)
    print(f"[FIGURE] {fig_path}")
    return True


def merge_session(directory, names, out_path, fig_path, subject):
    print(subject)
    print(f"session: {directory}")
    print(f"merged file: {out_path}")
    print(f"figure: {fig_path}")
    if not directory.is_dir():
        print("  folder not found")
        not_written(subject, out_path, "the session folder is missing.")
        return False
    found, missing = resolve_listed(directory, names)
    for name in missing:
        print(f"  missing {name}")
    extras = unlisted_temp(directory, names)
    if extras:
        print("  Temp files left out of this merge:")
        for name in extras:
            print(f"    {name}")
    if missing or len(found) != len(names):
        not_written(subject, out_path, "a raw chunk is missing.")
        return False
    paths = order_paths(found)
    if len({path.resolve() for path in paths}) != len(paths):
        not_written(subject, out_path, "a raw path is repeated.")
        return False
    print("  time order: " + ", ".join(path.name for path in paths))
    try:
        infos = load_infos(paths)
    except Exception as exc:
        print(f"[STOP] {subject}: could not read chunks: {exc}")
        not_written(subject, out_path, "the chunks could not be read.")
        return False
    for info in infos:
        dig = "none" if info["n_dig"] is None else str(info["n_dig"])
        print(
            f"  {info['path'].name}: amplifier_channels={info['n_amp']} "
            f"amplifier_samples={info['n_amp_time']} "
            f"dig_channels={dig} "
            f"duration={info['n_amp_time'] / rate_of(info) / 60.0:.2f} min"
        )
    _detail, reasons = mismatch_lines(infos)
    reported = False
    if can_scan(infos):
        reported = report_triggers(infos, fig_path, subject)
    else:
        print("board_dig_in_data cannot be concatenated across these chunks.")
    if reasons:
        print(f"[STOP] {subject}: chunks do not match.")
        for line in reasons:
            print(f"  {line}")
        not_written(subject, out_path, "the chunks do not match.")
        return False
    if not reported and any(info["n_dig"] is None for info in infos):
        print("board_dig_in_data is missing, so no trigger figure was saved.")
    return write_subject(infos, out_path, subject)


def main():
    if not merge_session(RAW_DIR, RAW_NAMES, OUT, FIG, SUBJECT):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
