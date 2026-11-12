#!/usr/bin/env python3
"""Merge confirmed audio-emotion-rating Intan chunks for four subjects.

Each subject has two Temp_*.mat chunks. They are concatenated in the
order listed below. board_dig_in_data (the Intan digital input, also
called broad_dig_in) is scanned with the same 25-stimulus, 3-response
report used for sub-012. These four recordings are confirmed task data,
so the merged file is still written when that pattern is absent. The
report is what shows whether the pattern is present.

Outputs, when the chunks agree and the requested microwires exist:

  /share/workspace3/ieeg/micro/audio_emotion_rating_pcm_1/sub-007/sub-007.mat
  /share/workspace3/ieeg/micro/audio_emotion_rating_pcm_1/sub-008/sub-008.mat
  /share/workspace3/ieeg/micro/audio_emotion_rating_pcm_1/sub-010/sub-010.mat
  /share/workspace3/ieeg/micro/audio_emotion_rating_pcm_1/sub-011/sub-011.mat

Raw chunks:

  sub-007: Temp_260323_102648.mat, Temp_260323_103648.mat
  sub-008/ses-01: Temp_260404_130758.mat, Temp_260404_131758.mat
  sub-010/ses-01: Temp_260604_102828.mat, Temp_260604_103828.mat
  sub-011/ses-02: Temp_260615_095633.mat, Temp_260615_100633.mat

A filename with a leading space is used only when the listed path is
missing and that spaced name is present.

Kept microwires use the same maps as these subjects in word_boun.
amplifier_data holds those channels. board_dig_in_data is copied in
full. channel_number and brain_region say which row is which. The
merged file is HDF5. Read it with h5py.

sub-012 is not merged here. Sorting results are not written here.
"""

from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/share/workspace3/ieeg/micro/audio_emotion_rating_pcm_1")
# Subject, session folder or None, chunk names in concatenation order.
SUBJECTS = [
    ("sub-007", None, ["Temp_260323_102648.mat", "Temp_260323_103648.mat"]),
    ("sub-008", "ses-01", ["Temp_260404_130758.mat", "Temp_260404_131758.mat"]),
    ("sub-010", "ses-01", ["Temp_260604_102828.mat", "Temp_260604_103828.mat"]),
    ("sub-011", "ses-02", ["Temp_260615_095633.mat", "Temp_260615_100633.mat"]),
]
# Inclusive 1-based channel numbers. Same maps as word_boun.
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
    "sub-010": [
        (17, 32, "Amygdala"),
        (33, 48, "VmPFC"),
    ],
    "sub-011": [
        (17, 32, "HG"),
        (33, 48, "ATL"),
    ],
}
BLOCK = 200000
N_STIMULI = 25
RESPONSES_PER_STIMULUS = 3
N_RESPONSES = N_STIMULI * RESPONSES_PER_STIMULUS
N_PULSES = N_STIMULI + N_RESPONSES
# Rising and falling indices are kept up to one stimulus plus its responses.
KEEP_EDGES = N_PULSES


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


def raw_directory(subject, session):
    if session is None:
        return ROOT / subject
    return ROOT / subject / session


def output_path(subject):
    return ROOT / subject / f"{subject}.mat"


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


def grouping(stim, resp, per_stim):
    stim = np.asarray(sorted(int(x) for x in stim), dtype=np.int64)
    resp = np.asarray(sorted(int(x) for x in resp), dtype=np.int64)
    if stim.size != N_STIMULI or resp.size != N_STIMULI * per_stim:
        return False, None, None
    before = int(np.sum(resp < stim[0]))
    counts = []
    for i in range(N_STIMULI - 1):
        counts.append(int(np.sum((resp > stim[i]) & (resp < stim[i + 1]))))
    counts.append(int(np.sum(resp > stim[-1])))
    exact = int(np.sum(np.isin(resp, stim)))
    ok = (
        before == 0
        and exact == 0
        and counts == [per_stim] * N_STIMULI
        and before + sum(counts) + exact == int(resp.size)
    )
    return ok, before, counts


def pulses_from_edges(rising, falling):
    if rising is None or falling is None:
        return None
    falling = [int(x) for x in falling]
    pulses = []
    j = 0
    for onset in (int(x) for x in rising):
        while j < len(falling) and falling[j] <= onset:
            j += 1
        if j >= len(falling):
            return None
        pulses.append((onset, falling[j] - onset))
        j += 1
    return pulses


def width_groups(pulses):
    """Split 100 pulses into 25 stimuli and 75 responses by width."""
    if pulses is None or len(pulses) != N_PULSES:
        return None
    widths = np.array([width for _, width in pulses], dtype=np.int64)
    onsets = np.array([onset for onset, _ in pulses], dtype=np.int64)
    order = np.argsort(widths, kind="mergesort")
    sorted_w = widths[order]
    gap_short = int(sorted_w[N_STIMULI] - sorted_w[N_STIMULI - 1])
    gap_long = int(sorted_w[N_RESPONSES] - sorted_w[N_RESPONSES - 1])
    if gap_short <= 0 and gap_long <= 0:
        return None
    if gap_short > 0 and gap_long > 0 and min(gap_short, gap_long) * 2 >= max(gap_short, gap_long):
        return None
    if gap_long > gap_short:
        stim_idx = order[N_RESPONSES:]
        resp_idx = order[:N_RESPONSES]
    else:
        stim_idx = order[:N_STIMULI]
        resp_idx = order[N_STIMULI:]
    stim = np.sort(onsets[stim_idx])
    resp = np.sort(onsets[resp_idx])
    return stim, resp


def edge_key(values):
    if values is None:
        return None
    return tuple(int(x) for x in values)


def decide(candidates):
    """Return ok, pattern, confirm lines, and diagnostics.

    A match is one of:
      one response line with 75 rising edges, three after each stimulus
      three response lines with 25 rising edges each, one after each stimulus
      one line whose pulse widths split into 25 stimuli and 75 responses
    Mirrored copies of the same edges count as one match.
    """
    matches = []
    diagnostics = []
    by_count = {}
    for cand in candidates:
        by_count.setdefault(cand["count"], []).append(cand)

    c25 = [c for c in by_count.get(N_STIMULI, []) if c["rising"] is not None]
    c75 = [c for c in by_count.get(N_RESPONSES, []) if c["rising"] is not None]
    for stim in c25:
        for resp in c75:
            ok, before, counts = grouping(stim["rising"], resp["rising"], RESPONSES_PER_STIMULUS)
            if ok:
                matches.append(("one_response_line", stim, [resp]))
            elif before is not None:
                diagnostics.append(
                    f"  {stim['name']} has 25 edges and {resp['name']} has 75, "
                    f"but the responses are not 3 after each stimulus "
                    f"(before first={before}, per stimulus={counts})"
                )

    if len(c25) >= 4:
        for stim in c25:
            good = []
            for resp in c25:
                if resp is stim:
                    continue
                ok, before, counts = grouping(stim["rising"], resp["rising"], 1)
                if ok:
                    good.append(resp)
            if len(good) == 3:
                matches.append(("three_response_lines", stim, good))

    for cand in candidates:
        if cand["count"] != N_PULSES or cand["rising"] is None:
            continue
        pulses = pulses_from_edges(cand["rising"], cand["falling"])
        grouped = width_groups(pulses)
        if grouped is None:
            if cand["falling"] is None:
                diagnostics.append(
                    f"  {cand['name']} has 100 rising edges, but the falling edges "
                    "were not kept, so pulse width could not be checked"
                )
            else:
                diagnostics.append(
                    f"  {cand['name']} has 100 rising edges, but the pulse widths do not "
                    "separate into 25 stimuli and 75 responses"
                )
            continue
        stim, resp = grouped
        ok, before, counts = grouping(stim, resp, RESPONSES_PER_STIMULUS)
        if ok:
            stim_cand = {"name": cand["name"], "count": N_STIMULI, "rising": stim.tolist()}
            resp_cand = {"name": cand["name"], "count": N_RESPONSES, "rising": resp.tolist()}
            matches.append(("pulse_width", stim_cand, [resp_cand]))
        else:
            diagnostics.append(
                f"  {cand['name']} pulse widths split into 25 and 75, but the 75 are not "
                f"3 after each of the 25 (before first={before}, per stimulus={counts})"
            )

    grouped = {}
    for pattern, stim, responses in matches:
        key = (
            pattern,
            edge_key(stim["rising"]),
            tuple(edge_key(item["rising"]) for item in responses),
        )
        bucket = grouped.setdefault(key, {"pattern": pattern, "stim": [], "responses": []})
        if stim["name"] not in bucket["stim"]:
            bucket["stim"].append(stim["name"])
        for item in responses:
            if item["name"] not in bucket["responses"]:
                bucket["responses"].append(item["name"])

    if len(grouped) == 1:
        chosen = next(iter(grouped.values()))
        pattern = chosen["pattern"]
        stim_names = ", ".join(chosen["stim"])
        resp_names = ", ".join(chosen["responses"])
        if pattern == "one_response_line":
            detail = "each of the 25 stimuli is followed by exactly 3 responses on that line"
            resp_line = f"[CONFIRM] response triggers: {resp_names}, {N_RESPONSES} rising edges"
        elif pattern == "three_response_lines":
            detail = (
                "each of the 25 stimuli is followed by exactly 1 response "
                "on each of those 3 lines"
            )
            resp_line = (
                f"[CONFIRM] response triggers: {resp_names}, "
                f"{N_STIMULI} rising edges on each line ({N_RESPONSES} total)"
            )
        else:
            detail = (
                "pulse width separates 25 stimuli from 75 responses, and each stimulus "
                "is followed by exactly 3 responses"
            )
            resp_line = f"[CONFIRM] response triggers: {resp_names}, {N_RESPONSES} rising edges"
        lines = [
            f"[CONFIRM] stimulus triggers: {stim_names}, {N_STIMULI} rising edges",
            resp_line,
            f"[CONFIRM] {detail}",
        ]
        return {"ok": True, "pattern": pattern, "lines": lines, "diagnostics": []}

    if len(grouped) > 1:
        diagnostics.insert(
            0,
            "  more than one trigger pattern matches, so the file was not written:",
        )
        for item in grouped.values():
            diagnostics.append(
                f"    {item['pattern']}: stimuli {', '.join(item['stim'])}; "
                f"responses {', '.join(item['responses'])}"
            )
    elif not diagnostics:
        counts = sorted((cand["count"], cand["name"]) for cand in candidates)
        if not any(count == N_STIMULI for count, _ in counts):
            diagnostics.append(f"  no line has {N_STIMULI} rising edges")
        if not any(count in (N_RESPONSES, N_PULSES) for count, _ in counts) and len(c25) < 4:
            diagnostics.append(
                f"  no line has {N_RESPONSES} rising edges, and there are not 4 lines with "
                f"{N_STIMULI}"
            )
        preview = ", ".join(f"{name}={count}" for count, name in counts[:12])
        if preview:
            diagnostics.append(f"  rising-edge counts: {preview}")
    return {"ok": False, "pattern": None, "lines": [], "diagnostics": diagnostics}


def evaluate_triggers(files_info):
    if not can_scan(files_info):
        if all(info["n_dig"] is None for info in files_info):
            reason = "board_dig_in_data is missing, so the stimulus triggers cannot be checked."
        else:
            reason = "board_dig_in_data cannot be concatenated across the raw chunks."
        return {
            "ok": False,
            "pattern": None,
            "lines": [],
            "diagnostics": [f"  {reason}"],
        }
    table, candidates = build_candidates(files_info)
    decision = decide(candidates)
    decision["lines"] = table + decision["lines"]
    rate = sample_rate(files_info[0]["path"])
    if rate is not None and decision["ok"]:
        decision["lines"].append(f"[CONFIRM] board_dig_in sample rate: {rate:.6f} Hz")
    return decision


def resolve_raw_files(subject, paths):
    resolved = []
    for path in paths:
        if path.is_file():
            resolved.append(path)
            continue
        spaced = path.parent / f" {path.name}"
        if spaced.is_file():
            print(f"  using {spaced} in place of missing {path}")
            resolved.append(spaced)
            continue
        print(f"[STOP] {subject}: raw file not found: {path}")
        parent = path.parent
        if parent.is_dir():
            nearby = sorted(item.name for item in parent.glob("Temp_*.mat"))
            if nearby:
                print("  Temp_*.mat files in that folder:")
                for name in nearby:
                    print(f"    {name}")
            else:
                print(f"  no Temp_*.mat files in {parent}")
        else:
            print(f"  folder not found: {parent}")
        return None
    return resolved


def not_written(subject, out_path, reason):
    print(f"[STOP] {subject}: {reason} {out_path} was not written.")
    if out_path.exists():
        print(f"  existing file left unchanged: {out_path}")


def merge_subject(paths, out_path, subject):
    print(f"\n{subject}")
    paths = resolve_raw_files(subject, paths)
    if paths is None:
        not_written(subject, out_path, "a raw chunk is missing.")
        return False
    if len({path.resolve() for path in paths}) != len(paths):
        not_written(subject, out_path, "a raw path is repeated.")
        return False
    if any(path.resolve() == out_path.resolve() for path in paths):
        not_written(subject, out_path, "the output path is one of the raw inputs.")
        return False
    try:
        files_info = [inspect_file(path) for path in paths]
    except Exception as exc:
        print(f"[STOP] {subject}: could not read chunks: {exc}")
        not_written(subject, out_path, "the chunks could not be read.")
        return False

    detail, reasons = mismatch_lines(files_info)
    for line in detail:
        print(line)
    triggers = evaluate_triggers(files_info)
    for line in triggers["lines"]:
        print(line)
    if not triggers["ok"]:
        print(
            f"[TRIGGERS] {subject}: board_dig_in does not show 25 stimuli "
            "with 3 responses each."
        )
        for line in triggers["diagnostics"]:
            print(line)
    if reasons:
        print(f"[STOP] {subject}: chunks do not match.")
        for line in reasons:
            print(f"  {line}")
        not_written(subject, out_path, "the chunks do not match.")
        return False

    numbers, regions = wanted(subject)
    indices, how = channel_indices(files_info[0], numbers)
    if indices is None:
        print(f"[STOP] {subject}: channels were not selected.")
        print(f"  {how}")
        not_written(subject, out_path, "the requested microwires are missing.")
        return False
    if how == "position":
        print("  channel names were not found; selected channels by 1-based position")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_keep, n_time = write_merged(files_info, out_path, indices, numbers, regions)
    print(f"[WROTE] {out_path}")
    print(f"  amplifier_channels={n_keep} samples={n_time} files={len(paths)}")
    for start, end, region in REGIONS[subject]:
        print(f"  {region}: {start}-{end}")
    return True


def main():
    ok = True
    for subject, session, names in SUBJECTS:
        folder = raw_directory(subject, session)
        paths = [folder / name for name in names]
        if not merge_subject(paths, output_path(subject), subject):
            ok = False
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
