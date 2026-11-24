#!/usr/bin/env python3
"""Find the 25 audio onsets in the merged sub-008 recording.

Reads the merged digital input:

  /share/workspace3/ieeg/micro/audio_emotion_rating_pcm_1/sub-008/sub-008.mat

Writes next to the Combinato results:

  /share/home/mitan/combinato_result/emo_rate/sub-008/sub-008_triggers.png
  /share/home/mitan/combinato_result/emo_rate/sub-008/sub-008_audio_onsets.csv
  /share/home/mitan/combinato_result/emo_rate/sub-008/sub-008_audio_onsets.txt

Time zero is the first sample of that merged file. An onset time of 12.5 s
is sample 12.5 * sample_rate on board_dig_in_data.

The first three audios last 15-16 s, 17-18 s, and about 16 s. Those three
lengths decide which edges are the audios. A digital line that stays high
for the whole sound, and a short onset pulse paired with a short offset
pulse, are both accepted. The audio length is the time from the onset
edge to the offset edge. The gap from one onset to the next onset is not
used, because that gap also contains the rating time.

Run from the emo_rate directory on the cluster:

  python audio_onsets_sub008.py
  python audio_onsets_sub008.py /path/to/sub-008.mat /path/to/output_dir

Exit 0 when one set of 25 audios matches. Exit 2 when more than one set
matches and they disagree, so the script does not pick one. Exit 1 when
nothing matches the three lengths. The trigger plot is written in every
case where the mat file can be read.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

SUBJECT = "sub-008"
DATA_ROOT = Path("/share/workspace3/ieeg/micro/audio_emotion_rating_pcm_1")
RESULT_ROOT = Path("/share/home/mitan/combinato_result/emo_rate")
MERGED = DATA_ROOT / SUBJECT / f"{SUBJECT}.mat"
OUT_DIR = RESULT_ROOT / SUBJECT
SR_FALLBACK = 30000.0
BLOCK = 200000
N_AUDIO = 25
# Inclusive windows, seconds. A tenth of a second of slack covers a length
# that lands on the stated boundary.
WINDOWS = (
    (14.75, 16.25),  # 15-16 s
    (16.75, 18.25),  # 17-18 s
    (15.25, 16.75),  # around 16 s
)
MIN_LENGTH = 2.0
MAX_LENGTH = 90.0
# A low gap shorter than this is a dropout inside one pulse, not a new event.
DROPOUT_S = 0.002
# Two matches that land inside this tolerance are the same 25 audios.
AGREE_S = 0.05


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


def dataset_names(h5, key):
    if key not in h5:
        return []
    names = []
    for item in np.asarray(h5[key]).ravel():
        if isinstance(item, bytes):
            names.append(item.decode("utf-8", "replace").strip())
        else:
            names.append(str(item).strip())
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


def sample_rate(h5):
    if "frequency_parameters" not in h5:
        return SR_FALLBACK
    group = h5["frequency_parameters"]
    if "board_dig_in_sample_rate" not in group:
        return SR_FALLBACK
    value = float(np.asarray(group["board_dig_in_sample_rate"]).ravel()[0])
    if not np.isfinite(value) or value <= 0:
        return SR_FALLBACK
    return value


def load_block(data, axis, pos, take):
    if axis == 1:
        block = np.asarray(data[:, pos:pos + take])
    else:
        block = np.asarray(data[pos:pos + take, :]).T
    return block


def edges_in_block(high, prev, at_start):
    """Rising and falling indices inside one block.

    The first sample of the recording is not a rising edge. Later blocks
    are compared with the sample before them.
    """
    n_ch, take = high.shape
    rise = [[] for _ in range(n_ch)]
    fall = [[] for _ in range(n_ch)]
    if take == 0:
        return rise, fall, prev
    if not at_start:
        for ch in np.flatnonzero(high[:, 0] & ~prev):
            rise[int(ch)].append(0)
        for ch in np.flatnonzero(prev & ~high[:, 0]):
            fall[int(ch)].append(0)
    if take > 1:
        up = high[:, 1:] & ~high[:, :-1]
        down = high[:, :-1] & ~high[:, 1:]
        for ch in range(n_ch):
            rise[ch].extend((1 + np.flatnonzero(up[ch])).tolist())
            fall[ch].extend((1 + np.flatnonzero(down[ch])).tolist())
    return rise, fall, high[:, -1].copy()


def make_line(label, channel, rise, fall, bit=None, synthetic=False):
    return {
        "label": label,
        "channel": channel,
        "bit": bit,
        "rise": np.asarray(rise, dtype=np.int64),
        "fall": np.asarray(fall, dtype=np.int64),
        "synthetic": synthetic,
    }


def default_label(index, names):
    if names and index < len(names) and names[index]:
        return f"{index + 1}  {names[index]}"
    return f"{index + 1}  DIGITAL-IN-{index + 1:02d}"


def read_digital(path, block=BLOCK):
    """Rising and falling edges of board_dig_in_data, in merged-file samples."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not h5py.is_hdf5(str(path)):
        raise ValueError(f"{path} is not an HDF5/v7.3 mat file")
    with h5py.File(path, "r") as h5:
        if "board_dig_in_data" not in h5:
            raise KeyError(f"{path} has no board_dig_in_data")
        data = h5["board_dig_in_data"]
        names = channel_names(h5, "board_dig_in_channels")
        if not any(names):
            names = dataset_names(h5, "board_dig_in_channel_names")
        shape = tuple(int(x) for x in data.shape)
        n_ch, n_time, axis = time_axis(shape, len(names))
        sr = sample_rate(h5)
        rises = [[] for _ in range(n_ch)]
        falls = [[] for _ in range(n_ch)]
        any_rise = []
        any_fall = []
        max_value = np.zeros(n_ch, dtype=np.float64)
        uniques = [set() for _ in range(n_ch)]
        many = [False] * n_ch
        prev = np.zeros(n_ch, dtype=bool)
        any_prev = np.zeros(1, dtype=bool)
        at_start = True
        pos = 0
        while pos < n_time:
            take = min(block, n_time - pos)
            samples = load_block(data, axis, pos, take)
            if samples.shape[0] != n_ch:
                raise ValueError(
                    f"digital block has {samples.shape[0]} channels, expected {n_ch}"
                )
            np.maximum(max_value, np.max(samples, axis=1), out=max_value)
            for ch in range(n_ch):
                if many[ch]:
                    continue
                found = np.unique(samples[ch])
                uniques[ch].update(float(x) for x in found.tolist())
                if len(uniques[ch]) > 8:
                    many[ch] = True
                    uniques[ch].clear()
            high = samples > 0.5
            rel_rise, rel_fall, prev = edges_in_block(high, prev, at_start)
            any_high = np.any(high, axis=0)[None, :]
            rel_any_rise, rel_any_fall, any_prev = edges_in_block(
                any_high, any_prev, at_start
            )
            for ch in range(n_ch):
                rises[ch].extend(pos + int(i) for i in rel_rise[ch])
                falls[ch].extend(pos + int(i) for i in rel_fall[ch])
            any_rise.extend(pos + int(i) for i in rel_any_rise[0])
            any_fall.extend(pos + int(i) for i in rel_any_fall[0])
            at_start = False
            pos += take
        kinds = []
        for ch in range(n_ch):
            kinds.append(channel_kind(float(max_value[ch]), uniques[ch], many[ch]))
        bit_plan = []
        for ch, kind in enumerate(kinds):
            if kind != "bitfield":
                continue
            n_bits = int(max_value[ch]).bit_length()
            n_bits = min(max(n_bits, 1), 16)
            bit_plan.append((ch, n_bits))
        bit_edges = {ch: None for ch, _n_bits in bit_plan}
        if bit_plan:
            bit_edges = scan_bits(h5["board_dig_in_data"], axis, n_time, bit_plan, block)
    lines = []
    for ch in range(n_ch):
        lines.append(
            make_line(default_label(ch, names), ch + 1, rises[ch], falls[ch])
        )
    for ch, n_bits in bit_plan:
        bit_rise, bit_fall = bit_edges[ch]
        for bit in range(n_bits):
            if len(bit_rise[bit]) == 0 and len(bit_fall[bit]) == 0:
                continue
            lines.append(
                make_line(
                    f"{ch + 1}  bit{bit}",
                    ch + 1,
                    bit_rise[bit],
                    bit_fall[bit],
                    bit=bit,
                )
            )
    lines.append(
        make_line("any DIGITAL-IN", 0, any_rise, any_fall, synthetic=True)
    )
    return {
        "sr": sr,
        "n_time": n_time,
        "n_channels": n_ch,
        "kinds": kinds,
        "lines": lines,
    }


def channel_kind(max_value, uniques, many):
    if many or (max_value > 1 and len(uniques) > 2):
        if max_value > 65535:
            return "unrecognized"
        if max_value > 1:
            return "bitfield"
    return "binary"


def scan_bits(data, axis, n_time, bit_plan, block):
    """Second pass: edges of each bit of a packed digital word."""
    found = {}
    prev = {}
    rises = {}
    falls = {}
    for ch, n_bits in bit_plan:
        prev[ch] = np.zeros(n_bits, dtype=bool)
        rises[ch] = [[] for _ in range(n_bits)]
        falls[ch] = [[] for _ in range(n_bits)]
    at_start = True
    pos = 0
    while pos < n_time:
        take = min(block, n_time - pos)
        samples = load_block(data, axis, pos, take)
        for ch, n_bits in bit_plan:
            vals = np.rint(samples[ch]).astype(np.int64)
            shifts = (1 << np.arange(n_bits)).astype(np.int64)
            high = (vals[None, :] & shifts[:, None]) != 0
            rel_rise, rel_fall, prev[ch] = edges_in_block(high, prev[ch], at_start)
            for bit in range(n_bits):
                rises[ch][bit].extend(pos + int(i) for i in rel_rise[bit])
                falls[ch][bit].extend(pos + int(i) for i in rel_fall[bit])
        at_start = False
        pos += take
    for ch, _n_bits in bit_plan:
        found[ch] = (rises[ch], falls[ch])
    return found


def pulse_intervals(rise, fall, sr):
    """Each rising edge paired with the next falling edge."""
    fall = np.asarray(fall, dtype=np.int64)
    intervals = []
    j = 0
    for onset in np.asarray(rise, dtype=np.int64):
        onset = int(onset)
        while j < fall.size and int(fall[j]) <= onset:
            j += 1
        if j >= fall.size:
            break
        offset = int(fall[j])
        intervals.append((onset, offset, (offset - onset) / sr))
        j += 1
    return merge_dropouts(intervals, sr)


def merge_dropouts(intervals, sr):
    """Join a pulse that a sub-millisecond low gap split in two."""
    if not intervals:
        return []
    gap = max(1, int(round(DROPOUT_S * sr)))
    merged = [list(intervals[0])]
    for onset, offset, _length in intervals[1:]:
        if onset - merged[-1][1] <= gap:
            merged[-1][1] = offset
            merged[-1][2] = (offset - merged[-1][0]) / sr
        else:
            merged.append([onset, offset, (offset - onset) / sr])
    return [tuple(item) for item in merged]


def first_three_ok(lengths):
    if len(lengths) < 3:
        return False
    for length, (low, high) in zip(lengths[:3], WINDOWS):
        if length < low or length > high:
            return False
    return True


def lengths_ok(lengths):
    if len(lengths) != N_AUDIO:
        return False
    if any(length < MIN_LENGTH or length > MAX_LENGTH for length in lengths):
        return False
    return first_three_ok(lengths)


def matching_runs(intervals):
    """Contiguous runs of 25 intervals whose first three lengths are the anchors."""
    if len(intervals) < N_AUDIO:
        return []
    found = []
    last = len(intervals) - N_AUDIO
    for start in range(last + 1):
        chunk = intervals[start:start + N_AUDIO]
        if lengths_ok([item[2] for item in chunk]):
            found.append(chunk)
    return found


def long_pulses(intervals):
    return [item for item in intervals if MIN_LENGTH <= item[2] <= MAX_LENGTH]


def add_runs(found, line, other, method, rank, detail, intervals):
    for chunk in matching_runs(intervals):
        found.append(
            {
                "method": method,
                "rank": rank,
                "bit_penalty": 0 if line["bit"] is None and other["bit"] is None else 1,
                "onset_line": line,
                "offset_line": other,
                "detail": detail,
                "intervals": chunk,
            }
        )


def cross_pairs(onsets, offsets, sr):
    """Pair each onset with the next offset that lands before the next onset."""
    onsets = np.asarray(onsets, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if onsets.size == 0 or offsets.size == 0:
        return []
    horizon = int(round(MAX_LENGTH * sr))
    pairs = []
    j = 0
    for i, onset in enumerate(onsets):
        onset = int(onset)
        limit = int(onsets[i + 1]) if i + 1 < onsets.size else onset + horizon
        while j < offsets.size and int(offsets[j]) <= onset:
            j += 1
        if j >= offsets.size or int(offsets[j]) >= limit:
            pairs.append(None)
            continue
        offset = int(offsets[j])
        pairs.append((onset, offset, (offset - onset) / sr))
        j += 1
    return pairs


def complete_runs(pairs):
    """Break the pair list at a missing offset, then take runs of 25."""
    runs = []
    current = []
    for item in pairs:
        if item is None:
            runs.extend(matching_runs(current))
            current = []
            continue
        current.append(item)
    runs.extend(matching_runs(current))
    return runs


def alternate_pairs(rises, sr):
    """Onset, offset, onset, offset on one line. Also the one-edge shift."""
    rises = np.asarray(rises, dtype=np.int64)
    runs = []
    for shift in (0, 1):
        pairs = []
        idx = shift
        while idx + 1 < rises.size:
            onset = int(rises[idx])
            offset = int(rises[idx + 1])
            pairs.append((onset, offset, (offset - onset) / sr))
            idx += 2
        runs.extend(matching_runs(pairs))
    return runs


def locate_audios(lines, sr):
    """Return every 25-audio match. Callers decide whether they agree."""
    found = []
    for line in lines:
        pulses = long_pulses(pulse_intervals(line["rise"], line["fall"], sr))
        add_runs(
            found,
            line,
            line,
            "pulse",
            0,
            f"{line['label']}: the line stays high for the length of each audio",
            pulses,
        )
        if line["rise"].size <= 5000:
            for chunk in alternate_pairs(line["rise"], sr):
                found.append(
                    {
                        "method": "alternate",
                        "rank": 1,
                        "bit_penalty": 0 if line["bit"] is None else 1,
                        "onset_line": line,
                        "offset_line": line,
                        "detail": (
                            f"{line['label']}: rising edges alternate "
                            "between audio onset and audio offset"
                        ),
                        "intervals": chunk,
                    }
                )
    for onset_line in lines:
        if onset_line["rise"].size == 0 or onset_line["rise"].size > 5000:
            continue
        for offset_line in lines:
            if offset_line is onset_line:
                continue
            if offset_line["rise"].size > 5000 and offset_line["fall"].size > 5000:
                continue
            for event, offsets, rank, how in (
                ("rise", offset_line["rise"], 2, "rising edge"),
                ("fall", offset_line["fall"], 3, "falling edge"),
            ):
                if offsets.size == 0 or offsets.size > 5000:
                    continue
                detail = (
                    f"onset is a rising edge on {onset_line['label']}; "
                    f"offset is a {how} on {offset_line['label']}"
                )
                for chunk in complete_runs(cross_pairs(onset_line["rise"], offsets, sr)):
                    found.append(
                        {
                            "method": "cross",
                            "rank": rank,
                            "bit_penalty": (
                                0
                                if onset_line["bit"] is None and offset_line["bit"] is None
                                else 1
                            ),
                            "onset_line": onset_line,
                            "offset_line": offset_line,
                            "detail": detail,
                            "intervals": chunk,
                        }
                    )
    return found


def agree(one, other, sr):
    tol = AGREE_S * sr
    for (onset_a, _off_a, len_a), (onset_b, _off_b, len_b) in zip(
        one["intervals"], other["intervals"]
    ):
        if abs(onset_a - onset_b) > tol or abs(len_a - len_b) > AGREE_S:
            return False
    return True


def cluster_matches(matches, sr):
    parent = list(range(len(matches)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            if agree(matches[i], matches[j], sr):
                left, right = find(i), find(j)
                if left != right:
                    parent[right] = left
    groups = {}
    for index, match in enumerate(matches):
        groups.setdefault(find(index), []).append(match)
    ordered = []
    for group in groups.values():
        group.sort(key=lambda item: (item["rank"], item["bit_penalty"], item["detail"]))
        ordered.append(group)
    ordered.sort(key=lambda group: group[0]["intervals"][0][0])
    return ordered


def line_summary(lines, sr):
    rows = ["board_dig_in edges (rising / falling), counted on the merged file:"]
    for line in lines:
        if line["synthetic"]:
            continue
        pulses = long_pulses(pulse_intervals(line["rise"], line["fall"], sr))
        text = (
            f"  {line['label']}: {line['rise'].size} rising, "
            f"{line['fall'].size} falling, {len(pulses)} pulses of "
            f"{MIN_LENGTH:.0f}-{MAX_LENGTH:.0f} s"
        )
        if pulses:
            shown = ", ".join(f"{item[2]:.3f}" for item in pulses[:3])
            text += f"; first long pulses {shown} s"
        rows.append(text)
    return rows


def table_lines(intervals, sr):
    header = (
        f"{'audio':>5}  {'onset_s':>12}  {'offset_s':>12}  "
        f"{'length_s':>10}  {'onset_sample':>14}  {'offset_sample':>14}"
    )
    rows = [header]
    for index, (onset, offset, length) in enumerate(intervals, start=1):
        rows.append(
            f"{index:5d}  {onset / sr:12.6f}  {offset / sr:12.6f}  "
            f"{length:10.6f}  {onset:14d}  {offset:14d}"
        )
    return rows


def write_csv(path, intervals, sr):
    lines = ["audio,onset_s,offset_s,length_s,onset_sample,offset_sample"]
    for index, (onset, offset, length) in enumerate(intervals, start=1):
        lines.append(
            f"{index},{onset / sr:.6f},{offset / sr:.6f},{length:.6f},{onset},{offset}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_match(match, sr):
    lengths = [item[2] for item in match["intervals"]]
    rows = [
        match["detail"],
        (
            "first three lengths: "
            f"{lengths[0]:.3f} s, {lengths[1]:.3f} s, {lengths[2]:.3f} s"
        ),
        (
            "windows used: "
            f"{WINDOWS[0][0]:.2f}-{WINDOWS[0][1]:.2f} s, "
            f"{WINDOWS[1][0]:.2f}-{WINDOWS[1][1]:.2f} s, "
            f"{WINDOWS[2][0]:.2f}-{WINDOWS[2][1]:.2f} s"
        ),
        "",
    ]
    rows.extend(table_lines(match["intervals"], sr))
    return rows


def header_lines(path, info):
    duration = info["n_time"] / info["sr"]
    return [
        f"subject: {SUBJECT}",
        f"merged file: {path}",
        f"sample rate: {info['sr']:.6f} Hz",
        f"samples: {info['n_time']}",
        f"duration: {duration:.3f} s ({duration / 60.0:.3f} min)",
        "time zero: first sample of the merged file",
        "length: offset sample minus onset sample, divided by the sample rate",
        "",
    ]


def same_line(one, other):
    return (
        not other["synthetic"]
        and one["channel"] == other["channel"]
        and one["bit"] == other["bit"]
    )


def nearest(samples, target, tol):
    if len(samples) == 0:
        return False
    samples = np.asarray(samples)
    index = int(np.searchsorted(samples, target))
    for pos in (index - 1, index):
        if 0 <= pos < samples.size and abs(int(samples[pos]) - int(target)) <= tol:
            return True
    return False


def offset_on_rise(chosen):
    """True when the offset event is a rising edge rather than a falling edge."""
    if chosen["method"] == "alternate":
        return True
    if chosen["method"] == "cross" and "falling" not in chosen["detail"]:
        return True
    return False


def mark_sets(plotted, chosen):
    """Sample indices to paint as onset (red) or offset (green) on each row."""
    onset = [[] for _ in plotted]
    offset_rise = [[] for _ in plotted]
    offset_fall = [[] for _ in plotted]
    on_rise = offset_on_rise(chosen)
    onset_line = chosen["onset_line"]
    offset_line = chosen["offset_line"]
    for start, stop, _length in chosen["intervals"]:
        for ch, line in enumerate(plotted):
            if same_line(line, onset_line) or onset_line["synthetic"]:
                if nearest(line["rise"], start, 1):
                    onset[ch].append(start)
            if same_line(line, offset_line):
                pool = line["rise"] if on_rise else line["fall"]
                if nearest(pool, stop, 1):
                    (offset_rise if on_rise else offset_fall)[ch].append(stop)
            elif offset_line["synthetic"] and nearest(line["fall"], stop, 1):
                offset_fall[ch].append(stop)
    return onset, offset_rise, offset_fall


def draw_edges(ax, row, samples, sr, onset, offset, plain_color, half):
    if samples.size == 0:
        return
    minutes = samples.astype(np.float64) / sr / 60.0
    kind = np.zeros(samples.size, dtype=np.int8)
    for index, sample in enumerate(samples):
        if nearest(onset, int(sample), 0):
            kind[index] = 1
        elif nearest(offset, int(sample), 0):
            kind[index] = 2
    if np.any(kind == 0):
        ax.vlines(
            minutes[kind == 0],
            row - half,
            row + half,
            color=plain_color,
            linewidth=0.8,
            zorder=2,
        )
    if np.any(kind == 2):
        ax.vlines(
            minutes[kind == 2],
            row - half,
            row + half,
            color="#1e8449",
            linewidth=1.3,
            zorder=3,
        )
    if np.any(kind == 1):
        ax.vlines(
            minutes[kind == 1],
            row - half,
            row + half,
            color="#c0392b",
            linewidth=1.4,
            zorder=4,
        )


def draw_audio_bars(ax, intervals, sr):
    onsets = np.array([item[0] / sr / 60.0 for item in intervals], dtype=np.float64)
    widths = np.array([item[2] / 60.0 for item in intervals], dtype=np.float64)
    ys = np.arange(1, len(intervals) + 1)
    ax.barh(ys, widths, left=onsets, height=0.62, color="#1f4e79", zorder=2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([str(y) for y in ys], fontsize=7)
    ax.set_ylim(len(intervals) + 0.7, 0.3)
    ax.set_xlabel("Time from the start of the merged recording (minutes)")
    ax.set_ylabel("audio")
    ax.grid(True, axis="x", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Estimated audio intervals, onset to offset", fontsize=10, loc="left")


def render_plot(path, info, chosen):
    """Every rising and falling edge, with the 25 audios marked when known."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sr = info["sr"]
    duration_min = info["n_time"] / sr / 60.0
    if duration_min <= 0:
        duration_min = 1.0
    plotted = [line for line in info["lines"] if not line["synthetic"]]
    n_ch = len(plotted)
    intervals = [] if chosen is None else chosen["intervals"]
    if chosen is None:
        onset = [[] for _ in plotted]
        offset_rise = [[] for _ in plotted]
        offset_fall = [[] for _ in plotted]
    else:
        onset, offset_rise, offset_fall = mark_sets(plotted, chosen)
    fig_h = max(4.6, 0.48 * n_ch + 1.7) + (5.6 if intervals else 0.3)
    fig = plt.figure(figsize=(12.8, fig_h), constrained_layout=True)
    if intervals:
        grid = fig.add_gridspec(2, 1, height_ratios=[max(2.4, 0.48 * n_ch), 5.4])
        ax = fig.add_subplot(grid[0, 0])
        ax_audio = fig.add_subplot(grid[1, 0], sharex=ax)
        ax.tick_params(labelbottom=False)
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax_audio = None
    for ch, line in enumerate(plotted):
        ax.hlines(ch, 0, duration_min, color="0.90", linewidth=0.8, zorder=0)
        draw_edges(ax, ch, line["fall"], sr, [], offset_fall[ch], "0.62", 0.28)
        draw_edges(ax, ch, line["rise"], sr, onset[ch], offset_rise[ch], "#1f4e79", 0.36)
    ax.set_yticks(list(range(n_ch)))
    ax.set_yticklabels([line["label"] for line in plotted])
    # Channel 1 is the top row, matching the edge table.
    ax.set_ylim(n_ch - 0.35, -0.75)
    pad = max(0.04, duration_min * 0.012)
    ax.set_xlim(-pad, duration_min + pad)
    ax.set_ylabel("board_dig_in channel")
    ax.grid(True, axis="x", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)
    handles = [
        Line2D([0], [0], color="#1f4e79", lw=1.6, label="rising edge"),
        Line2D([0], [0], color="0.62", lw=1.6, label="falling edge"),
    ]
    title = (
        f"{SUBJECT}  board_dig_in triggers\n"
        f"{info['n_time'] / sr / 60.0:.2f} min from the start of the merged file"
    )
    if intervals:
        handles.extend(
            [
                Line2D([0], [0], color="#c0392b", lw=1.8, label="audio onset"),
                Line2D([0], [0], color="#1e8449", lw=1.8, label="audio offset"),
            ]
        )
        lengths = [item[2] for item in intervals]
        title += (
            f"\n25 audios; first three lengths "
            f"{lengths[0]:.2f} s, {lengths[1]:.2f} s, {lengths[2]:.2f} s"
        )
        draw_audio_bars(ax_audio, intervals, sr)
    else:
        title += "\naudio onsets were not identified from the length anchors"
        ax.set_xlabel("Time from the start of the merged recording (minutes)")
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(title, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def output_paths(out_dir):
    out_dir = Path(out_dir)
    return {
        "png": out_dir / f"{SUBJECT}_triggers.png",
        "csv": out_dir / f"{SUBJECT}_audio_onsets.csv",
        "txt": out_dir / f"{SUBJECT}_audio_onsets.txt",
        "candidates": out_dir / f"{SUBJECT}_audio_onset_candidates.txt",
    }


def analyze(info):
    matches = locate_audios(info["lines"], info["sr"])
    groups = cluster_matches(matches, info["sr"])
    if len(groups) == 1:
        status = "ok"
    elif len(groups) > 1:
        status = "ambiguous"
    else:
        status = "none"
    return {"status": status, "groups": groups}


def emit(path, info, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir)
    result = analyze(info)
    status = result["status"]
    chosen = result["groups"][0][0] if status == "ok" else None
    render_plot(paths["png"], info, chosen)
    rows = header_lines(path, info)
    rows.extend(line_summary(info["lines"], info["sr"]))
    rows.append("")
    if status == "ok":
        rows.append(f"[OK] 25 audio onsets on the time axis of {Path(path).name}")
        rows.append("")
        rows.extend(format_match(chosen, info["sr"]))
        others = result["groups"][0][1:]
        if others:
            rows.append("")
            rows.append("The same 25 times also come from:")
            for other in others:
                rows.append(f"  {other['detail']}")
        rows.append("")
        rows.append(f"[WROTE] {paths['png']}")
        rows.append(f"[WROTE] {paths['csv']}")
        rows.append(f"[WROTE] {paths['txt']}")
        write_csv(paths["csv"], chosen["intervals"], info["sr"])
        if paths["candidates"].exists():
            paths["candidates"].unlink()
    elif status == "ambiguous":
        rows.append(
            "[STOP] more than one set of 25 audios matches the first three "
            "lengths, and the onset times disagree. No onset table was written."
        )
        rows.append("")
        for index, group in enumerate(result["groups"], start=1):
            rows.append(f"candidate {index}")
            rows.extend(format_match(group[0], info["sr"]))
            rows.append("")
        rows.append(f"[WROTE] {paths['png']}")
        rows.append(f"[WROTE] {paths['candidates']}")
        paths["candidates"].write_text("\n".join(rows) + "\n", encoding="utf-8")
        if paths["csv"].exists():
            paths["csv"].unlink()
    else:
        rows.append(
            "[STOP] no run of 25 trigger intervals has first lengths in "
            f"{WINDOWS[0][0]:.2f}-{WINDOWS[0][1]:.2f} s, "
            f"{WINDOWS[1][0]:.2f}-{WINDOWS[1][1]:.2f} s, and "
            f"{WINDOWS[2][0]:.2f}-{WINDOWS[2][1]:.2f} s."
        )
        rows.append("The trigger plot was still written.")
        rows.append(f"[WROTE] {paths['png']}")
        if paths["csv"].exists():
            paths["csv"].unlink()
        if paths["candidates"].exists():
            paths["candidates"].unlink()
    text = "\n".join(rows) + "\n"
    paths["txt"].write_text(text, encoding="utf-8")
    print(text, end="")
    if status == "ok":
        return 0
    if status == "ambiguous":
        return 2
    return 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        return 0
    mat = Path(argv[0]) if argv else MERGED
    out_dir = Path(argv[1]) if len(argv) > 1 else OUT_DIR
    try:
        info = read_digital(mat)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[STOP] {exc}")
        return 1
    return emit(mat, info, out_dir)


if __name__ == "__main__":
    sys.exit(main())
