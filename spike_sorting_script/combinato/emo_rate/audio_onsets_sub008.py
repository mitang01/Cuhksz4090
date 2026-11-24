#!/usr/bin/env python3
"""Find the audio onsets in the merged sub-008 recording.

Reads the merged digital input:

  /share/workspace3/ieeg/micro/audio_emotion_rating_pcm_1/sub-008/sub-008.mat

Writes next to the Combinato results:

  /share/home/mitan/combinato_result/emo_rate/sub-008/sub-008_triggers.png
  /share/home/mitan/combinato_result/emo_rate/sub-008/sub-008_audio_onsets.csv
  /share/home/mitan/combinato_result/emo_rate/sub-008/sub-008_audio_onsets.txt
  /share/home/mitan/combinato_result/emo_rate/sub-008/sub-008_trigger_events.csv

Time zero is the first sample of that merged file.

Each trigger is one parallel-port byte, read down the digital channels at
one moment. Channel 1 is bit 0 (value 1), channel 2 is bit 1 (value 2),
and so on. Channels that rise within 1 ms are the same byte. A code from
1 to 25 is an audio onset. Any other code is listed with the events and
is not an audio onset.

Run from the emo_rate directory on the cluster:

  python audio_onsets_sub008.py
  python audio_onsets_sub008.py /path/to/sub-008.mat /path/to/output_dir

Exit 0 when at least one audio code is found. Exit 1 when the file cannot
be read or no code in 1..25 appears. The trigger plot is written whenever
the mat file can be read.
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
# Port bytes 1..25 are the audio onsets. Channel 1 is the least significant bit.
AUDIO_CODE_MIN = 1
AUDIO_CODE_MAX = 25
# Rising edges this close belong to one port write.
CLUSTER_S = 0.001


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


def physical_lines(lines):
    """One row per digital input. Unpacked bits and the OR line are not rows."""
    return [line for line in lines if line["bit"] is None and not line["synthetic"]]


def vertical_events(lines, sr):
    """One port byte for each moment when one or more channels rise together.

    Channel 1 is bit 0. Edges within CLUSTER_S are one write to the port.
    """
    stamps = []
    for line in physical_lines(lines):
        channel = int(line["channel"])
        for sample in np.asarray(line["rise"], dtype=np.int64):
            stamps.append((int(sample), channel))
    stamps.sort()
    if not stamps:
        return []
    tol = max(1, int(round(CLUSTER_S * sr)))
    groups = [[stamps[0]]]
    for sample, channel in stamps[1:]:
        if sample - groups[-1][0][0] <= tol:
            groups[-1].append((sample, channel))
        else:
            groups.append([(sample, channel)])
    events = []
    for group in groups:
        channels = sorted({channel for _sample, channel in group})
        code = 0
        for channel in channels:
            if 1 <= channel <= 16:
                code |= 1 << (channel - 1)
        events.append(
            {
                "sample": min(sample for sample, _channel in group),
                "channels": channels,
                "code": int(code),
            }
        )
    return events


def is_audio(event):
    return AUDIO_CODE_MIN <= int(event["code"]) <= AUDIO_CODE_MAX


def channel_text(channels):
    return "+".join(str(channel) for channel in channels)


def pulse_width_s(lines, sr):
    """Median rising-to-falling width, in seconds, across the physical lines."""
    widths = []
    for line in physical_lines(lines):
        falls = np.asarray(line["fall"], dtype=np.int64)
        j = 0
        for onset in np.asarray(line["rise"], dtype=np.int64):
            while j < falls.size and int(falls[j]) <= int(onset):
                j += 1
            if j >= falls.size:
                break
            widths.append((int(falls[j]) - int(onset)) / sr)
            j += 1
    if not widths:
        return None
    return float(np.median(np.asarray(widths, dtype=np.float64)))


def line_summary(lines):
    rows = ["board_dig_in rising edges, counted on the merged file:"]
    for line in physical_lines(lines):
        rows.append(f"  {line['label']}: {line['rise'].size} rising, {line['fall'].size} falling")
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
        "code: channel 1 is bit 0 (value 1), channel 2 is bit 1 (value 2), and so on",
        f"one code: rising edges within {CLUSTER_S * 1000:.0f} ms",
        f"audio onset: port code {AUDIO_CODE_MIN} through {AUDIO_CODE_MAX}",
        "audio length: not estimated; each code is a short pulse at the onset",
        "",
    ]


def event_table(events, sr, numbered_as):
    header = (
        f"{numbered_as:>5}  {'code':>6}  {'onset_s':>12}  "
        f"{'onset_sample':>14}  channels"
    )
    rows = [header]
    for index, event in enumerate(events, start=1):
        rows.append(
            f"{index:5d}  {event['code']:6d}  {event['sample'] / sr:12.6f}  "
            f"{event['sample']:14d}  {channel_text(event['channels'])}"
        )
    return rows


def write_event_csv(path, events, sr, index_name):
    lines = [f"{index_name},code,onset_s,onset_sample,channels"]
    for index, event in enumerate(events, start=1):
        lines.append(
            f"{index},{event['code']},{event['sample'] / sr:.6f},"
            f"{event['sample']},{channel_text(event['channels'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def nearest(samples, target, tol):
    """Return the stored sample within tol of target, or None."""
    if len(samples) == 0:
        return None
    samples = np.sort(np.asarray(samples, dtype=np.int64))
    index = int(np.searchsorted(samples, target))
    found = None
    best = tol + 1
    for pos in (index - 1, index):
        if 0 <= pos < samples.size:
            gap = abs(int(samples[pos]) - int(target))
            if gap <= tol and gap < best:
                best = gap
                found = int(samples[pos])
    return found


def audio_marks(plotted, audios, sr):
    tol = max(1, int(round(CLUSTER_S * sr)))
    marks = [[] for _ in plotted]
    for event in audios:
        for row, line in enumerate(plotted):
            if int(line["channel"]) not in event["channels"]:
                continue
            hit = nearest(line["rise"], event["sample"], tol)
            if hit is not None:
                marks[row].append(hit)
    return marks


def draw_edges(ax, row, samples, sr, marked, plain_color, mark_color, half):
    if samples.size == 0:
        return
    minutes = samples.astype(np.float64) / sr / 60.0
    chosen = np.zeros(samples.size, dtype=bool)
    for index, sample in enumerate(samples):
        if nearest(marked, int(sample), 0) is not None:
            chosen[index] = True
    if np.any(~chosen):
        ax.vlines(
            minutes[~chosen],
            row - half,
            row + half,
            color=plain_color,
            linewidth=0.8,
            zorder=2,
        )
    if np.any(chosen):
        ax.vlines(
            minutes[chosen],
            row - half,
            row + half,
            color=mark_color,
            linewidth=1.4,
            zorder=4,
        )


def draw_audio_onsets(ax, audios, sr):
    minutes = np.array([event["sample"] / sr / 60.0 for event in audios], dtype=np.float64)
    ys = np.arange(1, len(audios) + 1)
    ax.plot(minutes, ys, linestyle="None", marker="|", markersize=14, color="#c0392b", zorder=2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([str(event["code"]) for event in audios], fontsize=7)
    ax.set_ylim(len(audios) + 0.7, 0.3)
    ax.set_xlabel("Time from the start of the merged recording (minutes)")
    ax.set_ylabel("port code")
    ax.grid(True, axis="x", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Audio onsets, one port code read down the channels", fontsize=10, loc="left")


def render_plot(path, info, audios):
    """Every edge. Red ticks are the bits of an audio code."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sr = info["sr"]
    duration_min = info["n_time"] / sr / 60.0
    if duration_min <= 0:
        duration_min = 1.0
    plotted = physical_lines(info["lines"])
    n_ch = len(plotted)
    marks = audio_marks(plotted, audios, sr) if audios else [[] for _ in plotted]
    fig_h = max(4.6, 0.48 * n_ch + 1.7) + (5.6 if audios else 0.3)
    fig = plt.figure(figsize=(12.8, fig_h), constrained_layout=True)
    if audios:
        grid = fig.add_gridspec(2, 1, height_ratios=[max(2.4, 0.48 * n_ch), 5.4])
        ax = fig.add_subplot(grid[0, 0])
        ax_audio = fig.add_subplot(grid[1, 0], sharex=ax)
        ax.tick_params(labelbottom=False)
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax_audio = None
    for row, line in enumerate(plotted):
        ax.hlines(row, 0, duration_min, color="0.90", linewidth=0.8, zorder=0)
        draw_edges(ax, row, line["fall"], sr, [], "0.62", "#c0392b", 0.28)
        draw_edges(ax, row, line["rise"], sr, marks[row], "#1f4e79", "#c0392b", 0.36)
    ax.set_yticks(list(range(n_ch)))
    ax.set_yticklabels([line["label"] for line in plotted])
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
    if audios:
        handles.append(Line2D([0], [0], color="#c0392b", lw=1.8, label="audio onset"))
        codes = [event["code"] for event in audios]
        title += (
            f"\n{len(audios)} audio onsets, port codes {min(codes)}-{max(codes)}"
        )
        draw_audio_onsets(ax_audio, audios, sr)
    else:
        title += "\nno port code in 1..25"
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
        "events": out_dir / f"{SUBJECT}_trigger_events.csv",
        "txt": out_dir / f"{SUBJECT}_audio_onsets.txt",
        "candidates": out_dir / f"{SUBJECT}_audio_onset_candidates.txt",
    }


def emit(path, info, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir)
    events = vertical_events(info["lines"], info["sr"])
    audios = [event for event in events if is_audio(event)]
    others = [event for event in events if not is_audio(event)]
    render_plot(paths["png"], info, audios)
    rows = header_lines(path, info)
    rows.extend(line_summary(info["lines"]))
    width = pulse_width_s(info["lines"], info["sr"])
    if width is not None:
        rows.append(f"median pulse width: {width * 1000:.2f} ms")
    rows.append("")
    rows.append(f"vertical events: {len(events)}")
    rows.append(f"audio codes {AUDIO_CODE_MIN}..{AUDIO_CODE_MAX}: {len(audios)}")
    present = {event["code"] for event in audios}
    missing = [code for code in range(AUDIO_CODE_MIN, AUDIO_CODE_MAX + 1) if code not in present]
    if audios and missing:
        shown = ", ".join(str(code) for code in missing)
        rows.append(f"codes in that range with no pulse: {shown}")
    rows.append("")
    if audios:
        rows.append("audio onsets, in the order the pulses arrived:")
        rows.extend(event_table(audios, info["sr"], "audio"))
        rows.append("")
    if others:
        rows.append("other port codes:")
        rows.extend(event_table(others, info["sr"], "event"))
        rows.append("")
    rows.append(f"[WROTE] {paths['png']}")
    if audios:
        rows.append(f"[OK] {len(audios)} audio onsets on the time axis of {Path(path).name}")
        rows.append(f"[WROTE] {paths['csv']}")
        write_event_csv(paths["csv"], audios, info["sr"], "audio")
    elif paths["csv"].exists():
        paths["csv"].unlink()
    if events:
        rows.append(f"[WROTE] {paths['events']}")
        write_event_csv(paths["events"], events, info["sr"], "event")
    elif paths["events"].exists():
        paths["events"].unlink()
    rows.append(f"[WROTE] {paths['txt']}")
    if paths["candidates"].exists():
        paths["candidates"].unlink()
    if not audios:
        rows.append("[STOP] no rising-edge code in 1..25. The event list above is the full port record.")
    text = "\n".join(rows) + "\n"
    paths["txt"].write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if audios else 1


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
