#!/usr/bin/env python3
"""Plot spike rate against prosodic boundary-strength quartile.

Uses the positive Combinato groups in sort_pos_simple. A spike counts when
it falls in the 0–200 ms after a word boundary. Boundaries come from
storyNN.prosodic_word_depth.tsv. Their times are placed on the recording
clock by matching story durations to board_dig_in_data.

Outputs are written under
/share/home/mitan/combinato_result/boundary_strength_spike_rate/
"""

import csv
import os
import re
import wave
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = Path("/share/workspace3/ieeg/micro/story_listen_v2")
RESULT_ROOT = Path("/share/home/mitan/combinato_result")
STORY_ROOT = Path("/share/workspace3/ieeg/seeg/story_listen_v2")
PROSODY_ROOT = Path("/share/home/mitan/seeg_story_listen_v2/prosodic_word_depth")
SUBJECTS = ["sub-007", "sub-008", "sub-010", "sub-011"]
SR_DEFAULT = 30000.0
WINDOW_S = 0.2
BLOCK = 1000000


def story_key(name):
    stem = Path(name).name
    lower = stem.lower()
    for suffix in (
        ".prosodic_word_depth.tsv",
        ".textgrid",
        ".wav",
        ".wave",
        ".tsv",
    ):
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        stem = Path(stem).stem
    compact = re.sub(r"[\s_.-]+", "", stem).lower()
    match = re.match(r"^(.*?)(\d+)$", compact)
    if match and match.group(1):
        return (match.group(1), int(match.group(2)))
    return (compact, None)


def read_table(path, delimiter):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"empty table: {path}")
    header = [cell.strip() for cell in rows[0]]
    records = []
    for raw in rows[1:]:
        if not any(cell.strip() for cell in raw):
            continue
        records.append(
            {header[i]: (raw[i].strip() if i < len(raw) else "") for i in range(len(header))}
        )
    return header, records


def column(header, options):
    lookup = {re.sub(r"[\s_.-]+", "", name).lower(): name for name in header}
    for option in options:
        key = re.sub(r"[\s_.-]+", "", option).lower()
        if key in lookup:
            return lookup[key]
    return None


def index_files(root, suffixes):
    found = {}
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            found.setdefault(story_key(path.name), path)
    return found


def wav_duration(path):
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / float(audio.getframerate())
    except (wave.Error, EOFError):
        pass
    try:
        import soundfile as sf
    except ImportError as error:
        raise ValueError(f"cannot read duration of {path}") from error
    return float(sf.info(str(path)).duration)


def story_order(event_csv):
    header, rows = read_table(event_csv, ",")
    label_name = column(header, ("trigger_label", "trigger", "event", "label"))
    file_name = column(header, ("stimuli_filename", "stimulus", "audio", "filename", "wav"))
    silence_name = column(header, ("silence", "silence_s"))
    if label_name is None or file_name is None:
        raise ValueError(f"{event_csv} needs a trigger column and an audio column")
    stories = []
    seen = set()
    for row in rows:
        label = row[label_name]
        audio_name = row[file_name]
        if not label or not audio_name:
            continue
        lowered = label.lower().replace(" ", "")
        if "offset" in lowered or lowered.startswith("off"):
            continue
        if "onset" not in lowered and not lowered.startswith("on"):
            continue
        key = story_key(audio_name)
        if key in seen:
            continue
        seen.add(key)
        numbers = re.findall(r"\d+", label)
        order = int(numbers[-1]) if numbers else len(stories) + 1
        silence = float(row[silence_name]) if silence_name and row[silence_name] else 0.0
        stories.append((order, key, audio_name, silence))
    stories.sort(key=lambda item: item[0])
    if not stories:
        raise ValueError(f"no onset rows in {event_csv}")
    return stories


def load_boundaries(path):
    header, rows = read_table(path, "\t")
    time_name = column(header, ("boundary_time_s", "offset_s", "end"))
    strength_name = column(header, ("boundary_strength_after", "boundary_strength"))
    if time_name is None or strength_name is None:
        raise ValueError(f"{path} needs boundary time and boundary strength columns, found {header}")
    times = []
    strengths = []
    for row in rows:
        if not row[time_name] or not row[strength_name]:
            continue
        times.append(float(row[time_name]))
        strengths.append(float(row[strength_name]))
    if not times:
        raise ValueError(f"no boundaries in {path}")
    return np.asarray(times, dtype=float), np.asarray(strengths, dtype=float)


def assign_quartiles(strength):
    """1 is the lowest boundary strength and 4 is the highest."""
    edges = np.quantile(strength, [0.25, 0.5, 0.75])
    quartile = np.ones(len(strength), dtype=int)
    quartile[strength > edges[0]] = 2
    quartile[strength > edges[1]] = 3
    quartile[strength > edges[2]] = 4
    return quartile


def sample_rate(h5):
    if "frequency_parameters" in h5 and "amplifier_sample_rate" in h5["frequency_parameters"]:
        return float(np.array(h5["frequency_parameters"]["amplifier_sample_rate"]).ravel()[0])
    return SR_DEFAULT


def time_axis(shape):
    if len(shape) != 2:
        raise ValueError(f"digital data is not 2D: {shape}")
    if shape[0] < shape[1]:
        return 1
    return 0


def channel_signal(dataset, channel, axis, bit):
    rises = []
    durations = []
    high_start = None
    previous = 0
    n = dataset.shape[axis]
    pos = 0
    while pos < n:
        take = min(BLOCK, n - pos)
        if axis == 1:
            piece = np.asarray(dataset[channel, pos:pos + take])
        else:
            piece = np.asarray(dataset[pos:pos + take, channel])
        if bit is None:
            signal = piece > 0
        else:
            signal = (piece.astype(np.int64) >> bit) & 1
        signal = np.asarray(signal, dtype=np.int8)
        changes = np.diff(signal, prepend=previous)
        for index in np.flatnonzero(changes != 0):
            sample = pos + int(index)
            if changes[index] == 1:
                rises.append(sample)
                high_start = sample
            elif high_start is not None:
                durations.append((high_start, sample - high_start))
                high_start = None
        previous = int(signal[-1])
        pos += take
    if high_start is not None:
        durations.append((high_start, n - high_start))
    return np.asarray(rises, dtype=np.int64), durations


def digital_candidates(path):
    candidates = []
    with h5py.File(path, "r") as h5:
        if "board_dig_in_data" not in h5:
            raise KeyError(f"{path} has no board_dig_in_data")
        data = h5["board_dig_in_data"]
        sr = sample_rate(h5)
        axis = time_axis(tuple(int(x) for x in data.shape))
        n_channels = data.shape[0] if axis == 1 else data.shape[1]
        probe_n = min(data.shape[axis], 20000)
        if axis == 1:
            probe = np.asarray(data[:, :probe_n])
        else:
            probe = np.asarray(data[:probe_n, :])
        use_bits = int(np.nanmax(probe)) > 1
        for channel in range(n_channels):
            bits = range(16) if use_bits else (None,)
            for bit in bits:
                rises, highs = channel_signal(data, channel, axis, bit)
                name = f"channel {channel}" if bit is None else f"channel {channel} bit {bit}"
                candidates.append((name, sr, rises, highs))
    return candidates


def best_alignment(durations, candidates):
    expected = np.asarray(durations, dtype=float)
    best = None
    for name, sr, rises, highs in candidates:
        rise_s = rises / sr
        options = []
        if len(rise_s) >= 2:
            options.append(("intervals", rise_s[:-1], np.diff(rise_s)))
        if highs:
            high_onset = np.asarray([item[0] for item in highs], dtype=float) / sr
            high_dur = np.asarray([item[1] for item in highs], dtype=float) / sr
            options.append(("high-periods", high_onset, high_dur))
        for kind, onsets, observed in options:
            n = len(expected)
            if len(observed) < n:
                continue
            for start in range(len(observed) - n + 1):
                error = np.abs(observed[start:start + n] - expected)
                median_error = float(np.median(error))
                if best is None or median_error < best["median_error"]:
                    best = {
                        "name": name,
                        "kind": kind,
                        "median_error": median_error,
                        "onsets": onsets[start:start + n],
                        "observed": observed[start:start + n],
                    }
    if best is None:
        raise ValueError("no digital channel has enough pulses for the story list")
    limit = max(1.0, 0.05 * float(np.median(expected)))
    if best["median_error"] > limit:
        raise ValueError(
            f"closest digital match is {best['name']} {best['kind']} "
            f"with median error {best['median_error']:.3f} s"
        )
    return best


def group_times(sort_path, data_path):
    with h5py.File(sort_path, "r") as sorting, h5py.File(data_path, "r") as data:
        classes = np.asarray(sorting["classes"]).ravel().astype(np.int64)
        index = np.asarray(sorting["index"]).ravel().astype(np.int64)
        groups = np.asarray(sorting["groups"])
        times = np.asarray(data["pos"]["times"]).ravel().astype(float) / 1000.0
    if classes.shape != index.shape:
        raise ValueError(f"{sort_path} classes and index have different lengths")
    if groups.ndim == 1:
        if groups.shape[0] != classes.shape[0]:
            raise ValueError(f"{sort_path} has an unexpected groups array")
        group_per_spike = groups.astype(np.int64)
    else:
        if groups.shape[0] == 2 and groups.shape[1] != 2:
            groups = groups.T
        class_id = groups[:, 0].astype(np.int64)
        group_id = groups[:, 1].astype(np.int64)
        order = np.argsort(class_id)
        class_id = class_id[order]
        group_id = group_id[order]
        position = np.clip(np.searchsorted(class_id, classes), 0, len(class_id) - 1)
        matched = class_id[position] == classes
        group_per_spike = np.full(classes.shape, -999, dtype=np.int64)
        group_per_spike[matched] = group_id[position][matched]
    result = {}
    valid = (index >= 0) & (index < len(times))
    for gid in np.unique(group_per_spike):
        if gid == -999:
            continue
        chosen = valid & (group_per_spike == gid)
        result[int(gid)] = np.sort(times[index[chosen]])
    return result


def rates_for_group(spike_times, boundary_times):
    if len(spike_times) == 0:
        return np.zeros(len(boundary_times), dtype=float)
    stop = np.searchsorted(spike_times, boundary_times + WINDOW_S, side="left")
    start = np.searchsorted(spike_times, boundary_times, side="left")
    return (stop - start) / WINDOW_S


def group_name(gid):
    if gid == -1:
        return "artifact"
    if gid == 0:
        return "unassigned"
    return f"group {gid}"


def mean_by_quartile(rates, quartile):
    rows = []
    for level in (1, 2, 3, 4):
        chosen = rates[quartile == level]
        if len(chosen) == 0:
            mean = np.nan
            sem = np.nan
        else:
            mean = float(np.mean(chosen))
            sem = float(np.std(chosen, ddof=1) / np.sqrt(len(chosen))) if len(chosen) > 1 else 0.0
        rows.append((level, len(chosen), mean, sem))
    return rows


def plot_subject(subject, panels, strength, quartile, out_path):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.2), squeeze=False)
    for axis, (channel, region, lines) in zip(axes[0], panels):
        for gid, summary in lines:
            levels = [item[0] for item in summary]
            means = [item[2] if np.isfinite(item[2]) else 0.0 for item in summary]
            sems = [item[3] if np.isfinite(item[3]) else 0.0 for item in summary]
            axis.errorbar(levels, means, yerr=sems, marker="o", capsize=3, label=group_name(gid))
        labels = []
        for level in (1, 2, 3, 4):
            values = strength[quartile == level]
            if len(values) == 0:
                labels.append(f"Q{level}")
            else:
                labels.append(f"Q{level}\n{values.min():.3g}–{values.max():.3g}")
        axis.set_xticks([1, 2, 3, 4], labels)
        axis.set_xlabel("Boundary strength quartile")
        axis.set_ylabel("Spike rate (Hz)")
        title = f"{channel} {region}".strip()
        axis.set_title(title)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{subject}: spike rate 0–200 ms after the boundary")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def region_by_channel(merged):
    with h5py.File(merged, "r") as h5:
        numbers = [int(x) for x in h5["channel_number"][:]]
        if "brain_region" not in h5:
            return {number: "" for number in numbers}
        regions = []
        for item in h5["brain_region"][:]:
            regions.append(item.decode() if isinstance(item, bytes) else str(item))
    return dict(zip(numbers, regions))


def prepare_boundaries(story_root, prosody_root, merged):
    event_csv = story_root / "event_stimuli.csv"
    stories = story_order(event_csv)
    audio_files = index_files(story_root, {".wav", ".wave"})
    prosody_files = index_files(prosody_root, {".tsv"})
    durations = []
    story_rows = []
    for order, key, audio_name, silence in stories:
        audio = audio_files.get(key)
        prosody = prosody_files.get(key)
        if audio is None or prosody is None:
            raise FileNotFoundError(f"missing audio or prosody for {audio_name}")
        duration = wav_duration(audio)
        durations.append(duration)
        story_rows.append(
            {
                "order": order,
                "story": audio_name,
                "audio": str(audio),
                "prosody": str(prosody),
                "duration_s": duration,
                "silence_s": silence,
            }
        )
    alignment = best_alignment(durations, digital_candidates(merged))
    print(
        f"[ALIGN] {merged.parent.name}: {alignment['name']} {alignment['kind']} "
        f"median error {alignment['median_error']:.3f} s"
    )
    boundary_time = []
    boundary_strength = []
    boundary_story = []
    for row, onset, observed in zip(story_rows, alignment["onsets"], alignment["observed"]):
        row["onset_s"] = float(onset)
        row["digital_duration_s"] = float(observed)
        times, strengths = load_boundaries(Path(row["prosody"]))
        kept = (times >= 0) & (times <= row["duration_s"] + 0.05)
        times = times[kept]
        strengths = strengths[kept]
        boundary_time.append(times + onset)
        boundary_strength.append(strengths)
        boundary_story.extend([row["story"]] * len(times))
        print(f"  {row['story']}: onset {onset:.3f} s, boundaries {len(times)}")
    if not boundary_time:
        raise ValueError("no boundaries fell inside the story durations")
    return story_rows, np.concatenate(boundary_time), np.concatenate(boundary_strength), boundary_story


def sorted_channels(result_root, subject):
    subject_dir = result_root / subject
    if not subject_dir.is_dir():
        return []
    return sorted(subject_dir.glob("A-*/sort_pos_simple/sort_cat.h5"))


def analyze_subject(subject, boundary_time, strength, quartile, result_root, data_root):
    channels = sorted_channels(result_root, subject)
    if not channels:
        print(f"[SKIP] {subject}: no sort_pos_simple results")
        return []
    merged = data_root / subject / f"{subject}.mat"
    regions = region_by_channel(merged) if merged.is_file() else {}
    panels = []
    rate_rows = []
    for sort_path in channels:
        channel_dir = sort_path.parents[1]
        channel = channel_dir.name
        number = int(channel.split("-")[-1])
        region = regions.get(number, "")
        data_path = channel_dir / f"data_{channel}.h5"
        groups = group_times(sort_path, data_path)
        lines = []
        for gid in sorted(groups):
            rates = rates_for_group(groups[gid], boundary_time)
            summary = mean_by_quartile(rates, quartile)
            lines.append((gid, summary))
            for level, count, mean, sem in summary:
                rate_rows.append(
                    {
                        "subject": subject,
                        "channel": channel,
                        "region": region,
                        "group": group_name(gid),
                        "quartile": level,
                        "n_boundaries": count,
                        "mean_rate_hz": mean,
                        "sem_rate_hz": sem,
                    }
                )
        panels.append((channel, region, lines))
        print(f"[UNIT] {subject} {channel} {region}: {len(lines)} groups")
    out_dir = result_root / "boundary_strength_spike_rate"
    plot_subject(subject, panels, strength, quartile, out_dir / f"{subject}_boundary_strength_spike_rate.png")
    print(f"[PLOT] {out_dir / f'{subject}_boundary_strength_spike_rate.png'}")
    return rate_rows


def main(story_root=STORY_ROOT, prosody_root=PROSODY_ROOT, data_root=DATA_ROOT, result_root=RESULT_ROOT):
    out_dir = result_root / "boundary_strength_spike_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rates = []
    alignment_rows = []
    boundary_rows = []
    for subject in SUBJECTS:
        merged = data_root / subject / f"{subject}.mat"
        if not merged.is_file():
            print(f"[SKIP] {subject}: merged file not found")
            continue
        if not sorted_channels(result_root, subject):
            print(f"[SKIP] {subject}: no sort_pos_simple results")
            continue
        try:
            rows, times, strengths, names = prepare_boundaries(story_root, prosody_root, merged)
        except Exception as exc:
            print(f"[STOP] {subject}: {exc}")
            continue
        levels = assign_quartiles(strengths)
        for row in rows:
            copied = dict(row)
            copied["subject"] = subject
            alignment_rows.append(copied)
        for name, time_s, value, level in zip(names, times, strengths, levels):
            boundary_rows.append(
                {
                    "subject": subject,
                    "story": name,
                    "boundary_time_s": float(time_s),
                    "boundary_strength": float(value),
                    "quartile": int(level),
                }
            )
        all_rates.extend(
            analyze_subject(subject, times, strengths, levels, result_root, data_root)
        )
    write_csv(out_dir / "story_alignment.csv", alignment_rows)
    write_csv(out_dir / "boundaries.csv", boundary_rows)
    write_csv(out_dir / "spike_rates.csv", all_rates)
    if not all_rates:
        raise SystemExit("no sorted channels were found")
    print(f"[WROTE] {out_dir}")


if __name__ == "__main__":
    main()
