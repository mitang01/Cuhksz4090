#!/usr/bin/env python3
"""Preprocess story-listening sEEG and identify speech-responsive contacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import mne
import numpy as np
from scipy import signal, stats
from scipy.io import wavfile


DEFAULT_INPUT = Path("/share/workspace3/ieeg/seeg/story_listen_v2")
DEFAULT_OUTPUT = Path("/share/home/mitan/seeg_story_listen_v2")
DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 70.0),
    "high_gamma": (70.0, 150.0),
}
TIME_ALIASES = ("time", "timestamp", "seconds", "sec", "latency", "event_time")
LABEL_ALIASES = (
    "trigger",
    "trigger_label",
    "event",
    "event_id",
    "code",
    "value",
    "description",
)
STIMULUS_ALIASES = (
    "stimulus",
    "stimuli",
    "stimuli_filename",
    "stimulus_filename",
    "audio",
    "wav",
    "sound",
    "filename",
    "file",
    "story",
)
PHASE_ALIASES = ("phase", "type", "action", "onset_offset", "start_end")
SILENCE_ALIASES = ("silence", "silence_s", "silence_seconds")
ONSET_ALIASES = ("onset", "start", "start_time", "onset_time")
OFFSET_ALIASES = ("offset", "end", "end_time", "offset_time")
NON_SEEG_DEFAULT = r"(?i)(ecg|ekg|eog|emg|stim|trig|marker|event|status|dc|ref)"
CONTACT_RE = re.compile(r"^(.*?)[\s_.-]*([0-9]+)$")


@dataclass(frozen=True)
class TrackEvent:
    stimulus: str
    onset: float
    offset: float
    onset_trigger: str = ""
    offset_trigger: str = ""
    onset_trigger_time: float | None = None
    offset_trigger_time: float | None = None
    onset_silence_s: float = 0.0
    offset_silence_s: float = 0.0


@dataclass(frozen=True)
class MappingEntry:
    stimulus: str
    phase: str | None
    silence_s: float = 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean line noise, apply shaft-wise Laplacian referencing, extract "
            "canonical-band Hilbert amplitudes, and identify speech-responsive contacts."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-stimuli", type=Path)
    parser.add_argument("--stimuli-wav-dir", type=Path)
    parser.add_argument("--textgrid-dir", type=Path)
    parser.add_argument(
        "--bands",
        nargs="+",
        choices=tuple(DEFAULT_BANDS),
        default=list(DEFAULT_BANDS),
        help="Bands to process (default: all)",
    )
    parser.add_argument("--target-sfreq", type=float, default=128.0)
    parser.add_argument("--line-frequency", type=float, default=50.0)
    parser.add_argument("--line-harmonics", type=int, default=5)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--baseline-start", type=float, default=-1.0)
    parser.add_argument("--baseline-end", type=float, default=-0.05)
    parser.add_argument("--epoch-start", type=float, default=-1.0)
    parser.add_argument("--epoch-end", type=float, default=2.0)
    parser.add_argument("--epoch-baseline-start", type=float, default=-0.2)
    parser.add_argument("--epoch-baseline-end", type=float, default=-0.05)
    parser.add_argument("--response-start", type=float, default=0.05)
    parser.add_argument("--response-end", type=float, default=0.2)
    parser.add_argument("--fdr-q", type=float, default=0.01)
    parser.add_argument("--duration-tolerance", type=float, default=0.1)
    parser.add_argument("--channel-chunk-size", type=int, default=16)
    parser.add_argument("--exclude-channel-regex", default=NON_SEEG_DEFAULT)
    parser.add_argument(
        "--event-time-column",
        help="Override automatic event CSV time-column detection",
    )
    parser.add_argument(
        "--event-label-column",
        help="Override automatic event CSV trigger-column detection",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover files and validate event/audio/TextGrid metadata",
    )
    return parser.parse_args(argv)


def normalize(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().casefold())


def clean_stimulus(value: str) -> str:
    return Path(value.strip()).stem


def first_column(
    fieldnames: Iterable[str],
    aliases: Sequence[str],
    override: str | None = None,
) -> str | None:
    by_normalized = {normalize(name): name for name in fieldnames}
    if override:
        try:
            return by_normalized[normalize(override)]
        except KeyError as error:
            raise ValueError(
                f"column {override!r} was not found; available columns: "
                f"{', '.join(fieldnames)}"
            ) from error
    return next(
        (by_normalized[alias] for alias in aliases if alias in by_normalized), None
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]
        return list(reader.fieldnames), rows


def classify_phase(value: str) -> str | None:
    value = normalize(value)
    if re.search(r"(^|_)(onset|start|begin|play)($|_)", value):
        return "onset"
    if re.search(r"(^|_)(off|offset|stop|end|finish)($|_)", value):
        return "offset"
    return None


def infer_headerless_track_csv(
    path: Path, mapping: dict[str, MappingEntry]
) -> tuple[list[str], list[dict[str, str]]] | None:
    """Infer label/time columns when the first event row was mistaken for a header."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_rows: list[list[str]] = []
        for row in csv.reader(stream):
            if not any(value.strip() for value in row):
                continue
            # Some subject files quote the complete comma-separated row, so
            # the outer CSV parse yields one value such as
            # "onset_1, 98.4135, 0". Parse that value once more.
            if len(row) == 1 and "," in row[0]:
                nested = next(csv.reader([row[0]], skipinitialspace=True))
                if len(nested) > 1:
                    row = nested
            raw_rows.append([value.strip() for value in row])
    if not raw_rows:
        return None
    first = raw_rows[0]
    label_indices = [
        index
        for index, value in enumerate(first)
        if normalize(value) in mapping or classify_phase(value) is not None
    ]
    if len(label_indices) != 1:
        return None
    label_index = label_indices[0]
    numeric_indices: list[int] = []
    for index, value in enumerate(first):
        if index == label_index:
            continue
        try:
            if math.isfinite(float(value)):
                numeric_indices.append(index)
        except ValueError:
            pass
    if not numeric_indices:
        return None
    # In the cluster files the timestamp immediately follows the trigger label
    # (for example onset_1,16.743,0). Prefer that position over auxiliary
    # numeric columns such as the trailing zero.
    time_index = (
        label_index + 1
        if label_index + 1 in numeric_indices
        else numeric_indices[0]
    )
    width = max(len(row) for row in raw_rows)
    fields = [
        "trigger"
        if index == label_index
        else "time"
        if index == time_index
        else f"column_{index + 1}"
        for index in range(width)
    ]
    rows = [
        {
            field: (row[index].strip() if index < len(row) else "")
            for index, field in enumerate(fields)
        }
        for row in raw_rows
    ]
    return fields, rows


def load_event_stimuli(path: Path) -> dict[str, MappingEntry]:
    fields, rows = read_csv(path)
    label_column = first_column(fields, LABEL_ALIASES)
    stimulus_column = first_column(fields, STIMULUS_ALIASES)
    phase_column = first_column(fields, PHASE_ALIASES)
    silence_column = first_column(fields, SILENCE_ALIASES)
    if label_column is None or stimulus_column is None:
        raise ValueError(
            f"{path} needs trigger/event and stimulus/audio columns; found {fields}"
        )
    result: dict[str, MappingEntry] = {}
    for row in rows:
        label = normalize(row[label_column])
        stimulus = clean_stimulus(row[stimulus_column])
        if not label or not stimulus:
            continue
        phase = classify_phase(row[phase_column]) if phase_column else None
        if phase is None:
            phase = classify_phase(row[label_column])
        silence_s = (
            parse_float(row[silence_column], context=f"{path.name}:{silence_column}")
            if silence_column and row[silence_column]
            else 0.0
        )
        result[label] = MappingEntry(
            stimulus=stimulus, phase=phase, silence_s=silence_s
        )
    if not result:
        raise ValueError(f"no usable trigger-to-stimulus rows in {path}")
    return result


def parse_float(value: str, *, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"expected a numeric value for {context}, got {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"expected a finite value for {context}, got {value!r}")
    return parsed


def load_track_events(
    path: Path,
    mapping: dict[str, MappingEntry],
    *,
    time_override: str | None = None,
    label_override: str | None = None,
) -> tuple[list[TrackEvent], list[str]]:
    """Read either wide onset/offset rows or long trigger rows."""
    fields, rows = read_csv(path)
    stimulus_column = first_column(fields, STIMULUS_ALIASES)
    onset_column = first_column(fields, ONSET_ALIASES)
    offset_column = first_column(fields, OFFSET_ALIASES)
    warnings: list[str] = []

    if stimulus_column and onset_column and offset_column and onset_column != offset_column:
        events = [
            TrackEvent(
                stimulus=clean_stimulus(row[stimulus_column]),
                onset=parse_float(row[onset_column], context=onset_column),
                offset=parse_float(row[offset_column], context=offset_column),
            )
            for row in rows
            if row[stimulus_column] and row[onset_column] and row[offset_column]
        ]
        return validate_track_events(events, path), warnings

    time_column = first_column(fields, TIME_ALIASES, time_override)
    label_column = first_column(fields, LABEL_ALIASES, label_override)
    phase_column = first_column(fields, PHASE_ALIASES)
    if (
        time_column is None
        and label_column is None
        and time_override is None
        and label_override is None
    ):
        inferred = infer_headerless_track_csv(path, mapping)
        if inferred is not None:
            fields, rows = inferred
            time_column = first_column(fields, TIME_ALIASES)
            label_column = first_column(fields, LABEL_ALIASES)
            phase_column = first_column(fields, PHASE_ALIASES)
    if time_column is None or label_column is None:
        raise ValueError(
            f"{path} needs either stimulus/onset/offset columns or time/trigger "
            f"columns; found {fields}. Use --event-time-column and "
            "--event-label-column when names are nonstandard."
        )

    open_events: dict[str, tuple[float, str, float, float]] = {}
    events: list[TrackEvent] = []
    for row_number, row in enumerate(rows, start=2):
        label = normalize(row[label_column])
        if not row[time_column] or not label:
            continue
        entry = mapping.get(label)
        if entry is None:
            warnings.append(f"row {row_number}: unmapped trigger {row[label_column]!r}")
            continue
        event_time = parse_float(row[time_column], context=f"{path.name}:{row_number}")
        corrected_time = event_time + entry.silence_s
        phase = classify_phase(row[phase_column]) if phase_column else None
        phase = phase or entry.phase or classify_phase(row[label_column])
        current = open_events.get(entry.stimulus)
        if phase == "onset" or (phase is None and current is None):
            if current is not None:
                warnings.append(
                    f"row {row_number}: replaced unmatched onset for {entry.stimulus}"
                )
            open_events[entry.stimulus] = (
                corrected_time,
                row[label_column],
                event_time,
                entry.silence_s,
            )
        elif phase == "offset" or (phase is None and current is not None):
            if current is None:
                warnings.append(
                    f"row {row_number}: offset without onset for {entry.stimulus}"
                )
                continue
            (
                onset,
                onset_trigger,
                onset_trigger_time,
                onset_silence_s,
            ) = open_events.pop(entry.stimulus)
            events.append(
                TrackEvent(
                    stimulus=entry.stimulus,
                    onset=onset,
                    offset=corrected_time,
                    onset_trigger=onset_trigger,
                    offset_trigger=row[label_column],
                    onset_trigger_time=onset_trigger_time,
                    offset_trigger_time=event_time,
                    onset_silence_s=onset_silence_s,
                    offset_silence_s=entry.silence_s,
                )
            )
    for stimulus in sorted(open_events):
        warnings.append(f"unmatched onset for {stimulus}")
    return validate_track_events(events, path), warnings


def validate_track_events(events: list[TrackEvent], path: Path) -> list[TrackEvent]:
    if not events:
        raise ValueError(f"no complete onset/offset event pairs found in {path}")
    events = sorted(events, key=lambda event: event.onset)
    for event in events:
        if event.onset < 0 or event.offset <= event.onset:
            raise ValueError(f"invalid event interval in {path}: {event}")
    return events


def wav_duration(path: Path) -> float:
    sfreq, samples = wavfile.read(path, mmap=True)
    return len(samples) / float(sfreq)


def find_stimulus_file(directory: Path, stimulus: str, suffix: str) -> Path | None:
    direct = directory / f"{stimulus}{suffix}"
    if direct.is_file():
        return direct
    wanted = normalize(stimulus)
    return next(
        (
            path
            for path in directory.rglob(f"*{suffix}")
            if normalize(path.stem) == wanted
        ),
        None,
    )


def validate_audio_durations(
    events: Sequence[TrackEvent],
    wav_directory: Path,
    tolerance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        wav_path = find_stimulus_file(wav_directory, event.stimulus, ".wav")
        trigger_duration = event.offset - event.onset
        audio_duration = wav_duration(wav_path) if wav_path else None
        difference = (
            trigger_duration - audio_duration if audio_duration is not None else None
        )
        rows.append(
            {
                **asdict(event),
                "trigger_duration_s": trigger_duration,
                "wav_path": str(wav_path) if wav_path else "",
                "audio_duration_s": audio_duration,
                "difference_s": difference,
                "within_tolerance": (
                    abs(difference) <= tolerance if difference is not None else False
                ),
            }
        )
    return rows


def parse_first_interval_tier(path: Path) -> list[float]:
    """Return non-empty interval onsets from the first long-format IntervalTier."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    in_tier = False
    tier_indent = -1
    intervals: list[float] = []
    current_xmin: float | None = None
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if not in_tier and re.fullmatch(r'class\s*=\s*"IntervalTier"', stripped):
            in_tier = True
            tier_indent = indent
            continue
        if in_tier and stripped.startswith("item [") and indent < tier_indent:
            break
        if not in_tier:
            continue
        match = re.fullmatch(r"xmin\s*=\s*([-+0-9.eE]+)", stripped)
        if match:
            current_xmin = float(match.group(1))
            continue
        match = re.fullmatch(r'text\s*=\s*"(.*)"', stripped)
        if match and current_xmin is not None:
            text = match.group(1).replace('""', '"').strip()
            if text:
                intervals.append(current_xmin)
            current_xmin = None
    if not in_tier:
        raise ValueError(f"no long-format IntervalTier found in {path}")
    return intervals


def collect_token_onsets(
    events: Sequence[TrackEvent], textgrid_directory: Path
) -> tuple[np.ndarray, list[dict[str, object]]]:
    onsets: list[float] = []
    details: list[dict[str, object]] = []
    for event in events:
        if normalize(event.stimulus) == "story18":
            details.append(
                {"stimulus": event.stimulus, "status": "excluded_story18", "tokens": 0}
            )
            continue
        textgrid = find_stimulus_file(textgrid_directory, event.stimulus, ".TextGrid")
        if textgrid is None:
            # Accommodate lower-case extensions on case-sensitive Linux filesystems.
            textgrid = find_stimulus_file(textgrid_directory, event.stimulus, ".textgrid")
        if textgrid is None:
            details.append(
                {"stimulus": event.stimulus, "status": "missing_textgrid", "tokens": 0}
            )
            continue
        relative_onsets = parse_first_interval_tier(textgrid)
        valid = [
            event.onset + onset
            for onset in relative_onsets
            if event.onset + onset < event.offset
        ]
        onsets.extend(valid)
        details.append(
            {
                "stimulus": event.stimulus,
                "status": "ok",
                "textgrid": str(textgrid),
                "tokens": len(valid),
                "tokens_past_trigger_offset": len(relative_onsets) - len(valid),
            }
        )
    return np.asarray(sorted(onsets), dtype=float), details


def subject_number(value: str) -> int | None:
    match = re.fullmatch(
        r"(?:sub(?:ject)?[\s_-]*)?0*(\d+)", value.strip(), re.IGNORECASE
    )
    return int(match.group(1)) if match else None


def source_subject_number(path: Path) -> int | None:
    for component in (path.stem, *reversed(path.parent.parts)):
        number = subject_number(component)
        if number is not None:
            return number
    return None


def event_subject_number(path: Path) -> int | None:
    stem = re.sub(r"(?i)_event$", "", path.stem)
    return subject_number(stem)


def find_event_csv(source: Path, root: Path) -> Path:
    candidates = [
        path
        for path in root.rglob("*_event.csv")
        if path.name.casefold() != "event_stimuli.csv"
    ]
    number = source_subject_number(source)
    matches = [
        path for path in candidates if event_subject_number(path) == number
    ] if number is not None else []
    if not matches:
        local = [path for path in candidates if path.parent == source.parent]
        if len(local) == 1:
            return local[0]
        raise FileNotFoundError(f"no subject-matching *_event.csv found for {source}")
    return min(
        matches,
        key=lambda path: (
            path.parent != source.parent,
            len(path.parts),
            str(path).casefold(),
        ),
    )


def find_edfs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() == ".edf"
        and "_prepocessed_" not in path.stem
        and "_responsive_" not in path.stem
    )


def laplacian_matrix(
    channel_names: Sequence[str], exclude_pattern: str
) -> tuple[np.ndarray, list[str], list[str]]:
    excluded = re.compile(exclude_pattern) if exclude_pattern else None
    trajectories: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    dropped: list[str] = []
    for index, name in enumerate(channel_names):
        match = CONTACT_RE.fullmatch(name.strip())
        if not match or (excluded and excluded.search(name)):
            dropped.append(name)
            continue
        trajectory = normalize(match.group(1)).strip("_")
        if not trajectory:
            dropped.append(name)
            continue
        trajectories[trajectory].append((int(match.group(2)), index, name))

    rows: list[np.ndarray] = []
    output_names: list[str] = []
    for contacts in trajectories.values():
        contacts.sort()
        if len(contacts) < 2:
            dropped.extend(contact[2] for contact in contacts)
            continue
        for position, (_, source_index, name) in enumerate(contacts):
            neighbors = []
            if position:
                neighbors.append(contacts[position - 1][1])
            if position + 1 < len(contacts):
                neighbors.append(contacts[position + 1][1])
            row = np.zeros(len(channel_names), dtype=np.float64)
            row[source_index] = 1.0
            row[neighbors] = -1.0 / len(neighbors)
            rows.append(row)
            output_names.append(name)
    if not rows:
        raise ValueError(
            "no electrode trajectories with at least two numbered contacts were found"
        )
    return np.stack(rows), output_names, sorted(set(dropped))


def octave_edges(low: float, high: float, fraction: int = 7) -> np.ndarray:
    count = max(1, math.ceil(math.log2(high / low) * fraction))
    return np.geomspace(low, high, count + 1)


def extract_band_amplitude(
    data: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
    *,
    order: int,
) -> np.ndarray:
    if not 0 < low < high < sfreq / 2:
        raise ValueError(
            f"band {low:g}-{high:g} Hz requires sampling rate above {2 * high:g} Hz; "
            f"recording is {sfreq:g} Hz"
        )
    edges = octave_edges(low, high)
    amplitude = np.zeros_like(data, dtype=np.float64)
    for sub_low, sub_high in zip(edges[:-1], edges[1:]):
        sos = signal.butter(
            order, (sub_low, sub_high), btype="bandpass", fs=sfreq, output="sos"
        )
        filtered = signal.sosfiltfilt(sos, data, axis=-1)
        amplitude += np.abs(signal.hilbert(filtered, axis=-1))
    return amplitude / (len(edges) - 1)


def pooled_baseline_zscore(
    data: np.ndarray,
    sfreq: float,
    events: Sequence[TrackEvent],
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    for event in events:
        first = max(0, round((event.onset + start) * sfreq))
        last = min(data.shape[1], round((event.onset + end) * sfreq))
        if last > first:
            windows.append(data[:, first:last])
    if not windows:
        raise ValueError("no valid pre-track baseline samples fall inside the recording")
    baseline = np.concatenate(windows, axis=1)
    mean = baseline.mean(axis=1)
    std = baseline.std(axis=1, ddof=1)
    if np.any(~np.isfinite(std) | (std <= np.finfo(float).eps)):
        bad = np.flatnonzero(~np.isfinite(std) | (std <= np.finfo(float).eps))
        raise ValueError(f"zero/invalid baseline variance in channel indices {bad.tolist()}")
    return (data - mean[:, None]) / std[:, None], mean, std


def resample(data: np.ndarray, old_sfreq: float, target_sfreq: float) -> tuple[np.ndarray, float]:
    ratio = Fraction(target_sfreq / old_sfreq).limit_denominator(10000)
    result = signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)
    return result, old_sfreq * ratio.numerator / ratio.denominator


def fdr_bh(pvalues: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)
    reject = np.zeros(pvalues.shape, dtype=bool)
    finite = np.flatnonzero(np.isfinite(pvalues))
    if not finite.size:
        return reject, adjusted
    order = finite[np.argsort(pvalues[finite])]
    ranked = pvalues[order]
    n = len(ranked)
    adjusted_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    passing = np.flatnonzero(ranked <= q * np.arange(1, n + 1) / n)
    if passing.size:
        reject[order[: passing[-1] + 1]] = True
    return reject, adjusted


def speech_responsiveness(
    data: np.ndarray,
    sfreq: float,
    token_onsets: np.ndarray,
    *,
    epoch_start: float,
    epoch_end: float,
    baseline_start: float,
    baseline_end: float,
    response_start: float,
    response_end: float,
    fdr_q: float,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.arange(round(epoch_start * sfreq), round(epoch_end * sfreq))
    times = offsets / sfreq
    baseline_mask = (times >= baseline_start) & (times < baseline_end)
    response_mask = (times >= response_start) & (times < response_end)
    centers = np.round(token_onsets * sfreq).astype(int)
    centers = centers[
        (centers + offsets[0] >= 0) & (centers + offsets[-1] < data.shape[1])
    ]
    if not centers.size:
        raise ValueError("no speech-token epochs fit fully inside the recording")
    indices = centers[:, None] + offsets[None, :]
    pvalues = np.ones(data.shape[0], dtype=float)
    effects = np.zeros(data.shape[0], dtype=float)
    mean_erps = np.empty((data.shape[0], len(times)), dtype=np.float64)
    for channel in range(data.shape[0]):
        epochs = data[channel, indices]
        baseline_values = epochs[:, baseline_mask]
        baseline_mean = baseline_values.mean()
        baseline_std = baseline_values.std(ddof=1)
        zepochs = (epochs - baseline_mean) / baseline_std
        pre = zepochs[:, baseline_mask].mean(axis=1)
        post = zepochs[:, response_mask].mean(axis=1)
        effects[channel] = np.median(post - pre)
        mean_erps[channel] = zepochs.mean(axis=0)
        if np.any(post != pre):
            pvalues[channel] = stats.wilcoxon(
                post, pre, alternative="greater", zero_method="wilcox"
            ).pvalue
    reject, adjusted = fdr_bh(pvalues, fdr_q)
    rows = [
        {
            "channel_index": channel,
            "n_tokens": int(len(centers)),
            "median_response_minus_baseline": effects[channel],
            "p_value": pvalues[channel],
            "fdr_p_value": adjusted[channel],
            "responsive": bool(reject[channel] and effects[channel] > 0),
        }
        for channel in range(data.shape[0])
    ]
    responsive = np.flatnonzero(reject & (effects > 0))
    return rows, responsive, times, mean_erps


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_edf(raw: mne.io.BaseRaw, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.edf")
    temporary.unlink(missing_ok=True)
    try:
        mne.export.export_raw(
            temporary,
            raw,
            fmt="edf",
            physical_range="channelwise",
            overwrite=True,
            verbose="ERROR",
        )
        check = mne.io.read_raw_edf(temporary, preload=False, verbose="ERROR")
        try:
            if check.ch_names != raw.ch_names or check.n_times != raw.n_times:
                raise RuntimeError("EDF round-trip channel/sample validation failed")
        finally:
            check.close()
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def output_base(source: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / source.relative_to(input_root).parent / source.stem


def process_recording(
    source: Path,
    input_root: Path,
    output_root: Path,
    mapping: dict[str, MappingEntry],
    args: argparse.Namespace,
) -> None:
    event_csv = find_event_csv(source, input_root)
    events, event_warnings = load_track_events(
        event_csv,
        mapping,
        time_override=args.event_time_column,
        label_override=args.event_label_column,
    )
    duration_rows = validate_audio_durations(
        events, args.stimuli_wav_dir, args.duration_tolerance
    )
    token_onsets, token_details = collect_token_onsets(events, args.textgrid_dir)
    base = output_base(source, input_root, output_root)
    qc_dir = base.parent / f"{base.name}_qc"
    write_csv(qc_dir / "audio_trigger_duration_qc.csv", duration_rows)
    write_csv(qc_dir / "textgrid_token_qc.csv", token_details)
    (qc_dir / "event_warnings.txt").write_text(
        "\n".join(event_warnings) + ("\n" if event_warnings else ""), encoding="utf-8"
    )
    if args.dry_run:
        print(
            f"DRY  {source}: {len(events)} tracks, {len(token_onsets)} speech tokens, "
            f"{sum(not row['within_tolerance'] for row in duration_rows)} duration mismatches"
        )
        return
    missing_textgrids = [
        row["stimulus"]
        for row in token_details
        if row["status"] == "missing_textgrid"
    ]
    if missing_textgrids:
        raise FileNotFoundError(
            "missing TextGrid files for non-story18 stimuli: "
            + ", ".join(map(str, missing_textgrids))
        )
    if not token_onsets.size:
        raise ValueError("no non-story18 speech token onsets were found")

    mmap_path = qc_dir / f".{source.stem}_preload.dat"
    raw = mne.io.read_raw_edf(source, preload=str(mmap_path), verbose="ERROR")
    try:
        nyquist = raw.info["sfreq"] / 2
        line_frequencies = np.arange(
            args.line_frequency,
            args.line_frequency * (args.line_harmonics + 1),
            args.line_frequency,
        )
        line_frequencies = line_frequencies[line_frequencies < nyquist]
        if line_frequencies.size:
            raw.notch_filter(
                line_frequencies,
                method="spectrum_fit",
                filter_length="10s",
                verbose="ERROR",
            )

        matrix, channel_names, dropped = laplacian_matrix(
            raw.ch_names, args.exclude_channel_regex
        )
        referenced = matrix @ raw.get_data()
        del matrix
        metadata = {
            "source_edf": str(source),
            "event_csv": str(event_csv),
            "event_timing_correction": (
                "corrected onset/offset = subject trigger time + matching "
                "event_stimuli.csv silence value"
            ),
            "line_frequencies_hz": line_frequencies.tolist(),
            "reference": (
                "contact minus mean of immediate shaft neighbors; endpoint minus "
                "its single immediate neighbor"
            ),
            "retained_channels": channel_names,
            "dropped_channels": dropped,
            "bands_hz": {name: DEFAULT_BANDS[name] for name in args.bands},
            "responsive_electrode_selection": "independent_per_band",
            "target_sfreq": args.target_sfreq,
            "edf_value_convention": (
                "Derived data are dimensionless z-scores. EDF has no standard "
                "z-score unit, so 1 z-unit is stored as 1 microvolt; with MNE, "
                "use raw.get_data(units='uV') to recover numeric z-scores."
            ),
            "track_baseline_s": [args.baseline_start, args.baseline_end],
            "epoch_s": [args.epoch_start, args.epoch_end],
            "epoch_baseline_s": [
                args.epoch_baseline_start,
                args.epoch_baseline_end,
            ],
            "response_window_s": [args.response_start, args.response_end],
            "fdr_q": args.fdr_q,
        }
        (qc_dir / "processing_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        for band_name in args.bands:
            low, high = DEFAULT_BANDS[band_name]
            chunks: list[np.ndarray] = []
            for first in range(0, len(channel_names), args.channel_chunk_size):
                chunk = referenced[first : first + args.channel_chunk_size]
                amplitude = extract_band_amplitude(
                    chunk,
                    raw.info["sfreq"],
                    low,
                    high,
                    order=args.filter_order,
                )
                downsampled, actual_sfreq = resample(
                    amplitude, raw.info["sfreq"], args.target_sfreq
                )
                chunks.append(downsampled)
            band_data = np.concatenate(chunks)
            band_data, baseline_mean, baseline_std = pooled_baseline_zscore(
                band_data,
                actual_sfreq,
                events,
                args.baseline_start,
                args.baseline_end,
            )
            band_raw = mne.io.RawArray(
                band_data * 1e-6,
                mne.create_info(channel_names, actual_sfreq, ch_types="seeg"),
                verbose="ERROR",
            )
            band_raw.set_annotations(
                mne.Annotations(
                    onset=raw.annotations.onset,
                    duration=raw.annotations.duration,
                    description=raw.annotations.description,
                )
            )
            processed_path = base.with_name(
                f"{base.name}_prepocessed_{band_name}.edf"
            )
            export_edf(band_raw, processed_path, args.overwrite)

            rows, band_responsive, epoch_times, mean_erps = speech_responsiveness(
                band_data,
                actual_sfreq,
                token_onsets,
                epoch_start=args.epoch_start,
                epoch_end=args.epoch_end,
                baseline_start=args.epoch_baseline_start,
                baseline_end=args.epoch_baseline_end,
                response_start=args.response_start,
                response_end=args.response_end,
                fdr_q=args.fdr_q,
            )
            for row, name, mean, std in zip(
                rows, channel_names, baseline_mean, baseline_std
            ):
                row["channel"] = name
                row["track_baseline_mean_volts"] = mean
                row["track_baseline_std_volts"] = std
                row["passes_band_specific_fdr"] = row.pop("responsive")
            for row in rows:
                row["selected_for_band"] = row["passes_band_specific_fdr"]
            write_csv(qc_dir / f"{band_name}_speech_responsiveness.csv", rows)
            np.savez_compressed(
                qc_dir / f"{band_name}_mean_token_erp.npz",
                times=epoch_times,
                channel_names=np.asarray(channel_names),
                mean_zscored_erp=mean_erps,
                token_onsets_s=token_onsets,
                responsive_channel_indices=band_responsive,
            )
            positive_candidates = sorted(
                (
                    row
                    for row in rows
                    if float(row["median_response_minus_baseline"]) > 0
                ),
                key=lambda row: (
                    float(row["p_value"]),
                    -float(row["median_response_minus_baseline"]),
                ),
            )
            write_csv(
                qc_dir / f"{band_name}_top_candidates.csv",
                positive_candidates[:20],
            )
            finite_fdr = [
                float(row["fdr_p_value"])
                for row in rows
                if math.isfinite(float(row["fdr_p_value"]))
            ]
            diagnostics = {
                "selection_band": band_name,
                "line_frequencies_hz": line_frequencies.tolist(),
                "n_channels_tested": len(rows),
                "n_speech_tokens_used": int(rows[0]["n_tokens"]),
                "n_responsive_channels": int(len(band_responsive)),
                "fdr_q": args.fdr_q,
                "minimum_raw_p_value": min(float(row["p_value"]) for row in rows),
                "minimum_fdr_p_value": min(finite_fdr) if finite_fdr else None,
                "maximum_median_response_minus_baseline": max(
                    float(row["median_response_minus_baseline"]) for row in rows
                ),
                "interpretation": (
                    f"Selected {band_name} channels pass a one-sided paired "
                    "Wilcoxon test with positive median 50-200 ms response and "
                    "Benjamini-Hochberg FDR correction across contacts."
                ),
                "if_none_selected": (
                    f"Inspect {band_name}_top_candidates.csv, "
                    "event_warnings.txt, audio_trigger_duration_qc.csv, and "
                    "textgrid_token_qc.csv. No channel is forced to pass "
                    "q=0.01; persistent zero results can indicate incorrect "
                    "token alignment, too few usable tokens, or absent/weak "
                    f"{band_name} responses."
                ),
            }
            (qc_dir / f"{band_name}_speech_response_diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2), encoding="utf-8"
            )
            no_responsive_path = (
                qc_dir / f"{band_name}_no_responsive_channels.txt"
            )
            responsive_path = base.with_name(
                f"{base.name}_responsive_{band_name}.edf"
            )
            if band_responsive.size:
                no_responsive_path.unlink(missing_ok=True)
                epoch_offsets = np.arange(
                    round(args.epoch_start * actual_sfreq),
                    round(args.epoch_end * actual_sfreq),
                )
                epoch_centers = np.round(token_onsets * actual_sfreq).astype(int)
                epoch_centers = epoch_centers[
                    (epoch_centers + epoch_offsets[0] >= 0)
                    & (epoch_centers + epoch_offsets[-1] < band_data.shape[1])
                ]
                epoch_indices = epoch_centers[:, None] + epoch_offsets[None, :]
                responsive_epochs = band_data[band_responsive][:, epoch_indices]
                epoch_baseline = (
                    (epoch_times >= args.epoch_baseline_start)
                    & (epoch_times < args.epoch_baseline_end)
                )
                epoch_mean = responsive_epochs[:, :, epoch_baseline].mean(
                    axis=(1, 2)
                )
                epoch_std = responsive_epochs[:, :, epoch_baseline].std(
                    axis=(1, 2), ddof=1
                )
                responsive_epochs = (
                    responsive_epochs - epoch_mean[:, None, None]
                ) / epoch_std[:, None, None]
                np.savez_compressed(
                    qc_dir / f"{band_name}_responsive_token_epochs.npz",
                    times=epoch_times,
                    channel_names=np.asarray(channel_names)[band_responsive],
                    zscored_epochs=responsive_epochs,
                    token_onsets_s=epoch_centers / actual_sfreq,
                )
                responsive_raw = band_raw.copy().pick(band_responsive.tolist())
                export_edf(responsive_raw, responsive_path, args.overwrite)
                responsive_raw.close()
            else:
                if args.overwrite:
                    responsive_path.unlink(missing_ok=True)
                no_responsive_path.write_text(
                    f"No channels passed {band_name} one-sided Wilcoxon "
                    "signed-rank testing with a positive effect and "
                    f"Benjamini-Hochberg FDR q={args.fdr_q}. See "
                    f"{band_name}_speech_response_diagnostics.json and "
                    f"{band_name}_top_candidates.csv.\n",
                    encoding="utf-8",
                )
            band_raw.close()
            print(
                f"OK   {source.name} {band_name}: {len(channel_names)} contacts, "
                f"{len(token_onsets)} tokens, {len(band_responsive)} responsive"
            )
    finally:
        raw.close()
        mmap_path.unlink(missing_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {args.input_dir}")
    if args.target_sfreq <= 0 or args.channel_chunk_size < 1:
        raise ValueError("--target-sfreq and --channel-chunk-size must be positive")
    if not 0 < args.fdr_q < 1:
        raise ValueError("--fdr-q must be between 0 and 1")
    windows = (
        (args.baseline_start, args.baseline_end, "track baseline"),
        (args.epoch_start, args.epoch_end, "epoch"),
        (args.epoch_baseline_start, args.epoch_baseline_end, "epoch baseline"),
        (args.response_start, args.response_end, "response"),
    )
    for start, end, name in windows:
        if start >= end:
            raise ValueError(f"{name} start must be before its end")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.event_stimuli = (
        args.event_stimuli or args.input_dir / "event_stimuli.csv"
    ).expanduser().resolve()
    args.stimuli_wav_dir = (
        args.stimuli_wav_dir or args.input_dir / "stimuli_wav"
    ).expanduser().resolve()
    args.textgrid_dir = (
        args.textgrid_dir or args.input_dir / "stimuli_textgrid"
    ).expanduser().resolve()
    try:
        validate_args(args)
        for path in (args.event_stimuli, args.stimuli_wav_dir, args.textgrid_dir):
            if not path.exists():
                raise FileNotFoundError(f"required input does not exist: {path}")
        mapping = load_event_stimuli(args.event_stimuli)
        edfs = find_edfs(args.input_dir)
        if not edfs:
            raise FileNotFoundError(f"no raw EDF files found below {args.input_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failures = 0
        for source in edfs:
            try:
                process_recording(source, args.input_dir, args.output_dir, mapping, args)
            except Exception as error:
                failures += 1
                print(f"ERROR {source}: {error}", file=sys.stderr)
        print(f"Finished: {len(edfs) - failures} succeeded, {failures} failed")
        return 1 if failures else 0
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
