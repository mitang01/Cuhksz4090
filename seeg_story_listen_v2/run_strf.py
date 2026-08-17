#!/usr/bin/env python3
"""Fit L2 ridge-regularized STRFs to story-listening sEEG responses.

The pipeline aligns log-mel power, syllable onsets, prosodic boundary strength,
and prosodic structure depth to continuous, preprocessed neural data. Model
selection uses nested, stimulus-grouped cross-validation. Feature significance
is assessed with outer-fold-blocked permutation tests on held-out accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import ReceptiveField, TimeDelayingRidge
from scipy import signal
from scipy.io import wavfile

import preprocess_seeg as prep


DEFAULT_INPUT = Path("/share/workspace3/ieeg/seeg/story_listen_v2")
DEFAULT_PREPROCESSED = Path("/share/home/mitan/seeg_story_listen_v2")
DEFAULT_PROSODY = DEFAULT_PREPROCESSED / "prosodic_word_depth"
DEFAULT_OUTPUT = DEFAULT_PREPROCESSED / "strf"

FEATURE_FAMILIES = ("mel", "syl_onset", "boundary_strength", "struc_depth")
MODEL_FAMILIES = {
    "M1_mel": ("mel",),
    "M2_mel_syl": ("mel", "syl_onset"),
    "M3_mel_syl_boundary": ("mel", "syl_onset", "boundary_strength"),
    "M4_mel_syl_structure": ("mel", "syl_onset", "struc_depth"),
    "M5_full": FEATURE_FAMILIES,
}
MODEL_COMPARISONS = {
    "syl_onset_after_mel": ("M2_mel_syl", "M1_mel"),
    "boundary_after_syl": ("M3_mel_syl_boundary", "M2_mel_syl"),
    "structure_after_syl": ("M4_mel_syl_structure", "M2_mel_syl"),
    "structure_after_boundary": ("M5_full", "M3_mel_syl_boundary"),
    "boundary_after_structure": ("M5_full", "M4_mel_syl_structure"),
}
FEATURE_COMPARISONS = {
    "mel": ("M1_mel", "M0_null"),
    "syl_onset": ("M2_mel_syl", "M1_mel"),
    "boundary_strength": ("M5_full", "M4_mel_syl_structure"),
    "struc_depth": ("M5_full", "M3_mel_syl_boundary"),
}


@dataclass(frozen=True)
class ManifestRow:
    recording_id: str
    source_edf: Path
    neural_edf: Path
    responsiveness_csv: Path
    event_csv: Path
    stimulus_id: str
    audio_file: Path
    textgrid_file: Path
    prosody_file: Path
    neural_audio_onset_s: float
    neural_audio_offset_s: float


@dataclass
class TrackData:
    stimulus_id: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    channel_names: list[str]
    time: np.ndarray


@dataclass
class WindowedData:
    X: np.ndarray
    y: np.ndarray
    stimulus_ids: np.ndarray
    epoch_indices: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare aligned story features and fit nested, L2 ridge-regularized "
            "STRF encoding models to speech-responsive sEEG contacts."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED
    )
    parser.add_argument("--prosody-dir", type=Path, default=DEFAULT_PROSODY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-stimuli", type=Path)
    parser.add_argument("--stimuli-wav-dir", type=Path)
    parser.add_argument("--textgrid-dir", type=Path)
    parser.add_argument("--band", default="high_gamma")
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--target-sfreq", type=float, default=128.0)
    parser.add_argument("--n-mels", type=int, default=20)
    parser.add_argument("--fmin", type=float, default=50.0)
    parser.add_argument("--fmax", type=float, default=8000.0)
    parser.add_argument("--mel-window-s", type=float, default=0.025)
    parser.add_argument("--tmin", type=float, default=-0.1)
    parser.add_argument("--tmax", type=float, default=0.6)
    parser.add_argument("--epoch-duration", type=float, default=10.0)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=np.logspace(-3, 3, 7).tolist(),
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--exclude-stimuli",
        nargs="*",
        default=["story18"],
        help="Stimulus IDs to omit (default: story18)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Discover inputs and write the manifest without loading neural data",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        help="Process at most this many recordings (useful for a pilot run)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def normalize(value: str) -> str:
    return re.sub(r"[\s_.-]+", "", value.strip().casefold())


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "unnamed"


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_casefold_file(directory: Path, stem: str, suffixes: Iterable[str]) -> Path:
    wanted_stem = normalize(stem)
    wanted_suffixes = {suffix.casefold() for suffix in suffixes}
    matches = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in wanted_suffixes
        and normalize(path.stem.replace(".prosodic_word_depth", "")) == wanted_stem
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one file for {stem!r} below {directory}, found {len(matches)}: "
            + ", ".join(map(str, matches[:5]))
        )
    return matches[0]


def load_selected_channels(path: Path, threshold: float) -> list[str]:
    fields, rows = prep.read_csv(path)
    by_normalized = {prep.normalize(field): field for field in fields}
    p_column = by_normalized.get("fdr_p_value")
    channel_column = by_normalized.get("channel")
    effect_column = by_normalized.get("median_response_minus_baseline")
    if p_column is None or channel_column is None:
        raise ValueError(
            f"{path} needs channel and fdr_p_value columns; found {fields}"
        )
    selected: list[str] = []
    for row in rows:
        if not row[p_column]:
            continue
        p_value = prep.parse_float(row[p_column], context=f"{path.name}:fdr_p_value")
        effect = (
            prep.parse_float(row[effect_column], context=f"{path.name}:effect")
            if effect_column and row[effect_column]
            else 1.0
        )
        if p_value < threshold and effect > 0:
            selected.append(row[channel_column])
    return selected


def discover_manifest(args: argparse.Namespace) -> list[ManifestRow]:
    mapping = prep.load_event_stimuli(args.event_stimuli)
    excluded = {normalize(value) for value in args.exclude_stimuli}
    rows: list[ManifestRow] = []
    sources = prep.find_edfs(args.input_dir)
    if args.max_recordings is not None:
        sources = sources[: args.max_recordings]
    for source in sources:
        event_csv = prep.find_event_csv(source, args.input_dir)
        events, warnings = prep.load_track_events(event_csv, mapping)
        if warnings:
            print(
                f"WARNING {source}: " + "; ".join(warnings),
                file=sys.stderr,
            )
        base = prep.output_base(source, args.input_dir, args.preprocessed_dir)
        neural_edf = base.with_name(f"{base.name}_prepocessed_{args.band}.edf")
        responsiveness_csv = (
            base.parent / f"{base.name}_qc" / f"{args.band}_speech_responsiveness.csv"
        )
        for required in (neural_edf, responsiveness_csv):
            if not required.is_file():
                raise FileNotFoundError(f"required preprocessed input missing: {required}")
        recording_id = str(source.relative_to(args.input_dir).with_suffix(""))
        for event in events:
            if normalize(event.stimulus) in excluded:
                continue
            audio = find_casefold_file(args.stimuli_wav_dir, event.stimulus, [".wav"])
            textgrid = find_casefold_file(
                args.textgrid_dir, event.stimulus, [".TextGrid", ".textgrid"]
            )
            prosody = find_casefold_file(
                args.prosody_dir,
                event.stimulus,
                [".tsv"],
            )
            rows.append(
                ManifestRow(
                    recording_id=recording_id,
                    source_edf=source,
                    neural_edf=neural_edf,
                    responsiveness_csv=responsiveness_csv,
                    event_csv=event_csv,
                    stimulus_id=event.stimulus,
                    audio_file=audio,
                    textgrid_file=textgrid,
                    prosody_file=prosody,
                    neural_audio_onset_s=event.onset,
                    neural_audio_offset_s=event.offset,
                )
            )
    if not rows:
        raise ValueError("no usable recording/stimulus rows were discovered")
    return rows


def manifest_dict(row: ManifestRow) -> dict[str, object]:
    return {
        "recording_id": row.recording_id,
        "source_edf": row.source_edf,
        "neural_edf": row.neural_edf,
        "responsiveness_csv": row.responsiveness_csv,
        "event_csv": row.event_csv,
        "stimulus_id": row.stimulus_id,
        "audio_file": row.audio_file,
        "textgrid_file": row.textgrid_file,
        "prosody_file": row.prosody_file,
        "neural_audio_onset_s": row.neural_audio_onset_s,
        "neural_audio_offset_s": row.neural_audio_offset_s,
    }


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {path}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]
    return list(reader.fieldnames), rows


def load_prosody(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields, rows = read_tsv(path)
    by_normalized = {prep.normalize(field): field for field in fields}
    required = {
        "start": "start",
        "end": "end",
        "boundary_strength_after": "boundary_strength_after",
        "prosodic_word_depth": "prosodic_word_depth",
    }
    missing = [name for name in required if name not in by_normalized]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {fields}")
    starts: list[float] = []
    boundary_strength: list[float] = []
    structure_depth: list[float] = []
    ends: list[float] = []
    for index, row in enumerate(rows, start=2):
        values = {}
        for normalized in required:
            column = by_normalized[normalized]
            if not row[column]:
                raise ValueError(f"{path}:{index} has empty {column}")
            values[normalized] = prep.parse_float(
                row[column], context=f"{path.name}:{index}:{column}"
            )
        if values["start"] < 0 or values["end"] <= values["start"]:
            raise ValueError(f"{path}:{index} has an invalid start/end interval")
        starts.append(values["start"])
        ends.append(values["end"])
        boundary_strength.append(values["boundary_strength_after"])
        structure_depth.append(values["prosodic_word_depth"])
    if not starts:
        raise ValueError(f"no prosodic rows found in {path}")
    return (
        np.asarray(ends),
        np.asarray(boundary_strength),
        np.asarray(structure_depth),
    )


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    frequencies: np.ndarray, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    filters = np.zeros((n_mels, len(frequencies)), dtype=float)
    for index, (left, center, right) in enumerate(zip(edges, edges[1:], edges[2:])):
        ascending = (frequencies - left) / max(center - left, np.finfo(float).eps)
        descending = (right - frequencies) / max(right - center, np.finfo(float).eps)
        filters[index] = np.maximum(0.0, np.minimum(ascending, descending))
        total = filters[index].sum()
        if total:
            filters[index] /= total
    if np.any(filters.sum(axis=1) == 0):
        raise ValueError("mel configuration creates an empty frequency filter")
    return filters


def read_audio(path: Path) -> tuple[float, np.ndarray]:
    sfreq, audio = wavfile.read(path)
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        limit = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(float) / float(limit)
    else:
        audio = audio.astype(float)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"unsupported WAV shape {audio.shape} in {path}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"WAV contains non-finite samples: {path}")
    return float(sfreq), audio


def extract_log_mel(
    path: Path,
    target_times: np.ndarray,
    *,
    n_mels: int,
    fmin: float,
    fmax: float,
    window_s: float,
) -> np.ndarray:
    sfreq, audio = read_audio(path)
    nyquist = sfreq / 2.0
    fmax = min(fmax, nyquist)
    if not 0 <= fmin < fmax:
        raise ValueError(f"invalid mel frequency range {fmin}-{fmax} for {path}")
    nperseg = max(8, round(window_s * sfreq))
    nfft = 1 << int(math.ceil(math.log2(nperseg)))
    hop = max(1, round(sfreq / (1.0 / np.median(np.diff(target_times)))))
    noverlap = max(0, nperseg - hop)
    frequencies, frame_times, stft = signal.stft(
        audio,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    power = np.abs(stft) ** 2
    filters = mel_filterbank(frequencies, n_mels, fmin, fmax)
    mel_power = filters @ power
    floor = max(float(np.max(mel_power)) * 1e-10, np.finfo(float).tiny)
    log_mel = 10.0 * np.log10(np.maximum(mel_power, floor))
    return np.column_stack(
        [
            np.interp(
                target_times,
                frame_times,
                band,
                left=band[0],
                right=band[-1],
            )
            for band in log_mel
        ]
    )


def impulses(
    event_times: np.ndarray,
    values: np.ndarray,
    n_times: int,
    sfreq: float,
) -> tuple[np.ndarray, int]:
    result = np.zeros(n_times, dtype=float)
    collisions = 0
    for event_time, value in zip(event_times, values):
        sample = int(round(float(event_time) * sfreq))
        if not 0 <= sample < n_times:
            continue
        if result[sample] != 0:
            collisions += 1
        result[sample] = max(result[sample], float(value))
    return result, collisions


def prepare_track(
    row: ManifestRow,
    raw: mne.io.BaseRaw,
    channel_names: list[str],
    args: argparse.Namespace,
) -> tuple[TrackData, dict[str, object]]:
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, args.target_sfreq):
        raise ValueError(
            f"{row.neural_edf} has {sfreq} Hz, expected {args.target_sfreq} Hz"
        )
    audio_sfreq, audio = read_audio(row.audio_file)
    audio_duration = len(audio) / audio_sfreq
    event_duration = row.neural_audio_offset_s - row.neural_audio_onset_s
    duration = min(audio_duration, event_duration)
    n_times = int(math.floor(duration * sfreq))
    start = int(round(row.neural_audio_onset_s * sfreq))
    stop = start + n_times
    if start < 0 or stop > raw.n_times:
        raise ValueError(
            f"{row.stimulus_id} interval [{start}, {stop}) exceeds neural data "
            f"with {raw.n_times} samples"
        )
    target_times = np.arange(n_times) / sfreq
    mel = extract_log_mel(
        row.audio_file,
        target_times,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        window_s=args.mel_window_s,
    )
    syllable_onsets = np.asarray(prep.parse_first_interval_tier(row.textgrid_file))
    syl, syl_collisions = impulses(
        syllable_onsets, np.ones(len(syllable_onsets)), n_times, sfreq
    )
    boundary_times, boundary_values, depth_values = load_prosody(row.prosody_file)
    boundary, boundary_collisions = impulses(
        boundary_times, boundary_values, n_times, sfreq
    )
    depth, depth_collisions = impulses(boundary_times, depth_values, n_times, sfreq)
    X = np.column_stack([mel, syl, boundary, depth])
    feature_names = [
        *(f"mel_{index:02d}" for index in range(args.n_mels)),
        "syl_onset",
        "boundary_strength",
        "struc_depth",
    ]
    y = raw.get_data(
        picks=channel_names,
        start=start,
        stop=stop,
        units="uV",
    ).T
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError(f"non-finite aligned data for {row.recording_id}/{row.stimulus_id}")
    track = TrackData(
        stimulus_id=row.stimulus_id,
        X=X,
        y=y,
        feature_names=feature_names,
        channel_names=channel_names,
        time=target_times,
    )
    qc = {
        "recording_id": row.recording_id,
        "stimulus_id": row.stimulus_id,
        "audio_duration_s": audio_duration,
        "event_duration_s": event_duration,
        "used_duration_s": n_times / sfreq,
        "n_samples": n_times,
        "n_syllable_onsets": len(syllable_onsets),
        "n_prosody_rows": len(boundary_times),
        "syl_onset_sample_collisions": syl_collisions,
        "boundary_sample_collisions": boundary_collisions,
        "struc_depth_sample_collisions": depth_collisions,
    }
    return track, qc


def save_track(path: Path, track: TrackData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        stimulus_id=track.stimulus_id,
        X=track.X,
        y=track.y,
        feature_names=np.asarray(track.feature_names),
        channel_names=np.asarray(track.channel_names),
        time=track.time,
    )


def family_columns(feature_names: Sequence[str]) -> dict[str, np.ndarray]:
    names = np.asarray(feature_names)
    return {
        "mel": np.flatnonzero(np.char.startswith(names.astype(str), "mel_")),
        "syl_onset": np.flatnonzero(names == "syl_onset"),
        "boundary_strength": np.flatnonzero(names == "boundary_strength"),
        "struc_depth": np.flatnonzero(names == "struc_depth"),
    }


def assign_folds(
    groups: Sequence[str],
    n_folds: int,
    seed: int,
    weights: Sequence[float] | None = None,
) -> dict[str, int]:
    groups_array = np.asarray(groups)
    unique = np.unique(groups_array)
    if len(unique) < 2:
        raise ValueError("at least two different stimuli are required for CV")
    if weights is None:
        weights_array = np.ones(len(groups_array))
    else:
        weights_array = np.asarray(weights, dtype=float)
        if weights_array.shape != groups_array.shape or np.any(weights_array < 0):
            raise ValueError("fold weights must match groups and be non-negative")
    group_weights = np.asarray(
        [weights_array[groups_array == group].sum() for group in unique]
    )
    n_folds = min(n_folds, len(unique))
    rng = np.random.default_rng(seed)
    order = np.arange(len(unique))
    rng.shuffle(order)
    order = order[np.argsort(-group_weights[order], kind="stable")]
    loads = np.zeros(n_folds, dtype=float)
    result: dict[str, int] = {}
    for index in order:
        fold = int(np.argmin(loads))
        result[str(unique[index])] = fold
        loads[fold] += group_weights[index]
    return result


def window_tracks(
    tracks: Sequence[TrackData], epoch_samples: int, columns: np.ndarray
) -> WindowedData:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    stimuli: list[str] = []
    epoch_indices: list[int] = []
    for track in tracks:
        n_epochs = len(track.X) // epoch_samples
        for epoch in range(n_epochs):
            selection = slice(epoch * epoch_samples, (epoch + 1) * epoch_samples)
            xs.append(track.X[selection][:, columns])
            ys.append(track.y[selection])
            stimuli.append(track.stimulus_id)
            epoch_indices.append(epoch)
    if not xs:
        raise ValueError(
            "no complete analysis epochs; reduce --epoch-duration or check alignments"
        )
    return WindowedData(
        X=np.stack(xs, axis=1),
        y=np.stack(ys, axis=1),
        stimulus_ids=np.asarray(stimuli),
        epoch_indices=np.asarray(epoch_indices),
    )


def scale_from_training(
    train: WindowedData, test: WindowedData
) -> tuple[WindowedData, WindowedData]:
    mean = train.X.mean(axis=(0, 1), keepdims=True)
    std = train.X.std(axis=(0, 1), keepdims=True)
    std[std < np.finfo(float).eps] = 1.0

    def transformed(data: WindowedData) -> WindowedData:
        return WindowedData(
            X=(data.X - mean) / std,
            y=data.y,
            stimulus_ids=data.stimulus_ids,
            epoch_indices=data.epoch_indices,
        )

    return transformed(train), transformed(test)


def make_rf(args: argparse.Namespace, alpha: float, feature_names: list[str]) -> ReceptiveField:
    estimator = TimeDelayingRidge(
        args.tmin,
        args.tmax,
        args.target_sfreq,
        alpha=alpha,
        reg_type="ridge",
        fit_intercept=True,
        n_jobs=args.n_jobs,
    )
    return ReceptiveField(
        args.tmin,
        args.tmax,
        args.target_sfreq,
        feature_names=feature_names,
        estimator=estimator,
        scoring="r2",
    )


def valid_flat(
    rf: ReceptiveField, y: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = rf.valid_samples_
    return y[valid].reshape(-1, y.shape[-1]), prediction[valid].reshape(
        -1, prediction.shape[-1]
    )


def metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = y - prediction
    mse = np.mean(residual**2, axis=0)
    denominator = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum(residual**2, axis=0) / np.maximum(
        denominator, np.finfo(float).eps
    )
    y_centered = y - y.mean(axis=0)
    prediction_centered = prediction - prediction.mean(axis=0)
    correlation = np.sum(y_centered * prediction_centered, axis=0) / np.sqrt(
        np.maximum(
            np.sum(y_centered**2, axis=0)
            * np.sum(prediction_centered**2, axis=0),
            np.finfo(float).eps,
        )
    )
    return r2, correlation, mse


def select_alpha(
    tracks: Sequence[TrackData],
    columns: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[float, list[dict[str, object]]]:
    groups = [track.stimulus_id for track in tracks]
    epoch_samples = round(args.epoch_duration * args.target_sfreq)
    epoch_counts = [len(track.X) // epoch_samples for track in tracks]
    assignments = assign_folds(
        groups, args.inner_folds, seed, weights=epoch_counts
    )
    rows: list[dict[str, object]] = []
    for alpha in args.alphas:
        fold_scores: list[float] = []
        fold_samples: list[int] = []
        for fold in sorted(set(assignments.values())):
            inner_train = [
                track for track in tracks if assignments[track.stimulus_id] != fold
            ]
            inner_test = [
                track for track in tracks if assignments[track.stimulus_id] == fold
            ]
            train = window_tracks(inner_train, epoch_samples, columns)
            test = window_tracks(inner_test, epoch_samples, columns)
            train, test = scale_from_training(train, test)
            rf = make_rf(args, alpha, feature_names)
            rf.fit(train.X, train.y)
            y, prediction = valid_flat(rf, test.y, rf.predict(test.X))
            score = float(np.mean(metrics(y, prediction)[0]))
            fold_scores.append(score)
            fold_samples.append(len(y))
            rows.append(
                {
                    "alpha": alpha,
                    "inner_fold": fold,
                    "mean_r2": score,
                    "n_validation_samples": len(y),
                }
            )
        weighted_score = float(np.average(fold_scores, weights=fold_samples))
        rows.append(
            {
                "alpha": alpha,
                "inner_fold": "mean",
                "mean_r2": weighted_score,
                "n_validation_samples": int(np.sum(fold_samples)),
            }
        )
    mean_rows = [row for row in rows if row["inner_fold"] == "mean"]
    best = max(mean_rows, key=lambda row: (float(row["mean_r2"]), -float(row["alpha"])))
    return float(best["alpha"]), rows


def stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "|".join([str(seed), *(str(part) for part in parts)]).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def sign_flip_pvalue(values: np.ndarray, n_permutations: int, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))
    if observed <= 0 or not np.all(np.isfinite(values)):
        return 1.0
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for index in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=len(values))
        null[index] = np.mean(values * signs)
    return (1.0 + float(np.sum(null >= observed))) / (n_permutations + 1.0)


def fdr_bh_rows(
    rows: list[dict[str, object]], p_key: str, output_key: str
) -> None:
    if not rows:
        return
    pvalues = np.asarray([float(row[p_key]) for row in rows])
    _, adjusted = prep.fdr_bh(pvalues, 0.05)
    for row, value in zip(rows, adjusted):
        row[output_key] = float(value)
        row[f"{output_key}_significant_0.05"] = bool(value < 0.05)


def fit_recording(
    recording_id: str,
    tracks: Sequence[TrackData],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    feature_names = tracks[0].feature_names
    channel_names = tracks[0].channel_names
    if any(track.feature_names != feature_names for track in tracks):
        raise ValueError(f"inconsistent feature names in {recording_id}")
    if any(track.channel_names != channel_names for track in tracks):
        raise ValueError(f"inconsistent channel names in {recording_id}")
    n_stimuli = len({track.stimulus_id for track in tracks})
    if n_stimuli < 3:
        raise ValueError(
            f"{recording_id} has {n_stimuli} stimuli; nested CV requires at least 3"
        )
    recording_dir = output_dir / "recordings" / safe_name(recording_id)
    recording_dir.mkdir(parents=True, exist_ok=True)
    families = family_columns(feature_names)
    model_columns = {
        model: np.concatenate([families[family] for family in model_families])
        for model, model_families in MODEL_FAMILIES.items()
    }
    epoch_samples = round(args.epoch_duration * args.target_sfreq)
    outer_folds = 3 if n_stimuli == 3 else args.outer_folds
    assignments = assign_folds(
        [track.stimulus_id for track in tracks],
        outer_folds,
        args.seed,
        weights=[len(track.X) // epoch_samples for track in tracks],
    )
    fold_rows = [
        {
            "recording_id": recording_id,
            "stimulus_id": stimulus,
            "outer_fold": fold,
        }
        for stimulus, fold in sorted(assignments.items())
    ]
    write_csv(recording_dir / "cv_folds.csv", fold_rows)
    model_metric_rows: list[dict[str, object]] = []
    stimulus_metric_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    coefficient_data: dict[str, list[np.ndarray]] = defaultdict(list)

    for outer_fold in sorted(set(assignments.values())):
        train_tracks = [
            track for track in tracks if assignments[track.stimulus_id] != outer_fold
        ]
        test_tracks = [
            track for track in tracks if assignments[track.stimulus_id] == outer_fold
        ]
        fold_predictions: list[np.ndarray] = []
        fold_y: np.ndarray | None = None
        fold_stimuli: np.ndarray | None = None
        for model, columns in model_columns.items():
            selected_names = [feature_names[index] for index in columns]
            alpha, selection_rows = select_alpha(
                train_tracks,
                columns,
                selected_names,
                args,
                stable_seed(args.seed, recording_id, outer_fold, model),
            )
            for selection_row in selection_rows:
                alpha_rows.append(
                    {
                        "recording_id": recording_id,
                        "outer_fold": outer_fold,
                        "model": model,
                        **selection_row,
                        "selected": bool(
                            selection_row["inner_fold"] == "mean"
                            and float(selection_row["alpha"]) == alpha
                        ),
                    }
                )
            train = window_tracks(train_tracks, epoch_samples, columns)
            test = window_tracks(test_tracks, epoch_samples, columns)
            train, test = scale_from_training(train, test)
            rf = make_rf(args, alpha, selected_names)
            rf.fit(train.X, train.y)
            prediction = rf.predict(test.X)
            y_valid, prediction_valid = valid_flat(rf, test.y, prediction)
            valid = rf.valid_samples_
            valid_stimuli = np.repeat(
                test.stimulus_ids[np.newaxis, :], test.X.shape[0], axis=0
            )[valid].reshape(-1)
            r2, correlation, mse = metrics(y_valid, prediction_valid)
            for channel, channel_name in enumerate(channel_names):
                row = {
                    "recording_id": recording_id,
                    "channel": channel_name,
                    "outer_fold": outer_fold,
                    "test_stimuli": ";".join(sorted(set(test.stimulus_ids))),
                    "model": model,
                    "alpha": alpha,
                    "r2": float(r2[channel]),
                    "correlation": float(correlation[channel]),
                    "mse": float(mse[channel]),
                    "n_test_samples": len(y_valid),
                }
                model_metric_rows.append(row)
            for stimulus_id in sorted(set(valid_stimuli)):
                stimulus_mask = valid_stimuli == stimulus_id
                stimulus_r2, stimulus_correlation, stimulus_mse = metrics(
                    y_valid[stimulus_mask], prediction_valid[stimulus_mask]
                )
                for channel, channel_name in enumerate(channel_names):
                    stimulus_metric_rows.append(
                        {
                            "recording_id": recording_id,
                            "channel": channel_name,
                            "stimulus_id": stimulus_id,
                            "outer_fold": outer_fold,
                            "model": model,
                            "r2": float(stimulus_r2[channel]),
                            "correlation": float(stimulus_correlation[channel]),
                            "mse": float(stimulus_mse[channel]),
                            "n_test_samples": int(np.sum(stimulus_mask)),
                        }
                    )
            coefficients = np.asarray(rf.coef_)
            if coefficients.ndim == 2:
                coefficients = coefficients[np.newaxis, ...]
            full_coefficients = np.full(
                (len(channel_names), len(feature_names), coefficients.shape[-1]),
                np.nan,
            )
            full_coefficients[:, columns, :] = coefficients
            coefficient_data[model].append(full_coefficients)
            fold_predictions.append(prediction_valid)
            if fold_y is None:
                fold_y = y_valid
                fold_stimuli = valid_stimuli
                null_prediction = np.broadcast_to(
                    train.y.mean(axis=(0, 1)), y_valid.shape
                )
                for stimulus_id in sorted(set(valid_stimuli)):
                    stimulus_mask = valid_stimuli == stimulus_id
                    null_r2, null_correlation, null_mse = metrics(
                        y_valid[stimulus_mask], null_prediction[stimulus_mask]
                    )
                    for channel, channel_name in enumerate(channel_names):
                        stimulus_metric_rows.append(
                            {
                                "recording_id": recording_id,
                                "channel": channel_name,
                                "stimulus_id": stimulus_id,
                                "outer_fold": outer_fold,
                                "model": "M0_null",
                                "r2": float(null_r2[channel]),
                                "correlation": float(null_correlation[channel]),
                                "mse": float(null_mse[channel]),
                                "n_test_samples": int(np.sum(stimulus_mask)),
                            }
                        )
        assert fold_y is not None and fold_stimuli is not None
        np.savez_compressed(
            recording_dir / f"predictions_outer_fold_{outer_fold}.npz",
            y_true=fold_y,
            predictions=np.stack(fold_predictions),
            model_names=np.asarray(list(MODEL_FAMILIES)),
            channel_names=np.asarray(channel_names),
            stimulus_ids=fold_stimuli,
        )
        plot_prediction_excerpt(
            recording_dir,
            outer_fold,
            fold_y,
            fold_predictions[-1],
            channel_names,
            args.target_sfreq,
        )

    write_csv(recording_dir / "model_metrics.csv", model_metric_rows)
    write_csv(recording_dir / "stimulus_model_metrics.csv", stimulus_metric_rows)
    write_csv(recording_dir / "alpha_selection.csv", alpha_rows)
    lags = (
        np.arange(round(args.tmin * args.target_sfreq), round(args.tmax * args.target_sfreq) + 1)
        / args.target_sfreq
    )
    np.savez_compressed(
        recording_dir / "model_coefficients.npz",
        **{model: np.stack(values) for model, values in coefficient_data.items()},
        model_names=np.asarray(list(MODEL_FAMILIES)),
        channel_names=np.asarray(channel_names),
        feature_names=np.asarray(feature_names),
        lags_s=lags,
    )

    stimulus_metric_lookup = {
        (str(row["stimulus_id"]), str(row["model"]), str(row["channel"])): float(
            row["r2"]
        )
        for row in stimulus_metric_rows
    }
    stimulus_ids = sorted({str(row["stimulus_id"]) for row in stimulus_metric_rows})
    comparison_rows: list[dict[str, object]] = []
    for comparison, (full_model, reduced_model) in MODEL_COMPARISONS.items():
        for channel_name in channel_names:
            deltas = np.asarray(
                [
                    stimulus_metric_lookup[(stimulus, full_model, channel_name)]
                    - stimulus_metric_lookup[(stimulus, reduced_model, channel_name)]
                    for stimulus in stimulus_ids
                ]
            )
            comparison_rows.append(
                {
                    "recording_id": recording_id,
                    "channel": channel_name,
                    "comparison": comparison,
                    "full_model": full_model,
                    "reduced_model": reduced_model,
                    "mean_delta_r2": float(np.mean(deltas)),
                    "std_delta_r2": float(np.std(deltas, ddof=1))
                    if len(deltas) > 1
                    else 0.0,
                    "positive_stimuli": int(np.sum(deltas > 0)),
                    "n_stimuli": len(deltas),
                    "n_permutation_blocks": len(set(assignments.values())),
                    "permutation_p_value": sign_flip_pvalue(
                        np.asarray(
                            [
                                np.mean(
                                    [
                                        delta
                                        for delta, stimulus in zip(deltas, stimulus_ids)
                                        if assignments[stimulus] == fold
                                    ]
                                )
                                for fold in sorted(set(assignments.values()))
                            ]
                        ),
                        args.n_permutations,
                        stable_seed(args.seed, recording_id, comparison, channel_name),
                    ),
                }
            )
    fdr_bh_rows(comparison_rows, "permutation_p_value", "fdr_p_value")
    write_csv(recording_dir / "model_comparisons.csv", comparison_rows)

    contribution_rows: list[dict[str, object]] = []
    for family, (full_model, reduced_model) in FEATURE_COMPARISONS.items():
        for channel_name in channel_names:
            deltas = np.asarray(
                [
                    stimulus_metric_lookup[(stimulus, full_model, channel_name)]
                    - stimulus_metric_lookup[(stimulus, reduced_model, channel_name)]
                    for stimulus in stimulus_ids
                ]
            )
            contribution_rows.append(
                {
                    "recording_id": recording_id,
                    "channel": channel_name,
                    "feature": family,
                    "full_model": full_model,
                    "reduced_model": reduced_model,
                    "mean_delta_r2": float(np.mean(deltas)),
                    "std_delta_r2": float(np.std(deltas, ddof=1)),
                    "positive_stimuli": int(np.sum(deltas > 0)),
                    "n_stimuli": len(deltas),
                    "n_permutation_blocks": len(set(assignments.values())),
                    "permutation_p_value": sign_flip_pvalue(
                        np.asarray(
                            [
                                np.mean(
                                    [
                                        delta
                                        for delta, stimulus in zip(deltas, stimulus_ids)
                                        if assignments[stimulus] == fold
                                    ]
                                )
                                for fold in sorted(set(assignments.values()))
                            ]
                        ),
                        args.n_permutations,
                        stable_seed(args.seed, recording_id, family, channel_name),
                    ),
                    "n_permutations": args.n_permutations,
                    "test": "outer-fold-blocked sign flip of held-out delta R2",
                }
            )
    fdr_bh_rows(contribution_rows, "permutation_p_value", "fdr_p_value")
    write_csv(recording_dir / "feature_contributions.csv", contribution_rows)
    create_figures(
        recording_dir,
        model_metric_rows,
        comparison_rows,
        contribution_rows,
        coefficient_data,
        feature_names,
        channel_names,
        lags,
    )


def plot_prediction_excerpt(
    recording_dir: Path,
    outer_fold: int,
    y: np.ndarray,
    prediction: np.ndarray,
    channel_names: list[str],
    sfreq: float,
) -> None:
    figures = recording_dir / "figures"
    figures.mkdir(exist_ok=True)
    n_samples = min(len(y), round(10 * sfreq))
    time = np.arange(n_samples) / sfreq
    for channel, channel_name in enumerate(channel_names):
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        ax.plot(time, y[:n_samples, channel], color="black", alpha=0.65, label="Actual")
        ax.plot(
            time,
            prediction[:n_samples, channel],
            color="#d1495b",
            linewidth=1.2,
            label="M5 prediction",
        )
        ax.set(
            xlabel="Held-out sample time (s)",
            ylabel="Neural response (z)",
            title=f"{channel_name}: held-out prediction, fold {outer_fold}",
        )
        ax.legend()
        fig.savefig(
            figures
            / f"{safe_name(channel_name)}_outer_fold_{outer_fold}_prediction.png",
            dpi=160,
        )
        plt.close(fig)


def create_figures(
    recording_dir: Path,
    metric_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    contribution_rows: list[dict[str, object]],
    coefficient_data: dict[str, list[np.ndarray]],
    feature_names: list[str],
    channel_names: list[str],
    lags: np.ndarray,
) -> None:
    figures = recording_dir / "figures"
    figures.mkdir(exist_ok=True)
    models = list(MODEL_FAMILIES)
    for channel_index, channel_name in enumerate(channel_names):
        channel_metrics = [
            row for row in metric_rows if row["channel"] == channel_name
        ]
        means = [
            np.mean([float(row["r2"]) for row in channel_metrics if row["model"] == model])
            for model in models
        ]
        sems = [
            np.std(
                [float(row["r2"]) for row in channel_metrics if row["model"] == model],
                ddof=1,
            )
            / math.sqrt(
                len([row for row in channel_metrics if row["model"] == model])
            )
            for model in models
        ]
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.bar(np.arange(len(models)), means, yerr=sems, capsize=4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            xticks=np.arange(len(models)),
            xticklabels=models,
            ylabel="Held-out $R^2$",
            title=f"{channel_name}: STRF predictive accuracy",
        )
        ax.tick_params(axis="x", rotation=25)
        fig.savefig(figures / f"{safe_name(channel_name)}_model_accuracy.png", dpi=160)
        plt.close(fig)

        channel_comparisons = [
            row for row in comparison_rows if row["channel"] == channel_name
        ]
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        values = [float(row["mean_delta_r2"]) for row in channel_comparisons]
        colors = [
            "#2a9d8f" if row["fdr_p_value_significant_0.05"] else "#9aa0a6"
            for row in channel_comparisons
        ]
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            xticks=np.arange(len(values)),
            xticklabels=[str(row["comparison"]) for row in channel_comparisons],
            ylabel=r"Mean $\Delta R^2$",
            title=f"{channel_name}: nested model comparisons",
        )
        ax.tick_params(axis="x", rotation=25)
        fig.savefig(figures / f"{safe_name(channel_name)}_model_comparisons.png", dpi=160)
        plt.close(fig)

        channel_contributions = [
            row for row in contribution_rows if row["channel"] == channel_name
        ]
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        values = [float(row["mean_delta_r2"]) for row in channel_contributions]
        colors = [
            "#e76f51" if row["fdr_p_value_significant_0.05"] else "#9aa0a6"
            for row in channel_contributions
        ]
        bars = ax.bar(np.arange(len(values)), values, color=colors)
        for bar, row in zip(bars, channel_contributions):
            if row["fdr_p_value_significant_0.05"]:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            xticks=np.arange(len(values)),
            xticklabels=[str(row["feature"]) for row in channel_contributions],
            ylabel=r"Mean held-out $\Delta R^2$",
            title=f"{channel_name}: feature contributions",
        )
        fig.savefig(
            figures / f"{safe_name(channel_name)}_feature_contributions.png", dpi=160
        )
        plt.close(fig)

        for model, fold_coefficients in coefficient_data.items():
            stacked = np.stack(fold_coefficients)
            finite = np.isfinite(stacked)
            coefficient_sum = np.nansum(stacked, axis=0)
            coefficient_count = finite.sum(axis=0)
            averaged = np.full(coefficient_sum.shape, np.nan)
            np.divide(
                coefficient_sum,
                coefficient_count,
                out=averaged,
                where=coefficient_count > 0,
            )
            coefficients = averaged[channel_index]
            mel_indices = [
                index for index, name in enumerate(feature_names) if name.startswith("mel_")
            ]
            other_indices = [
                index
                for index, name in enumerate(feature_names)
                if not name.startswith("mel_") and np.any(np.isfinite(coefficients[index]))
            ]
            n_panels = 1 + bool(other_indices)
            fig, axes = plt.subplots(
                n_panels,
                1,
                figsize=(8, 4 + 2 * bool(other_indices)),
                layout="constrained",
                squeeze=False,
            )
            image = axes[0, 0].pcolormesh(
                lags,
                np.arange(len(mel_indices)),
                coefficients[mel_indices],
                cmap="RdBu_r",
                shading="auto",
            )
            axes[0, 0].set(
                ylabel="Mel band",
                title=f"{channel_name}: {model} coefficients",
            )
            fig.colorbar(image, ax=axes[0, 0], label="Normalized coefficient")
            if other_indices:
                for index in other_indices:
                    axes[1, 0].plot(
                        lags, coefficients[index], label=feature_names[index]
                    )
                axes[1, 0].axhline(0, color="black", linewidth=0.7)
                axes[1, 0].legend()
                axes[1, 0].set(ylabel="Coefficient", xlabel="Lag (s)")
            else:
                axes[0, 0].set_xlabel("Lag (s)")
            fig.savefig(
                figures / f"{safe_name(channel_name)}_{model}_coefficients.png",
                dpi=160,
            )
            plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.input_dir,
        args.preprocessed_dir,
        args.prosody_dir,
        args.event_stimuli,
        args.stimuli_wav_dir,
        args.textgrid_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(f"required input does not exist: {path}")
    if not 0 < args.fdr_threshold < 1:
        raise ValueError("--fdr-threshold must be between 0 and 1")
    if args.target_sfreq <= 0 or args.n_mels < 1 or args.epoch_duration <= 0:
        raise ValueError("sampling rate, mel count, and epoch duration must be positive")
    if args.tmin >= args.tmax:
        raise ValueError("--tmin must be before --tmax")
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("outer and inner folds must be at least 2")
    if args.n_permutations < 1:
        raise ValueError("permutation count must be positive")
    if not args.alphas or any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("all alpha values must be positive")


def run(args: argparse.Namespace) -> int:
    args.input_dir = args.input_dir.expanduser().resolve()
    args.preprocessed_dir = args.preprocessed_dir.expanduser().resolve()
    args.prosody_dir = args.prosody_dir.expanduser().resolve()
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
    validate_args(args)
    protected_inputs = {
        args.input_dir,
        args.preprocessed_dir,
        args.prosody_dir,
        args.event_stimuli,
        args.stimuli_wav_dir,
        args.textgrid_dir,
    }
    if any(
        input_path == args.output_dir or input_path.is_relative_to(args.output_dir)
        for input_path in protected_inputs
    ):
        raise ValueError(
            "--output-dir must not equal or contain an input path"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty (use --overwrite): {args.output_dir}"
        )
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = discover_manifest(args)
    write_csv(
        args.output_dir / "recording_manifest.csv",
        [manifest_dict(row) for row in manifest],
    )
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration["models"] = MODEL_FAMILIES
    configuration["regularization"] = "ridge"
    configuration["permutation_test"] = (
        "Outer-fold-blocked sign flips of held-out delta R2 between full and "
        "reduced models; stimulus deltas are averaged within fold before each "
        "sign flip; p=(1+# permuted mean >= observed mean)/(n_permutations+1)."
    )
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    if args.validate_only:
        print(f"Validated {len(manifest)} recording/stimulus rows")
        return 0

    by_recording: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in manifest:
        by_recording[row.recording_id].append(row)
    qc_rows: list[dict[str, object]] = []
    for recording_id, rows in by_recording.items():
        channel_names = load_selected_channels(
            rows[0].responsiveness_csv, args.fdr_threshold
        )
        if not channel_names:
            print(
                f"SKIP {recording_id}: no positive channels with "
                f"fdr_p_value < {args.fdr_threshold}",
                file=sys.stderr,
            )
            continue
        raw = mne.io.read_raw_edf(rows[0].neural_edf, preload=False, verbose="ERROR")
        try:
            missing = sorted(set(channel_names) - set(raw.ch_names))
            if missing:
                raise ValueError(
                    f"{recording_id} selected channels missing from neural EDF: {missing}"
                )
            tracks: list[TrackData] = []
            for presentation_index, row in enumerate(rows, start=1):
                track, qc = prepare_track(row, raw, channel_names, args)
                tracks.append(track)
                qc_rows.append(qc)
                save_track(
                    args.output_dir
                    / "aligned_data"
                    / safe_name(recording_id)
                    / (
                        f"{presentation_index:03d}_"
                        f"{safe_name(row.stimulus_id)}.npz"
                    ),
                    track,
                )
            fit_recording(recording_id, tracks, args.output_dir, args)
            print(
                f"OK {recording_id}: {len(tracks)} stimuli, "
                f"{len(channel_names)} channels"
            )
        finally:
            raw.close()
    write_csv(args.output_dir / "alignment_qc.csv", qc_rows)
    if not qc_rows:
        raise ValueError("no recordings had channels meeting the selection criterion")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
