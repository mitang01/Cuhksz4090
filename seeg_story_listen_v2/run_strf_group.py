#!/usr/bin/env python3
"""Fit one group STRF after pooling all subjects and signal electrodes.

For each stimulus, responsive non-MISC electrodes are placed on the same
audio-relative time grid and averaged with equal electrode weight. The existing
STRF pipeline is then applied to this group-mean neural response, producing the
same metrics, coefficient files, permutation tests, and figure types as the
per-recording analysis.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mne
import numpy as np

import run_strf as individual


DEFAULT_OUTPUT = individual.DEFAULT_PREPROCESSED / "strf_group"


@dataclass(frozen=True)
class MemberTrack:
    recording_id: str
    track: individual.TrackData


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pool responsive non-MISC electrodes across all recordings into a "
            "group-mean response and fit the same five L2 ridge STRF models."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=individual.DEFAULT_INPUT)
    parser.add_argument(
        "--preprocessed-dir", type=Path, default=individual.DEFAULT_PREPROCESSED
    )
    parser.add_argument("--prosody-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
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
    parser.add_argument(
        "--group-duration-tolerance",
        type=float,
        default=0.1,
        help="Maximum duration difference across copies of one stimulus (seconds)",
    )
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
        help="Pool at most this many recordings (useful for a pilot run)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def partition_signal_channels(
    channel_names: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Separate signal contacts from channels whose label starts with MISC."""
    signal_channels: list[str] = []
    excluded: list[str] = []
    for channel_name in channel_names:
        if channel_name.strip().casefold().startswith("misc"):
            excluded.append(channel_name)
        else:
            signal_channels.append(channel_name)
    return signal_channels, excluded


def load_fdr_channels(path: Path, threshold: float) -> list[str]:
    """Select every channel below the requested adjusted-p threshold."""
    fields, rows = individual.prep.read_csv(path)
    by_normalized = {individual.prep.normalize(field): field for field in fields}
    p_column = by_normalized.get("fdr_p_value")
    channel_column = by_normalized.get("channel")
    if p_column is None or channel_column is None:
        raise ValueError(
            f"{path} needs channel and fdr_p_value columns; found {fields}"
        )
    selected: list[str] = []
    for row in rows:
        if not row[p_column]:
            continue
        p_value = individual.prep.parse_float(
            row[p_column], context=f"{path.name}:fdr_p_value"
        )
        if p_value < threshold:
            selected.append(row[channel_column])
    return selected


@dataclass
class StimulusAccumulator:
    X: np.ndarray
    time: np.ndarray
    feature_names: list[str]
    response_sum: np.ndarray
    n_electrodes: int
    recordings: set[str]


class GroupAggregator:
    """Incrementally average electrodes without retaining all neural arrays."""

    def __init__(self, duration_tolerance_s: float = 0.1):
        if duration_tolerance_s < 0:
            raise ValueError("group duration tolerance must be non-negative")
        self.duration_tolerance_s = duration_tolerance_s
        self.accumulators: dict[str, StimulusAccumulator] = {}
        self.membership_rows: list[dict[str, object]] = []
        self.recording_stimuli: set[tuple[str, str]] = set()

    def add(self, member: MemberTrack) -> None:
        track = member.track
        key = (member.recording_id, track.stimulus_id)
        if key in self.recording_stimuli:
            raise ValueError(
                f"repeated presentation is ambiguous for group averaging: "
                f"{member.recording_id}/{track.stimulus_id}"
            )
        self.recording_stimuli.add(key)
        if len(track.X) < 1:
            raise ValueError(f"{track.stimulus_id} has no aligned samples")
        response_sum = track.y.sum(axis=1)
        accumulator = self.accumulators.get(track.stimulus_id)
        if accumulator is None:
            self.accumulators[track.stimulus_id] = StimulusAccumulator(
                X=track.X.copy(),
                time=track.time.copy(),
                feature_names=list(track.feature_names),
                response_sum=response_sum.copy(),
                n_electrodes=track.y.shape[1],
                recordings={member.recording_id},
            )
        else:
            if track.feature_names != accumulator.feature_names:
                raise ValueError(
                    f"inconsistent feature names for {track.stimulus_id}"
                )
            if track.X.shape[1] != accumulator.X.shape[1]:
                raise ValueError(
                    f"inconsistent feature count for {track.stimulus_id}"
                )
            sample_interval = (
                float(np.median(np.diff(accumulator.time)))
                if len(accumulator.time) > 1
                else 0.0
            )
            duration_difference = (
                abs(len(track.X) - len(accumulator.X)) * sample_interval
            )
            if duration_difference > self.duration_tolerance_s + 1e-12:
                raise ValueError(
                    f"duration mismatch for {track.stimulus_id}: "
                    f"{duration_difference:.6f}s exceeds "
                    f"{self.duration_tolerance_s:.6f}s"
                )
            minimum_samples = min(len(track.X), len(accumulator.X))
            if not np.allclose(
                track.time[:minimum_samples],
                accumulator.time[:minimum_samples],
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError(
                    f"time grids disagree across recordings for {track.stimulus_id}"
                )
            if not np.allclose(
                track.X[:minimum_samples],
                accumulator.X[:minimum_samples],
                rtol=1e-7,
                atol=1e-8,
            ):
                raise ValueError(
                    f"stimulus features disagree across recordings for "
                    f"{track.stimulus_id}"
                )
            accumulator.X = accumulator.X[:minimum_samples]
            accumulator.time = accumulator.time[:minimum_samples]
            accumulator.response_sum = (
                accumulator.response_sum[:minimum_samples]
                + response_sum[:minimum_samples]
            )
            accumulator.n_electrodes += track.y.shape[1]
            accumulator.recordings.add(member.recording_id)
        self.membership_rows.extend(
            {
                "recording_id": member.recording_id,
                "channel": channel_name,
                "stimulus_id": track.stimulus_id,
            }
            for channel_name in track.channel_names
        )

    def finalize(
        self,
    ) -> tuple[
        list[individual.TrackData],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        if not self.accumulators:
            raise ValueError(
                "no subject-electrode tracks are available for group fitting"
            )
        group_tracks: list[individual.TrackData] = []
        aggregation_rows: list[dict[str, object]] = []
        for stimulus_id, accumulator in sorted(self.accumulators.items()):
            group_tracks.append(
                individual.TrackData(
                    stimulus_id=stimulus_id,
                    X=accumulator.X,
                    y=(
                        accumulator.response_sum / accumulator.n_electrodes
                    )[:, np.newaxis],
                    feature_names=accumulator.feature_names,
                    channel_names=["GROUP"],
                    time=accumulator.time,
                )
            )
            aggregation_rows.append(
                {
                    "stimulus_id": stimulus_id,
                    "n_recordings": len(accumulator.recordings),
                    "n_electrodes": accumulator.n_electrodes,
                    "n_samples": len(accumulator.X),
                    "duration_s": (
                        float(accumulator.time[-1])
                        + (
                            float(accumulator.time[1] - accumulator.time[0])
                            if len(accumulator.time) > 1
                            else 0.0
                        )
                    ),
                    "aggregation": "equal-weight mean across electrodes",
                }
            )
        return group_tracks, aggregation_rows, list(self.membership_rows)


def aggregate_group_tracks(
    members: Sequence[MemberTrack],
    duration_tolerance_s: float = 0.1,
) -> tuple[
    list[individual.TrackData],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Average all electrode responses separately for each stimulus."""
    aggregator = GroupAggregator(duration_tolerance_s)
    for member in members:
        aggregator.add(member)
    return aggregator.finalize()


def resolve_args(args: argparse.Namespace) -> None:
    args.input_dir = args.input_dir.expanduser().resolve()
    args.preprocessed_dir = args.preprocessed_dir.expanduser().resolve()
    args.prosody_dir = (
        args.prosody_dir
        or args.preprocessed_dir / "prosodic_word_depth"
    ).expanduser().resolve()
    args.output_dir = (
        args.output_dir or args.preprocessed_dir / "strf_group"
    ).expanduser().resolve()
    args.event_stimuli = (
        args.event_stimuli or args.input_dir / "event_stimuli.csv"
    ).expanduser().resolve()
    args.stimuli_wav_dir = (
        args.stimuli_wav_dir or args.input_dir / "stimuli_wav"
    ).expanduser().resolve()
    args.textgrid_dir = (
        args.textgrid_dir or args.input_dir / "stimuli_textgrid"
    ).expanduser().resolve()


def validate_configuration(args: argparse.Namespace) -> None:
    individual.validate_args(args)
    if args.group_duration_tolerance < 0:
        raise ValueError("--group-duration-tolerance must be non-negative")
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
        raise ValueError("--output-dir must not equal or contain an input path")


def initialize_output(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty (use --overwrite): {args.output_dir}"
        )
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def validate_manifest_cohort(
    manifest: Sequence[individual.ManifestRow],
) -> dict[str, set[str]]:
    stimuli_by_recording: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        if row.stimulus_id in stimuli_by_recording[row.recording_id]:
            raise ValueError(
                f"repeated stimulus presentation is ambiguous for group analysis: "
                f"{row.recording_id}/{row.stimulus_id}"
            )
        stimuli_by_recording[row.recording_id].add(row.stimulus_id)
    if not stimuli_by_recording:
        raise ValueError("no recordings are available for group analysis")
    reference_recording, reference_stimuli = next(iter(stimuli_by_recording.items()))
    if len(reference_stimuli) < 3:
        raise ValueError("group nested CV requires at least three stimuli")
    for recording_id, stimuli in stimuli_by_recording.items():
        if stimuli != reference_stimuli:
            missing = sorted(reference_stimuli - stimuli)
            extra = sorted(stimuli - reference_stimuli)
            raise ValueError(
                f"group cohort differs for {recording_id} relative to "
                f"{reference_recording}; missing={missing}, extra={extra}"
            )
    return stimuli_by_recording


def run(args: argparse.Namespace) -> int:
    resolve_args(args)
    validate_configuration(args)
    manifest = individual.discover_manifest(args)
    validate_manifest_cohort(manifest)
    initialize_output(args)
    individual.write_csv(
        args.output_dir / "recording_manifest.csv",
        [individual.manifest_dict(row) for row in manifest],
    )
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration.update(
        {
            "analysis_level": "group",
            "group_response": (
                "For each stimulus and sample, equal-weight mean of every "
                "responsive electrode from every recording after excluding "
                "channel labels starting with MISC."
            ),
            "models": individual.MODEL_FAMILIES,
            "regularization": "ridge",
            "permutation_test": (
                "Outer-fold-blocked sign flips of held-out delta R2 between "
                "full and reduced models."
            ),
        }
    )
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    if args.validate_only:
        print(f"Validated {len(manifest)} recording/stimulus rows for group analysis")
        return 0

    by_recording: dict[str, list[individual.ManifestRow]] = defaultdict(list)
    for row in manifest:
        by_recording[row.recording_id].append(row)

    aggregator = GroupAggregator(args.group_duration_tolerance)
    alignment_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    recording_rows: list[dict[str, object]] = []
    for recording_id, rows in by_recording.items():
        selected = load_fdr_channels(
            rows[0].responsiveness_csv, args.fdr_threshold
        )
        channel_names, excluded = partition_signal_channels(selected)
        excluded_rows.extend(
            {
                "recording_id": recording_id,
                "channel": channel_name,
                "reason": "channel label starts with MISC",
            }
            for channel_name in excluded
        )
        if not channel_names:
            recording_rows.append(
                {
                    "recording_id": recording_id,
                    "status": "skipped",
                    "n_selected_channels": len(selected),
                    "n_misc_excluded": len(excluded),
                    "n_group_channels": 0,
                }
            )
            print(
                f"SKIP {recording_id}: no non-MISC channels with "
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
            for row in rows:
                track, qc = individual.prepare_track(
                    row, raw, channel_names, args
                )
                aggregator.add(
                    MemberTrack(recording_id=recording_id, track=track)
                )
                alignment_rows.append(qc)
            recording_rows.append(
                {
                    "recording_id": recording_id,
                    "status": "included",
                    "n_selected_channels": len(selected),
                    "n_misc_excluded": len(excluded),
                    "n_group_channels": len(channel_names),
                    "n_stimuli": len(rows),
                }
            )
        finally:
            raw.close()

    if not aggregator.accumulators:
        raise ValueError("no non-MISC responsive electrodes were available")
    group_tracks, aggregation_rows, membership_rows = aggregator.finalize()
    if len(group_tracks) < 3:
        raise ValueError("group nested CV requires at least three stimuli")
    for track in group_tracks:
        individual.save_track(
            args.output_dir
            / "aligned_group_data"
            / f"{individual.safe_name(track.stimulus_id)}.npz",
            track,
        )
    individual.write_csv(args.output_dir / "alignment_qc.csv", alignment_rows)
    individual.write_csv(
        args.output_dir / "recording_inclusion.csv", recording_rows
    )
    individual.write_csv(
        args.output_dir / "excluded_channels.csv", excluded_rows
    )
    individual.write_csv(
        args.output_dir / "group_membership.csv", membership_rows
    )
    individual.write_csv(
        args.output_dir / "group_aggregation.csv", aggregation_rows
    )
    individual.fit_recording("GROUP", group_tracks, args.output_dir, args)
    print(
        f"OK GROUP: {len(group_tracks)} stimuli, "
        f"{len({row['recording_id'] for row in membership_rows})} recordings, "
        f"{len({(row['recording_id'], row['channel']) for row in membership_rows})} "
        "electrodes"
    )
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
