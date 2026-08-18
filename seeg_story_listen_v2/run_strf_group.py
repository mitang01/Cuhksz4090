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
    parser.add_argument("--prosody-dir", type=Path, default=individual.DEFAULT_PROSODY)
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


def aggregate_group_tracks(
    members: Sequence[MemberTrack],
) -> tuple[
    list[individual.TrackData],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Average all electrode responses separately for each stimulus."""
    if not members:
        raise ValueError("no subject-electrode tracks are available for group fitting")
    by_stimulus: dict[str, list[MemberTrack]] = defaultdict(list)
    for member in members:
        by_stimulus[member.track.stimulus_id].append(member)

    group_tracks: list[individual.TrackData] = []
    aggregation_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    for stimulus_id, stimulus_members in sorted(by_stimulus.items()):
        reference = stimulus_members[0].track
        minimum_samples = min(len(member.track.X) for member in stimulus_members)
        if minimum_samples < 1:
            raise ValueError(f"{stimulus_id} has no aligned samples")
        responses: list[np.ndarray] = []
        recordings: set[str] = set()
        for member in stimulus_members:
            track = member.track
            if track.feature_names != reference.feature_names:
                raise ValueError(f"inconsistent feature names for {stimulus_id}")
            if track.X.shape[1] != reference.X.shape[1]:
                raise ValueError(f"inconsistent feature count for {stimulus_id}")
            if not np.allclose(
                track.X[:minimum_samples],
                reference.X[:minimum_samples],
                rtol=1e-7,
                atol=1e-8,
            ):
                raise ValueError(
                    f"stimulus features disagree across recordings for {stimulus_id}"
                )
            recordings.add(member.recording_id)
            responses.append(track.y[:minimum_samples])
            for channel_name in track.channel_names:
                membership_rows.append(
                    {
                        "recording_id": member.recording_id,
                        "channel": channel_name,
                        "stimulus_id": stimulus_id,
                    }
                )
        all_responses = np.concatenate(responses, axis=1)
        group_response = all_responses.mean(axis=1, keepdims=True)
        group_tracks.append(
            individual.TrackData(
                stimulus_id=stimulus_id,
                X=reference.X[:minimum_samples].copy(),
                y=group_response,
                feature_names=list(reference.feature_names),
                channel_names=["GROUP"],
                time=reference.time[:minimum_samples].copy(),
            )
        )
        aggregation_rows.append(
            {
                "stimulus_id": stimulus_id,
                "n_recordings": len(recordings),
                "n_electrodes": all_responses.shape[1],
                "n_samples": minimum_samples,
                "duration_s": (
                    float(reference.time[minimum_samples - 1])
                    + (
                        float(reference.time[1] - reference.time[0])
                        if minimum_samples > 1
                        else 0.0
                    )
                ),
                "aggregation": "equal-weight mean across electrodes",
            }
        )
    return group_tracks, aggregation_rows, membership_rows


def resolve_args(args: argparse.Namespace) -> None:
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


def initialize_output(args: argparse.Namespace) -> None:
    individual.validate_args(args)
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
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty (use --overwrite): {args.output_dir}"
        )
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> int:
    resolve_args(args)
    initialize_output(args)
    manifest = individual.discover_manifest(args)
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

    members: list[MemberTrack] = []
    alignment_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    recording_rows: list[dict[str, object]] = []
    for recording_id, rows in by_recording.items():
        selected = individual.load_selected_channels(
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
                members.append(MemberTrack(recording_id=recording_id, track=track))
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

    if not members:
        raise ValueError("no non-MISC responsive electrodes were available")
    group_tracks, aggregation_rows, membership_rows = aggregate_group_tracks(members)
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
