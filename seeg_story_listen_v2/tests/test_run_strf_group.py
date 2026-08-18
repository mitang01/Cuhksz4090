from __future__ import annotations

import csv
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_strf as individual
import run_strf_group as group


def make_track(
    stimulus_id: str,
    X: np.ndarray,
    y: np.ndarray,
    channel_names: list[str],
    sfreq: float = 20.0,
) -> individual.TrackData:
    return individual.TrackData(
        stimulus_id=stimulus_id,
        X=X,
        y=y,
        feature_names=[
            "mel_00",
            "mel_01",
            "syl_onset",
            "boundary_strength",
            "struc_depth",
        ],
        channel_names=channel_names,
        time=np.arange(len(X)) / sfreq,
    )


def test_partition_signal_channels_excludes_misc_prefix_case_insensitively() -> None:
    included, excluded = group.partition_signal_channels(
        ["LA1", "MISC", "misc01", " MISC_aux", "RA2"]
    )

    assert included == ["LA1", "RA2"]
    assert excluded == ["MISC", "misc01", " MISC_aux"]


def test_load_fdr_channels_includes_both_effect_directions(tmp_path: Path) -> None:
    path = tmp_path / "high_gamma_speech_responsiveness.csv"
    path.write_text(
        "channel,fdr_p_value,median_response_minus_baseline\n"
        "LA1,0.01,1.2\n"
        "LA2,0.02,-0.8\n"
        "LA3,0.20,2.0\n",
        encoding="utf-8",
    )

    assert group.load_fdr_channels(path, 0.05) == ["LA1", "LA2"]


def test_group_feature_figure_omits_mel_only() -> None:
    rows = [
        {"channel": channel, "feature": feature}
        for channel in ("GROUP", "LA1")
        for feature in individual.FEATURE_FAMILIES
    ]

    group_rows = individual.feature_rows_for_plot(rows, "GROUP")
    electrode_rows = individual.feature_rows_for_plot(rows, "LA1")

    assert [row["feature"] for row in group_rows] == [
        "syl_onset",
        "boundary_strength",
        "struc_depth",
    ]
    assert [row["feature"] for row in electrode_rows] == list(
        individual.FEATURE_FAMILIES
    )


def test_aggregate_group_tracks_uses_equal_electrode_mean_and_shortest_time() -> None:
    X = np.arange(50, dtype=float).reshape(10, 5)
    first = make_track(
        "story01",
        X,
        np.column_stack([np.ones(10), np.full(10, 3.0)]),
        ["LA1", "LA2"],
    )
    second = make_track(
        "story01",
        X[:8],
        np.full((8, 1), 5.0),
        ["RA1"],
    )

    tracks, aggregation, membership = group.aggregate_group_tracks(
        [
            group.MemberTrack("sub01", first),
            group.MemberTrack("sub02", second),
        ]
    )

    assert len(tracks) == 1
    assert tracks[0].channel_names == ["GROUP"]
    assert tracks[0].X.shape == (8, 5)
    np.testing.assert_allclose(tracks[0].y[:, 0], 3.0)
    assert aggregation[0]["n_recordings"] == 2
    assert aggregation[0]["n_electrodes"] == 3
    assert {(row["recording_id"], row["channel"]) for row in membership} == {
        ("sub01", "LA1"),
        ("sub01", "LA2"),
        ("sub02", "RA1"),
    }


def test_group_fit_writes_matching_metrics_and_figure_types(tmp_path: Path) -> None:
    sfreq = 20.0
    members: list[group.MemberTrack] = []
    for recording_index in range(2):
        for story_index in range(6):
            feature_rng = np.random.default_rng(story_index)
            noise_rng = np.random.default_rng(recording_index * 20 + story_index)
            n_times = 80
            X = feature_rng.normal(size=(n_times, 5))
            X[:, 2:] = 0
            for sample in range(6 + story_index % 3, n_times, 12):
                X[sample, 2] = 1
                X[sample, 3] = 0.2 + 0.1 * (sample % 5)
                X[sample, 4] = 1 + sample % 3
            base_y = (
                0.7 * np.roll(X[:, 0], 2)
                - 0.3 * np.roll(X[:, 1], 4)
                + 0.6 * np.roll(X[:, 3], 3)
            )
            y = np.column_stack(
                [
                    base_y + noise_rng.normal(scale=0.1, size=n_times),
                    base_y + noise_rng.normal(scale=0.1, size=n_times),
                ]
            )
            members.append(
                group.MemberTrack(
                    f"sub{recording_index + 1:02d}",
                    make_track(
                        f"story{story_index + 1}",
                        X,
                        y,
                        [f"L{recording_index}1", f"L{recording_index}2"],
                        sfreq,
                    ),
                )
            )
    group_tracks, _, _ = group.aggregate_group_tracks(members)
    args = Namespace(
        inner_folds=2,
        outer_folds=3,
        epoch_duration=2.0,
        target_sfreq=sfreq,
        tmin=0.0,
        tmax=0.3,
        alphas=[0.1, 1.0],
        n_jobs=1,
        seed=23,
        n_permutations=19,
    )

    individual.fit_recording("GROUP", group_tracks, tmp_path, args)

    result = tmp_path / "recordings" / "GROUP"
    metrics = list(csv.DictReader((result / "model_metrics.csv").open()))
    contributions = list(
        csv.DictReader((result / "feature_contributions.csv").open())
    )
    assert {row["channel"] for row in metrics} == {"GROUP"}
    assert {row["feature"] for row in contributions} == set(
        individual.FEATURE_FAMILIES
    )
    expected_figures = {
        "GROUP_model_accuracy.png",
        "GROUP_model_comparisons.png",
        "GROUP_feature_contributions.png",
        *(
            f"GROUP_{model}_coefficients.png"
            for model in individual.MODEL_FAMILIES
        ),
        *(f"GROUP_outer_fold_{fold}_prediction.png" for fold in range(3)),
    }
    assert expected_figures == {
        path.name for path in (result / "figures").glob("*.png")
    }
