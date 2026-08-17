from __future__ import annotations

import csv
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_strf as strf


def write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_prosody_uses_end_as_boundary_time(tmp_path: Path) -> None:
    path = tmp_path / "story01.prosodic_word_depth.tsv"
    path.write_text(
        "word\tstart\tend\tboundary_strength_after\tprosodic_word_depth\n"
        "one\t0.1\t0.4\t0.195\t2\n"
        "two\t0.5\t0.8\t0.662\t3\n",
        encoding="utf-8",
    )

    times, strengths, depths = strf.load_prosody(path)

    np.testing.assert_allclose(times, [0.4, 0.8])
    np.testing.assert_allclose(strengths, [0.195, 0.662])
    np.testing.assert_allclose(depths, [2, 3])


def test_impulses_use_requested_names_and_keep_largest_collision() -> None:
    values, collisions = strf.impulses(
        np.asarray([0.1, 0.101, 0.5]),
        np.asarray([0.2, 0.7, 2.0]),
        n_times=20,
        sfreq=10.0,
    )

    assert collisions == 1
    assert values[1] == 0.7
    assert values[5] == 2.0
    assert set(strf.FEATURE_FAMILIES) == {
        "mel",
        "syl_onset",
        "boundary_strength",
        "struc_depth",
    }


def test_make_rf_uses_l2_ridge_regularization() -> None:
    args = Namespace(
        tmin=-0.1,
        tmax=0.6,
        target_sfreq=128.0,
        n_jobs=1,
    )

    rf = strf.make_rf(args, alpha=1.0, feature_names=["mel_00"])

    assert rf.estimator.reg_type == "ridge"


def test_discover_manifest_joins_cluster_naming_conventions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    processed = tmp_path / "processed"
    prosody = processed / "prosodic_word_depth"
    subject = source / "sub001"
    subject.mkdir(parents=True)
    raw_edf = subject / "recording.edf"
    raw_edf.touch()
    write_csv(
        source / "event_stimuli.csv",
        ["trigger_label", "stimuli_filename"],
        [
            {"trigger_label": "onset_1", "stimuli_filename": "story01.wav"},
            {"trigger_label": "offset_1", "stimuli_filename": "story01.wav"},
        ],
    )
    write_csv(
        subject / "sub001_event.csv",
        ["trigger", "time"],
        [
            {"trigger": "onset_1", "time": 2},
            {"trigger": "offset_1", "time": 4},
        ],
    )
    wav_dir = source / "stimuli_wav"
    wav_dir.mkdir()
    wavfile.write(wav_dir / "story01.wav", 1000, np.zeros(2000, dtype=np.int16))
    textgrid_dir = source / "stimuli_textgrid"
    textgrid_dir.mkdir()
    (textgrid_dir / "story01.TextGrid").write_text(
        'class = "IntervalTier"\n'
        "    intervals [1]:\n"
        "        xmin = 0.1\n"
        '        text = "syl"\n',
        encoding="utf-8",
    )
    prosody.mkdir(parents=True)
    (prosody / "story01.prosodic_word_depth.tsv").write_text(
        "start\tend\tboundary_strength_after\tprosodic_word_depth\n"
        "0.1\t0.5\t0.2\t2\n",
        encoding="utf-8",
    )
    processed_subject = processed / "sub001"
    processed_subject.mkdir()
    (processed_subject / "recording_prepocessed_high_gamma.edf").touch()
    qc = processed_subject / "recording_qc"
    write_csv(
        qc / "high_gamma_speech_responsiveness.csv",
        ["channel", "fdr_p_value"],
        [{"channel": "LA1", "fdr_p_value": 0.01}],
    )
    args = Namespace(
        input_dir=source,
        preprocessed_dir=processed,
        prosody_dir=prosody,
        event_stimuli=source / "event_stimuli.csv",
        stimuli_wav_dir=wav_dir,
        textgrid_dir=textgrid_dir,
        band="high_gamma",
        exclude_stimuli=["story18"],
        max_recordings=None,
    )

    manifest = strf.discover_manifest(args)

    assert len(manifest) == 1
    assert manifest[0].recording_id == "sub001/recording"
    assert manifest[0].stimulus_id == "story01"
    assert manifest[0].neural_audio_onset_s == 2
    assert manifest[0].neural_audio_offset_s == 4
    assert manifest[0].prosody_file.name == "story01.prosodic_word_depth.tsv"


def test_fit_recording_writes_metrics_permutations_and_figures(
    tmp_path: Path,
) -> None:
    sfreq = 20.0
    feature_names = [
        "mel_00",
        "mel_01",
        "syl_onset",
        "boundary_strength",
        "struc_depth",
    ]
    tracks: list[strf.TrackData] = []
    for track_index in range(6):
        rng = np.random.default_rng(track_index)
        n_times = 80
        X = rng.normal(size=(n_times, len(feature_names)))
        X[:, 2:] = 0
        for sample in range(5 + track_index % 3, n_times, 12):
            X[sample, 2] = 1
            X[sample, 3] = 0.2 + 0.1 * (sample % 5)
            X[sample, 4] = 1 + sample % 3
        y = (
            0.7 * np.roll(X[:, 0], 2)
            - 0.3 * np.roll(X[:, 1], 4)
            + 0.6 * np.roll(X[:, 3], 3)
            + rng.normal(scale=0.1, size=n_times)
        )[:, np.newaxis]
        tracks.append(
            strf.TrackData(
                stimulus_id=f"story{track_index + 1}",
                X=X,
                y=y,
                feature_names=feature_names,
                channel_names=["LA1"],
                time=np.arange(n_times) / sfreq,
            )
        )
    args = Namespace(
        inner_folds=2,
        outer_folds=3,
        epoch_duration=2.0,
        target_sfreq=sfreq,
        tmin=0.0,
        tmax=0.3,
        alphas=[0.1, 1.0],
        n_jobs=1,
        seed=17,
        n_permutations=19,
    )

    strf.fit_recording("sub001/run1", tracks, tmp_path, args)

    result = tmp_path / "recordings" / "sub001_run1"
    metrics = list(csv.DictReader((result / "model_metrics.csv").open()))
    contributions = list(
        csv.DictReader((result / "feature_contributions.csv").open())
    )
    comparisons = list(csv.DictReader((result / "model_comparisons.csv").open()))
    assert len(metrics) == 3 * len(strf.MODEL_FAMILIES)
    assert {row["feature"] for row in contributions} == set(strf.FEATURE_FAMILIES)
    assert all(int(row["n_permutations"]) == 19 for row in contributions)
    assert len(comparisons) == len(strf.MODEL_COMPARISONS)
    assert (result / "model_coefficients.npz").is_file()
    assert (
        result / "figures" / "LA1_M5_full_coefficients.png"
    ).is_file()
    assert (
        result / "figures" / "LA1_feature_contributions.png"
    ).is_file()
    stimulus_metrics = list(
        csv.DictReader((result / "stimulus_model_metrics.csv").open())
    )
    assert {row["model"] for row in stimulus_metrics} == {
        "M0_null",
        *strf.MODEL_FAMILIES,
    }
