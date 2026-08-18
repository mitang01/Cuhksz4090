from __future__ import annotations

import csv
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prosody_epoch_stats as stats
import run_prosody_epochs
import run_prosody_epochs_group
import run_strf


def make_epoch_dataset(recording_id: str, seed: int = 1) -> stats.EpochDataset:
    rng = np.random.default_rng(seed)
    n_epochs, n_times, n_channels = 48, 21, 2
    times = np.linspace(-0.5, 0.5, n_times)
    boundary = rng.uniform(0, 1, n_epochs)
    depth = np.tile([1.0, 2.0, 3.0], n_epochs // 3)
    stimulus_ids = np.asarray(
        [f"story{index % 4}" for index in range(n_epochs)]
    )
    observed = rng.normal(scale=0.25, size=(n_epochs, n_times, n_channels))
    observed[:, 8:13, :] += boundary[:, None, None] * 1.5
    observed[:, 10:15, :] += (depth[:, None, None] - 2) * 0.8
    model_names = list(run_strf.MODEL_FAMILIES)
    predictions = np.repeat(
        observed[:, :, np.newaxis, :] * 0.8,
        len(model_names),
        axis=2,
    )
    residual = observed - predictions[:, :, 1, :]
    return stats.EpochDataset(
        recording_id=recording_id,
        channel_names=["LA1", "LA2"],
        times=times,
        observed=observed,
        predictions=predictions,
        model_names=model_names,
        residual_m2=residual,
        boundary_strength=boundary,
        struc_depth=depth,
        stimulus_ids=stimulus_ids,
    )


def test_boundary_and_depth_statistics_detect_synthetic_effects() -> None:
    dataset = make_epoch_dataset("sub001/run1")
    boundary_f, boundary_effect = stats.boundary_f_stat(
        dataset.observed[:, :, 0], dataset.boundary_strength
    )
    depth_f, depth_effect = stats.depth_f_stat(
        dataset.observed[:, :, 0], dataset.struc_depth
    )

    assert boundary_f[10] > boundary_f[0]
    assert boundary_effect[10] > 0
    assert depth_f[12] > depth_f[0]
    assert depth_effect[12] > 0


def test_fdr_clusters_requires_contiguous_samples() -> None:
    times = np.arange(8) / 128
    statistic = np.arange(8, dtype=float)
    pvalues = np.asarray([0.001, 0.001, 0.5, 0.001, 0.001, 0.001, 0.001, 0.5])

    retained, _, clusters = stats.fdr_clusters(
        times,
        statistic,
        pvalues,
        q=0.01,
        minimum_samples=4,
    )

    assert retained.tolist() == [False, False, False, True, True, True, True, False]
    assert len(clusters) == 1
    assert clusters[0]["n_samples"] == 4


def test_extract_epochs_uses_held_out_sample_indices(tmp_path: Path) -> None:
    strf_dir = tmp_path / "strf"
    aligned = strf_dir / "aligned_data" / "sub001_run1"
    recording = strf_dir / "recordings" / "sub001_run1"
    aligned.mkdir(parents=True)
    recording.mkdir(parents=True)
    (recording / "model_metrics.csv").write_text(
        "recording_id,channel,model,r2,correlation,mse\n"
        "sub001/run1,LA1,M1_mel,0.1,0.2,1.0\n",
        encoding="utf-8",
    )
    n_times = 100
    feature_names = [
        "mel_00",
        "syl_onset",
        "boundary_strength",
        "struc_depth",
    ]
    X = np.zeros((n_times, len(feature_names)))
    X[[20, 50, 80], 2] = [0.0, 0.5, 1.0]
    X[[20, 50, 80], 3] = [1, 2, 3]
    y = np.arange(n_times, dtype=float)[:, np.newaxis]
    np.savez_compressed(
        aligned / "001_story1.npz",
        stimulus_id="story1",
        X=X,
        y=y,
        feature_names=np.asarray(feature_names),
        channel_names=np.asarray(["LA1"]),
        time=np.arange(n_times) / 20,
        prosody_event_mask=np.isin(np.arange(n_times), [20, 50, 80]),
    )
    model_names = np.asarray(list(run_strf.MODEL_FAMILIES))
    predictions = np.repeat(
        y[np.newaxis, :, :] * 0.5, len(model_names), axis=0
    )
    np.savez_compressed(
        recording / "predictions_outer_fold_0.npz",
        y_true=y,
        predictions=predictions,
        model_names=model_names,
        channel_names=np.asarray(["LA1"]),
        stimulus_ids=np.asarray(["story1"] * n_times),
        stimulus_sample_indices=np.arange(n_times),
    )

    dataset = stats.extract_recording_epochs(
        strf_dir,
        "sub001_run1",
        sfreq=20,
        epoch_start=-0.1,
        epoch_end=0.1,
    )

    assert dataset.observed.shape == (3, 5, 1)
    assert dataset.boundary_strength.tolist() == [0.0, 0.5, 1.0]
    np.testing.assert_allclose(
        dataset.residual_m2, dataset.observed * 0.5
    )


def test_individual_and_group_epoch_outputs(tmp_path: Path) -> None:
    individual_root = tmp_path / "individual"
    for subject_index in range(2):
        dataset = make_epoch_dataset(
            f"sub{subject_index + 1:03d}/run1", seed=subject_index + 3
        )
        output = (
            individual_root
            / "recordings"
            / f"sub{subject_index + 1:03d}_run1"
        )
        stats.save_epoch_dataset(output / "epoch_data.npz", dataset)
    args = Namespace(
        analysis_start=-0.3,
        analysis_end=0.3,
        fdr_q=0.05,
        minimum_cluster_samples=2,
        n_permutations=19,
        seed=7,
        individual_epoch_dir=individual_root,
        output_dir=tmp_path / "group",
        overwrite=False,
    )

    result = run_prosody_epochs_group.run(args)

    assert result == 0
    rows = list(
        csv.DictReader(
            (args.output_dir / "group_timepoint_statistics.csv").open()
        )
    )
    assert {row["source"] for row in rows} == {
        "observed",
        "m2_residual",
        *(f"prediction_{model}" for model in run_strf.MODEL_FAMILIES),
    }
    assert {row["predictor"] for row in rows} == {
        "boundary_strength",
        "struc_depth",
    }
    assert all(row["baseline_correction"] == "none" for row in rows)
    assert (
        args.output_dir / "figures" / "group_observed_separability.png"
    ).is_file()
