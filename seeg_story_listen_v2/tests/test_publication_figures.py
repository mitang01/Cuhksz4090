from __future__ import annotations

import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_nature_figures
import prosody_epoch_stats as epoch_stats
import run_strf
from figure_significance import (
    annotate_bars,
    annotate_time_significance,
    contiguous_true_runs,
    significance_stars,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_significance_annotations_use_adjusted_p_and_contiguous_runs() -> None:
    assert [significance_stars(value) for value in (0.2, 0.04, 0.009, 0.0009)] == [
        "",
        "*",
        "**",
        "***",
    ]
    assert contiguous_true_runs([False, True, True, False, True]) == [(1, 3), (4, 5)]
    fig, (bar_ax, time_ax) = plt.subplots(1, 2)
    bars = bar_ax.bar([0, 1], [0.1, 0.2])
    bar_marks = annotate_bars(
        bar_ax,
        bars,
        [{"q": 0.2}, {"q": 0.004}],
        p_key="q",
    )
    time_marks = annotate_time_significance(
        time_ax,
        np.arange(5),
        [False, True, True, False, True],
        color="black",
        level=0,
    )
    plt.close(fig)
    assert bar_marks == 1
    assert time_marks == 2


def make_epoch_dataset(recording_id: str, seed: int) -> epoch_stats.EpochDataset:
    rng = np.random.default_rng(seed)
    n_epochs, n_times = 12, 9
    times = np.linspace(-0.5, 0.5, n_times)
    observed = rng.normal(size=(n_epochs, n_times, 1))
    models = list(run_strf.MODEL_FAMILIES)
    predictions = np.repeat(
        observed[:, :, np.newaxis, :] * 0.7,
        len(models),
        axis=2,
    )
    return epoch_stats.EpochDataset(
        recording_id=recording_id,
        channel_names=["LA1"],
        times=times,
        observed=observed,
        predictions=predictions,
        model_names=models,
        residual_m2=observed - predictions[:, :, 1, :],
        boundary_strength=np.linspace(0.1, 0.9, n_epochs),
        struc_depth=np.tile([1.0, 2.0, 3.0], 4),
        stimulus_ids=np.asarray([f"story{index % 3}" for index in range(n_epochs)]),
    )


def build_publication_fixture(root: Path) -> Namespace:
    strf_dir = root / "strf"
    group_dir = root / "strf_group"
    epochs_dir = root / "prosody_epochs"
    epochs_group_dir = root / "prosody_epochs_group"
    members: list[dict[str, object]] = []
    for subject_index in range(2):
        recording_id = f"sub{subject_index + 1:03d}/run1"
        channel = f"L{subject_index + 1}A1"
        members.append(
            {
                "recording_id": recording_id,
                "channel": channel,
                "stimulus_id": "story01",
            }
        )
        rows = [
            {
                "recording_id": recording_id,
                "channel": channel,
                "feature": feature,
                "mean_delta_correlation": 0.01 * (feature_index + subject_index + 1),
            }
            for feature_index, feature in enumerate(run_strf.FEATURE_FAMILIES)
        ]
        write_csv(
            strf_dir
            / "recordings"
            / f"sub{subject_index + 1:03d}_run1"
            / "feature_contributions.csv",
            rows,
        )
        dataset = make_epoch_dataset(recording_id, subject_index + 1)
        epoch_stats.save_epoch_dataset(
            epochs_dir
            / "recordings"
            / f"sub{subject_index + 1:03d}_run1"
            / "epoch_data.npz",
            dataset,
        )
    write_csv(group_dir / "group_membership.csv", members)
    write_csv(
        group_dir / "recording_inclusion.csv",
        [
            {"recording_id": row["recording_id"], "status": "included"}
            for row in members
        ],
    )
    write_csv(
        group_dir / "subject_feature_contributions.csv",
        [
            {
                "feature": feature,
                "mean_subject_delta_correlation": 0.02 * (index + 1),
                "std_subject_delta_correlation": 0.01,
                "n_subjects": 2,
                "fdr_p_value_correlation": 0.01 if index == 2 else 0.2,
                "fdr_p_value_correlation_significant_0.05": index == 2,
            }
            for index, feature in enumerate(run_strf.FEATURE_FAMILIES)
        ],
    )
    group_recording = group_dir / "recordings" / "GROUP"
    write_csv(
        group_recording / "model_metrics.csv",
        [
            {
                "channel": "GROUP",
                "model": model,
                "outer_fold": fold,
                "correlation": 0.05 * (model_index + 1) + fold * 0.005,
            }
            for model_index, model in enumerate(run_strf.MODEL_FAMILIES)
            for fold in range(3)
        ],
    )
    features = ["mel_00", "mel_01", "syl_onset", "boundary_strength", "struc_depth"]
    lags = np.linspace(-0.3, 0.3, 7)
    coefficients = np.random.default_rng(4).normal(
        scale=0.1, size=(3, 1, len(features), len(lags))
    )
    group_recording.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        group_recording / "model_coefficients.npz",
        M5_full=coefficients,
        feature_names=np.asarray(features),
        lags_s=lags,
    )
    times = np.linspace(-0.3, 0.3, 7)
    sources = {
        "observed",
        "m2_residual",
        *(f"prediction_{model}" for model in run_strf.MODEL_FAMILIES),
    }
    write_csv(
        epochs_group_dir / "group_timepoint_statistics.csv",
        [
            {
                "source": source,
                "predictor": predictor,
                "time_s": time,
                "mean_subject_f_statistic": 2 + index,
                "mean_subject_effect_size": 0.1,
                "permutation_p_value": 0.001,
                "fdr_p_value": 0.005 if index < 4 else 0.2,
                "passes_fdr_and_cluster": (
                    source == "observed"
                    and predictor == "boundary_strength"
                    and index < 4
                ),
                "n_subjects": 2,
            }
            for source in sorted(sources)
            for predictor in ("boundary_strength", "struc_depth")
            for index, time in enumerate(times)
        ],
    )
    return Namespace(
        strf_dir=strf_dir,
        strf_group_dir=group_dir,
        epochs_dir=epochs_dir,
        epochs_group_dir=epochs_group_dir,
        output_dir=root / "nature_figures",
        overwrite=False,
    )


def test_publication_bundle_contains_vector_source_and_audit_outputs(
    tmp_path: Path,
) -> None:
    args = build_publication_fixture(tmp_path)

    assert make_nature_figures.run(args) == 0

    expected = {
        "main_figure.pdf",
        "main_figure.svg",
        "main_figure.png",
        "supplementary_epoch_sources.pdf",
        "panel_notes.md",
        "source_data_mapping.json",
        "exclusion_counts.csv",
        "panel_visual_audit.json",
        "alignment.json",
        "alignment_diagnostic.svg",
        "collision.json",
        "collision_diagnostic.pdf",
        "pre_submission_qa.json",
    }
    assert expected <= {path.name for path in args.output_dir.iterdir()}
    audit = json.loads((args.output_dir / "panel_visual_audit.json").read_text())
    assert audit["C"]["expected_significance_marks"] == 1
    assert audit["C"]["rendered_significance_marks"] == 1
    assert audit["F"]["expected_significance_intervals"] == 1
    assert audit["F"]["rendered_significance_intervals"] == 1
    mapping = json.loads((args.output_dir / "source_data_mapping.json").read_text())
    assert mapping["no_sampling"] is True
    assert set(mapping["panels"]) == {"A", "B", "C", "D", "E", "F", "Supplementary"}
    qa = json.loads((args.output_dir / "pre_submission_qa.json").read_text())
    assert qa["checks"]["alignment_status"] == "pass"
    assert qa["checks"]["collision_status"] == "pass"
    assert qa["status"] == "pass"
