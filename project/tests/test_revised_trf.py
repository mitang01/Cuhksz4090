from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from speech_strf.revised_trf import (
    FitSpec,
    PRIMARY_REDUCTIONS,
    RevisedTRFRunner,
    _fit_equal_rank,
    _model_masks,
    _prepare_predictors,
    fit_revised_encoding,
    load_revised_config,
)
from speech_strf.revised_plots import load_kernel_series, load_trajectory


MODELS = [
    "hubert_base",
    "hubert_large_refactor_rerun",
    "wav2vec2_base",
    "wav2vec2_large",
    "wavlm_base_plus",
    "wavlm_large",
    "data2vec_audio_base",
    "xls_r_300m",
    "whisper_medium_encoder",
]
ROOT = Path(__file__).parents[1]


def _records(count=6, frames=48):
    generator = np.random.default_rng(27)
    records = {}
    for index in range(count):
        recording_id = f"r{index}"
        dense = generator.normal(size=(frames, 7))
        lexical = generator.normal(size=(8, 6))
        onsets = np.arange(2, 42, 5)
        records[recording_id] = {
            "dense": dense,
            "families": [
                "acoustic",
                "acoustic",
                "prosodic",
                "phonetic",
                "word",
                "word",
                "word",
            ],
            "blocks": [
                "acoustic",
                "acoustic",
                "prosodic",
                "phonetic",
                "word_timing",
                "word_timing",
                "word_timing",
            ],
            "lexical_vectors": lexical,
            "lexical_onset_indices": onsets,
            "targets": np.column_stack(
                [
                    dense[:, 0],
                    dense[:, 2],
                    dense[:, 3],
                    dense[:, 4] + dense[:, 5],
                ]
            ),
            "duration_seconds": frames / 50,
        }
    return records


def _fit(records, **kwargs):
    return fit_revised_encoding(
        records,
        {key: key for key in records},
        outer_mode="grouped",
        outer_folds=5,
        inner_folds=4,
        alphas=[1.0],
        lags_seconds=[0.0],
        rate_hz=50,
        lexical_components=2,
        target_components=2,
        reductions=("word", "lexical"),
        seed=17,
        **kwargs,
    )


def test_nested_fit_refits_lexical_projection_and_reports_recording_scores(
    monkeypatch,
):
    records = _records()
    from speech_strf import revised_trf

    original = revised_trf._fit_lexical_projection
    calls = []

    def recording_call(values, train_ids, eval_ids, *args, **kwargs):
        assert not set(train_ids) & set(eval_ids)
        calls.append((tuple(train_ids), tuple(eval_ids)))
        return original(values, train_ids, eval_ids, *args, **kwargs)

    monkeypatch.setattr(revised_trf, "_fit_lexical_projection", recording_call)
    result = _fit(records)

    # Five outer fits plus four inner fits inside every outer fold.
    assert len(calls) == 5 + 5 * 4
    assert set(result["scores"]["recording_id"]) == set(records)
    assert set(result["scores"]["feature_family"]) == {"word", "lexical"}
    assert np.allclose(
        result["scores"]["delta_r2"],
        result["scores"]["full_r2"] - result["scores"]["reduced_r2"],
    )
    assert all(
        report["inner_cv_repeated"] for report in result["split_reports"]
    )


def test_lexical_ablation_keeps_all_word_timing_columns():
    records = _records()
    designs, families, blocks, names, _, _ = _prepare_predictors(
        records,
        ["r0", "r1", "r2", "r3"],
        ["r4", "r5"],
        lexical_components=2,
        lags_seconds=[-0.02, 0.0, 0.02],
        rate_hz=50,
        equal_rank_cap=None,
        seed=17,
        shift=None,
        minimum_zero_seconds=0.1,
        max_abs_lag_seconds=0.02,
    )
    masks = _model_masks(families, blocks, PRIMARY_REDUCTIONS)
    block_array = np.asarray(blocks)
    assert np.all(masks["lexical"][block_array == "word_timing"])
    assert not np.any(masks["lexical"][block_array == "lexical"])
    assert not np.any(masks["word"][np.asarray(families) == "word"])
    assert all(value.shape[1] == len(families) for value in designs.values())
    assert len(names) == len(families)


def test_equal_rank_jointly_projects_timing_and_lexical_word_inputs():
    records = _records()
    lexical = {
        key: np.zeros((len(value["dense"]), 2)) for key, value in records.items()
    }
    matrices = {
        key: np.column_stack([value["dense"], lexical[key]])
        for key, value in records.items()
    }
    families = [*records["r0"]["families"], "word", "word"]
    transformed, output_families, output_blocks, output_names, report = (
        _fit_equal_rank(
            matrices,
            families,
            ["r0", "r1", "r2", "r3"],
            list(records),
            cap=3,
            seed=17,
        )
    )
    assert report["joint_word_projection"] is True
    assert 1 <= report["achieved_common_rank"] <= 3
    assert set(report["families"]) == {
        "acoustic",
        "prosodic",
        "phonetic",
        "word",
    }
    assert output_families == output_blocks
    assert len(output_names) == len(output_families)
    assert all(
        value.shape[1] == 4 * report["achieved_common_rank"]
        for value in transformed.values()
    )


def test_shift_fit_reuses_observed_alphas_folds_and_target_coordinates():
    records = _records(frames=48)
    observed = _fit(records)
    shifted = _fit(
        records,
        shift=2,
        fixed_split_reports=observed["split_reports"],
        fixed_target_assets=observed["fold_assets"],
        minimum_zero_seconds=0.1,
        max_abs_lag_seconds=0.0,
    )
    assert all(
        report["inner_cv_repeated"] is False
        for report in shifted["split_reports"]
    )
    assert all(
        report["fixed_observed_coordinates"]
        for report in shifted["target_pca_reports"]
    )
    for observed_report, shifted_report in zip(
        observed["split_reports"], shifted["split_reports"]
    ):
        assert (
            observed_report["selected_alphas"]
            == shifted_report["selected_alphas"]
        )
        assert (
            observed_report["train_recording_ids"]
            == shifted_report["train_recording_ids"]
        )
    for fold_manifest in shifted["shift_manifest"].values():
        assert set(fold_manifest) == set(records)
        assert all(
            value["circular_distance_frames"] >= 5
            for value in fold_manifest.values()
        )


def _write_scope_fixture(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    output_root = tmp_path / "outputs"
    for model in MODELS:
        destination = output_root / model
        destination.mkdir(parents=True)
        with h5py.File(destination / "activations.h5", "w") as store:
            group = store.create_group("recording")
            group.attrs["complete"] = True
            layers = [
                "layer_00_input",
                "layer_01_transformer",
                "layer_02_transformer",
            ]
            group.attrs["layer_names_json"] = json.dumps(layers)
            canonical = group.create_group("canonical")
            for layer in layers:
                canonical.create_dataset(layer, data=np.zeros((2, 2)))
    config = {
        "schema_version": 1,
        "random_seed": 17,
        "analysis_rate_hz": 50,
        "expected_recording_count": 12,
        "manifest": "outputs/manifest.csv",
        "outer_group_column": "story_id",
        "model_root": "outputs",
        "revised_features": "outputs/stage4_revision/revised_predictors",
        "output_root": "outputs/stage4_revision/revised_trf",
        "models": MODELS,
        "encoding": {
            "outer_folds": 5,
            "inner_folds": 4,
            "alphas": [0.1, 1.0, 10.0, 100.0],
            "lags_seconds": [-0.2, -0.1, 0.0, 0.1, 0.2],
            "target_pca_components": 30,
            "lexical_pca_components": 32,
        },
        "controls": {
            "selected_depths": ["input", "middle", "final"],
            "equal_rank_cap": 3,
        },
        "nulls": {"shifts": 20, "minimum_zero_seconds": 2.0},
        "statistics": {
            "bootstrap_samples": 10000,
            "confidence": 0.95,
            "test_families": {
                "primary": [
                    "acoustic",
                    "prosodic",
                    "phonetic",
                    "word",
                    "lexical",
                ],
                "rich": [
                    "acoustic",
                    "prosodic",
                    "phonetic",
                    "word",
                    "lexical",
                ],
                "equal_rank": ["acoustic", "prosodic", "phonetic", "word"],
                "zero_lag": [
                    "acoustic",
                    "prosodic",
                    "phonetic",
                    "word",
                    "lexical",
                ],
                "null": ["word"],
            },
        },
    }
    config_path = config_dir / "revised_trf.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_exact_scope_manifests_have_27_units_per_control_and_180_nulls(
    tmp_path,
):
    config_path = _write_scope_fixture(tmp_path)
    assert len(load_revised_config(config_path)["models"]) == 9
    result = RevisedTRFRunner(config_path).build_manifests()
    assert result["primary_units"] == 30  # 27 grouped + 3 HuBERT LORO
    assert result["control_units"] == 81  # 27 for each of three controls
    assert result["null_units"] == 180
    assert result["expected_units"] == 291
    resolved = result["resolved_layers"]
    assert all(
        value["selected_depths"]
        == {
            "input": "layer_00_input",
            "middle": "layer_01_transformer",
            "final": "layer_02_transformer",
        }
        for value in resolved.values()
    )


def test_runner_publishes_and_resumes_an_atomic_primary_unit(tmp_path):
    config_path = _write_scope_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["expected_recording_count"] = 6
    config["encoding"]["alphas"] = [1.0]
    config["encoding"]["target_pca_components"] = 2
    config["encoding"]["lexical_pca_components"] = 2
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    records = _records()
    outputs = tmp_path / "outputs"
    manifest = pd.DataFrame(
        {
            "recording_id": list(records),
            "story_id": list(records),
        }
    )
    manifest.to_csv(outputs / "manifest.csv", index=False)
    revised_root = outputs / "stage4_revision" / "revised_predictors"
    for design in ("original", "rich"):
        (revised_root / design).mkdir(parents=True)
    archive_rows = []
    store_path = outputs / "hubert_base" / "activations.h5"
    with h5py.File(store_path, "w") as store:
        for recording_id, record in records.items():
            times = np.arange(len(record["dense"])) / 50
            group = store.require_group(recording_id)
            group.attrs["complete"] = True
            group.attrs["layer_names_json"] = json.dumps(
                ["layer_00_input", "layer_01_transformer"]
            )
            group.create_dataset("canonical_timestamps", data=times)
            canonical = group.create_group("canonical")
            canonical.create_dataset("layer_00_input", data=record["targets"])
            canonical.create_dataset(
                "layer_01_transformer", data=record["targets"] * 0.9
            )
            for design in ("original", "rich"):
                destination = revised_root / design / f"{recording_id}.npz"
                np.savez(
                    destination,
                    matrix=record["dense"],
                    times=times,
                    names=np.asarray(
                        [
                            "broadband_envelope",
                            "acoustic_1",
                            "prosody",
                            "phone",
                            "word_onset",
                            "word_boundary",
                            "word_duration",
                        ]
                    ),
                    families=np.asarray(record["families"]),
                    blocks=np.asarray(record["blocks"]),
                    lexical_vectors=record["lexical_vectors"],
                    lexical_onset_indices=record["lexical_onset_indices"],
                    lexical_word_indices=np.arange(
                        len(record["lexical_vectors"])
                    ),
                    lexical_start_seconds=record[
                        "lexical_onset_indices"
                    ] / 50,
                    lexical_end_seconds=(
                        record["lexical_onset_indices"] + 1
                    ) / 50,
                )
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                destination.with_name(f"{destination.name}.sha256").write_text(
                    f"{digest}  {destination.name}\n", encoding="ascii"
                )
                archive_rows.append(
                    {
                        "recording_id": recording_id,
                        "design": design,
                        "archive": str(destination),
                        "archive_sha256": digest,
                    }
                )
    pd.DataFrame(archive_rows).to_csv(
        revised_root / "archive_manifest.csv", index=False
    )

    runner = RevisedTRFRunner(config_path)
    spec = FitSpec(
        model="hubert_base",
        layer="layer_00_input",
        variant="primary",
    )
    first = runner.fit(spec)
    assert first["state"] == "complete"
    destination = Path(first["path"])
    assert (destination / "scores.csv").is_file()
    assert (destination / "predictions.npz").is_file()
    assert (destination / "model_assets.npz").is_file()
    scores = pd.read_csv(destination / "scores.csv")
    assert set(scores["feature_family"]) == set(PRIMARY_REDUCTIONS)
    assert set(scores["recording_id"]) == set(records)
    trajectory, trajectory_metadata, _ = load_trajectory(
        destination, "r0", 0
    )
    assert len(trajectory) == len(records["r0"]["dense"])
    assert trajectory_metadata["display_target"] == "PC1"
    kernel, kernel_metadata, _ = load_kernel_series(
        destination, "r0", "broadband_envelope", 0
    )
    assert kernel["lag_ms"].tolist() == [-200, -100, 0, 100, 200]
    assert "per one outer-training standard deviation" in kernel_metadata[
        "coefficient_units_plotted"
    ]
    second = runner.fit(spec)
    assert second == {"state": "resumed", "path": str(destination)}

    runner.fit(
        FitSpec(
            model="hubert_base",
            layer="layer_01_transformer",
            variant="primary",
        )
    )
    plot_root = tmp_path / "plot_outputs"
    commands = [
        [
            sys.executable,
            "scripts/plot_layer_profiles.py",
            "--config",
            str(config_path),
            "--model",
            "hubert_base",
            "--output",
            str(plot_root / "profiles"),
        ],
        [
            sys.executable,
            "scripts/plot_predicted_activations.py",
            "--config",
            str(config_path),
            "--model",
            "hubert_base",
            "--recording-id",
            "r0",
            "--start-seconds",
            "0",
            "--end-seconds",
            "0.2",
            "--selection",
            "best-lexical",
            "--output",
            str(plot_root / "trajectories"),
        ],
        [
            sys.executable,
            "scripts/plot_trf_kernels.py",
            "--config",
            str(config_path),
            "--model",
            "hubert_base",
            "--layer",
            "layer_00_input",
            "--recording-id",
            "r0",
            "--predictor",
            "broadband_envelope",
            "--output",
            str(plot_root / "kernels"),
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert len(list(plot_root.rglob("*.pdf"))) == 4
    assert len(list(plot_root.rglob("*.png"))) == 4
    assert list(plot_root.rglob("*.series.csv"))
    assert len(list(plot_root.rglob("*.metadata.json"))) == 4
