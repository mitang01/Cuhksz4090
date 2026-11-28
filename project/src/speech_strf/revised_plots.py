"""Auditable plotting helpers for revised TRF artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .revised_trf import (
    FIT_SCHEMA_VERSION,
    PRIMARY_REDUCTIONS,
    FitSpec,
    RevisedTRFRunner,
)
from .stage4_statistics import recording_cluster_bootstrap_ci


FAMILY_LABELS = {
    "acoustic": "Acoustic",
    "prosodic": "Prosodic",
    "phonetic": "Phonetic",
    "word": "Joint word",
    "lexical": "Lexical only",
}
FAMILY_COLORS = {
    "acoustic": "#1f77b4",
    "prosodic": "#ff7f0e",
    "phonetic": "#2ca02c",
    "word": "#9467bd",
    "lexical": "#d62728",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def deterministic_seed(seed: int, *parts: str) -> int:
    payload = "\0".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def primary_unit(runner: RevisedTRFRunner, model: str, layer: str) -> Path:
    spec = FitSpec(model=model, layer=layer, variant="primary")
    path = runner.unit_path(spec)
    required = (
        "scores.csv",
        "split_reports.json",
        "fold_assets.npz",
        "predictions.npz",
        "model_assets.npz",
        "status.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Primary fit unit {model}/{layer} lacks {missing}; rerun this "
            f"revised fit with schema version {FIT_SCHEMA_VERSION}"
        )
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    if (
        status.get("state") != "complete"
        or status.get("fit_schema_version") != FIT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Primary fit unit {model}/{layer} is not a complete plotting-schema unit"
        )
    if not runner._valid_unit(path, runner._expected_metadata(spec)):
        raise ValueError(
            f"Primary fit unit {model}/{layer} failed provenance or file validation"
        )
    return path


def collect_layer_profile_series(
    runner: RevisedTRFRunner,
    *,
    models: Sequence[str],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    expected_ids = set(runner._recording_ids_groups()[0])
    for model in models:
        layers = runner.layers(model)
        if len(layers) < 2:
            raise ValueError(f"{model}: layer profile requires at least two layers")
        for layer_index, layer in enumerate(layers):
            unit = primary_unit(runner, model, layer)
            score_path = unit / "scores.csv"
            sources[str(score_path)] = sha256_file(score_path)
            scores = pd.read_csv(score_path)
            scores = scores.loc[
                (scores["outer_mode"] == "grouped")
                & scores["feature_family"].isin(PRIMARY_REDUCTIONS)
            ]
            for family in PRIMARY_REDUCTIONS:
                family_rows = scores.loc[
                    scores["feature_family"] == family
                ].sort_values("recording_id")
                if family_rows["recording_id"].duplicated().any():
                    raise ValueError(
                        f"{model}/{layer}/{family}: duplicate recording scores"
                    )
                values = family_rows["delta_r2"].to_numpy(dtype=float)
                if set(family_rows["recording_id"]) != expected_ids:
                    raise ValueError(
                        f"{model}/{layer}/{family}: incomplete recording vector"
                    )
                low, high = recording_cluster_bootstrap_ci(
                    values,
                    n_bootstrap=n_bootstrap,
                    seed=deterministic_seed(seed, model, layer, family),
                    confidence=confidence,
                )
                rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "native_layer_index": layer_index,
                        "relative_depth": layer_index / (len(layers) - 1),
                        "feature_family": family,
                        "equal_recording_mean_delta_r2": float(values.mean()),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "n_recordings": len(values),
                    }
                )
    return pd.DataFrame(rows), sources


def held_out_fold(unit: Path, recording_id: str) -> tuple[int, dict[str, Any]]:
    reports = json.loads(
        (unit / "split_reports.json").read_text(encoding="utf-8")
    )
    matches = [
        value
        for value in reports
        if recording_id in value["test_recording_ids"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{recording_id}: expected one held-out fold in {unit}, found {len(matches)}"
        )
    if recording_id in matches[0]["train_recording_ids"]:
        raise ValueError(f"{recording_id} appears in both training and held-out sets")
    return int(matches[0]["outer_fold"]), matches[0]


def selected_inner_score(
    report: Mapping[str, Any], model_name: str
) -> float:
    alpha = float(report["selected_alphas"][model_name])
    candidates = report["inner_cv_mean_scores"][model_name]
    matches = [
        float(value)
        for key, value in candidates.items()
        if np.isclose(float(key), alpha, rtol=0, atol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot resolve selected alpha {alpha} for {model_name}"
        )
    return matches[0]


def layer_selection_table(
    runner: RevisedTRFRunner,
    model: str,
    recording_id: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    sources: dict[str, str] = {}
    reference_split = None
    for layer_index, layer in enumerate(runner.layers(model)):
        unit = primary_unit(runner, model, layer)
        report_path = unit / "split_reports.json"
        sources[str(report_path)] = sha256_file(report_path)
        fold, report = held_out_fold(unit, recording_id)
        split = (fold, report["train_recording_ids"], report["test_recording_ids"])
        if reference_split is None:
            reference_split = split
        elif split != reference_split:
            raise ValueError(f"{model}/{layer}: layer selection requires matched recording splits")
        full = selected_inner_score(report, "full")
        lexical_omitted = selected_inner_score(report, "lexical")
        rows.append(
            {
                "model": model,
                "layer": layer,
                "native_layer_index": layer_index,
                "outer_fold": fold,
                "selected_full_alpha": float(
                    report["selected_alphas"]["full"]
                ),
                "selected_lexical_omitted_alpha": float(
                    report["selected_alphas"]["lexical"]
                ),
                "inner_validation_full_r2": full,
                "inner_validation_lexical_omitted_r2": lexical_omitted,
                "inner_validation_lexical_delta_r2": full - lexical_omitted,
            }
        )
    return pd.DataFrame(rows), sources


def load_trajectory(
    unit: Path,
    recording_id: str,
    target_component: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    fold, report = held_out_fold(unit, recording_id)
    prefix = f"fold_{fold:02d}::{recording_id}"
    predictions_path = unit / "predictions.npz"
    model_assets_path = unit / "model_assets.npz"
    fold_assets_path = unit / "fold_assets.npz"
    with np.load(predictions_path, allow_pickle=False) as predictions:
        required = {
            f"{prefix}::times",
            f"{prefix}::target",
            f"{prefix}::full",
            f"{prefix}::lexical_omitted",
        }
        missing = required - set(predictions.files)
        if missing:
            raise ValueError(f"Prediction artifact lacks {sorted(missing)}")
        times = predictions[f"{prefix}::times"]
        target = predictions[f"{prefix}::target"]
        full = predictions[f"{prefix}::full"]
        reduced = predictions[f"{prefix}::lexical_omitted"]
    if not (target.ndim == 2 and target.shape == full.shape == reduced.shape):
        raise ValueError("Observed/full/lexical-omitted target coordinates differ")
    if times.shape != (len(target),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Prediction timestamps must match the target rows and increase strictly")
    if not 0 <= target_component < target.shape[1]:
        raise ValueError(
            f"Target component {target_component} is outside 0..{target.shape[1] - 1}"
        )
    fold_prefix = f"fold_{fold:02d}"
    with np.load(model_assets_path, allow_pickle=False) as assets:
        signs = assets[f"{fold_prefix}::target_component_signs"]
        retained = assets[
            f"{fold_prefix}::lexical_omitted_retained_mask"
        ].astype(bool)
        full_mean = assets[f"{fold_prefix}::scaler_mean"]
        full_scale = assets[f"{fold_prefix}::scaler_scale"]
        reduced_mean = assets[
            f"{fold_prefix}::lexical_omitted_scaler_mean"
        ]
        reduced_scale = assets[
            f"{fold_prefix}::lexical_omitted_scaler_scale"
        ]
    if not (
        np.array_equal(full_mean[retained], reduced_mean)
        and np.array_equal(full_scale[retained], reduced_scale)
    ):
        raise ValueError("Full and lexical-omitted retained-column scaling differs")
    with np.load(fold_assets_path, allow_pickle=False) as assets:
        components = assets[f"{fold_prefix}_target_components"]
    anchor = int(np.argmax(np.abs(components[target_component])))
    if components[target_component, anchor] < 0:
        raise ValueError("Saved target component does not satisfy the sign rule")
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "observed_target_pc": target[:, target_component],
            "full_prediction": full[:, target_component],
            "lexical_omitted_prediction": reduced[:, target_component],
        }
    )
    metadata = {
        "outer_fold": fold,
        "target_component_zero_based": target_component,
        "display_target": f"PC{target_component + 1}",
        "target_component_sign_rule": (
            "largest_absolute_outer_training_loading_positive"
        ),
        "target_component_original_pca_sign_multiplier": float(
            signs[target_component]
        ),
        "target_component_anchor_loading_index": anchor,
        "predictor_scaling": (
            "outer-training StandardScaler; retained-column means/scales are "
            "exactly identical in full and lexical-omitted models"
        ),
        "full_alpha": float(report["selected_alphas"]["full"]),
        "lexical_omitted_alpha": float(
            report["selected_alphas"]["lexical"]
        ),
        "train_recording_ids": report["train_recording_ids"],
        "test_recording_ids": report["test_recording_ids"],
    }
    sources = {
        str(path): sha256_file(path)
        for path in (predictions_path, model_assets_path, fold_assets_path, unit / "split_reports.json")
    }
    return frame, metadata, sources


def trajectory_scores(
    frame: pd.DataFrame, mask: np.ndarray, scope: str
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if len(selected) < 2:
        raise ValueError(f"{scope}: at least two timestamps are required")
    observed = selected["observed_target_pc"].to_numpy(dtype=np.float64)
    constant_target = bool(np.all(observed == observed[0]))
    full_r2 = reduced_r2 = delta = None
    if not constant_target:
        full_r2 = float(r2_score(observed, selected["full_prediction"]))
        reduced_r2 = float(r2_score(observed, selected["lexical_omitted_prediction"]))
        delta = full_r2 - reduced_r2
    return {
        "scope": scope,
        "frames": len(selected),
        "start_seconds": float(selected["time_seconds"].iloc[0]),
        "end_seconds": float(selected["time_seconds"].iloc[-1]),
        "full_r2": full_r2,
        "lexical_omitted_r2": reduced_r2,
        "lexical_delta_r2": delta,
        "r2_status": "undefined_constant_target" if constant_target else "defined",
    }


def load_kernel_series(
    unit: Path,
    recording_id: str,
    predictor: str,
    target_component: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    fold, report = held_out_fold(unit, recording_id)
    prefix = f"fold_{fold:02d}"
    assets_path = unit / "model_assets.npz"
    fold_assets_path = unit / "fold_assets.npz"
    with np.load(assets_path, allow_pickle=False) as assets:
        predictors = assets[f"{prefix}::predictor_names"].astype(str)
        feature_names = assets[f"{prefix}::feature_names"].astype(str)
        families = assets[f"{prefix}::feature_families"].astype(str)
        blocks = assets[f"{prefix}::feature_blocks"].astype(str)
        lags = assets[f"{prefix}::lag_ms"]
        means = assets[f"{prefix}::scaler_mean"]
        scales = assets[f"{prefix}::scaler_scale"]
        standardized = assets[f"{prefix}::full_coef_standardized"]
        original = assets[
            f"{prefix}::full_coef_original_predictor_units"
        ]
        signs = assets[f"{prefix}::target_component_signs"]
    if not 0 <= target_component < standardized.shape[0]:
        raise ValueError("Requested target component is unavailable")
    selected = np.flatnonzero(predictors == predictor)
    if len(selected) == 0:
        available = sorted(set(predictors))
        raise ValueError(
            f"Predictor {predictor!r} is absent; available predictors include "
            f"{available[:20]}"
        )
    series = (
        pd.DataFrame(
            {
                "predictor": predictors[selected],
                "feature_column": feature_names[selected],
                "feature_family": families[selected],
                "feature_block": blocks[selected],
                "lag_ms": lags[selected],
                "coefficient_standardized_predictor": standardized[
                    target_component, selected
                ],
                "coefficient_original_predictor_units": original[
                    target_component, selected
                ],
                "training_scaler_mean": means[selected],
                "training_scaler_scale": scales[selected],
                "target_component_original_pca_sign_multiplier": signs[
                    target_component
                ],
            }
        )
        .sort_values("lag_ms")
        .reset_index(drop=True)
    )
    if series["lag_ms"].duplicated().any():
        raise ValueError(f"{predictor}: multiple columns exist at one lag")
    with np.load(fold_assets_path, allow_pickle=False) as assets:
        components = assets[f"{prefix}_target_components"]
    anchor = int(np.argmax(np.abs(components[target_component])))
    if components[target_component, anchor] < 0:
        raise ValueError("Saved kernel target component does not satisfy the sign rule")
    metadata = {
        "outer_fold": fold,
        "display_target": f"PC{target_component + 1}",
        "target_component_zero_based": target_component,
        "target_component_sign_rule": (
            "largest_absolute_outer_training_loading_positive"
        ),
        "target_component_anchor_loading_index": anchor,
        "full_alpha": float(report["selected_alphas"]["full"]),
        "coefficient_units_plotted": (
            "outer-training target PCA score units per one outer-training "
            "standard deviation of the named predictor"
        ),
        "coefficient_units_exported_original": (
            "outer-training target PCA score units per original predictor unit"
        ),
        "visualization_fit": "revised_primary_fit",
    }
    sources = {
        str(path): sha256_file(path)
        for path in (assets_path, fold_assets_path, unit / "split_reports.json")
    }
    return series, metadata, sources
