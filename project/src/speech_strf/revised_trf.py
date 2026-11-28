"""Leakage-safe revised TRF fits with sparse contextual lexical events."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .design_matrix import lagged_design
from .lexical_predictors import TOP_LEVEL_FAMILIES, rasterize_lexical_events
from .stage4_compute_scope import resolve_depth_layers
from .stage4_runner import discover_layers
from .stage4_statistics import (
    benjamini_hochberg,
    exact_paired_sign_flip,
    recording_cluster_bootstrap_ci,
)


PRIMARY_REDUCTIONS = ("acoustic", "prosodic", "phonetic", "word", "lexical")
FAMILY_REDUCTIONS = PRIMARY_REDUCTIONS[:-1]
CONTROL_VARIANTS = ("rich", "equal_rank", "zero_lag")
FIT_SCHEMA_VERSION = 3
PAPER_MODELS = (
    "hubert_base",
    "hubert_large_refactor_rerun",
    "wav2vec2_base",
    "wav2vec2_large",
    "wavlm_base_plus",
    "wavlm_large",
    "data2vec_audio_base",
    "xls_r_300m",
    "whisper_medium_encoder",
)
FIT_FILES = (
    "scores.csv",
    "split_reports.json",
    "target_pca_coverage.csv",
    "predictor_coverage.csv",
    "fold_assets.npz",
    "status.json",
)
PLOTTING_FIT_FILES = ("predictions.npz", "model_assets.npz")


@dataclass(frozen=True)
class FitSpec:
    model: str
    layer: str
    variant: str
    outer_mode: str = "grouped"
    shift: int | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_revised_config(
    path: str | Path = "configs/revised_trf.yaml",
) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Revised TRF config must be a mapping")
    models = config.get("models")
    if tuple(models or []) != PAPER_MODELS:
        raise ValueError(
            "Revised TRF config must contain the fixed ordered nine paper models"
        )
    encoding = config.get("encoding", {})
    if int(encoding.get("outer_folds", 0)) != 5:
        raise ValueError("Primary revised fits require exactly five outer folds")
    if int(encoding.get("inner_folds", 0)) != 4:
        raise ValueError("Revised fits require exactly four inner folds")
    if [float(value) for value in encoding.get("lags_seconds", [])] != [
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
    ]:
        raise ValueError("Primary lag convention must be -200,-100,0,100,200 ms")
    alphas = [float(value) for value in encoding.get("alphas", [])]
    if not alphas or any(value < 0 or not np.isfinite(value) for value in alphas):
        raise ValueError("Ridge alpha grid must contain finite nonnegative values")
    if not 1 <= int(encoding.get("target_pca_components", 0)) <= 30:
        raise ValueError("Target PCA requires 1 to 30 components")
    if int(encoding.get("lexical_pca_components", 0)) < 1:
        raise ValueError("Lexical PCA component count must be positive")
    controls = config.get("controls", {})
    if controls.get("selected_depths") != ["input", "middle", "final"]:
        raise ValueError("Controls require input, middle, and final depth")
    if int(controls.get("equal_rank_cap", 0)) != 3:
        raise ValueError("Equal-rank control must be capped at three")
    nulls = config.get("nulls", {})
    if int(nulls.get("shifts", 0)) != 20:
        raise ValueError("Revised alignment control requires exactly 20 shifts")
    if float(nulls.get("minimum_zero_seconds", 0)) < 2:
        raise ValueError("Null exclusion around zero must be at least two seconds")
    expected_tests = {
        "primary": list(PRIMARY_REDUCTIONS),
        "rich": list(PRIMARY_REDUCTIONS),
        "equal_rank": list(FAMILY_REDUCTIONS),
        "zero_lag": list(PRIMARY_REDUCTIONS),
        "null": ["word"],
    }
    if config.get("statistics", {}).get("test_families") != expected_tests:
        raise ValueError(
            "Statistics test families must explicitly match the revised panels"
        )
    return config


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _safe(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\x00" in value:
        raise ValueError(f"Unsafe path component {value!r}")
    return value


def _duration(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    return float(times[-1] + np.median(np.diff(times)))


def _load_recordings(
    feature_root: Path,
    activation_store: Path,
    layer: str,
    recording_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    reference_schema = None
    with h5py.File(activation_store, "r") as store:
        for recording_id in recording_ids:
            feature_path = feature_root / f"{recording_id}.npz"
            sidecar = feature_path.with_name(f"{feature_path.name}.sha256")
            if not sidecar.is_file():
                raise FileNotFoundError(f"Missing revised archive sidecar {sidecar}")
            sidecar_fields = sidecar.read_text(encoding="ascii").split()
            if (
                len(sidecar_fields) != 2
                or sidecar_fields[1] != feature_path.name
                or sidecar_fields[0] != _sha256(feature_path)
            ):
                raise ValueError(f"Revised archive integrity failed: {feature_path}")
            with np.load(feature_path, allow_pickle=False) as archive:
                required = {
                    "matrix",
                    "times",
                    "names",
                    "families",
                    "blocks",
                    "lexical_vectors",
                    "lexical_onset_indices",
                    "lexical_word_indices",
                }
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(
                        f"{recording_id}: revised archive lacks {sorted(missing)}"
                    )
                dense = np.asarray(archive["matrix"], dtype=np.float64)
                times = np.asarray(archive["times"], dtype=np.float64)
                names = archive["names"].astype(str).tolist()
                families = archive["families"].astype(str).tolist()
                blocks = archive["blocks"].astype(str).tolist()
                lexical = np.asarray(archive["lexical_vectors"], dtype=np.float64)
                # Old archives encode missing token mappings as all-zero vectors.
                mapped = (
                    archive["lexical_mapped"]
                    if "lexical_mapped" in archive.files
                    else np.any(lexical != 0, axis=1)
                )
                onset_indices = np.asarray(
                    archive["lexical_onset_indices"], dtype=np.int64
                )
                word_indices = np.asarray(
                    archive["lexical_word_indices"], dtype=np.int64
                )
            if recording_id not in store:
                raise ValueError(f"{activation_store} lacks {recording_id}")
            group = store[recording_id]
            if layer not in group["canonical"]:
                raise ValueError(f"{recording_id}: target layer {layer!r} is absent")
            targets = np.asarray(group["canonical"][layer][:], dtype=np.float64)
            target_times = np.asarray(
                group["canonical_timestamps"][:], dtype=np.float64
            )
            if not np.array_equal(times, target_times):
                raise ValueError(
                    f"{recording_id}: revised features and activations differ in time"
                )
            if (
                dense.shape != (len(times), len(names))
                or len(names) != len(families)
                or len(names) != len(blocks)
                or targets.shape[0] != len(times)
                or lexical.ndim != 2
                or len(lexical) != len(onset_indices)
                or len(lexical) != len(word_indices)
                or mapped.shape != (len(lexical),)
                or mapped.dtype.kind != "b"
            ):
                raise ValueError(f"{recording_id}: revised archive dimensions disagree")
            if set(families) != set(TOP_LEVEL_FAMILIES):
                raise ValueError(
                    f"{recording_id}: expected four top-level predictor families"
                )
            expected_blocks = ["word_timing" if family == "word" else family for family in families]
            if blocks != expected_blocks or len(set(names)) != len(names) or "recording_onset" in names:
                raise ValueError(f"{recording_id}: invalid dense predictor schema")
            schema = (names, families, blocks)
            if reference_schema is None:
                reference_schema = schema
            elif schema != reference_schema:
                raise ValueError(f"{recording_id}: predictor column order differs across recordings")
            if not np.array_equal(word_indices, np.arange(len(lexical))):
                raise ValueError(f"{recording_id}: lexical word indexes are invalid")
            if any(not np.isfinite(values).all() for values in (dense, lexical, targets)):
                raise ValueError(f"{recording_id}: predictors or activations contain nonfinite values")
            records[recording_id] = {
                "dense": dense,
                "times": times,
                "names": names,
                "families": families,
                "blocks": blocks,
                "lexical_vectors": lexical,
                "lexical_mapped": mapped,
                "lexical_onset_indices": onset_indices,
                "targets": targets,
                "duration_seconds": _duration(times),
            }
    target_widths = {record["targets"].shape[1] for record in records.values()}
    lexical_widths = {
        record["lexical_vectors"].shape[1] for record in records.values()
    }
    if len(target_widths) != 1 or len(lexical_widths) != 1:
        raise ValueError("Target or lexical dimensions differ between recordings")
    return records


def _validate_shared_lexical_events(
    original_root: Path,
    rich_root: Path,
    recording_ids: Sequence[str],
) -> None:
    fields = (
        "lexical_vectors",
        "lexical_onset_indices",
        "lexical_word_indices",
        "lexical_start_seconds",
        "lexical_end_seconds",
    )
    for recording_id in recording_ids:
        with np.load(
            original_root / f"{recording_id}.npz", allow_pickle=False
        ) as original, np.load(
            rich_root / f"{recording_id}.npz", allow_pickle=False
        ) as rich:
            for field in fields:
                if field not in original.files or field not in rich.files:
                    raise ValueError(
                        f"{recording_id}: revised design lacks lexical field {field}"
                    )
                if not np.array_equal(original[field], rich[field]):
                    raise ValueError(
                        f"{recording_id}: rich and original lexical events differ "
                        f"for {field}"
                    )
            if "lexical_mapped" in original.files or "lexical_mapped" in rich.files:
                if (
                    "lexical_mapped" not in original.files
                    or "lexical_mapped" not in rich.files
                    or not np.array_equal(original["lexical_mapped"], rich["lexical_mapped"])
                ):
                    raise ValueError(f"{recording_id}: original and rich lexical masks differ")


def _fit_lexical_projection(
    records: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
    requested: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    mapped = {
        key: records[key].get(
            "lexical_mapped", np.ones(len(records[key]["lexical_vectors"]), dtype=bool)
        )
        for key in [*train_ids, *eval_ids]
    }
    training = np.vstack([records[key]["lexical_vectors"][mapped[key]] for key in train_ids])
    maximum = min(training.shape[0] - 1, training.shape[1])
    achieved = min(int(requested), maximum)
    if achieved < 1:
        raise ValueError("Lexical PCA requires at least two mapped training words")
    projector = PCA(n_components=achieved, svd_solver="full", random_state=seed)
    projector.fit(training)
    if not np.isfinite(projector.explained_variance_ratio_).all():
        raise ValueError("Mapped training word embeddings have no usable variance")
    transformed: dict[str, np.ndarray] = {}
    for recording_id in [*train_ids, *eval_ids]:
        valid = mapped[recording_id]
        events = (
            projector.transform(records[recording_id]["lexical_vectors"][valid])
            if valid.any() else np.empty((0, achieved))
        )
        transformed[recording_id] = rasterize_lexical_events(
            events,
            records[recording_id]["lexical_onset_indices"][valid],
            len(records[recording_id]["dense"]),
        )
    return transformed, {
        "requested_components": int(requested),
        "achieved_components": int(achieved),
        "training_word_occurrences": int(len(training)),
        "excluded_unmapped_training_words": int(sum((~mapped[key]).sum() for key in train_ids)),
        "input_dimensions": int(training.shape[1]),
        "explained_variance_coverage": float(
            projector.explained_variance_ratio_.sum()
        ),
        "train_recording_ids": list(train_ids),
        "test_recording_ids": list(eval_ids),
    }


def _word_shift_offset(
    recording_id: str,
    frames: int,
    *,
    shift: int,
    seed: int,
    rate_hz: float,
    minimum_zero_seconds: float,
    max_abs_lag_seconds: float,
) -> tuple[int, dict[str, Any]]:
    minimum = max(
        1,
        int(np.ceil(minimum_zero_seconds * rate_hz)),
        int(np.floor(max_abs_lag_seconds * rate_hz)) + 1,
    )
    maximum = frames - minimum
    if maximum < minimum:
        raise ValueError(
            f"{recording_id}: {frames} frames cannot support a shift with "
            f"minimum circular distance {minimum}"
        )
    payload = (
        f"speech-strf-stage4-null-v1\0{seed}\0{shift}\0"
        f"{recording_id}\0word"
    ).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    generator = np.random.default_rng(derived_seed)
    offset = int(generator.integers(minimum, maximum + 1))
    return offset, {
        "offset_frames": offset,
        "offset_seconds": offset / rate_hz,
        "circular_distance_frames": min(offset, frames - offset),
        "circular_distance_seconds": min(offset, frames - offset) / rate_hz,
        "permitted_offset_frames": [minimum, maximum],
        "derived_seed": derived_seed,
    }


def _fit_equal_rank(
    matrices: Mapping[str, np.ndarray],
    families: Sequence[str],
    train_ids: Sequence[str],
    all_ids: Sequence[str],
    *,
    cap: int,
    seed: int,
) -> tuple[
    dict[str, np.ndarray],
    list[str],
    list[str],
    list[str],
    dict[str, Any],
]:
    family_array = np.asarray(families)
    stacked = {
        family: np.vstack(
            [matrices[key][:, family_array == family] for key in train_ids]
        )
        for family in TOP_LEVEL_FAMILIES
    }
    available = {
        family: int(np.linalg.matrix_rank(values))
        for family, values in stacked.items()
    }
    rank = min(int(cap), *available.values())
    if rank < 1:
        raise ValueError("Equal-rank control found a zero-rank family")
    reducers: dict[str, TruncatedSVD] = {}
    padded: dict[str, np.ndarray] = {}
    for family, values in stacked.items():
        padded[family] = (
            values
            if values.shape[1] >= 2
            else np.column_stack([values, np.zeros(len(values))])
        )
        reducers[family] = TruncatedSVD(
            n_components=rank,
            algorithm="randomized",
            n_iter=7,
            random_state=seed,
        ).fit(padded[family])
    output: dict[str, np.ndarray] = {}
    for recording_id in all_ids:
        blocks = []
        for family in TOP_LEVEL_FAMILIES:
            values = matrices[recording_id][:, family_array == family]
            if values.shape[1] == 1:
                values = np.column_stack([values, np.zeros(len(values))])
            blocks.append(reducers[family].transform(values))
        output[recording_id] = np.column_stack(blocks)
    output_families = [
        family for family in TOP_LEVEL_FAMILIES for _ in range(rank)
    ]
    output_names = [
        f"{family}_equal_rank_{component + 1:02d}"
        for family in TOP_LEVEL_FAMILIES
        for component in range(rank)
    ]
    return output, output_families, list(output_families), output_names, {
        "enabled": True,
        "achieved_common_rank": int(rank),
        "rank_cap": int(cap),
        "train_recording_ids": list(train_ids),
        "families": {
            family: {
                "input_columns": int(stacked[family].shape[1]),
                "available_training_rank": available[family],
                "achieved_rank": int(rank),
                "explained_variance_coverage": float(
                    reducers[family].explained_variance_ratio_.sum()
                ),
            }
            for family in TOP_LEVEL_FAMILIES
        },
        "joint_word_projection": True,
    }


def _prepare_predictors(
    records: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
    *,
    lexical_components: int,
    lags_seconds: Sequence[float],
    rate_hz: float,
    equal_rank_cap: int | None,
    seed: int,
    shift: int | None,
    minimum_zero_seconds: float,
    max_abs_lag_seconds: float,
) -> tuple[
    dict[str, np.ndarray],
    list[str],
    list[str],
    list[str],
    dict[str, Any],
    dict[str, Any],
]:
    all_ids = [*train_ids, *eval_ids]
    lexical_grids, lexical_report = _fit_lexical_projection(
        records, train_ids, eval_ids, lexical_components, seed
    )
    dense_families = list(next(iter(records.values()))["families"])
    dense_blocks = list(next(iter(records.values()))["blocks"])
    dense_names = list(
        next(iter(records.values())).get(
            "names", [f"dense_{index:03d}" for index in range(len(dense_families))]
        )
    )
    lexical_width = next(iter(lexical_grids.values())).shape[1]
    families = [*dense_families, *(["word"] * lexical_width)]
    blocks = [*dense_blocks, *(["lexical"] * lexical_width)]
    names = [
        *dense_names,
        *[f"lexical_pc_{index + 1:02d}" for index in range(lexical_width)],
    ]
    matrices = {
        key: np.column_stack([records[key]["dense"], lexical_grids[key]])
        for key in all_ids
    }

    shift_manifest: dict[str, Any] = {}
    if shift is not None:
        word_columns = np.asarray(families) == "word"
        for recording_id in all_ids:
            offset, report = _word_shift_offset(
                recording_id,
                len(matrices[recording_id]),
                shift=shift,
                seed=seed,
                rate_hz=rate_hz,
                minimum_zero_seconds=minimum_zero_seconds,
                max_abs_lag_seconds=max_abs_lag_seconds,
            )
            matrices[recording_id][:, word_columns] = np.roll(
                matrices[recording_id][:, word_columns], offset, axis=0
            )
            shift_manifest[recording_id] = report

    capacity_report: dict[str, Any] = {"enabled": False}
    if equal_rank_cap is not None:
        matrices, families, blocks, names, capacity_report = _fit_equal_rank(
            matrices,
            families,
            train_ids,
            all_ids,
            cap=equal_rank_cap,
            seed=seed,
        )

    designs: dict[str, np.ndarray] = {}
    lagged_families: list[str] | None = None
    lagged_blocks: list[str] | None = None
    lagged_names: list[str] | None = None
    for recording_id in all_ids:
        design, current = lagged_design(
            matrices[recording_id],
            np.full(len(matrices[recording_id]), recording_id, dtype=object),
            list(families),
            list(lags_seconds),
            rate_hz,
        )
        current_blocks = list(blocks) * len(lags_seconds)
        current_names = [
            f"{name}__lag_{int(round(lag * 1000)):+d}ms"
            for lag in lags_seconds
            for name in names
        ]
        if lagged_families is not None and current != lagged_families:
            raise RuntimeError("Lagged family schema changed between recordings")
        designs[recording_id] = design
        lagged_families = current
        lagged_blocks = current_blocks
        lagged_names = current_names
    return (
        designs,
        lagged_families or [],
        lagged_blocks or [],
        lagged_names or [],
        lexical_report,
        {
            "capacity": capacity_report,
            "shift": shift_manifest,
        },
    )


def _fit_target_projection(
    records: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
    requested: int,
    seed: int,
    *,
    fixed_mean: np.ndarray | None = None,
    fixed_components: np.ndarray | None = None,
    fixed_signs: np.ndarray | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    training = np.vstack([records[key]["targets"] for key in train_ids])
    if fixed_mean is None or fixed_components is None:
        achieved = min(int(requested), training.shape[1], max(1, len(training) - 1))
        projector = PCA(
            n_components=achieved, svd_solver="full", random_state=seed
        ).fit(training)
        mean = projector.mean_
        components = np.asarray(projector.components_, dtype=np.float64)
        anchor_indexes = np.argmax(np.abs(components), axis=1)
        signs = np.sign(
            components[np.arange(len(components)), anchor_indexes]
        )
        signs[signs == 0] = 1.0
        components = components * signs[:, None]
        coverage_curve = np.cumsum(projector.explained_variance_ratio_)
        coverage = float(coverage_curve[-1])
    else:
        mean = np.asarray(fixed_mean, dtype=np.float64)
        components = np.asarray(fixed_components, dtype=np.float64)
        signs = (
            np.asarray(fixed_signs, dtype=np.float64)
            if fixed_signs is not None
            else np.ones(components.shape[0], dtype=np.float64)
        )
        if mean.shape != (training.shape[1],) or components.shape[1] != training.shape[1]:
            raise ValueError("Fixed target PCA assets have incompatible dimensions")
        if signs.shape != (components.shape[0],):
            raise ValueError("Fixed target PCA signs have incompatible dimensions")
        achieved = components.shape[0]
        centered = training - mean
        total_variance = np.var(centered, axis=0, ddof=1).sum()
        projected_variance = np.var(centered @ components.T, axis=0, ddof=1)
        coverage_curve = np.cumsum(projected_variance) / total_variance
        coverage = float(coverage_curve[-1])
    transform = lambda value: (records[value]["targets"] - mean) @ components.T
    train_values = {key: transform(key) for key in train_ids}
    eval_values = {key: transform(key) for key in eval_ids}
    report = {
        "target_units_before_pca": int(training.shape[1]),
        "requested_components": int(requested),
        "achieved_components": int(achieved),
        "total_explained_variance_ratio": coverage,
        "cumulative_explained_variance_ratio": coverage_curve.tolist(),
        "train_recording_ids": list(train_ids),
        "test_recording_ids": list(eval_ids),
        "fixed_observed_coordinates": fixed_mean is not None,
        "component_sign_rule": "largest_absolute_training_loading_positive",
        "component_signs": signs.tolist(),
    }
    assets = {
        "target_mean": np.asarray(mean, dtype=np.float64),
        "target_components": np.asarray(components, dtype=np.float64),
        "target_component_signs": np.asarray(signs, dtype=np.float64),
    }
    return train_values, eval_values, report, assets


def _score(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(target, prediction, multioutput="variance_weighted"))


def _model_masks(
    families: Sequence[str],
    blocks: Sequence[str],
    reductions: Sequence[str],
) -> dict[str, np.ndarray]:
    family_array = np.asarray(families)
    block_array = np.asarray(blocks)
    masks = {"full": np.ones(len(families), dtype=bool)}
    for reduction in reductions:
        if reduction == "lexical":
            masks[reduction] = block_array != "lexical"
        else:
            masks[reduction] = family_array != reduction
        if not masks[reduction].any() or masks[reduction].all():
            raise ValueError(f"Reduction {reduction!r} does not remove a valid block")
    return masks


def _fit_scaled_ridge_with_assets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: Mapping[str, np.ndarray],
    mask: np.ndarray,
    alpha: float,
) -> tuple[dict[str, np.ndarray], StandardScaler, Ridge]:
    scaler = StandardScaler().fit(x_train[:, mask])
    model = Ridge(alpha=float(alpha)).fit(
        scaler.transform(x_train[:, mask]), y_train
    )
    predictions = {
        recording_id: model.predict(scaler.transform(values[:, mask]))
        for recording_id, values in x_eval.items()
    }
    return predictions, scaler, model


def _scaled_designs(
    x_train: np.ndarray,
    x_eval: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scaler = StandardScaler().fit(x_train[:, mask])
    return (
        scaler.transform(x_train[:, mask]),
        {
            recording_id: scaler.transform(values[:, mask])
            for recording_id, values in x_eval.items()
        },
    )


def _outer_splits(
    recording_ids: Sequence[str],
    groups: Mapping[str, str],
    *,
    mode: str,
    folds: int,
) -> list[tuple[int, list[str], list[str]]]:
    ids = list(recording_ids)
    if mode == "loro":
        return [
            (index, [value for value in ids if value != held], [held])
            for index, held in enumerate(ids)
        ]
    if mode != "grouped":
        raise ValueError(f"Unknown outer mode {mode!r}")
    labels = np.asarray([groups[value] for value in ids])
    if len(np.unique(labels)) < folds:
        raise ValueError(f"Need at least {folds} distinct outer groups")
    splitter = GroupKFold(n_splits=folds)
    return [
        (
            fold,
            [ids[index] for index in train],
            [ids[index] for index in test],
        )
        for fold, (train, test) in enumerate(
            splitter.split(np.zeros(len(ids)), groups=labels)
        )
    ]


def _inner_splits(
    train_ids: Sequence[str],
    groups: Mapping[str, str],
    folds: int,
) -> list[tuple[list[str], list[str]]]:
    ids = list(train_ids)
    labels = np.asarray([groups[value] for value in ids])
    unique = len(np.unique(labels))
    if unique < folds:
        raise ValueError(
            f"Outer training split has {unique} groups; {folds} inner folds required"
        )
    splitter = GroupKFold(n_splits=folds)
    return [
        (
            [ids[index] for index in train],
            [ids[index] for index in test],
        )
        for train, test in splitter.split(np.zeros(len(ids)), groups=labels)
    ]


def _select_alphas(
    records: Mapping[str, Mapping[str, Any]],
    outer_train: Sequence[str],
    groups: Mapping[str, str],
    *,
    reductions: Sequence[str],
    alphas: Sequence[float],
    inner_folds: int,
    lexical_components: int,
    target_components: int,
    lags_seconds: Sequence[float],
    rate_hz: float,
    equal_rank_cap: int | None,
    seed: int,
    shift: int | None,
    minimum_zero_seconds: float,
    max_abs_lag_seconds: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    values = {
        name: {float(alpha): [] for alpha in alphas}
        for name in ("full", *reductions)
    }
    for inner_train, inner_test in _inner_splits(
        outer_train, groups, inner_folds
    ):
        designs, families, blocks, _, _, _ = _prepare_predictors(
            records,
            inner_train,
            inner_test,
            lexical_components=lexical_components,
            lags_seconds=lags_seconds,
            rate_hz=rate_hz,
            equal_rank_cap=equal_rank_cap,
            seed=seed,
            shift=shift,
            minimum_zero_seconds=minimum_zero_seconds,
            max_abs_lag_seconds=max_abs_lag_seconds,
        )
        y_train, y_test, _, _ = _fit_target_projection(
            records, inner_train, inner_test, target_components, seed
        )
        x_train = np.vstack([designs[key] for key in inner_train])
        target_train = np.vstack([y_train[key] for key in inner_train])
        eval_designs = {key: designs[key] for key in inner_test}
        masks = _model_masks(families, blocks, reductions)
        for name, mask in masks.items():
            scaled_train, scaled_eval = _scaled_designs(
                x_train, eval_designs, mask
            )
            for alpha in values[name]:
                model = Ridge(alpha=float(alpha)).fit(
                    scaled_train, target_train
                )
                values[name][alpha].extend(
                    _score(y_test[key], model.predict(scaled_eval[key]))
                    for key in inner_test
                )
    means = {
        name: {alpha: float(np.mean(scores)) for alpha, scores in candidates.items()}
        for name, candidates in values.items()
    }
    selected = {
        name: max(sorted(candidates), key=lambda alpha: candidates[alpha])
        for name, candidates in means.items()
    }
    return selected, means


def fit_revised_encoding(
    records: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, str],
    *,
    outer_mode: str,
    outer_folds: int,
    inner_folds: int,
    alphas: Sequence[float],
    lags_seconds: Sequence[float],
    rate_hz: float,
    lexical_components: int,
    target_components: int,
    reductions: Sequence[str],
    seed: int,
    equal_rank_cap: int | None = None,
    shift: int | None = None,
    fixed_split_reports: Sequence[Mapping[str, Any]] | None = None,
    fixed_target_assets: Mapping[str, np.ndarray] | None = None,
    minimum_zero_seconds: float = 2.0,
    max_abs_lag_seconds: float = 0.2,
    capture_plot_assets: bool = True,
) -> dict[str, Any]:
    """Fit nested revised encoding models and score held recordings separately."""
    recording_ids = sorted(records)
    if set(groups) != set(recording_ids):
        raise ValueError("Recording group mapping differs from loaded recordings")
    splits = _outer_splits(
        recording_ids, groups, mode=outer_mode, folds=outer_folds
    )
    fixed_reports = {
        int(value["outer_fold"]): value for value in (fixed_split_reports or [])
    }
    if shift is not None:
        expected_folds = {fold for fold, _, _ in splits}
        if (
            set(fixed_reports) != expected_folds
            or len(fixed_split_reports or []) != len(expected_folds)
            or fixed_target_assets is None
        ):
            raise ValueError("Shifted fits require fixed observed splits and PCA for every fold")
        required = {
            f"fold_{fold:02d}_{name}"
            for fold in expected_folds
            for name in ("target_mean", "target_components", "target_component_signs")
        }
        if missing := required - fixed_target_assets.keys():
            raise ValueError(f"Observed target PCA assets are missing {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    split_reports: list[dict[str, Any]] = []
    target_reports: list[dict[str, Any]] = []
    predictor_reports: list[dict[str, Any]] = []
    asset_arrays: dict[str, np.ndarray] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    model_arrays: dict[str, np.ndarray] = {}
    shift_manifest: dict[str, Any] = {}

    for fold, train_ids, test_ids in splits:
        fixed = fixed_reports.get(fold)
        if fixed is not None:
            if (
                list(fixed["train_recording_ids"]) != train_ids
                or list(fixed["test_recording_ids"]) != test_ids
            ):
                raise ValueError(f"Observed fold {fold} assignments do not match")
            selected = {
                "full": float(fixed["selected_alphas"]["full"]),
                **{
                    reduction: float(fixed["selected_alphas"][reduction])
                    for reduction in reductions
                },
            }
            inner_scores: dict[str, dict[str, float]] = {}
        else:
            selected, inner_scores = _select_alphas(
                records,
                train_ids,
                groups,
                reductions=reductions,
                alphas=alphas,
                inner_folds=inner_folds,
                lexical_components=lexical_components,
                target_components=target_components,
                lags_seconds=lags_seconds,
                rate_hz=rate_hz,
                equal_rank_cap=equal_rank_cap,
                seed=seed,
                shift=shift,
                minimum_zero_seconds=minimum_zero_seconds,
                max_abs_lag_seconds=max_abs_lag_seconds,
            )

        designs, families, blocks, column_names, lexical_report, preparation_report = (
            _prepare_predictors(
                records,
                train_ids,
                test_ids,
                lexical_components=lexical_components,
                lags_seconds=lags_seconds,
                rate_hz=rate_hz,
                equal_rank_cap=equal_rank_cap,
                seed=seed,
                shift=shift,
                minimum_zero_seconds=minimum_zero_seconds,
                max_abs_lag_seconds=max_abs_lag_seconds,
            )
        )
        prefix = f"fold_{fold:02d}"
        fixed_mean = (
            fixed_target_assets.get(f"{prefix}_target_mean")
            if fixed_target_assets is not None
            else None
        )
        fixed_components = (
            fixed_target_assets.get(f"{prefix}_target_components")
            if fixed_target_assets is not None
            else None
        )
        fixed_signs = (
            fixed_target_assets.get(f"{prefix}_target_component_signs")
            if fixed_target_assets is not None
            else None
        )
        y_train, y_test, target_report, target_assets = _fit_target_projection(
            records,
            train_ids,
            test_ids,
            target_components,
            seed,
            fixed_mean=fixed_mean,
            fixed_components=fixed_components,
            fixed_signs=fixed_signs,
        )
        for name, values in target_assets.items():
            asset_arrays[f"{prefix}_{name}"] = values
        target_report.update({"outer_fold": fold, "outer_mode": outer_mode})
        lexical_report.update({"outer_fold": fold, "transform": "lexical_pca"})
        target_reports.append(target_report)
        predictor_reports.append(lexical_report)
        capacity = preparation_report["capacity"]
        if capacity.get("enabled"):
            for family, report in capacity["families"].items():
                predictor_reports.append(
                    {
                        "outer_fold": fold,
                        "transform": "equal_rank",
                        "family": family,
                        **report,
                        "achieved_common_rank": capacity["achieved_common_rank"],
                    }
                )
        if shift is not None:
            shift_manifest[str(fold)] = preparation_report["shift"]

        x_train = np.vstack([designs[key] for key in train_ids])
        target_train = np.vstack([y_train[key] for key in train_ids])
        eval_designs = {key: designs[key] for key in test_ids}
        masks = _model_masks(families, blocks, reductions)
        fitted = {
            name: _fit_scaled_ridge_with_assets(
                x_train, target_train, eval_designs, mask, selected[name]
            )
            for name, mask in masks.items()
        }
        predictions = {name: value[0] for name, value in fitted.items()}
        full_predictions = predictions["full"]
        full_scaler = fitted["full"][1]
        full_model = fitted["full"][2]
        base_column_count = len(column_names) // len(lags_seconds)
        column_lags_ms = np.repeat(
            np.asarray(
                [int(round(value * 1000)) for value in lags_seconds],
                dtype=np.int32,
            ),
            base_column_count,
        )
        if capture_plot_assets:
            if "lexical" in fitted:
                lexical_mask = masks["lexical"]
                lexical_scaler = fitted["lexical"][1]
                if not (
                    np.array_equal(
                        full_scaler.mean_[lexical_mask],
                        lexical_scaler.mean_,
                    )
                    and np.array_equal(
                        full_scaler.scale_[lexical_mask],
                        lexical_scaler.scale_,
                    )
                ):
                    raise RuntimeError(
                        "Full and lexical-omitted retained-column scaling differs"
                    )
            model_arrays.update(
                {
                    f"{prefix}::feature_names": np.asarray(column_names),
                    f"{prefix}::predictor_names": np.asarray(
                        [
                            name.rsplit("__lag_", 1)[0]
                            for name in column_names
                        ]
                    ),
                    f"{prefix}::feature_families": np.asarray(families),
                    f"{prefix}::feature_blocks": np.asarray(blocks),
                    f"{prefix}::lag_ms": column_lags_ms,
                    f"{prefix}::scaler_mean": np.asarray(
                        full_scaler.mean_, dtype=np.float64
                    ),
                    f"{prefix}::scaler_scale": np.asarray(
                        full_scaler.scale_, dtype=np.float64
                    ),
                    f"{prefix}::full_coef_standardized": np.asarray(
                        full_model.coef_, dtype=np.float64
                    ),
                    f"{prefix}::full_coef_original_predictor_units": np.asarray(
                        full_model.coef_ / full_scaler.scale_[None, :],
                        dtype=np.float64,
                    ),
                    f"{prefix}::full_intercept_standardized": np.asarray(
                        full_model.intercept_, dtype=np.float64
                    ),
                    f"{prefix}::full_alpha": np.asarray(
                        [selected["full"]], dtype=np.float64
                    ),
                    f"{prefix}::train_recording_ids": np.asarray(train_ids),
                    f"{prefix}::test_recording_ids": np.asarray(test_ids),
                    f"{prefix}::target_component_signs": np.asarray(
                        target_assets["target_component_signs"],
                        dtype=np.float64,
                    ),
                    **(
                        {
                            f"{prefix}::lexical_omitted_retained_mask": np.asarray(
                                lexical_mask, dtype=bool
                            ),
                            f"{prefix}::lexical_omitted_scaler_mean": np.asarray(
                                lexical_scaler.mean_, dtype=np.float64
                            ),
                            f"{prefix}::lexical_omitted_scaler_scale": np.asarray(
                                lexical_scaler.scale_, dtype=np.float64
                            ),
                        }
                        if "lexical" in fitted
                        else {}
                    ),
                }
            )
        if len(column_names) != len(families) or len(column_names) != len(blocks):
            raise RuntimeError("Prepared predictor column metadata disagree")
        if capture_plot_assets:
            for recording_id in test_ids:
                prediction_prefix = f"{prefix}::{recording_id}"
                prediction_arrays.update(
                    {
                        f"{prediction_prefix}::times": np.asarray(
                            records[recording_id].get(
                                "times",
                                np.arange(len(y_test[recording_id])) / rate_hz,
                            ),
                            dtype=np.float64,
                        ),
                        f"{prediction_prefix}::target": np.asarray(
                            y_test[recording_id], dtype=np.float32
                        ),
                        f"{prediction_prefix}::full": np.asarray(
                            full_predictions[recording_id], dtype=np.float32
                        ),
                    }
                )
                if "lexical" in predictions:
                    prediction_arrays[
                        f"{prediction_prefix}::lexical_omitted"
                    ] = np.asarray(
                        predictions["lexical"][recording_id],
                        dtype=np.float32,
                    )
        full_scores = {
            key: _score(y_test[key], full_predictions[key]) for key in test_ids
        }
        for reduction in reductions:
            for recording_id in test_ids:
                full_r2 = full_scores[recording_id]
                reduced_r2 = _score(
                    y_test[recording_id], predictions[reduction][recording_id]
                )
                rows.append(
                    {
                        "outer_mode": outer_mode,
                        "outer_fold": fold,
                        "recording_id": recording_id,
                        "feature_family": reduction,
                        "full_r2": full_r2,
                        "reduced_r2": reduced_r2,
                        "delta_r2": full_r2 - reduced_r2,
                        "full_alpha": selected["full"],
                        "reduced_alpha": selected[reduction],
                        "frames": len(y_test[recording_id]),
                        "duration_seconds": records[recording_id][
                            "duration_seconds"
                        ],
                    }
                )
        split_reports.append(
            {
                "outer_fold": fold,
                "outer_mode": outer_mode,
                "train_recording_ids": train_ids,
                "test_recording_ids": test_ids,
                "selected_alphas": selected,
                "inner_cv_mean_scores": inner_scores,
                "inner_cv_repeated": fixed is None,
            }
        )
    return {
        "scores": pd.DataFrame(rows),
        "split_reports": split_reports,
        "target_pca_reports": target_reports,
        "predictor_reports": predictor_reports,
        "fold_assets": asset_arrays,
        "predictions": prediction_arrays,
        "model_assets": model_arrays,
        "shift_manifest": shift_manifest,
    }


class RevisedTRFRunner:
    def __init__(self, config_path: str | Path = "configs/revised_trf.yaml") -> None:
        self.config_path = Path(config_path).resolve()
        self.root = self.config_path.parents[1]
        self.config = load_revised_config(self.config_path)
        self.config_hash = _sha256(self.config_path)
        self.output_root = _resolve(self.root, self.config["output_root"])
        self.model_root = _resolve(self.root, self.config["model_root"])
        self.revised_root = _resolve(self.root, self.config["revised_features"])

    @property
    def models(self) -> list[str]:
        return [str(value) for value in self.config["models"]]

    def _manifest(self) -> pd.DataFrame:
        frame = pd.read_csv(_resolve(self.root, self.config["manifest"]))
        required = {"recording_id", self.config["outer_group_column"]}
        if not required <= set(frame):
            raise ValueError(f"Stimulus manifest lacks columns {sorted(required)}")
        if len(frame) != int(self.config["expected_recording_count"]):
            raise ValueError("Stimulus manifest has an unexpected recording count")
        if (
            frame[list(required)].isna().any().any()
            or frame["recording_id"].astype(str).duplicated().any()
            or any(frame[column].astype(str).str.strip().eq("").any() for column in required)
        ):
            raise ValueError("Manifest requires unique recording IDs and nonempty group IDs")
        return frame

    def _recording_ids_groups(self) -> tuple[list[str], dict[str, str]]:
        frame = self._manifest()
        ids = frame["recording_id"].astype(str).tolist()
        groups = dict(
            zip(
                ids,
                frame[self.config["outer_group_column"]].astype(str),
            )
        )
        return ids, groups

    def activation_store(self, model: str) -> Path:
        if model not in self.models:
            raise ValueError(f"Unknown revised model {model!r}")
        return self.model_root / model / "activations.h5"

    def layers(self, model: str) -> list[str]:
        return discover_layers(self.activation_store(model))

    def resolved_depths(self, model: str) -> dict[str, str]:
        return resolve_depth_layers(self.layers(model))

    def unit_path(self, spec: FitSpec) -> Path:
        if spec.shift is not None:
            return (
                self.output_root
                / "null"
                / _safe(spec.model)
                / _safe(spec.layer)
                / f"shift_{spec.shift:02d}"
            )
        namespace = (
            "hubert_loro"
            if spec.outer_mode == "loro"
            else (
                "primary"
                if spec.variant == "primary"
                else f"controls/{spec.variant}"
            )
        )
        return (
            self.output_root
            / namespace
            / _safe(spec.model)
            / _safe(spec.layer)
        )

    def _source_identity(self, model: str, design: str) -> dict[str, Any]:
        store = self.activation_store(model)
        stat = store.stat()
        manifest = self.revised_root / "archive_manifest.csv"
        return {
            "fit_schema_version": FIT_SCHEMA_VERSION,
            "fit_implementation_sha256": _sha256(Path(__file__)),
            "config_sha256": self.config_hash,
            "stimulus_manifest_sha256": _sha256(_resolve(self.root, self.config["manifest"])),
            "revised_archive_manifest_sha256": _sha256(manifest),
            "activation_store": str(store),
            "activation_bytes": stat.st_size,
            "activation_mtime_ns": stat.st_mtime_ns,
            "design": design,
        }

    def _valid_unit(self, path: Path, expected: Mapping[str, Any]) -> bool:
        try:
            status = json.loads((path / "status.json").read_text(encoding="utf-8"))
            plot_files = (
                list(PLOTTING_FIT_FILES)
                if expected.get("variant") == "primary"
                and expected.get("outer_mode") == "grouped"
                and expected.get("shift") is None
                else []
            )
            required = [
                *[name for name in FIT_FILES if name != "status.json"],
                *plot_files,
                *(["shift_manifest.json"] if expected.get("shift") is not None else []),
            ]
            return (
                status.get("state") == "complete"
                and status.get("expected") == dict(expected)
                and (path / "status.json").is_file()
                and all((path / name).is_file() for name in required)
                and status.get("file_sha256")
                == {
                    name: _sha256(path / name)
                    for name in required
                }
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _preserve_invalid(self, path: Path) -> None:
        if not path.exists():
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = path.parent / f"{path.name}.invalid.{stamp}"
        path.rename(destination)

    def _observed_spec_for_null(self, model: str, layer: str) -> FitSpec:
        return FitSpec(model=model, layer=layer, variant="primary")

    def _load_observed_null_contract(
        self, model: str, layer: str
    ) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], str]:
        path = self.unit_path(self._observed_spec_for_null(model, layer))
        expected = self._expected_metadata(
            self._observed_spec_for_null(model, layer)
        )
        if not self._valid_unit(path, expected):
            raise RuntimeError(
                f"Matched observed primary unit is incomplete or invalid: {path}"
            )
        reports = json.loads(
            (path / "split_reports.json").read_text(encoding="utf-8")
        )
        with np.load(path / "fold_assets.npz", allow_pickle=False) as archive:
            assets = {name: archive[name] for name in archive.files}
        return reports, assets, _sha256(path / "status.json")

    def _expected_metadata(self, spec: FitSpec) -> dict[str, Any]:
        design = "rich" if spec.variant == "rich" else "original"
        source = self._source_identity(spec.model, design)
        if spec.shift is not None:
            observed = self.unit_path(
                self._observed_spec_for_null(spec.model, spec.layer)
            ) / "status.json"
            source["observed_status_sha256"] = (
                _sha256(observed) if observed.is_file() else None
            )
        return {
            "model": spec.model,
            "layer": spec.layer,
            "variant": spec.variant,
            "outer_mode": spec.outer_mode,
            "shift": spec.shift,
            "source": source,
        }

    def fit(self, spec: FitSpec) -> dict[str, Any]:
        if spec.model not in self.models:
            raise ValueError(f"Unknown model {spec.model!r}")
        available = self.layers(spec.model)
        if spec.layer not in available:
            raise ValueError(
                f"Requested layer {spec.layer!r} is absent; available={available}"
            )
        if spec.variant not in ("primary", *CONTROL_VARIANTS, "null"):
            raise ValueError(f"Unknown fit variant {spec.variant!r}")
        if spec.outer_mode not in {"grouped", "loro"}:
            raise ValueError(f"Unknown outer mode {spec.outer_mode!r}")
        if spec.variant != "primary":
            depths = resolve_depth_layers(available)
            permitted = [depths["middle"]] if spec.variant == "null" else depths.values()
            if spec.layer not in permitted:
                raise ValueError(f"{spec.variant} requires one of these layers: {list(permitted)}")
        if spec.outer_mode == "loro" and (
            spec.model != "hubert_base" or spec.variant != "primary"
        ):
            raise ValueError("LORO is restricted to HuBERT Base primary sensitivity")
        if spec.shift is not None and spec.variant != "null":
            raise ValueError("Shift indexes are valid only for null fits")
        if spec.variant == "null" and spec.shift is None:
            raise ValueError("Null fits require a shift index")
        if spec.shift is not None and not (
            0 <= spec.shift < int(self.config["nulls"]["shifts"])
        ):
            raise ValueError("Null shift index is outside the configured ensemble")

        expected = self._expected_metadata(spec)
        destination = self.unit_path(spec)
        if self._valid_unit(destination, expected):
            return {"state": "resumed", "path": str(destination)}
        self._preserve_invalid(destination)

        started = time.perf_counter()
        usage_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ids, groups = self._recording_ids_groups()
        design = "rich" if spec.variant == "rich" else "original"
        feature_root = self.revised_root / design
        if design == "rich":
            _validate_shared_lexical_events(
                self.revised_root / "original", feature_root, ids
            )
        records = _load_recordings(
            feature_root, self.activation_store(spec.model), spec.layer, ids
        )
        encoding = self.config["encoding"]
        lags = (
            [0.0]
            if spec.variant == "zero_lag"
            else [float(value) for value in encoding["lags_seconds"]]
        )
        reductions = self.config["statistics"]["test_families"][spec.variant]
        equal_rank_cap = (
            int(self.config["controls"]["equal_rank_cap"])
            if spec.variant == "equal_rank"
            else None
        )
        fixed_reports = None
        fixed_assets = None
        observed_status_sha256 = None
        if spec.variant == "null":
            fixed_reports, fixed_assets, observed_status_sha256 = (
                self._load_observed_null_contract(spec.model, spec.layer)
            )
        capture_plot_assets = (
            spec.variant == "primary"
            and spec.outer_mode == "grouped"
            and spec.shift is None
        )

        result = fit_revised_encoding(
            records,
            groups,
            outer_mode=spec.outer_mode,
            outer_folds=int(encoding["outer_folds"]),
            inner_folds=int(encoding["inner_folds"]),
            alphas=[float(value) for value in encoding["alphas"]],
            lags_seconds=lags,
            rate_hz=float(self.config["analysis_rate_hz"]),
            lexical_components=int(encoding["lexical_pca_components"]),
            target_components=int(encoding["target_pca_components"]),
            reductions=reductions,
            seed=int(self.config["random_seed"]),
            equal_rank_cap=equal_rank_cap,
            shift=spec.shift,
            fixed_split_reports=fixed_reports,
            fixed_target_assets=fixed_assets,
            minimum_zero_seconds=float(
                self.config["nulls"]["minimum_zero_seconds"]
            ),
            max_abs_lag_seconds=max(abs(value) for value in lags),
            capture_plot_assets=capture_plot_assets,
        )
        result["scores"].insert(0, "model", spec.model)
        result["scores"].insert(1, "layer", spec.layer)
        result["scores"].insert(2, "variant", spec.variant)
        result["scores"]["shift"] = spec.shift
        for report in result["target_pca_reports"]:
            report.update(
                {"model": spec.model, "layer": spec.layer, "variant": spec.variant}
            )
        for report in result["predictor_reports"]:
            report.update(
                {"model": spec.model, "layer": spec.layer, "variant": spec.variant}
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            result["scores"].to_csv(temporary / "scores.csv", index=False)
            _write_json(temporary / "split_reports.json", result["split_reports"])
            pd.DataFrame(result["target_pca_reports"]).to_csv(
                temporary / "target_pca_coverage.csv", index=False
            )
            pd.DataFrame(result["predictor_reports"]).to_csv(
                temporary / "predictor_coverage.csv", index=False
            )
            np.savez(temporary / "fold_assets.npz", **result["fold_assets"])
            if capture_plot_assets:
                np.savez_compressed(
                    temporary / "predictions.npz", **result["predictions"]
                )
                np.savez(
                    temporary / "model_assets.npz", **result["model_assets"]
                )
            if spec.shift is not None:
                _write_json(temporary / "shift_manifest.json", result["shift_manifest"])
            elapsed = time.perf_counter() - started
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if os.uname().sysname == "Darwin":
                peak_bytes = int(peak)
            else:
                peak_bytes = int(peak * 1024)
            hashed_files = [
                *[name for name in FIT_FILES if name != "status.json"],
                *(list(PLOTTING_FIT_FILES) if capture_plot_assets else []),
                *(["shift_manifest.json"] if spec.shift is not None else []),
            ]
            file_hashes = {
                name: _sha256(temporary / name) for name in hashed_files
            }
            status = {
                "state": "complete",
                "fit_schema_version": FIT_SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "expected": expected,
                "recording_count": len(ids),
                "score_rows": len(result["scores"]),
                "wall_seconds": elapsed,
                "peak_rss_bytes": peak_bytes,
                "peak_rss_start": usage_before,
                "file_sha256": file_hashes,
                "observed_status_sha256": observed_status_sha256,
                "fixed_observed_target_coordinates": spec.shift is not None,
                "fixed_observed_alphas": spec.shift is not None,
                "null_ridge_weights_refit": spec.shift is not None,
            }
            _write_json(temporary / "status.json", status)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return {
            "state": "complete",
            "path": str(destination),
            "wall_seconds": elapsed,
            "peak_rss_bytes": peak_bytes,
        }

    def build_manifests(self) -> dict[str, Any]:
        manifest_root = self.output_root / "manifests"
        manifest_root.mkdir(parents=True, exist_ok=True)
        resolved: dict[str, Any] = {}
        primary_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        null_rows: list[dict[str, Any]] = []
        expected_rows: list[dict[str, Any]] = []
        for model in self.models:
            layers = self.layers(model)
            depths = resolve_depth_layers(layers)
            resolved[model] = {"all_layers": layers, "selected_depths": depths}
            for layer in layers:
                row = {
                    "task_id": len(primary_rows),
                    "model": model,
                    "layer": layer,
                    "variant": "primary",
                    "outer_mode": "grouped",
                }
                primary_rows.append(row)
                expected_rows.append({**row, "shift": ""})
            if model == "hubert_base":
                for layer in layers:
                    row = {
                        "task_id": len(primary_rows),
                        "model": model,
                        "layer": layer,
                        "variant": "primary",
                        "outer_mode": "loro",
                    }
                    primary_rows.append(row)
                    expected_rows.append({**row, "shift": ""})
            for variant in CONTROL_VARIANTS:
                for depth, layer in depths.items():
                    row = {
                        "task_id": len(control_rows),
                        "model": model,
                        "layer": layer,
                        "depth": depth,
                        "variant": variant,
                        "outer_mode": "grouped",
                    }
                    control_rows.append(row)
                    expected_rows.append({**row, "shift": ""})
            for shift in range(int(self.config["nulls"]["shifts"])):
                row = {
                    "task_id": len(null_rows),
                    "model": model,
                    "layer": depths["middle"],
                    "variant": "null",
                    "outer_mode": "grouped",
                    "shift": shift,
                }
                null_rows.append(row)
                expected_rows.append(row)
        _write_json(manifest_root / "resolved_layers.json", resolved)
        pd.DataFrame(primary_rows).to_csv(
            manifest_root / "primary_units.tsv", sep="\t", index=False
        )
        pd.DataFrame(control_rows).to_csv(
            manifest_root / "control_units.tsv", sep="\t", index=False
        )
        pd.DataFrame(null_rows).to_csv(
            manifest_root / "null_units.tsv", sep="\t", index=False
        )
        pd.DataFrame(expected_rows).to_csv(
            manifest_root / "expected_coverage.csv", index=False
        )
        return {
            "manifest_root": str(manifest_root),
            "primary_units": len(primary_rows),
            "control_units": len(control_rows),
            "null_units": len(null_rows),
            "expected_units": len(expected_rows),
            "resolved_layers": resolved,
        }

    def _spec_from_expected_row(self, row: Mapping[str, Any]) -> FitSpec:
        shift_value = row.get("shift")
        shift = (
            None
            if shift_value is None or str(shift_value).strip() in {"", "nan"}
            else int(float(shift_value))
        )
        return FitSpec(
            model=str(row["model"]),
            layer=str(row["layer"]),
            variant=str(row["variant"]),
            outer_mode=str(row["outer_mode"]),
            shift=shift,
        )

    def coverage(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        expected_path = self.output_root / "manifests" / "expected_coverage.csv"
        if not expected_path.is_file():
            self.build_manifests()
        expected = pd.read_csv(expected_path, keep_default_na=False)
        rows = []
        for row in expected.to_dict("records"):
            spec = self._spec_from_expected_row(row)
            path = self.unit_path(spec)
            valid = self._valid_unit(path, self._expected_metadata(spec))
            rows.append(
                {
                    **row,
                    "status": "PASS" if valid else "MISSING_OR_INVALID",
                    "path": str(path),
                }
            )
        coverage = pd.DataFrame(rows)
        panel_columns = ["variant", "outer_mode"]
        panel = {
            "|".join(map(str, key if isinstance(key, tuple) else (key,))): {
                "expected_units": len(group),
                "passed_units": int((group["status"] == "PASS").sum()),
                "complete": bool((group["status"] == "PASS").all()),
            }
            for key, group in coverage.groupby(panel_columns, sort=True)
        }
        return coverage, panel

    def summarize(self) -> dict[str, Any]:
        coverage, panels = self.coverage()
        summary_root = self.output_root / "summaries"
        summary_root.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(summary_root / "coverage_status.csv", index=False)
        _write_json(summary_root / "panel_status.json", panels)
        complete_paths = coverage.loc[
            coverage["status"] == "PASS", "path"
        ].astype(str)
        runtime_rows = []
        for row in coverage.loc[
            coverage["status"] == "PASS"
        ].to_dict("records"):
            status = json.loads(
                (Path(row["path"]) / "status.json").read_text(encoding="utf-8")
            )
            runtime_rows.append(
                {
                    "model": row["model"],
                    "layer": row["layer"],
                    "variant": row["variant"],
                    "outer_mode": row["outer_mode"],
                    "shift": row.get("shift", ""),
                    "wall_seconds": status["wall_seconds"],
                    "peak_rss_bytes": status["peak_rss_bytes"],
                }
            )
        pd.DataFrame(runtime_rows).to_csv(
            summary_root / "runtime_memory.csv", index=False
        )
        all_scores = pd.concat(
            [pd.read_csv(Path(path) / "scores.csv") for path in complete_paths],
            ignore_index=True,
        ) if len(complete_paths) else pd.DataFrame()
        if not all_scores.empty:
            all_scores.to_csv(summary_root / "all_recording_scores.csv", index=False)
        target_tables = [
            pd.read_csv(Path(path) / "target_pca_coverage.csv")
            for path in complete_paths
        ]
        if target_tables:
            target = pd.concat(target_tables, ignore_index=True)
            target.to_csv(summary_root / "target_pca_coverage.csv", index=False)
            primary_target = target.loc[
                (target["variant"] == "primary")
                & (target["outer_mode"] == "grouped")
            ]
            (
                primary_target.groupby(["model", "layer"], as_index=False)
                .agg(
                    outer_folds=("outer_fold", "count"),
                    original_target_dimensions=(
                        "target_units_before_pca",
                        "first",
                    ),
                    retained_components=("achieved_components", "mean"),
                    coverage_mean=("total_explained_variance_ratio", "mean"),
                    coverage_sd=("total_explained_variance_ratio", "std"),
                    coverage_min=("total_explained_variance_ratio", "min"),
                    coverage_max=("total_explained_variance_ratio", "max"),
                )
                .to_csv(summary_root / "target_pca_by_checkpoint_layer.csv", index=False)
            )
            (
                primary_target.groupby("model", as_index=False)
                .agg(
                    model_layer_folds=("outer_fold", "count"),
                    coverage_mean=("total_explained_variance_ratio", "mean"),
                    coverage_sd=("total_explained_variance_ratio", "std"),
                    coverage_min=("total_explained_variance_ratio", "min"),
                    coverage_max=("total_explained_variance_ratio", "max"),
                )
                .to_csv(summary_root / "target_pca_nine_checkpoints.csv", index=False)
            )
        if all_scores.empty:
            return {"panels": panels, "scores": 0, "summary_root": str(summary_root)}
        self._write_shift_offset_summary(coverage, summary_root)
        self._write_inference(all_scores, panels, summary_root)
        self._write_null_summary(all_scores, panels, summary_root)
        return {
            "panels": panels,
            "scores": len(all_scores),
            "summary_root": str(summary_root),
        }

    def _write_shift_offset_summary(
        self, coverage: pd.DataFrame, root: Path
    ) -> None:
        rows = []
        null_units = coverage.loc[
            (coverage["variant"] == "null") & (coverage["status"] == "PASS")
        ]
        for unit in null_units.to_dict("records"):
            manifest = json.loads(
                (Path(unit["path"]) / "shift_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            first_fold: dict[str, Any] | None = None
            for fold, recordings in sorted(manifest.items()):
                if first_fold is None:
                    first_fold = recordings
                elif recordings != first_fold:
                    raise ValueError(
                        f"{unit['model']} shift {unit['shift']}: offsets differ "
                        f"between outer folds"
                    )
            for recording_id, report in (first_fold or {}).items():
                rows.append(
                    {
                        "model": unit["model"],
                        "layer": unit["layer"],
                        "shift": int(unit["shift"]),
                        "recording_id": recording_id,
                        **report,
                    }
                )
        offsets = pd.DataFrame(rows)
        if offsets.empty:
            return
        consistency = offsets.groupby(
            ["shift", "recording_id"], sort=True
        )["offset_frames"].nunique()
        inconsistent = consistency[consistency != 1]
        if not inconsistent.empty:
            raise ValueError(
                "Null offsets differ across checkpoints for "
                f"{[tuple(value) for value in inconsistent.index]}"
            )
        offsets.to_csv(root / "null_shift_offsets.csv", index=False)

    def _inference_rows(
        self, frame: pd.DataFrame, panel: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        keys = ["model", "layer", "recording_id", "feature_family"]
        if frame.duplicated(keys).any():
            raise ValueError(f"{panel}: duplicate model/layer/recording/family scores")
        expected_ids = set(self._recording_ids_groups()[0])
        for cell, scores in frame.groupby(["model", "layer", "feature_family"]):
            if set(scores["recording_id"]) != expected_ids:
                raise ValueError(f"{panel}: incomplete recording scores for {cell}")
        per_checkpoint = (
            frame.groupby(["model", "recording_id", "feature_family"], as_index=False)
            .agg(
                effect=("delta_r2", "mean"),
                duration_seconds=("duration_seconds", "first"),
                model_layer_cells=("delta_r2", "size"),
            )
        )
        # Give each checkpoint equal weight when their layer counts differ.
        recording = (
            per_checkpoint.groupby(["recording_id", "feature_family"], as_index=False)
            .agg(
                effect=("effect", "mean"),
                duration_seconds=("duration_seconds", "first"),
                model_layer_cells=("model_layer_cells", "sum"),
                checkpoint_count=("model", "nunique"),
            )
        )
        rows = []
        for family, group in recording.groupby("feature_family", sort=True):
            values = group["effect"].to_numpy(dtype=float)
            test = exact_paired_sign_flip(values)
            low, high = recording_cluster_bootstrap_ci(
                values,
                n_bootstrap=int(self.config["statistics"]["bootstrap_samples"]),
                seed=int(self.config["random_seed"]),
                confidence=float(self.config["statistics"]["confidence"]),
            )
            weights = group["duration_seconds"].to_numpy(dtype=float)
            weighted_low, weighted_high = recording_cluster_bootstrap_ci(
                values,
                weights=weights,
                n_bootstrap=int(self.config["statistics"]["bootstrap_samples"]),
                seed=int(self.config["random_seed"]),
                confidence=float(self.config["statistics"]["confidence"]),
            )
            rows.append(
                {
                    "panel": panel,
                    "feature_family": family,
                    "equal_recording_effect": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "duration_weighted_effect": float(
                        np.average(values, weights=weights)
                    ),
                    "duration_weighted_ci_low": weighted_low,
                    "duration_weighted_ci_high": weighted_high,
                    "p_value": test["p_value"],
                    "positive_recordings": test["positive_count"],
                    "n_recordings": len(values),
                }
            )
        inference = pd.DataFrame(rows)
        if not inference.empty:
            inference["q_value"] = benjamini_hochberg(inference["p_value"])
            inference["survives_bh_0_05"] = inference["q_value"] <= 0.05
        return recording, inference

    def _write_inference(
        self, scores: pd.DataFrame, panels: Mapping[str, Any], root: Path
    ) -> None:
        inference_tables = []
        checkpoint_tables = []
        for (variant, outer_mode), frame in scores.loc[
            scores["variant"] != "null"
        ].groupby(["variant", "outer_mode"], sort=True):
            panel = f"{variant}|{outer_mode}"
            if not panels.get(panel, {}).get("complete", False):
                continue
            expected_families = set(
                self.config["statistics"]["test_families"][variant]
            )
            observed_families = set(frame["feature_family"].astype(str))
            if observed_families != expected_families:
                raise ValueError(
                    f"{panel}: expected test families {sorted(expected_families)}, "
                    f"found {sorted(observed_families)}"
                )
            recording, inference = self._inference_rows(frame, panel)
            recording.to_csv(
                root / f"{variant}_{outer_mode}_recording_effects.csv", index=False
            )
            inference_tables.append(inference)
            for model, model_scores in frame.groupby("model", sort=True):
                _, model_inference = self._inference_rows(model_scores, panel)
                checkpoint_tables.append(model_inference.assign(model=model))

        primary = scores.query("variant == 'primary' and outer_mode == 'grouped'")
        for variant in CONTROL_VARIANTS:
            control = scores.loc[scores["variant"] == variant]
            panel = f"{variant}|grouped"
            if not panels.get(panel, {}).get("complete", False):
                continue
            keys = ["model", "layer", "recording_id", "feature_family"]
            paired = control.merge(
                primary,
                on=keys,
                suffixes=("_control", "_primary"),
                validate="one_to_one",
            )
            if len(paired) != len(control):
                raise ValueError(f"{variant}: some control scores have no matched primary score")
            if not paired["outer_fold_control"].eq(paired["outer_fold_primary"]).all():
                raise ValueError(f"{variant}: matched control/primary folds differ")
            paired["delta_difference"] = (
                paired["delta_r2_control"] - paired["delta_r2_primary"]
            )
            if not np.allclose(
                paired["duration_seconds_control"],
                paired["duration_seconds_primary"],
                rtol=0,
                atol=0,
            ):
                raise ValueError(
                    f"{variant}: matching control/primary durations differ"
                )
            paired["duration_seconds"] = paired["duration_seconds_control"]
            paired.to_csv(
                root / f"{variant}_versus_primary_matched.csv", index=False
            )
            contrast = paired.rename(columns={"delta_difference": "delta_r2"})
            recording, inference = self._inference_rows(
                contrast, f"{variant}_versus_primary"
            )
            recording.to_csv(
                root / f"{variant}_versus_primary_recording_effects.csv",
                index=False,
            )
            inference_tables.append(inference)
            for model, model_scores in contrast.groupby("model", sort=True):
                _, model_inference = self._inference_rows(
                    model_scores, f"{variant}_versus_primary"
                )
                checkpoint_tables.append(model_inference.assign(model=model))
        if inference_tables:
            pd.concat(inference_tables, ignore_index=True).to_csv(
                root / "inference_summary.csv", index=False
            )
        if checkpoint_tables:
            pd.concat(checkpoint_tables, ignore_index=True).to_csv(
                root / "inference_by_checkpoint.csv", index=False
            )

    def _write_null_summary(
        self, scores: pd.DataFrame, panels: Mapping[str, Any], root: Path
    ) -> None:
        if not panels.get("null|grouped", {}).get("complete", False):
            return
        null = scores.loc[
            (scores["variant"] == "null")
            & (scores["feature_family"] == "word")
        ]
        observed = scores.loc[
            (scores["variant"] == "primary")
            & (scores["outer_mode"] == "grouped")
            & (scores["feature_family"] == "word")
        ]
        keys = ["model", "layer", "recording_id"]
        fold_keys = [*keys, "outer_fold"]
        shifts = set(range(int(self.config["nulls"]["shifts"])))
        if null.duplicated([*fold_keys, "shift"]).any() or any(
            set(group["shift"]) != shifts for _, group in null.groupby(fold_keys)
        ):
            raise ValueError("Null scores require exactly one result per configured shift")
        mean_null = (
            null.groupby(fold_keys, as_index=False)
            .agg(
                mean_shifted_delta_r2=("delta_r2", "mean"),
                completed_shifts=("shift", "nunique"),
            )
        )
        paired = observed.merge(mean_null, on=fold_keys, validate="one_to_one")
        expected_ids = set(self._recording_ids_groups()[0])
        expected_pairs = {
            (model, self.resolved_depths(model)["middle"], recording)
            for model in self.models for recording in expected_ids
        }
        if (
            set(paired[keys].itertuples(index=False, name=None)) != expected_pairs
            or len(paired) != len(mean_null)
        ):
            raise ValueError("Null inference requires all matched recordings at all nine middle layers")
        paired["observed_minus_mean_shifted"] = (
            paired["delta_r2"] - paired["mean_shifted_delta_r2"]
        )
        paired.to_csv(root / "null_observed_minus_shifted.csv", index=False)
        checkpoint_rows = []
        for model, group in paired.groupby("model", sort=True):
            _, inference = self._inference_rows(
                group.assign(delta_r2=group["observed_minus_mean_shifted"]),
                "null_observed_minus_shifted",
            )
            checkpoint_rows.append(inference.assign(model=model))
        by_checkpoint = pd.concat(checkpoint_rows, ignore_index=True)
        by_checkpoint["q_value"] = benjamini_hochberg(by_checkpoint["p_value"])
        by_checkpoint["survives_bh_0_05"] = by_checkpoint["q_value"] <= 0.05
        by_checkpoint.to_csv(root / "null_inference_by_checkpoint.csv", index=False)
        recording = (
            paired.groupby("recording_id", as_index=False)
            .agg(
                effect=("observed_minus_mean_shifted", "mean"),
                duration_seconds=("duration_seconds", "first"),
                checkpoint_count=("model", "nunique"),
            )
        )
        values = recording["effect"].to_numpy(dtype=float)
        test = exact_paired_sign_flip(values)
        low, high = recording_cluster_bootstrap_ci(
            values,
            n_bootstrap=int(self.config["statistics"]["bootstrap_samples"]),
            seed=int(self.config["random_seed"]),
            confidence=float(self.config["statistics"]["confidence"]),
        )
        weights = recording["duration_seconds"].to_numpy(dtype=float)
        weighted_low, weighted_high = recording_cluster_bootstrap_ci(
            values,
            weights=weights,
            n_bootstrap=int(self.config["statistics"]["bootstrap_samples"]),
            seed=int(self.config["random_seed"]),
            confidence=float(self.config["statistics"]["confidence"]),
        )
        _write_json(
            root / "null_inference.json",
            {
                "effect": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "duration_weighted_effect": float(
                    np.average(values, weights=weights)
                ),
                "duration_weighted_ci_low": weighted_low,
                "duration_weighted_ci_high": weighted_high,
                "q_value": test["p_value"],
                "survives_bh_0_05": test["p_value"] <= 0.05,
                **test,
            },
        )
