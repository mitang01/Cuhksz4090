"""Shared syllable-offset epoch extraction and separability statistics."""

from __future__ import annotations

import csv
import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import preprocess_seeg as prep


@dataclass
class EpochDataset:
    recording_id: str
    channel_names: list[str]
    times: np.ndarray
    observed: np.ndarray
    predictions: np.ndarray
    model_names: list[str]
    residual_m2: np.ndarray
    boundary_strength: np.ndarray
    struc_depth: np.ndarray
    stimulus_ids: np.ndarray


def stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "|".join([str(seed), *(str(part) for part in parts)]).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


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


def _scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def load_prediction_arrays(
    recording_dir: Path,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    list[str],
    list[str],
]:
    """Return stimulus -> (sample indices, predictions) from outer folds."""
    result: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    model_names: list[str] | None = None
    channel_names: list[str] | None = None
    paths = sorted(recording_dir.glob("predictions_outer_fold_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no outer-fold predictions in {recording_dir}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "stimulus_sample_indices" not in data:
                raise ValueError(
                    f"{path} predates epoch-compatible predictions; rerun STRF"
                )
            current_models = data["model_names"].astype(str).tolist()
            current_channels = data["channel_names"].astype(str).tolist()
            if model_names is None:
                model_names = current_models
                channel_names = current_channels
            elif (
                current_models != model_names
                or current_channels != channel_names
            ):
                raise ValueError(f"inconsistent prediction metadata in {path}")
            predictions = data["predictions"]
            stimuli = data["stimulus_ids"].astype(str)
            sample_indices = data["stimulus_sample_indices"].astype(int)
            for stimulus in np.unique(stimuli):
                mask = stimuli == stimulus
                result.setdefault(stimulus, []).append(
                    (sample_indices[mask], predictions[:, mask, :])
                )
    assert model_names is not None and channel_names is not None
    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for stimulus, pieces in result.items():
        indices = np.concatenate([piece[0] for piece in pieces])
        predictions = np.concatenate([piece[1] for piece in pieces], axis=1)
        order = np.argsort(indices)
        indices = indices[order]
        predictions = predictions[:, order, :]
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"duplicate prediction samples for {stimulus}")
        merged[stimulus] = (indices, predictions)
    return merged, model_names, channel_names


def extract_recording_epochs(
    strf_dir: Path,
    recording_name: str,
    *,
    sfreq: float,
    epoch_start: float,
    epoch_end: float,
) -> EpochDataset:
    """Extract common observed/predicted epochs around prosodic offsets."""
    aligned_dir = strf_dir / "aligned_data" / recording_name
    recording_dir = strf_dir / "recordings" / recording_name
    metrics_path = recording_dir / "model_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing model metrics: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        first_metric = next(csv.DictReader(stream), None)
    if first_metric is None:
        raise ValueError(f"empty model metrics: {metrics_path}")
    recording_id = first_metric["recording_id"]
    prediction_data, model_names, predicted_channels = load_prediction_arrays(
        recording_dir
    )
    offsets = np.arange(
        round(epoch_start * sfreq), round(epoch_end * sfreq) + 1
    )
    times = offsets / sfreq
    observed_epochs: list[np.ndarray] = []
    predicted_epochs: list[np.ndarray] = []
    boundary_values: list[float] = []
    depth_values: list[float] = []
    stimuli: list[str] = []
    channel_names: list[str] | None = None
    for path in sorted(aligned_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            stimulus = _scalar_string(data["stimulus_id"])
            X = data["X"]
            y = data["y"]
            feature_names = data["feature_names"].astype(str).tolist()
            current_channels = data["channel_names"].astype(str).tolist()
            track_times = data["time"]
            if "prosody_event_mask" not in data:
                raise ValueError(
                    f"{path} predates explicit prosody events; rerun STRF"
                )
            prosody_event_mask = data["prosody_event_mask"].astype(bool)
        if len(track_times) > 1:
            differences = np.diff(track_times)
            if not np.allclose(differences, differences[0], atol=1e-10, rtol=0):
                raise ValueError(f"nonuniform aligned time grid in {path}")
            actual_sfreq = 1.0 / differences[0]
            if not np.isclose(actual_sfreq, sfreq):
                raise ValueError(
                    f"{path} has {actual_sfreq:g} Hz, expected {sfreq:g} Hz"
                )
        if channel_names is None:
            channel_names = current_channels
        elif current_channels != channel_names:
            raise ValueError(f"inconsistent channels in {aligned_dir}")
        if current_channels != predicted_channels:
            raise ValueError(
                f"aligned/predicted channels differ for {recording_name}"
            )
        if stimulus not in prediction_data:
            raise ValueError(f"no held-out predictions for {stimulus}")
        sample_indices, stimulus_predictions = prediction_data[stimulus]
        prediction_grid = np.full(
            (len(model_names), len(X), len(channel_names)), np.nan
        )
        valid_indices = (sample_indices >= 0) & (sample_indices < len(X))
        prediction_grid[:, sample_indices[valid_indices], :] = (
            stimulus_predictions[:, valid_indices, :]
        )
        boundary_index = feature_names.index("boundary_strength")
        depth_index = feature_names.index("struc_depth")
        centers = np.flatnonzero(prosody_event_mask)
        for center in centers:
            indices = center + offsets
            if indices[0] < 0 or indices[-1] >= len(X):
                continue
            predicted = prediction_grid[:, indices, :]
            if not np.all(np.isfinite(predicted)):
                continue
            observed_epochs.append(y[indices])
            predicted_epochs.append(np.moveaxis(predicted, 0, 1))
            boundary_values.append(float(X[center, boundary_index]))
            depth_values.append(float(X[center, depth_index]))
            stimuli.append(stimulus)
    if not observed_epochs or channel_names is None:
        raise ValueError(f"no complete predicted prosody epochs for {recording_name}")
    observed = np.stack(observed_epochs)
    predictions = np.stack(predicted_epochs)
    m2_index = model_names.index("M2_mel_syl")
    residual = observed - predictions[:, :, m2_index, :]
    return EpochDataset(
        recording_id=recording_id,
        channel_names=channel_names,
        times=times,
        observed=observed,
        predictions=predictions,
        model_names=model_names,
        residual_m2=residual,
        boundary_strength=np.asarray(boundary_values),
        struc_depth=np.asarray(depth_values),
        stimulus_ids=np.asarray(stimuli),
    )


def save_epoch_dataset(path: Path, dataset: EpochDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        recording_id=dataset.recording_id,
        channel_names=np.asarray(dataset.channel_names),
        times=dataset.times,
        observed=dataset.observed,
        predictions=dataset.predictions,
        model_names=np.asarray(dataset.model_names),
        residual_m2=dataset.residual_m2,
        boundary_strength=dataset.boundary_strength,
        struc_depth=dataset.struc_depth,
        stimulus_ids=dataset.stimulus_ids,
        baseline_correction="none",
    )


def load_epoch_dataset(path: Path) -> EpochDataset:
    with np.load(path, allow_pickle=False) as data:
        return EpochDataset(
            recording_id=_scalar_string(data["recording_id"]),
            channel_names=data["channel_names"].astype(str).tolist(),
            times=data["times"],
            observed=data["observed"],
            predictions=data["predictions"],
            model_names=data["model_names"].astype(str).tolist(),
            residual_m2=data["residual_m2"],
            boundary_strength=data["boundary_strength"],
            struc_depth=data["struc_depth"],
            stimulus_ids=data["stimulus_ids"].astype(str),
        )


def boundary_f_stat(
    epochs: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """F statistic and standardized slope for one continuous predictor."""
    x = np.asarray(values, dtype=float)
    x = (x - x.mean()) / max(x.std(ddof=1), np.finfo(float).eps)
    y = np.asarray(epochs, dtype=float)
    y_centered = y - y.mean(axis=0)
    slope = np.sum(x[:, np.newaxis] * y_centered, axis=0) / max(
        np.sum(x**2), np.finfo(float).eps
    )
    fitted = x[:, np.newaxis] * slope
    residual = y_centered - fitted
    model_ss = np.sum(fitted**2, axis=0)
    residual_ss = np.sum(residual**2, axis=0)
    denominator_df = max(len(x) - 2, 1)
    f_stat = model_ss / np.maximum(
        residual_ss / denominator_df, np.finfo(float).eps
    )
    standardized_slope = slope / np.maximum(
        y.std(axis=0, ddof=1), np.finfo(float).eps
    )
    return f_stat, standardized_slope


def depth_f_stat(
    epochs: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One-way F statistic and eta-squared across structure depths."""
    y = np.asarray(epochs, dtype=float)
    groups = np.unique(values)
    if len(groups) < 2:
        raise ValueError("at least two structure-depth levels are required")
    grand_mean = y.mean(axis=0)
    between = np.zeros(y.shape[1])
    within = np.zeros(y.shape[1])
    for group in groups:
        group_y = y[values == group]
        group_mean = group_y.mean(axis=0)
        between += len(group_y) * (group_mean - grand_mean) ** 2
        within += np.sum((group_y - group_mean) ** 2, axis=0)
    df_between = len(groups) - 1
    df_within = max(len(y) - len(groups), 1)
    f_stat = (between / df_between) / np.maximum(
        within / df_within, np.finfo(float).eps
    )
    eta_squared = between / np.maximum(
        between + within, np.finfo(float).eps
    )
    return f_stat, eta_squared


def permute_within_stimulus(
    values: np.ndarray, stimulus_ids: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    permuted = np.asarray(values).copy()
    for stimulus in np.unique(stimulus_ids):
        indices = np.flatnonzero(stimulus_ids == stimulus)
        permuted[indices] = rng.permutation(permuted[indices])
    return permuted


def fdr_clusters(
    times: np.ndarray,
    statistic: np.ndarray,
    pvalues: np.ndarray,
    *,
    q: float,
    minimum_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    reject, adjusted = prep.fdr_bh(pvalues, q)
    retained = np.zeros_like(reject)
    clusters: list[dict[str, object]] = []
    starts = np.flatnonzero(reject & np.r_[True, ~reject[:-1]])
    stops = np.flatnonzero(reject & np.r_[~reject[1:], True]) + 1
    for start, stop in zip(starts, stops):
        if stop - start < minimum_samples:
            continue
        retained[start:stop] = True
        clusters.append(
            {
                "start_s": float(times[start]),
                "end_s": float(times[stop - 1]),
                "n_samples": int(stop - start),
                "cluster_statistic": float(np.sum(statistic[start:stop])),
                "maximum_statistic": float(np.max(statistic[start:stop])),
            }
        )
    return retained, adjusted, clusters


def time_resolved_test(
    epochs: np.ndarray,
    values: np.ndarray,
    stimulus_ids: np.ndarray,
    times: np.ndarray,
    *,
    kind: str,
    n_permutations: int,
    seed: int,
    fdr_q: float,
    minimum_cluster_samples: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    statistic_function: Callable[
        [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ] = boundary_f_stat if kind == "boundary_strength" else depth_f_stat
    statistic, effect = statistic_function(epochs, values)
    exceedances = np.ones(len(times), dtype=int)
    rng = np.random.default_rng(seed)
    for _ in range(n_permutations):
        permuted = permute_within_stimulus(values, stimulus_ids, rng)
        null_statistic, _ = statistic_function(epochs, permuted)
        exceedances += null_statistic >= statistic
    pvalues = exceedances / (n_permutations + 1.0)
    retained, adjusted, clusters = fdr_clusters(
        times,
        statistic,
        pvalues,
        q=fdr_q,
        minimum_samples=minimum_cluster_samples,
    )
    rows = [
        {
            "time_s": float(time),
            "f_statistic": float(f_value),
            "effect_size": float(effect_value),
            "permutation_p_value": float(pvalue),
            "fdr_p_value": float(adjusted_p),
            "passes_fdr_and_cluster": bool(significant),
        }
        for time, f_value, effect_value, pvalue, adjusted_p, significant in zip(
            times, statistic, effect, pvalues, adjusted, retained
        )
    ]
    return rows, clusters


def quantile_bins(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Equal-count bins for visualization only."""
    order = np.argsort(values, kind="stable")
    bins = np.empty(len(values), dtype=int)
    for bin_index, indices in enumerate(np.array_split(order, n_bins)):
        bins[indices] = bin_index
    return bins
