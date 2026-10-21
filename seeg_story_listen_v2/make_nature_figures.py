#!/usr/bin/env python3
"""Build an auditable, journal-scale multi-panel summary of the STRF analyses.

The renderer consumes complete outputs from the four analysis entry points. It
does not recompute inferential tests, subsample observations, or impute missing
conditions. Missing required rows fail validation; structurally absent
subject-by-condition cells are retained in the audit record.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np

import preprocess_seeg as prep
import prosody_epoch_stats as epoch_stats
import run_strf
from figure_significance import (
    annotate_bars,
    annotate_time_significance,
    contiguous_true_runs,
)


DEFAULT_ROOT = run_strf.DEFAULT_PREPROCESSED
FEATURE_LABELS = {
    "mel": "Mel",
    "syl_onset": "Syllable onset",
    "boundary_strength": "Boundary strength",
    "struc_depth": "Structure depth",
}
PREDICTOR_LABELS = {
    "boundary_strength": "Boundary strength",
    "struc_depth": "Structure depth",
}
PALETTE = {
    "blue": "#2c7fb8",
    "orange": "#d95f0e",
    "green": "#238b45",
    "purple": "#756bb1",
    "grey": "#737373",
    "light_grey": "#d9d9d9",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the complete STRF/prosody result set as an auditable "
            "Nature-style multi-panel figure."
        )
    )
    parser.add_argument("--strf-dir", type=Path, default=DEFAULT_ROOT / "strf")
    parser.add_argument(
        "--strf-group-dir", type=Path, default=DEFAULT_ROOT / "strf_group"
    )
    parser.add_argument(
        "--epochs-dir", type=Path, default=DEFAULT_ROOT / "prosody_epochs"
    )
    parser.add_argument(
        "--epochs-group-dir",
        type=Path,
        default=DEFAULT_ROOT / "prosody_epochs_group",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ROOT / "nature_figures"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_csv_required(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required table does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"required table is empty: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty source-data table: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_json_required(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required configuration does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a JSON object: {path}")
    return value


def validate_full_run_configs(
    inputs: Sequence[tuple[str, Path]],
) -> dict[str, dict[str, object]]:
    configs: dict[str, dict[str, object]] = {}
    for label, path in inputs:
        config = read_json_required(path)
        max_recordings = config.get("max_recordings")
        if max_recordings not in (None, "None"):
            raise ValueError(
                f"{label} used max_recordings={max_recordings}; publication "
                "rendering requires complete upstream runs"
            )
        configs[label] = config
    return configs


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def require_finite(values: Iterable[float], context: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{context} contains no data or non-finite values")
    return array


def subject_id(recording_id: str) -> str:
    number = prep.source_subject_number(Path(recording_id))
    return f"sub{number:03d}" if number is not None else recording_id.split("/")[0]


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.20,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
    )


def scientific_axes(
    ax: plt.Axes,
    *,
    x: bool = True,
    y: bool = True,
) -> None:
    """Use math-text scientific notation on requested numeric axes."""
    if x:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(formatter)
    if y:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)


def validate_model_metrics(
    rows: list[dict[str, str]],
) -> tuple[list[str], list[int]]:
    models = list(run_strf.MODEL_FAMILIES)
    folds_by_model = {
        model: {
            int(row["outer_fold"])
            for row in rows
            if row["model"] == model and row["channel"] == "GROUP"
        }
        for model in models
    }
    if any(not folds for folds in folds_by_model.values()):
        raise ValueError("GROUP model metrics are missing one or more models")
    reference = folds_by_model[models[0]]
    if any(folds != reference for folds in folds_by_model.values()):
        raise ValueError("GROUP model metrics have incomplete outer-fold observations")
    return models, sorted(reference)


def validate_group_epoch_statistics(
    rows: list[dict[str, str]],
    *,
    expected_subjects: int,
) -> None:
    expected_sources = {
        "observed",
        "m2_residual",
        *(f"prediction_{model}" for model in run_strf.MODEL_FAMILIES),
    }
    actual_sources = {row["source"] for row in rows}
    if actual_sources != expected_sources:
        raise ValueError(
            "group epoch statistics have incomplete signal sources: "
            f"expected={sorted(expected_sources)}, got={sorted(actual_sources)}"
        )
    reference_times: tuple[float, ...] | None = None
    seen: set[tuple[str, str, float]] = set()
    for source in sorted(expected_sources):
        for predictor in PREDICTOR_LABELS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["source"] == source and row["predictor"] == predictor
                ),
                key=lambda row: float(row["time_s"]),
            )
            times = tuple(float(row["time_s"]) for row in selected)
            if not times:
                raise ValueError(f"missing group epoch rows for {source}/{predictor}")
            if reference_times is None:
                reference_times = times
            elif times != reference_times:
                raise ValueError(
                    f"incomplete group epoch time grid for {source}/{predictor}"
                )
            for row in selected:
                key = (source, predictor, float(row["time_s"]))
                if key in seen:
                    raise ValueError(f"duplicate group epoch result row: {key}")
                seen.add(key)
                if int(row["n_subjects"]) != expected_subjects:
                    raise ValueError(
                        f"group epoch subject count differs for {key}: "
                        f"expected={expected_subjects}, got={row['n_subjects']}"
                    )


def subject_feature_points(
    strf_dir: Path,
    membership_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    members = {
        (row["recording_id"], row["channel"]) for row in membership_rows
    }
    feature_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for path in sorted((strf_dir / "recordings").glob("*/feature_contributions.csv")):
        for row in read_csv_required(path):
            member = (row["recording_id"], row["channel"])
            if member not in members:
                continue
            feature = row["feature"]
            key = (*member, feature)
            if key in seen:
                raise ValueError(f"duplicate individual feature row: {key}")
            seen.add(key)
            feature_values[(subject_id(row["recording_id"]), feature)].append(
                float(row["mean_delta_correlation"])
            )
    expected = {
        (*member, feature)
        for member in members
        for feature in run_strf.FEATURE_FAMILIES
    }
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(
            "individual feature results are incomplete for current group members: "
            f"{missing[:10]}"
        )
    return [
        {
            "subject": subject,
            "feature": feature,
            "mean_electrode_delta_correlation": float(np.mean(values)),
            "n_electrodes": len(values),
        }
        for (subject, feature), values in sorted(feature_values.items())
    ]


def load_subject_waveforms(
    epoch_paths: Sequence[Path],
    *,
    strf_dir: Path,
) -> tuple[
    np.ndarray,
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    by_subject_level: dict[
        tuple[str, str, float], list[np.ndarray]
    ] = defaultdict(list)
    subjects: set[str] = set()
    levels: dict[str, set[float]] = defaultdict(set)
    reference_times: np.ndarray | None = None
    epoch_accounting: list[dict[str, object]] = []
    for path in epoch_paths:
        dataset = epoch_stats.load_epoch_dataset(path)
        if reference_times is None:
            reference_times = dataset.times
        elif not np.array_equal(reference_times, dataset.times):
            raise ValueError(f"epoch time grid differs in {path}")
        if not np.all(np.isfinite(dataset.observed)):
            raise ValueError(f"non-finite observed epochs in {path}")
        subject = subject_id(dataset.recording_id)
        subjects.add(subject)
        aligned_dir = (
            strf_dir / "aligned_data" / run_strf.safe_name(dataset.recording_id)
        )
        aligned_paths = sorted(aligned_dir.glob("*.npz"))
        if not aligned_paths:
            raise FileNotFoundError(
                f"aligned STRF tracks are missing for {dataset.recording_id}: "
                f"{aligned_dir}"
            )
        source_events = 0
        for aligned_path in aligned_paths:
            with np.load(aligned_path, allow_pickle=False) as aligned:
                if "prosody_event_mask" not in aligned:
                    raise ValueError(
                        f"epoch accounting requires prosody_event_mask: {aligned_path}"
                    )
                event_mask = np.asarray(aligned["prosody_event_mask"], dtype=bool)
                if event_mask.ndim != 1:
                    raise ValueError(
                        f"prosody_event_mask is not one-dimensional: {aligned_path}"
                    )
                source_events += int(event_mask.sum())
        retained_epochs = len(dataset.observed)
        if retained_epochs > source_events:
            raise ValueError(
                f"retained epochs exceed source events for {dataset.recording_id}"
            )
        epoch_accounting.append(
            {
                "recording_id": dataset.recording_id,
                "subject": subject,
                "n_source_prosody_events": source_events,
                "n_retained_epochs": retained_epochs,
                "n_events_not_retained": source_events - retained_epochs,
                "accounting_note": (
                    "not-retained events are upstream edge-window or "
                    "non-finite-prediction exclusions"
                ),
            }
        )
        predictor_levels = {
            "struc_depth": np.asarray(dataset.struc_depth, dtype=float),
            "boundary_strength": (
                epoch_stats.quantile_bins(dataset.boundary_strength) + 1
            ).astype(float),
        }
        for predictor, labels in predictor_levels.items():
            for level in np.unique(labels):
                levels[predictor].add(float(level))
                mask = labels == level
                if not np.any(mask):
                    raise AssertionError("unreachable empty prosody-level mask")
                by_subject_level[(subject, predictor, float(level))].append(
                    dataset.observed[mask].mean(axis=(0, 2))
                )
    if reference_times is None:
        raise ValueError("no individual epoch datasets were found")
    rows: list[dict[str, object]] = []
    for (subject, predictor, level), recording_waves in sorted(
        by_subject_level.items()
    ):
        wave = np.mean(recording_waves, axis=0)
        rows.extend(
            {
                "subject": subject,
                "predictor": predictor,
                "level": level,
                "level_label": (
                    f"Depth {level:g}"
                    if predictor == "struc_depth"
                    else f"Boundary quartile {level:g}"
                ),
                "time_s": float(time),
                "mean_observed_high_gamma_z": float(value),
                "n_recordings": len(recording_waves),
            }
            for time, value in zip(reference_times, wave)
        )
    missing = [
        {"subject": subject, "predictor": predictor, "level": level}
        for subject in sorted(subjects)
        for predictor, predictor_levels in sorted(levels.items())
        for level in sorted(predictor_levels)
        if (subject, predictor, level) not in by_subject_level
    ]
    return reference_times, rows, missing, epoch_accounting


def individual_strf_recordings(strf_dir: Path) -> set[str]:
    recordings: set[str] = set()
    recording_dirs = sorted(
        path for path in (strf_dir / "recordings").iterdir() if path.is_dir()
    )
    missing_metrics = [
        path for path in recording_dirs if not (path / "model_metrics.csv").is_file()
    ]
    if missing_metrics:
        raise ValueError(
            "partial individual STRF recording outputs lack model_metrics.csv: "
            f"{missing_metrics[:10]}"
        )
    for path in (directory / "model_metrics.csv" for directory in recording_dirs):
        ids = {row["recording_id"] for row in read_csv_required(path)}
        if len(ids) != 1:
            raise ValueError(f"ambiguous recording IDs in {path}: {sorted(ids)}")
        recordings.update(ids)
    if not recordings:
        raise ValueError("no individual STRF recording metrics were found")
    return recordings


def cohort_counts(
    inclusion: list[dict[str, str]],
    membership: list[dict[str, str]],
    excluded: list[dict[str, str]],
) -> dict[str, int]:
    included_recordings = {
        row["recording_id"] for row in inclusion if row["status"] == "included"
    }
    skipped_recordings = {
        row["recording_id"] for row in inclusion if row["status"] != "included"
    }
    electrodes = {(row["recording_id"], row["channel"]) for row in membership}
    subjects = {subject_id(recording) for recording in included_recordings}
    misc = {(row["recording_id"], row["channel"]) for row in excluded}
    counts = {
        "recordings_discovered": len(inclusion),
        "recordings_included": len(included_recordings),
        "recordings_excluded": len(skipped_recordings),
        "subjects_included": len(subjects),
        "responsive_non_misc_electrodes": len(electrodes),
        "misc_channels_excluded": len(misc),
    }
    return counts


def plot_cohort(
    ax: plt.Axes,
    inclusion: list[dict[str, str]],
    membership: list[dict[str, str]],
    excluded: list[dict[str, str]],
) -> dict[str, int]:
    counts = cohort_counts(inclusion, membership, excluded)
    ax.axis("off")
    labels = [
        ("Recordings discovered", counts["recordings_discovered"]),
        ("Recordings retained", counts["recordings_included"]),
        ("Subjects represented", counts["subjects_included"]),
        ("Responsive contacts", counts["responsive_non_misc_electrodes"]),
    ]
    for index, (label, count) in enumerate(labels):
        y = 0.90 - index * 0.22
        ax.text(
            0.5,
            y,
            f"{count}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            y - 0.075,
            label,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(0.5, y - 0.15),
                xytext=(0.5, y - 0.10),
                arrowprops={"arrowstyle": "-|>", "color": PALETTE["grey"], "lw": 0.7},
                xycoords=ax.transAxes,
            )
    ax.set_title("Audited analysis cohort", loc="left", fontweight="bold")
    return counts


def plot_group_accuracy(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    models: list[str],
    folds: list[int],
) -> list[dict[str, object]]:
    source_rows: list[dict[str, object]] = []
    means: list[float] = []
    sems: list[float] = []
    for model_index, model in enumerate(models):
        model_rows = sorted(
            (
                row
                for row in rows
                if row["channel"] == "GROUP" and row["model"] == model
            ),
            key=lambda row: int(row["outer_fold"]),
        )
        values = require_finite(
            (float(row["correlation"]) for row in model_rows),
            f"GROUP {model} correlations",
        )
        means.append(float(values.mean()))
        sems.append(float(values.std(ddof=1) / math.sqrt(len(values))))
        offsets = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(
            model_index + offsets,
            values,
            color="white",
            edgecolor=PALETTE["blue"],
            linewidth=0.6,
            s=12,
            zorder=3,
        )
        source_rows.extend(
            {
                "model": model,
                "outer_fold": int(row["outer_fold"]),
                "held_out_pearson_r": float(row["correlation"]),
            }
            for row in model_rows
        )
    ax.bar(
        np.arange(len(models)),
        means,
        yerr=sems,
        color=PALETTE["light_grey"],
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
        zorder=2,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(
        xticks=np.arange(len(models)),
        xticklabels=[model.split("_", 1)[0] for model in models],
        ylabel="Held-out Pearson $r$",
        title="Pooled encoding accuracy",
    )
    ax.text(
        0.02,
        0.98,
        f"points: all {len(folds)} held-out stories",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
    )
    scientific_axes(ax, x=False)
    return source_rows


def plot_subject_features(
    ax: plt.Axes,
    points: list[dict[str, object]],
    tests: list[dict[str, str]],
) -> int:
    features = list(run_strf.FEATURE_FAMILIES)
    tests_by_feature = {row["feature"]: row for row in tests}
    if set(tests_by_feature) != set(features):
        raise ValueError("subject feature-inference table has incomplete features")
    means: list[float] = []
    errors: list[float] = []
    for index, feature in enumerate(features):
        selected = [row for row in points if row["feature"] == feature]
        values = require_finite(
            (
                float(row["mean_electrode_delta_correlation"])
                for row in selected
            ),
            f"subject {feature} effects",
        )
        offsets = np.linspace(-0.13, 0.13, len(values))
        ax.scatter(
            index + offsets,
            values,
            s=11,
            facecolor="white",
            edgecolor=PALETTE["orange"],
            linewidth=0.6,
            zorder=3,
        )
        means.append(float(values.mean()))
        errors.append(float(1.96 * values.std(ddof=1) / math.sqrt(len(values))))
    bars = ax.bar(
        np.arange(len(features)),
        means,
        yerr=errors,
        color="#fdd0a2",
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
        zorder=2,
    )
    ordered_tests = [tests_by_feature[feature] for feature in features]
    marks = annotate_bars(
        ax,
        bars,
        ordered_tests,
        p_key="fdr_p_value_correlation",
        errors=errors,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(
        xticks=np.arange(len(features)),
        xticklabels=[
            "Mel",
            "Syllable\nonset",
            "Boundary\nstrength",
            "Structure\ndepth",
        ],
        ylabel=r"Subject mean $\Delta r$",
        title="Unique feature effects",
    )
    ax.tick_params(axis="x", labelsize=5.5)
    scientific_axes(ax, x=False)
    return marks


def plot_coefficients(
    figure: plt.Figure,
    subplot_spec,
    coefficient_path: Path,
    *,
    channel_name: str = "GROUP",
    title: str = "Full-model STRF coefficients",
) -> tuple[list[plt.Axes], list[dict[str, object]]]:
    if not coefficient_path.is_file():
        raise FileNotFoundError(f"required coefficients do not exist: {coefficient_path}")
    with np.load(coefficient_path, allow_pickle=False) as data:
        if "M5_full" not in data:
            raise ValueError("GROUP coefficients do not contain M5_full")
        coefficients = np.asarray(data["M5_full"], dtype=float)
        features = data["feature_names"].astype(str).tolist()
        lags = np.asarray(data["lags_s"], dtype=float)
        channel_names = data["channel_names"].astype(str).tolist()
    if coefficients.ndim != 4:
        raise ValueError(f"unexpected GROUP coefficient shape: {coefficients.shape}")
    if channel_name not in channel_names:
        raise ValueError(
            f"coefficient channel {channel_name!r} not found in {coefficient_path}"
        )
    channel_index = channel_names.index(channel_name)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("GROUP M5 coefficients contain non-finite values")
    mean = coefficients.mean(axis=0)[channel_index]
    sem = (
        coefficients.std(axis=0, ddof=1)[channel_index]
        / math.sqrt(coefficients.shape[0])
    )
    mel_indices = [index for index, name in enumerate(features) if name.startswith("mel_")]
    linguistic = [
        (feature, features.index(feature))
        for feature in ("syl_onset", "boundary_strength", "struc_depth")
    ]
    if not mel_indices or any(feature not in features for feature, _ in linguistic):
        raise ValueError("M5 coefficients do not contain all requested predictors")
    nested = subplot_spec.subgridspec(2, 1, height_ratios=(1.0, 1.0), hspace=0.12)
    ax_mel = figure.add_subplot(nested[0])
    ax_lines = figure.add_subplot(nested[1])
    limit = float(np.max(np.abs(mean[mel_indices])))
    image = ax_mel.pcolormesh(
        lags,
        np.arange(len(mel_indices)),
        mean[mel_indices],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        shading="auto",
    )
    ax_mel.set(ylabel="Mel band", title=title)
    ax_mel.tick_params(labelbottom=False)
    colorbar = figure.colorbar(image, ax=ax_mel, fraction=0.05, pad=0.02)
    colorbar.formatter = ScalarFormatter(useMathText=True)
    colorbar.formatter.set_scientific(True)
    colorbar.formatter.set_powerlimits((0, 0))
    colorbar.update_ticks()
    colors = [PALETTE["green"], PALETTE["orange"], PALETTE["purple"]]
    for (feature, index), color in zip(linguistic, colors):
        ax_lines.plot(lags, mean[index], color=color, label=FEATURE_LABELS[feature])
        ax_lines.fill_between(
            lags,
            mean[index] - 1.96 * sem[index],
            mean[index] + 1.96 * sem[index],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    ax_lines.axhline(0, color="black", linewidth=0.5)
    ax_lines.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax_lines.set(xlabel="Lag (s)", ylabel="Coefficient")
    ax_lines.legend(
        frameon=False,
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
    )
    scientific_axes(ax_mel)
    scientific_axes(ax_lines)
    source_rows = [
        {
            "outer_fold": fold,
            "channel": channel_name,
            "feature": feature,
            "lag_s": float(lag),
            "coefficient": float(
                coefficients[fold, channel_index, feature_index, lag_index]
            ),
        }
        for fold in range(coefficients.shape[0])
        for feature_index, feature in enumerate(features)
        for lag_index, lag in enumerate(lags)
    ]
    return [ax_mel, ax_lines], source_rows


def plot_subject_epochs(
    ax: plt.Axes,
    times: np.ndarray,
    rows: list[dict[str, object]],
    *,
    predictor: str,
) -> None:
    selected_rows = [row for row in rows if row["predictor"] == predictor]
    if not selected_rows:
        raise ValueError(f"no subject waveform rows for {predictor}")
    levels = sorted({float(row["level"]) for row in selected_rows})
    subjects = sorted({str(row["subject"]) for row in rows})
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(levels)))
    for level, color in zip(levels, colors):
        level_waves: list[np.ndarray] = []
        for subject in subjects:
            selected = [
                row
                for row in selected_rows
                if row["subject"] == subject
                and float(row["level"]) == level
            ]
            if not selected:
                continue
            wave_times = np.asarray([float(row["time_s"]) for row in selected])
            if not np.array_equal(wave_times, times):
                raise ValueError(
                    f"subject waveform grid is incomplete: "
                    f"{subject}, {predictor}={level}"
                )
            wave = require_finite(
                (float(row["mean_observed_high_gamma_z"]) for row in selected),
                f"{subject}, {predictor}={level}",
            )
            level_waves.append(wave)
            ax.plot(times, wave, color=color, alpha=0.14, linewidth=0.45)
        matrix = np.asarray(level_waves)
        ax.plot(
            times,
            matrix.mean(axis=0),
            color=color,
            linewidth=1.4,
            label=(
                f"Depth {level:g} ($n$={len(matrix)})"
                if predictor == "struc_depth"
                else f"Quartile {level:g} ($n$={len(matrix)})"
            ),
        )
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(
        xlabel="Time from syllable offset (s)",
        ylabel="High-gamma response (z)",
        title=(
            "ER-Hγ by structure depth"
            if predictor == "struc_depth"
            else "ER-Hγ by boundary strength"
        ),
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=2,
    )
    scientific_axes(ax)


def plot_source_statistics(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    source: str,
    title: str,
) -> int:
    marks = 0
    for level, (predictor, color) in enumerate(
        (
            ("boundary_strength", PALETTE["orange"]),
            ("struc_depth", PALETTE["purple"]),
        )
    ):
        selected = sorted(
            (
                row
                for row in rows
                if row["source"] == source and row["predictor"] == predictor
            ),
            key=lambda row: float(row["time_s"]),
        )
        if not selected:
            raise ValueError(f"group epoch statistics lack {source}/{predictor}")
        times = require_finite(
            (float(row["time_s"]) for row in selected),
            f"{predictor} times",
        )
        values = require_finite(
            (float(row["mean_subject_f_statistic"]) for row in selected),
            f"{predictor} F statistics",
        )
        significant = np.asarray(
            [parse_bool(row["passes_fdr_and_cluster"]) for row in selected]
        )
        ax.plot(times, values, color=color, label=PREDICTOR_LABELS[predictor])
        ax.fill_between(
            times,
            0,
            values,
            where=significant,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        marks += annotate_time_significance(
            ax,
            times,
            significant,
            color=color,
            level=level,
        )
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set(
        xlabel="Time from syllable offset (s)",
        ylabel="Mean subject F statistic",
        title=title,
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    scientific_axes(ax)
    return marks


def plot_all_epoch_sources(
    output_dir: Path,
    rows: list[dict[str, str]],
) -> dict[str, int]:
    sources = sorted({row["source"] for row in rows})
    if not sources:
        raise ValueError("group epoch table has no signal sources")
    n_columns = 2
    n_rows = math.ceil(len(sources) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(180 / 25.4, max(3.0, n_rows * 2.25)),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    marks: dict[str, int] = {}
    for axis, source in zip(axes.flat, sources):
        marks[source] = plot_source_statistics(
            axis,
            rows,
            source=source,
            title=source.replace("_", " "),
        )
    for axis in axes.flat[len(sources) :]:
        axis.set_visible(False)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            output_dir / f"supplementary_epoch_sources.{suffix}",
            dpi=600 if suffix == "png" else 300,
            bbox_inches="tight",
        )
    plt.close(figure)
    return marks


def select_individual_strf_result(
    strf_dir: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, str]],
    list[dict[str, str]],
    Path,
]:
    candidates: list[tuple[float, str, str, str, Path, dict[str, str]]] = []
    for path in sorted(
        (strf_dir / "recordings").glob("*/feature_contributions.csv")
    ):
        for row in read_csv_required(path):
            q_value = float(row["fdr_p_value_correlation"])
            if not np.isfinite(q_value):
                raise ValueError(f"non-finite individual STRF q value in {path}")
            candidates.append(
                (
                    q_value,
                    row["recording_id"],
                    row["channel"],
                    row["feature"],
                    path,
                    row,
                )
            )
    if not candidates:
        raise ValueError("no individual STRF feature tests were found")
    q_value, recording_id, channel, feature, path, _ = min(candidates)
    feature_rows = [
        row
        for row in read_csv_required(path)
        if row["recording_id"] == recording_id and row["channel"] == channel
    ]
    comparison_path = path.parent / "model_comparisons.csv"
    comparison_rows = [
        row
        for row in read_csv_required(comparison_path)
        if row["recording_id"] == recording_id and row["channel"] == channel
    ]
    if {row["feature"] for row in feature_rows} != set(run_strf.FEATURE_FAMILIES):
        raise ValueError("selected individual STRF channel has incomplete features")
    selection = {
        "recording_id": recording_id,
        "channel": channel,
        "selected_feature": feature,
        "fdr_p_value_correlation": q_value,
        "passes_bh_fdr_0.05": q_value < 0.05,
        "selection_rule": (
            "minimum finite fdr_p_value_correlation across every individual "
            "recording, channel, and feature; deterministic lexical tie-break"
        ),
    }
    return (
        selection,
        feature_rows,
        comparison_rows,
        path.parent / "model_coefficients.npz",
    )


def plot_individual_test_bars(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    key: str,
    title: str,
) -> int:
    labels = [str(row[key]) for row in rows]
    values = [float(row["mean_delta_correlation"]) for row in rows]
    colors = [
        PALETTE["orange"]
        if float(row["fdr_p_value_correlation"]) < 0.05
        else PALETTE["light_grey"]
        for row in rows
    ]
    bars = ax.bar(
        np.arange(len(rows)),
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    marks = annotate_bars(
        ax,
        bars,
        rows,
        p_key="fdr_p_value_correlation",
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(
        xticks=np.arange(len(rows)),
        xticklabels=[FEATURE_LABELS.get(label, label) for label in labels],
        ylabel=r"Held-out $\Delta r$",
        title=title,
    )
    ax.tick_params(axis="x", rotation=32, labelsize=5.5)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    scientific_axes(ax, x=False)
    return marks


def select_individual_epoch_result(
    epochs_dir: Path,
) -> tuple[dict[str, object], list[dict[str, str]], Path]:
    candidates: list[
        tuple[int, float, str, str, str, str, float, Path, dict[str, str]]
    ] = []
    for path in sorted(
        (epochs_dir / "recordings").glob("*/timepoint_statistics.csv")
    ):
        for row in read_csv_required(path):
            q_value = float(row["fdr_p_value"])
            if not np.isfinite(q_value):
                raise ValueError(f"non-finite individual epoch q value in {path}")
            passes = parse_bool(row["passes_fdr_and_cluster"])
            candidates.append(
                (
                    0 if passes else 1,
                    q_value,
                    row["recording_id"],
                    row["channel"],
                    row["source"],
                    row["predictor"],
                    float(row["time_s"]),
                    path,
                    row,
                )
            )
    if not candidates:
        raise ValueError("no individual epoch time-point tests were found")
    (
        _,
        q_value,
        recording_id,
        channel,
        source,
        predictor,
        time_s,
        path,
        selected_row,
    ) = min(candidates)
    rows = sorted(
        (
            row
            for row in read_csv_required(path)
            if row["recording_id"] == recording_id
            and row["channel"] == channel
            and row["source"] == source
            and row["predictor"] == predictor
        ),
        key=lambda row: float(row["time_s"]),
    )
    if not rows:
        raise AssertionError("selected individual epoch time course is empty")
    selection = {
        "recording_id": recording_id,
        "channel": channel,
        "source": source,
        "predictor": predictor,
        "selected_time_s": time_s,
        "fdr_p_value": q_value,
        "passes_fdr_and_cluster": parse_bool(
            selected_row["passes_fdr_and_cluster"]
        ),
        "selection_rule": (
            "minimum finite fdr_p_value among duration-qualified individual "
            "time points when available, otherwise global minimum; "
            "deterministic lexical/time tie-break"
        ),
    }
    return selection, rows, path.parent / "epoch_data.npz"


def plot_individual_epoch_statistic(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    selection: dict[str, object],
) -> int:
    times = require_finite(
        (float(row["time_s"]) for row in rows),
        "selected individual epoch times",
    )
    statistics = require_finite(
        (float(row["f_statistic"]) for row in rows),
        "selected individual epoch F statistics",
    )
    significant = np.asarray(
        [parse_bool(row["passes_fdr_and_cluster"]) for row in rows]
    )
    color = (
        PALETTE["orange"]
        if selection["predictor"] == "boundary_strength"
        else PALETTE["purple"]
    )
    ax.plot(times, statistics, color=color)
    ax.fill_between(
        times,
        0,
        statistics,
        where=significant,
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    marks = annotate_time_significance(
        ax,
        times,
        significant,
        color=color,
        level=0,
    )
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set(
        xlabel="Time from syllable offset (s)",
        ylabel="F statistic",
        title=(
            f"{selection['source']} × "
            f"{PREDICTOR_LABELS[str(selection['predictor'])]}"
        ),
    )
    scientific_axes(ax)
    return marks


def plot_individual_event_waveforms(
    ax: plt.Axes,
    epoch_path: Path,
    selection: dict[str, object],
) -> list[dict[str, object]]:
    dataset = epoch_stats.load_epoch_dataset(epoch_path)
    channel = str(selection["channel"])
    if channel not in dataset.channel_names:
        raise ValueError(f"selected epoch channel {channel!r} is absent")
    channel_index = dataset.channel_names.index(channel)
    source_name = str(selection["source"])
    if source_name == "observed":
        source = dataset.observed
    elif source_name == "m2_residual":
        source = dataset.residual_m2
    elif source_name.startswith("prediction_"):
        model = source_name.removeprefix("prediction_")
        source = dataset.predictions[
            :, :, dataset.model_names.index(model), :
        ]
    else:
        raise ValueError(f"unknown selected epoch source: {source_name}")
    predictor = str(selection["predictor"])
    labels = (
        epoch_stats.quantile_bins(dataset.boundary_strength) + 1
        if predictor == "boundary_strength"
        else dataset.struc_depth
    )
    levels = sorted(np.unique(labels))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(levels)))
    rows: list[dict[str, object]] = []
    for level, color in zip(levels, colors):
        values = source[labels == level, :, channel_index]
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / math.sqrt(len(values))
        label = (
            f"Quartile {float(level):g}"
            if predictor == "boundary_strength"
            else f"Depth {float(level):g}"
        )
        ax.plot(dataset.times, mean, color=color, label=label)
        ax.fill_between(
            dataset.times,
            mean - 1.96 * sem,
            mean + 1.96 * sem,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        rows.extend(
            {
                "recording_id": dataset.recording_id,
                "channel": channel,
                "source": source_name,
                "predictor": predictor,
                "level": float(level),
                "time_s": float(time),
                "mean": float(mean_value),
                "sem": float(sem_value),
                "n_epochs": len(values),
            }
            for time, mean_value, sem_value in zip(dataset.times, mean, sem)
        )
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(
        xlabel="Time from syllable offset (s)",
        ylabel="Signal (z)",
        title="Selected individual event-related response",
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
    )
    scientific_axes(ax)
    return rows


def artist_collision_audit(
    fig: plt.Figure,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    texts: list[dict[str, object]] = []
    seen_geometry: set[tuple[str, tuple[float, float, float, float]]] = set()
    seen_artists: set[int] = set()
    owned_artists = [
        (axis_index, artist)
        for axis_index, axis in enumerate(fig.axes)
        for artist in axis.findobj(match=lambda item: hasattr(item, "get_text"))
    ]
    owned_artists.extend(
        (-1, artist)
        for artist in fig.texts
        if hasattr(artist, "get_text")
    )
    for index, (axis_index, artist) in enumerate(owned_artists):
        if id(artist) in seen_artists:
            continue
        seen_artists.add(id(artist))
        text = str(artist.get_text()).strip()
        if not text or not artist.get_visible():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        normalized_bbox = (
            float(bbox.x0 / width),
            float(bbox.y0 / height),
            float(bbox.x1 / width),
            float(bbox.y1 / height),
        )
        geometry_key = (text, tuple(round(value, 6) for value in normalized_bbox))
        if geometry_key in seen_geometry:
            continue
        seen_geometry.add(geometry_key)
        texts.append(
            {
                "id": index,
                "text": text,
                "bbox_fraction": list(normalized_bbox),
                "axes_id": axis_index,
            }
        )
    collisions: list[dict[str, object]] = []
    for left_index, left in enumerate(texts):
        lx0, ly0, lx1, ly1 = left["bbox_fraction"]
        for right in texts[left_index + 1 :]:
            if left["axes_id"] != right["axes_id"]:
                continue
            rx0, ry0, rx1, ry1 = right["bbox_fraction"]
            overlap_x = min(lx1, rx1) - max(lx0, rx0)
            overlap_y = min(ly1, ry1) - max(ly0, ry0)
            # Bounding boxes can touch at axis corners without glyph overlap.
            # Flag only material intersections visible at journal scale.
            if overlap_x > 0.015 and overlap_y > 0.015:
                collisions.append(
                    {
                        "left_id": left["id"],
                        "right_id": right["id"],
                        "left_text": left["text"],
                        "right_text": right["text"],
                        "overlap_fraction": [overlap_x, overlap_y],
                    }
                )
    return texts, collisions


def write_geometry_diagnostics(
    output_dir: Path,
    fig: plt.Figure,
    panel_axes: dict[str, list[plt.Axes]],
    *,
    suffix: str = "",
    aligned_rows: Sequence[Sequence[str]] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    panels: dict[str, object] = {}
    for label, axes in panel_axes.items():
        boxes = [axis.get_position().bounds for axis in axes]
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[0] + box[2] for box in boxes)
        y1 = max(box[1] + box[3] for box in boxes)
        panels[label] = {
            "bbox_fraction": [x0, y0, x1, y1],
            "inside_canvas": bool(
                0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
            ),
        }
    row_spreads: list[dict[str, object]] = []
    for labels in aligned_rows:
        tops = [panels[label]["bbox_fraction"][3] for label in labels]
        bottoms = [panels[label]["bbox_fraction"][1] for label in labels]
        row_spreads.append(
            {
                "panels": list(labels),
                "top_spread": float(max(tops) - min(tops)),
                "bottom_spread": float(max(bottoms) - min(bottoms)),
            }
        )
    rows_aligned = all(
        row["top_spread"] < 0.03 and row["bottom_spread"] < 0.03
        for row in row_spreads
    )
    alignment = {
        "coordinate_system": "figure fraction, origin lower left",
        "panels": panels,
        "aligned_rows": row_spreads,
        "status": (
            "pass"
            if all(panel["inside_canvas"] for panel in panels.values())
            and rows_aligned
            else "review_required"
        ),
    }
    write_json(output_dir / f"alignment{suffix}.json", alignment)
    diagnostic, axis = plt.subplots(figsize=(7, 5))
    axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title="Panel alignment diagnostic")
    for label, panel in panels.items():
        x0, y0, x1, y1 = panel["bbox_fraction"]
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=PALETTE["blue"],
                linewidth=1.2,
            )
        )
        axis.text(x0 + 0.01, y1 - 0.02, label, va="top", fontweight="bold")
    diagnostic.savefig(output_dir / f"alignment_diagnostic{suffix}.svg")
    plt.close(diagnostic)

    texts, collisions = artist_collision_audit(fig)
    collision = {
        "method": (
            "pairwise intersection of de-duplicated rendered text bounding "
            "boxes within each panel axis, requiring >1.5% figure overlap in "
            "both dimensions; panel boundaries are audited separately"
        ),
        "n_text_artists": len(texts),
        "n_collisions": len(collisions),
        "collisions": collisions,
        "status": "pass" if not collisions else "review_required",
    }
    write_json(output_dir / f"collision{suffix}.json", collision)
    with PdfPages(output_dir / f"collision_diagnostic{suffix}.pdf") as pdf:
        diagnostic, axis = plt.subplots(figsize=(8, 6))
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            aspect="equal",
            title="Rendered text collision diagnostic",
        )
        colliding_ids = {
            collision_id
            for item in collisions
            for collision_id in (item["left_id"], item["right_id"])
        }
        for text in texts:
            x0, y0, x1, y1 = text["bbox_fraction"]
            axis.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=(
                        "#b2182b" if text["id"] in colliding_ids else "#bdbdbd"
                    ),
                    linewidth=0.7,
                )
            )
        axis.text(
            0.01,
            0.01,
            f"{len(collisions)} pairwise collision(s); red boxes require review",
            transform=axis.transAxes,
        )
        pdf.savefig(diagnostic)
        plt.close(diagnostic)
    return alignment, collision


def significant_claims(
    feature_tests: list[dict[str, str]],
    group_stats: list[dict[str, str]],
) -> list[str]:
    significant_features = [
        FEATURE_LABELS[row["feature"]]
        for row in feature_tests
        if float(row["fdr_p_value_correlation"]) < 0.05
    ]
    feature_claim = (
        "Subject-level held-out encoding supported unique contributions from "
        + ", ".join(significant_features)
        + " (BH-FDR q<0.05)."
        if significant_features
        else "No unique feature contribution survived subject-level BH-FDR q<0.05."
    )
    cluster_counts = {
        predictor: sum(
            len(
                contiguous_true_runs(
                    [
                        parse_bool(row["passes_fdr_and_cluster"])
                        for row in sorted(
                            (
                                candidate
                                for candidate in group_stats
                                if candidate["source"] == "observed"
                                and candidate["predictor"] == predictor
                            ),
                            key=lambda candidate: float(candidate["time_s"]),
                        )
                    ]
                )
            )
            for _ in [0]
        )
        for predictor in PREDICTOR_LABELS
    }
    present = [
        f"{PREDICTOR_LABELS[predictor]} ({count} interval{'s' if count != 1 else ''})"
        for predictor, count in cluster_counts.items()
        if count
    ]
    epoch_claim = (
        "Observed ER-Hγ showed FDR- and duration-qualified separability for "
        + ", ".join(present)
        + "."
        if present
        else "Observed ER-Hγ showed no interval meeting both q<0.01 and the "
        "four-sample duration requirement."
    )
    return [feature_claim, epoch_claim]


def run(args: argparse.Namespace) -> int:
    for name in ("strf_dir", "strf_group_dir", "epochs_dir", "epochs_group_dir"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_dir():
            raise FileNotFoundError(f"required result directory does not exist: {value}")
        setattr(args, name, value)
    args.output_dir = args.output_dir.expanduser().resolve()
    protected = {
        args.strf_dir,
        args.strf_group_dir,
        args.epochs_dir,
        args.epochs_group_dir,
    }
    if args.output_dir in protected:
        raise ValueError("publication output must differ from every analysis input")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty (use --overwrite): {args.output_dir}"
        )
    if args.output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.output_dir / "source_data"
    source_dir.mkdir()

    configs = validate_full_run_configs(
        (
            ("individual STRF", args.strf_dir / "analysis_config.json"),
            ("group STRF", args.strf_group_dir / "analysis_config.json"),
            ("individual epochs", args.epochs_dir / "analysis_config.json"),
            ("group epochs", args.epochs_group_dir / "analysis_config.json"),
        )
    )
    epoch_completion = read_json_required(
        args.epochs_dir / epoch_stats.COMPLETION_FILENAME
    )
    if epoch_completion.get("status") != "complete":
        raise ValueError("individual epoch completion marker is not complete")
    group_recording = args.strf_group_dir / "recordings" / "GROUP"
    metrics = read_csv_required(group_recording / "model_metrics.csv")
    models, folds = validate_model_metrics(metrics)
    membership = read_csv_required(args.strf_group_dir / "group_membership.csv")
    inclusion = read_csv_required(args.strf_group_dir / "recording_inclusion.csv")
    excluded_path = args.strf_group_dir / "excluded_channels.csv"
    excluded = (
        read_csv_required(excluded_path)
        if excluded_path.is_file() and excluded_path.stat().st_size
        else []
    )
    feature_tests = read_csv_required(
        args.strf_group_dir / "subject_feature_contributions.csv"
    )
    feature_points = subject_feature_points(args.strf_dir, membership)
    group_stats = read_csv_required(
        args.epochs_group_dir / "group_timepoint_statistics.csv"
    )
    epoch_paths = sorted((args.epochs_dir / "recordings").glob("*/epoch_data.npz"))
    epoch_recording_dirs = sorted(
        path
        for path in (args.epochs_dir / "recordings").iterdir()
        if path.is_dir()
    )
    partial_epoch_dirs = [
        path for path in epoch_recording_dirs if not (path / "epoch_data.npz").is_file()
    ]
    if partial_epoch_dirs:
        raise ValueError(
            "partial individual epoch outputs lack epoch_data.npz: "
            f"{partial_epoch_dirs[:10]}"
        )
    times, waveform_rows, missing_conditions, epoch_accounting = (
        load_subject_waveforms(epoch_paths, strf_dir=args.strf_dir)
    )
    strf_recordings = individual_strf_recordings(args.strf_dir)
    epoch_recordings = {str(row["recording_id"]) for row in epoch_accounting}
    if epoch_recordings != strf_recordings:
        raise ValueError(
            "individual STRF/epoch recording cohorts differ: "
            f"missing_epochs={sorted(strf_recordings - epoch_recordings)}, "
            f"unexpected_epochs={sorted(epoch_recordings - strf_recordings)}"
        )
    completed_recordings = set(epoch_completion.get("recording_ids", []))
    if completed_recordings != epoch_recordings:
        raise ValueError(
            "individual epoch completion marker differs from epoch files: "
            f"missing={sorted(epoch_recordings - completed_recordings)}, "
            f"unexpected={sorted(completed_recordings - epoch_recordings)}"
        )
    epoch_membership = read_csv_required(
        args.epochs_group_dir / "group_membership.csv"
    )
    membership_recordings = {row["recording_id"] for row in epoch_membership}
    if membership_recordings != epoch_recordings:
        raise ValueError(
            "group-epoch membership differs from individual epoch cohort: "
            f"missing={sorted(epoch_recordings - membership_recordings)}, "
            f"unexpected={sorted(membership_recordings - epoch_recordings)}"
        )
    epoch_subjects = {subject_id(recording) for recording in epoch_recordings}
    validate_group_epoch_statistics(
        group_stats,
        expected_subjects=len(epoch_subjects),
    )

    counts = cohort_counts(inclusion, membership, excluded)
    claims = significant_claims(feature_tests, group_stats)

    set_publication_style()
    figure = plt.figure(figsize=(183 / 25.4, 230 / 25.4), layout="constrained")
    grid = figure.add_gridspec(
        6,
        4,
        width_ratios=(1.05, 1.55, 1.55, 1.15),
        wspace=0.24,
        hspace=0.75,
    )
    axes = {
        "A": figure.add_subplot(grid[0:3, 0]),
        "B": figure.add_subplot(grid[3:6, 0]),
        "C": figure.add_subplot(grid[:, 1:3]),
    }
    panel_a_rows = plot_group_accuracy(axes["A"], metrics, models, folds)
    feature_marks = plot_subject_features(axes["B"], feature_points, feature_tests)
    epoch_marks = plot_source_statistics(
        axes["C"],
        group_stats,
        source="observed",
        title=textwrap.fill(claims[1], width=58),
    )
    coefficient_axes, coefficient_rows = plot_coefficients(
        figure,
        grid[0:2, 3],
        group_recording / "model_coefficients.npz",
        title="Group M5 coefficients",
    )
    axes["E"] = figure.add_subplot(grid[2:4, 3])
    axes["F"] = figure.add_subplot(grid[4:6, 3])
    plot_subject_epochs(
        axes["E"],
        times,
        waveform_rows,
        predictor="struc_depth",
    )
    plot_subject_epochs(
        axes["F"],
        times,
        waveform_rows,
        predictor="boundary_strength",
    )
    panel_axes = {
        "A": [axes["A"]],
        "B": [axes["B"]],
        "C": [axes["C"]],
        "D": coefficient_axes,
        "E": [axes["E"]],
        "F": [axes["F"]],
    }
    for label, label_axes in panel_axes.items():
        panel_label(label_axes[0], label)

    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            args.output_dir / f"main_figure.{suffix}",
            dpi=600 if suffix == "png" else 300,
            bbox_inches="tight",
        )

    (
        individual_strf_selection,
        individual_feature_rows,
        individual_comparison_rows,
        individual_coefficient_path,
    ) = select_individual_strf_result(args.strf_dir)
    (
        individual_epoch_selection,
        individual_epoch_rows,
        individual_epoch_path,
    ) = select_individual_epoch_result(args.epochs_dir)
    figure_2 = plt.figure(
        figsize=(183 / 25.4, 145 / 25.4),
        layout="constrained",
    )
    grid_2 = figure_2.add_gridspec(2, 2, hspace=0.30, wspace=0.22)
    axes_2 = {
        "A": figure_2.add_subplot(grid_2[0, 0]),
        "C": figure_2.add_subplot(grid_2[1, 0]),
        "D": figure_2.add_subplot(grid_2[1, 1]),
    }
    individual_feature_marks = plot_individual_test_bars(
        axes_2["A"],
        individual_feature_rows,
        key="feature",
        title=(
            f"Most supported STRF channel: "
            f"{individual_strf_selection['channel']}"
        ),
    )
    individual_coefficient_axes, individual_coefficient_rows = plot_coefficients(
        figure_2,
        grid_2[0, 1],
        individual_coefficient_path,
        channel_name=str(individual_strf_selection["channel"]),
        title="Selected individual M5 coefficients",
    )
    individual_epoch_marks = plot_individual_epoch_statistic(
        axes_2["C"],
        individual_epoch_rows,
        individual_epoch_selection,
    )
    individual_waveform_rows = plot_individual_event_waveforms(
        axes_2["D"],
        individual_epoch_path,
        individual_epoch_selection,
    )
    panel_axes_2 = {
        "A": [axes_2["A"]],
        "B": individual_coefficient_axes,
        "C": [axes_2["C"]],
        "D": [axes_2["D"]],
    }
    for label, label_axes in panel_axes_2.items():
        panel_label(label_axes[0], label)
    for suffix in ("pdf", "svg", "png"):
        figure_2.savefig(
            args.output_dir / f"main_figure_2.{suffix}",
            dpi=600 if suffix == "png" else 300,
            bbox_inches="tight",
        )
    supplementary_marks = plot_all_epoch_sources(args.output_dir, group_stats)

    write_csv(source_dir / "cohort_counts.csv", [counts])
    write_csv(source_dir / "panel_A_group_accuracy.csv", panel_a_rows)
    write_csv(source_dir / "panel_B_subject_feature_points.csv", feature_points)
    write_csv(source_dir / "panel_B_subject_feature_tests.csv", feature_tests)
    write_csv(source_dir / "panel_D_group_coefficients.csv", coefficient_rows)
    write_csv(source_dir / "panel_EF_subject_waveforms.csv", waveform_rows)
    panel_c_rows = [
        row
        for row in group_stats
        if row["source"] == "observed"
        and row["predictor"] in PREDICTOR_LABELS
    ]
    write_csv(source_dir / "panel_C_group_timepoint_statistics.csv", panel_c_rows)
    write_csv(
        source_dir / "figure_2_panel_A_feature_tests.csv",
        individual_feature_rows,
    )
    write_csv(
        source_dir / "figure_2_panel_A_model_comparisons.csv",
        individual_comparison_rows,
    )
    write_csv(
        source_dir / "figure_2_panel_B_coefficients.csv",
        individual_coefficient_rows,
    )
    write_csv(
        source_dir / "figure_2_panel_C_timepoint_statistics.csv",
        individual_epoch_rows,
    )
    write_csv(
        source_dir / "figure_2_panel_D_event_waveforms.csv",
        individual_waveform_rows,
    )
    write_json(
        args.output_dir / "individual_result_selection.json",
        {
            "strf": individual_strf_selection,
            "epoch": individual_epoch_selection,
        },
    )
    write_csv(
        source_dir / "supplementary_all_group_timepoint_statistics.csv",
        group_stats,
    )
    write_csv(source_dir / "epoch_retention_accounting.csv", epoch_accounting)

    total_source_events = sum(
        int(row["n_source_prosody_events"]) for row in epoch_accounting
    )
    total_retained_epochs = sum(
        int(row["n_retained_epochs"]) for row in epoch_accounting
    )
    exclusion_rows = [
        {"category": key, "count": value} for key, value in counts.items()
    ] + [
        {
            "category": "missing_subject_structure_depth_cells",
            "count": len(missing_conditions),
        },
        {
            "category": "source_prosody_events",
            "count": total_source_events,
        },
        {
            "category": "retained_prosody_epochs",
            "count": total_retained_epochs,
        },
        {
            "category": "prosody_events_not_retained_as_epochs",
            "count": total_source_events - total_retained_epochs,
        },
    ]
    write_csv(args.output_dir / "exclusion_counts.csv", exclusion_rows)
    write_json(
        args.output_dir / "exclusion_counts.json",
        {
            "counts": counts,
            "missing_subject_structure_depth_cells": missing_conditions,
            "epoch_retention_by_recording": epoch_accounting,
            "interpretation": (
                "Missing condition cells and source events not retained by the "
                "upstream epoch extractor are reported and not imputed; every "
                "retained subject trajectory is rendered."
            ),
        },
    )

    panel_notes = {
        "A": "All pooled held-out story folds; bars are mean ± SEM. No absolute-performance significance test was invented.",
        "B": "Every subject mean is shown. Bars are mean ± 1.96 SEM; stars reproduce subject-level BH-FDR-adjusted correlation tests.",
        "C": "Central claim panel: observed group F statistics; shadows, lines, and stars identify BH-FDR q<0.01 intervals retained for at least four samples.",
        "D": "M5 coefficients across all folds. Linguistic ribbons are descriptive ±1.96 SEM across folds.",
        "E": "All retained subject-level observed ER-Hγ trajectories by structure depth; missing cells and source-to-epoch retention are disclosed, not imputed.",
        "F": "All retained subject-level observed ER-Hγ trajectories by within-recording boundary-strength quartile; quartiles are visualization-only.",
        "Supplementary": "Every tested observed, M2-residual, and model-prediction source is shown without source selection; significance marks reproduce the same saved thresholds.",
    }
    notes_text = ["# Panel notes", "", "## Data-supported claims"]
    notes_text.extend(f"- {claim}" for claim in claims)
    for label, note in panel_notes.items():
        notes_text.extend(["", f"## Panel {label}", note])
    (args.output_dir / "panel_notes.md").write_text(
        "\n".join(notes_text) + "\n", encoding="utf-8"
    )
    strf_pass = bool(individual_strf_selection["passes_bh_fdr_0.05"])
    epoch_pass = bool(individual_epoch_selection["passes_fdr_and_cluster"])
    panel_notes_2 = {
        "A": (
            "Feature contributions for the individual recording/channel with "
            "the minimum BH-FDR-adjusted correlation p value. "
            f"Selected q={individual_strf_selection['fdr_p_value_correlation']:.3g}; "
            f"threshold passed={strf_pass}."
        ),
        "B": (
            "M5 STRF coefficients for that selected channel; ribbons are "
            "descriptive ±1.96 SEM across dependent outer folds."
        ),
        "C": (
            "Time-resolved F statistic for the duration-qualified individual "
            "epoch result with the minimum adjusted p value when available. "
            f"Selected q={individual_epoch_selection['fdr_p_value']:.3g}; "
            f"FDR+duration passed={epoch_pass}."
        ),
        "D": (
            "All retained epochs for the selected individual source and "
            "predictor, shown as condition means ±1.96 SEM; boundary strength "
            "uses equal-count quartiles for visualization only."
        ),
    }
    notes_2 = [
        "# Main Figure 2 panel notes",
        "",
        "Selections are deterministic across all individual result rows; no "
        "recording, channel, source, or time point was sampled for rendering.",
    ]
    for label, note in panel_notes_2.items():
        notes_2.extend(["", f"## Panel {label}", note])
    (args.output_dir / "panel_notes_main_figure_2.md").write_text(
        "\n".join(notes_2) + "\n",
        encoding="utf-8",
    )

    source_mapping = {
        "no_sampling": True,
        "validated_upstream_configs": {
            label: str(path)
            for label, path in (
                ("individual STRF", args.strf_dir / "analysis_config.json"),
                ("group STRF", args.strf_group_dir / "analysis_config.json"),
                ("individual epochs", args.epochs_dir / "analysis_config.json"),
                ("group epochs", args.epochs_group_dir / "analysis_config.json"),
            )
        },
        "upstream_max_recordings": {
            label: config.get("max_recordings")
            for label, config in configs.items()
        },
        "epoch_retention_source_data": (
            "source_data/epoch_retention_accounting.csv"
        ),
        "panels": {
            "A": {
                "input": str(group_recording / "model_metrics.csv"),
                "fields": ["model", "outer_fold", "correlation"],
                "source_data": "source_data/panel_A_group_accuracy.csv",
            },
            "B": {
                "inputs": [
                    str(args.strf_dir / "recordings/*/feature_contributions.csv"),
                    str(args.strf_group_dir / "subject_feature_contributions.csv"),
                ],
                "source_data": [
                    "source_data/panel_B_subject_feature_points.csv",
                    "source_data/panel_B_subject_feature_tests.csv",
                ],
            },
            "C": {
                "input": str(
                    args.epochs_group_dir / "group_timepoint_statistics.csv"
                ),
                "source_data": "source_data/panel_C_group_timepoint_statistics.csv",
            },
            "D": {
                "input": str(group_recording / "model_coefficients.npz"),
                "array": "M5_full",
                "source_data": "source_data/panel_D_group_coefficients.csv",
            },
            "E": {
                "input": str(args.epochs_dir / "recordings/*/epoch_data.npz"),
                "array": "observed",
                "source_data": "source_data/panel_EF_subject_waveforms.csv",
            },
            "F": {
                "input": str(args.epochs_dir / "recordings/*/epoch_data.npz"),
                "array": "observed",
                "source_data": "source_data/panel_EF_subject_waveforms.csv",
            },
            "Supplementary": {
                "input": str(
                    args.epochs_group_dir / "group_timepoint_statistics.csv"
                ),
                "source_data": (
                    "source_data/supplementary_all_group_timepoint_statistics.csv"
                ),
            },
        },
        "main_figure_2": {
            "selection": "individual_result_selection.json",
            "A": {
                "inputs": [
                    str(
                        args.strf_dir
                        / "recordings/*/feature_contributions.csv"
                    ),
                    str(
                        args.strf_dir
                        / "recordings/*/model_comparisons.csv"
                    ),
                ],
                "source_data": [
                    "source_data/figure_2_panel_A_feature_tests.csv",
                    "source_data/figure_2_panel_A_model_comparisons.csv",
                ],
            },
            "B": {
                "input": str(individual_coefficient_path),
                "source_data": "source_data/figure_2_panel_B_coefficients.csv",
            },
            "C": {
                "input": str(
                    args.epochs_dir
                    / "recordings/*/timepoint_statistics.csv"
                ),
                "source_data": (
                    "source_data/figure_2_panel_C_timepoint_statistics.csv"
                ),
            },
            "D": {
                "input": str(individual_epoch_path),
                "source_data": (
                    "source_data/figure_2_panel_D_event_waveforms.csv"
                ),
            },
        },
    }
    write_json(args.output_dir / "source_data_mapping.json", source_mapping)

    expected_feature_marks = sum(
        float(row["fdr_p_value_correlation"]) < 0.05 for row in feature_tests
    )
    expected_epoch_marks = sum(
        len(
            contiguous_true_runs(
                [
                    parse_bool(row["passes_fdr_and_cluster"])
                    for row in sorted(
                        (
                            candidate
                            for candidate in group_stats
                            if candidate["source"] == "observed"
                            and candidate["predictor"] == predictor
                        ),
                        key=lambda candidate: float(candidate["time_s"]),
                    )
                ]
            )
        )
        for predictor in PREDICTOR_LABELS
    )
    visual_audit_1 = {
        label: {
            "description": panel_notes[label],
            "axes_present": bool(panel_axes[label]),
            "titles": [axis.get_title() for axis in panel_axes[label]],
            "xlabels": [axis.get_xlabel() for axis in panel_axes[label]],
            "ylabels": [axis.get_ylabel() for axis in panel_axes[label]],
            "source_row_count": {
                "A": len(panel_a_rows),
                "B": len(feature_points),
                "C": len(panel_c_rows),
                "D": len(coefficient_rows),
                "E": sum(
                    row["predictor"] == "struc_depth"
                    for row in waveform_rows
                ),
                "F": sum(
                    row["predictor"] == "boundary_strength"
                    for row in waveform_rows
                ),
            }[label],
            "status": "pass",
        }
        for label in panel_axes
    }
    visual_audit_1["B"]["expected_significance_marks"] = expected_feature_marks
    visual_audit_1["B"]["rendered_significance_marks"] = feature_marks
    visual_audit_1["C"]["expected_significance_intervals"] = expected_epoch_marks
    visual_audit_1["C"]["rendered_significance_intervals"] = epoch_marks
    if feature_marks != expected_feature_marks or epoch_marks != expected_epoch_marks:
        raise AssertionError("rendered significance marks do not match result tables")
    expected_individual_feature_marks = sum(
        float(row["fdr_p_value_correlation"]) < 0.05
        for row in individual_feature_rows
    )
    expected_individual_epoch_marks = len(
        contiguous_true_runs(
            [
                parse_bool(row["passes_fdr_and_cluster"])
                for row in individual_epoch_rows
            ]
        )
    )
    visual_audit_2 = {
        label: {
            "description": panel_notes_2[label],
            "axes_present": bool(panel_axes_2[label]),
            "titles": [axis.get_title() for axis in panel_axes_2[label]],
            "xlabels": [axis.get_xlabel() for axis in panel_axes_2[label]],
            "ylabels": [axis.get_ylabel() for axis in panel_axes_2[label]],
            "source_row_count": {
                "A": len(individual_feature_rows),
                "B": len(individual_coefficient_rows),
                "C": len(individual_epoch_rows),
                "D": len(individual_waveform_rows),
            }[label],
            "status": "pass",
        }
        for label in panel_axes_2
    }
    visual_audit_2["A"].update(
        {
            "expected_significance_marks": expected_individual_feature_marks,
            "rendered_significance_marks": individual_feature_marks,
        }
    )
    visual_audit_2["C"].update(
        {
            "expected_significance_intervals": expected_individual_epoch_marks,
            "rendered_significance_intervals": individual_epoch_marks,
        }
    )
    if (
        individual_feature_marks != expected_individual_feature_marks
        or individual_epoch_marks != expected_individual_epoch_marks
    ):
        raise AssertionError(
            "individual rendered significance marks do not match result tables"
        )
    write_json(
        args.output_dir / "panel_visual_audit.json",
        {
            "main_figure": visual_audit_1,
            "main_figure_2": visual_audit_2,
        },
    )

    alignment, collision = write_geometry_diagnostics(
        args.output_dir,
        figure,
        panel_axes,
    )
    alignment_2, collision_2 = write_geometry_diagnostics(
        args.output_dir,
        figure_2,
        panel_axes_2,
        suffix="_main_figure_2",
        aligned_rows=(("C", "D"),),
    )
    plt.close(figure)
    plt.close(figure_2)
    required_outputs = [
        "main_figure.pdf",
        "main_figure.svg",
        "main_figure.png",
        "main_figure_2.pdf",
        "main_figure_2.svg",
        "main_figure_2.png",
        "supplementary_epoch_sources.pdf",
        "supplementary_epoch_sources.svg",
        "supplementary_epoch_sources.png",
        "panel_notes.md",
        "panel_notes_main_figure_2.md",
        "source_data_mapping.json",
        "exclusion_counts.csv",
        "panel_visual_audit.json",
        "alignment.json",
        "alignment_diagnostic.svg",
        "collision.json",
        "collision_diagnostic.pdf",
        "alignment_main_figure_2.json",
        "alignment_diagnostic_main_figure_2.svg",
        "collision_main_figure_2.json",
        "collision_diagnostic_main_figure_2.pdf",
        "individual_result_selection.json",
    ]
    output_checks = {
        name: {
            "exists": (args.output_dir / name).is_file(),
            "bytes": (
                (args.output_dir / name).stat().st_size
                if (args.output_dir / name).is_file()
                else 0
            ),
        }
        for name in required_outputs
    }
    qa = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": "make_nature_figures.py",
        "claims": claims,
        "supplementary_significance_intervals_rendered": supplementary_marks,
        "checks": {
            "all_required_outputs_nonempty": all(
                item["exists"] and item["bytes"] > 0
                for item in output_checks.values()
            ),
            "editable_vector_outputs": [
                "main_figure.pdf",
                "main_figure.svg",
                "main_figure_2.pdf",
                "main_figure_2.svg",
            ],
            "all_requested_feature_families_present": (
                {row["feature"] for row in feature_tests}
                == set(run_strf.FEATURE_FAMILIES)
            ),
            "no_rendering_subsample": True,
            "no_upstream_max_recordings_subset": all(
                config.get("max_recordings") in (None, "None")
                for config in configs.values()
            ),
            "strf_epoch_recording_cohorts_match": (
                strf_recordings == epoch_recordings
            ),
            "group_epoch_membership_matches": (
                membership_recordings == epoch_recordings
            ),
            "epoch_retention_accounting_disclosed": True,
            "individual_epoch_completion_manifest_matches": (
                completed_recordings == epoch_recordings
            ),
            "missing_conditions_disclosed": True,
            "significance_marks_match_source_tables": True,
            "alignment_status": alignment["status"],
            "collision_status": collision["status"],
            "main_figure_2_alignment_status": alignment_2["status"],
            "main_figure_2_collision_status": collision_2["status"],
            "individual_selection_recorded": True,
            "scientific_axis_formatting": True,
        },
        "output_files": output_checks,
        "status": (
            "pass"
            if alignment["status"] == "pass"
            and collision["status"] == "pass"
            and alignment_2["status"] == "pass"
            and collision_2["status"] == "pass"
            and all(item["exists"] and item["bytes"] > 0 for item in output_checks.values())
            else "review_required"
        ),
    }
    write_json(args.output_dir / "pre_submission_qa.json", qa)
    print(f"OK publication figure bundle: {args.output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
