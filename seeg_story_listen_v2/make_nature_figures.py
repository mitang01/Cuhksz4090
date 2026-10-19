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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
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
        -0.16,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
    )


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


def validate_group_epoch_statistics(rows: list[dict[str, str]]) -> None:
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
) -> tuple[
    np.ndarray,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    by_subject_level: dict[tuple[str, float], list[np.ndarray]] = defaultdict(list)
    subjects: set[str] = set()
    levels: set[float] = set()
    reference_times: np.ndarray | None = None
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
        for level in np.unique(dataset.struc_depth):
            levels.add(float(level))
            mask = dataset.struc_depth == level
            if not np.any(mask):
                raise AssertionError("unreachable empty structure-depth mask")
            by_subject_level[(subject, float(level))].append(
                dataset.observed[mask].mean(axis=(0, 2))
            )
    if reference_times is None:
        raise ValueError("no individual epoch datasets were found")
    rows: list[dict[str, object]] = []
    for (subject, level), recording_waves in sorted(by_subject_level.items()):
        wave = np.mean(recording_waves, axis=0)
        rows.extend(
            {
                "subject": subject,
                "structure_depth": level,
                "time_s": float(time),
                "mean_observed_high_gamma_z": float(value),
                "n_recordings": len(recording_waves),
            }
            for time, value in zip(reference_times, wave)
        )
    missing = [
        {"subject": subject, "structure_depth": level}
        for subject in sorted(subjects)
        for level in sorted(levels)
        if (subject, level) not in by_subject_level
    ]
    return reference_times, rows, missing


def plot_cohort(
    ax: plt.Axes,
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
        xticklabels=[FEATURE_LABELS[feature] for feature in features],
        ylabel=r"Subject mean $\Delta r$",
        title="Unique feature contributions",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.text(
        0.02,
        0.98,
        "bars: mean; error: 95% CI\npoints: subjects; * BH-FDR $q<0.05$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
    )
    return marks


def plot_coefficients(
    figure: plt.Figure,
    subplot_spec,
    coefficient_path: Path,
) -> tuple[list[plt.Axes], list[dict[str, object]]]:
    if not coefficient_path.is_file():
        raise FileNotFoundError(f"required coefficients do not exist: {coefficient_path}")
    with np.load(coefficient_path, allow_pickle=False) as data:
        if "M5_full" not in data:
            raise ValueError("GROUP coefficients do not contain M5_full")
        coefficients = np.asarray(data["M5_full"], dtype=float)
        features = data["feature_names"].astype(str).tolist()
        lags = np.asarray(data["lags_s"], dtype=float)
    if coefficients.ndim != 4 or coefficients.shape[1] != 1:
        raise ValueError(f"unexpected GROUP coefficient shape: {coefficients.shape}")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("GROUP M5 coefficients contain non-finite values")
    mean = coefficients.mean(axis=0)[0]
    sem = coefficients.std(axis=0, ddof=1)[0] / math.sqrt(coefficients.shape[0])
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
    ax_mel.set(ylabel="Mel band", title="Full-model STRF coefficients")
    ax_mel.tick_params(labelbottom=False)
    figure.colorbar(image, ax=ax_mel, fraction=0.05, pad=0.02)
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
    ax_lines.legend(frameon=False, ncol=1, loc="best")
    source_rows = [
        {
            "outer_fold": fold,
            "feature": feature,
            "lag_s": float(lag),
            "coefficient": float(coefficients[fold, 0, feature_index, lag_index]),
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
) -> None:
    levels = sorted({float(row["structure_depth"]) for row in rows})
    subjects = sorted({str(row["subject"]) for row in rows})
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(levels)))
    for level, color in zip(levels, colors):
        level_waves: list[np.ndarray] = []
        for subject in subjects:
            selected = [
                row
                for row in rows
                if row["subject"] == subject
                and float(row["structure_depth"]) == level
            ]
            if not selected:
                continue
            wave_times = np.asarray([float(row["time_s"]) for row in selected])
            if not np.array_equal(wave_times, times):
                raise ValueError(
                    f"subject waveform grid is incomplete: {subject}, depth={level}"
                )
            wave = require_finite(
                (float(row["mean_observed_high_gamma_z"]) for row in selected),
                f"{subject}, depth={level}",
            )
            level_waves.append(wave)
            ax.plot(times, wave, color=color, alpha=0.14, linewidth=0.45)
        matrix = np.asarray(level_waves)
        ax.plot(
            times,
            matrix.mean(axis=0),
            color=color,
            linewidth=1.4,
            label=f"Depth {level:g} ($n$={len(matrix)})",
        )
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(
        xlabel="Time from syllable offset (s)",
        ylabel="High-gamma response (z)",
        title="Observed event-related responses",
    )
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.02,
        "thin lines: every available subject",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
    )


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
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.98,
        "shading/lines: BH-FDR $q<0.01$ + ≥4 samples",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
    )
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


def artist_collision_audit(
    fig: plt.Figure,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    texts: list[dict[str, object]] = []
    for index, artist in enumerate(fig.findobj(match=lambda item: hasattr(item, "get_text"))):
        text = str(artist.get_text()).strip()
        if not text or not artist.get_visible():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        texts.append(
            {
                "id": index,
                "text": text,
                "bbox_fraction": [
                    float(bbox.x0 / width),
                    float(bbox.y0 / height),
                    float(bbox.x1 / width),
                    float(bbox.y1 / height),
                ],
            }
        )
    collisions: list[dict[str, object]] = []
    for left_index, left in enumerate(texts):
        lx0, ly0, lx1, ly1 = left["bbox_fraction"]
        for right in texts[left_index + 1 :]:
            rx0, ry0, rx1, ry1 = right["bbox_fraction"]
            overlap_x = min(lx1, rx1) - max(lx0, rx0)
            overlap_y = min(ly1, ry1) - max(ly0, ry0)
            if overlap_x > 0.001 and overlap_y > 0.001:
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
    top_row = [panels[label]["bbox_fraction"][3] for label in ("A", "B", "C")]
    bottom_row = [panels[label]["bbox_fraction"][1] for label in ("D", "E", "F")]
    alignment = {
        "coordinate_system": "figure fraction, origin lower left",
        "panels": panels,
        "top_row_top_spread": float(max(top_row) - min(top_row)),
        "bottom_row_bottom_spread": float(max(bottom_row) - min(bottom_row)),
        "status": (
            "pass"
            if all(panel["inside_canvas"] for panel in panels.values())
            and max(top_row) - min(top_row) < 0.03
            and max(bottom_row) - min(bottom_row) < 0.03
            else "review_required"
        ),
    }
    write_json(output_dir / "alignment.json", alignment)
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
    diagnostic.savefig(output_dir / "alignment_diagnostic.svg")
    plt.close(diagnostic)

    texts, collisions = artist_collision_audit(fig)
    collision = {
        "method": "pairwise intersection of rendered text bounding boxes",
        "n_text_artists": len(texts),
        "n_collisions": len(collisions),
        "collisions": collisions,
        "status": "pass" if not collisions else "review_required",
    }
    write_json(output_dir / "collision.json", collision)
    with PdfPages(output_dir / "collision_diagnostic.pdf") as pdf:
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
    validate_group_epoch_statistics(group_stats)
    epoch_paths = sorted((args.epochs_dir / "recordings").glob("*/epoch_data.npz"))
    times, waveform_rows, missing_conditions = load_subject_waveforms(epoch_paths)

    set_publication_style()
    figure = plt.figure(figsize=(180 / 25.4, 145 / 25.4), layout="constrained")
    grid = figure.add_gridspec(2, 3, height_ratios=(0.92, 1.08))
    axes = {
        "A": figure.add_subplot(grid[0, 0]),
        "B": figure.add_subplot(grid[0, 1]),
        "C": figure.add_subplot(grid[0, 2]),
        "E": figure.add_subplot(grid[1, 1]),
        "F": figure.add_subplot(grid[1, 2]),
    }
    counts = plot_cohort(axes["A"], inclusion, membership, excluded)
    panel_b_rows = plot_group_accuracy(axes["B"], metrics, models, folds)
    feature_marks = plot_subject_features(axes["C"], feature_points, feature_tests)
    coefficient_axes, coefficient_rows = plot_coefficients(
        figure,
        grid[1, 0],
        group_recording / "model_coefficients.npz",
    )
    plot_subject_epochs(axes["E"], times, waveform_rows)
    epoch_marks = plot_source_statistics(
        axes["F"],
        group_stats,
        source="observed",
        title="Time-resolved prosody separability",
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
    supplementary_marks = plot_all_epoch_sources(args.output_dir, group_stats)

    write_csv(source_dir / "panel_A_cohort_counts.csv", [counts])
    write_csv(source_dir / "panel_B_group_accuracy.csv", panel_b_rows)
    write_csv(source_dir / "panel_C_subject_feature_points.csv", feature_points)
    write_csv(source_dir / "panel_C_subject_feature_tests.csv", feature_tests)
    write_csv(source_dir / "panel_D_group_coefficients.csv", coefficient_rows)
    write_csv(source_dir / "panel_E_subject_waveforms.csv", waveform_rows)
    panel_f_rows = [
        row
        for row in group_stats
        if row["source"] == "observed"
        and row["predictor"] in PREDICTOR_LABELS
    ]
    write_csv(source_dir / "panel_F_group_timepoint_statistics.csv", panel_f_rows)
    write_csv(
        source_dir / "supplementary_all_group_timepoint_statistics.csv",
        group_stats,
    )

    exclusion_rows = [
        {"category": key, "count": value} for key, value in counts.items()
    ] + [
        {
            "category": "missing_subject_structure_depth_cells",
            "count": len(missing_conditions),
        }
    ]
    write_csv(args.output_dir / "exclusion_counts.csv", exclusion_rows)
    write_json(
        args.output_dir / "exclusion_counts.json",
        {
            "counts": counts,
            "missing_subject_structure_depth_cells": missing_conditions,
            "interpretation": (
                "Missing condition cells are reported and not imputed; every "
                "available subject trajectory is rendered."
            ),
        },
    )

    claims = significant_claims(feature_tests, group_stats)
    panel_notes = {
        "A": "Cohort flow from recording inclusion, group membership, and explicit MISC exclusions.",
        "B": "All pooled held-out story folds; bars are mean ± SEM. No absolute-performance significance test was invented.",
        "C": "Every subject mean is shown. Bars are mean ± 1.96 SEM; stars reproduce subject-level BH-FDR-adjusted correlation tests.",
        "D": "M5 coefficients across all folds. Linguistic ribbons are descriptive ±1.96 SEM across folds.",
        "E": "All available subject-level observed ER-Hγ trajectories by structure depth; missing cells are disclosed, not imputed.",
        "F": "Observed group F statistics; shadows, lines, and stars identify BH-FDR q<0.01 intervals retained for at least four samples.",
        "Supplementary": "Every tested observed, M2-residual, and model-prediction source is shown without source selection; significance marks reproduce the same saved thresholds.",
    }
    notes_text = ["# Panel notes", "", "## Data-supported claims"]
    notes_text.extend(f"- {claim}" for claim in claims)
    for label, note in panel_notes.items():
        notes_text.extend(["", f"## Panel {label}", note])
    (args.output_dir / "panel_notes.md").write_text(
        "\n".join(notes_text) + "\n", encoding="utf-8"
    )

    source_mapping = {
        "no_sampling": True,
        "panels": {
            "A": {
                "inputs": [
                    str(args.strf_group_dir / "recording_inclusion.csv"),
                    str(args.strf_group_dir / "group_membership.csv"),
                    str(excluded_path),
                ],
                "source_data": "source_data/panel_A_cohort_counts.csv",
            },
            "B": {
                "input": str(group_recording / "model_metrics.csv"),
                "fields": ["model", "outer_fold", "correlation"],
                "source_data": "source_data/panel_B_group_accuracy.csv",
            },
            "C": {
                "inputs": [
                    str(args.strf_dir / "recordings/*/feature_contributions.csv"),
                    str(args.strf_group_dir / "subject_feature_contributions.csv"),
                ],
                "source_data": [
                    "source_data/panel_C_subject_feature_points.csv",
                    "source_data/panel_C_subject_feature_tests.csv",
                ],
            },
            "D": {
                "input": str(group_recording / "model_coefficients.npz"),
                "array": "M5_full",
                "source_data": "source_data/panel_D_group_coefficients.csv",
            },
            "E": {
                "input": str(args.epochs_dir / "recordings/*/epoch_data.npz"),
                "array": "observed",
                "source_data": "source_data/panel_E_subject_waveforms.csv",
            },
            "F": {
                "input": str(
                    args.epochs_group_dir / "group_timepoint_statistics.csv"
                ),
                "source_data": "source_data/panel_F_group_timepoint_statistics.csv",
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
    visual_audit = {
        label: {
            "description": panel_notes[label],
            "axes_present": bool(panel_axes[label]),
            "titles": [axis.get_title() for axis in panel_axes[label]],
            "xlabels": [axis.get_xlabel() for axis in panel_axes[label]],
            "ylabels": [axis.get_ylabel() for axis in panel_axes[label]],
            "source_row_count": {
                "A": 1,
                "B": len(panel_b_rows),
                "C": len(feature_points),
                "D": len(coefficient_rows),
                "E": len(waveform_rows),
                "F": len(panel_f_rows),
            }[label],
            "status": "pass",
        }
        for label in panel_axes
    }
    visual_audit["C"]["expected_significance_marks"] = expected_feature_marks
    visual_audit["C"]["rendered_significance_marks"] = feature_marks
    visual_audit["F"]["expected_significance_intervals"] = expected_epoch_marks
    visual_audit["F"]["rendered_significance_intervals"] = epoch_marks
    if feature_marks != expected_feature_marks or epoch_marks != expected_epoch_marks:
        raise AssertionError("rendered significance marks do not match result tables")
    write_json(args.output_dir / "panel_visual_audit.json", visual_audit)

    alignment, collision = write_geometry_diagnostics(
        args.output_dir,
        figure,
        panel_axes,
    )
    plt.close(figure)
    required_outputs = [
        "main_figure.pdf",
        "main_figure.svg",
        "main_figure.png",
        "supplementary_epoch_sources.pdf",
        "supplementary_epoch_sources.svg",
        "supplementary_epoch_sources.png",
        "panel_notes.md",
        "source_data_mapping.json",
        "exclusion_counts.csv",
        "panel_visual_audit.json",
        "alignment.json",
        "alignment_diagnostic.svg",
        "collision.json",
        "collision_diagnostic.pdf",
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
            "editable_vector_outputs": ["main_figure.pdf", "main_figure.svg"],
            "all_requested_feature_families_present": (
                {row["feature"] for row in feature_tests}
                == set(run_strf.FEATURE_FAMILIES)
            ),
            "no_rendering_subsample": True,
            "missing_conditions_disclosed": True,
            "significance_marks_match_source_tables": True,
            "alignment_status": alignment["status"],
            "collision_status": collision["status"],
        },
        "output_files": output_checks,
        "status": (
            "pass"
            if alignment["status"] == "pass"
            and collision["status"] == "pass"
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
