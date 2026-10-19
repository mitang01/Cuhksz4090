"""Shared, deterministic significance annotations for analysis figures."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.axes
import numpy as np


def significance_stars(adjusted_p: float) -> str:
    """Return the conventional mark for an already-adjusted p value."""
    if not np.isfinite(adjusted_p) or adjusted_p >= 0.05:
        return ""
    if adjusted_p < 0.001:
        return "***"
    if adjusted_p < 0.01:
        return "**"
    return "*"


def contiguous_true_runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """Return half-open index intervals for contiguous true values."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("significance mask must be one-dimensional")
    starts = np.flatnonzero(values & np.r_[True, ~values[:-1]])
    stops = np.flatnonzero(values & np.r_[~values[1:], True]) + 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def annotate_bars(
    ax: matplotlib.axes.Axes,
    bars: Sequence[object],
    rows: Sequence[dict[str, object]],
    *,
    p_key: str,
    errors: Sequence[float] | None = None,
) -> int:
    """Draw a line and stars above bars passing adjusted-p thresholds."""
    if len(bars) != len(rows):
        raise ValueError("bar and result-row counts differ")
    if errors is not None and len(errors) != len(rows):
        raise ValueError("error and result-row counts differ")
    heights = np.asarray([float(bar.get_height()) for bar in bars], dtype=float)
    finite = heights[np.isfinite(heights)]
    span = max(float(np.ptp(finite)) if len(finite) else 0.0, 0.05)
    pad = 0.045 * span
    marks = 0
    annotation_tops: list[float] = []
    for index, (bar, row) in enumerate(zip(bars, rows)):
        stars = significance_stars(float(row[p_key]))
        if not stars:
            continue
        error = 0.0 if errors is None else float(errors[index])
        y = max(float(bar.get_height()) + error, 0.0) + pad
        center = float(bar.get_x() + bar.get_width() / 2)
        half_width = float(bar.get_width()) * 0.28
        ax.plot(
            [center - half_width, center + half_width],
            [y, y],
            color="black",
            linewidth=0.8,
            clip_on=False,
        )
        ax.text(
            center,
            y + 0.25 * pad,
            stars,
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False,
        )
        annotation_tops.append(y + 1.5 * pad)
        marks += 1
    if annotation_tops:
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, max(upper, max(annotation_tops)))
    return marks


def annotate_time_significance(
    ax: matplotlib.axes.Axes,
    times: Sequence[float],
    significant: Sequence[bool],
    *,
    color: str,
    level: int,
    label: str | None = None,
) -> int:
    """Draw cluster lines and asterisks above a time-resolved axis."""
    time_values = np.asarray(times, dtype=float)
    mask = np.asarray(significant, dtype=bool)
    if time_values.ndim != 1 or len(time_values) != len(mask):
        raise ValueError("times and significance mask must be matching vectors")
    runs = contiguous_true_runs(mask)
    y = 1.025 + 0.055 * level
    for run_index, (start, stop) in enumerate(runs):
        kwargs = {"label": label} if run_index == 0 and label else {}
        ax.plot(
            [time_values[start], time_values[stop - 1]],
            [y, y],
            color=color,
            linewidth=2.2,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            **kwargs,
        )
        midpoint = (time_values[start] + time_values[stop - 1]) / 2
        ax.text(
            midpoint,
            y + 0.006,
            "*",
            color=color,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            clip_on=False,
        )
    return len(runs)
