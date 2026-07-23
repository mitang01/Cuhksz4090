#!/usr/bin/env python3
"""Plot one quality-selected Bombcell good unit per analysis and brain region.

This script produces 12 non-interactive figures:

    syllable / word / sub4-session03 percept switch
        x ATL / HG / VMPFC / Amygdala

Each figure contains a trial raster, a trial-by-time heatmap sorted by
post-onset peak latency, and the selected unit's mean firing rate ± SEM across
trials.  Unit selection is independent of task responses: among Bombcell
``good`` units, it combines high total spike count with a low fraction of
adjacent inter-spike intervals shorter than 2 ms.  This avoids selecting a unit
because it happens to show an attractive task response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata

try:
    import plot_bistable_firing_rasters as population
except ModuleNotFoundError:
    from spike_sorting_script import plot_bistable_firing_rasters as population


REGIONS = population.REGIONS
ANALYSES = population.ANALYSES
REFRACTORY_PERIOD_S = 0.002


@dataclass
class UnitCandidate:
    analysis: str
    region: str
    subject: str
    session_key: str
    sorting_dir: Path
    unit_id: str
    spikes_s: np.ndarray
    events_s: np.ndarray
    n_spikes: int
    isi_violation_count: int
    isi_violation_fraction: float
    median_isi_ms: float
    spike_count_rank: float = math.nan
    isi_rank: float = math.nan
    quality_score: float = math.nan
    pareto_optimal: bool = False
    selected: bool = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot quality-selected representative good units for the bistable experiment."
    )
    parser.add_argument("--sorting-root", type=Path, default=population.DEFAULT_SORTING_ROOT)
    parser.add_argument("--log-root", type=Path, default=population.DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--raw-root",
        action="append",
        type=Path,
        default=None,
        help="Raw-data search root; repeatable. Defaults to known sub4/sub5/sub6 roots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=population.DEFAULT_SORTING_ROOT / "representative_unit_firing_rasters",
    )
    parser.add_argument("--stimulus-column", default=None)
    parser.add_argument("--condition-column", default=None)
    parser.add_argument("--response-column", default="mapped_response")
    parser.add_argument("--trigger-channel", type=int, default=None)
    parser.add_argument("--sample-rate", type=float, default=30000.0)
    parser.add_argument("--strict-trigger-count", action="store_true")
    parser.add_argument("--t-before", type=float, default=0.5)
    parser.add_argument("--t-after", type=float, default=1.35)
    parser.add_argument("--baseline-start", type=float, default=-0.5)
    parser.add_argument("--baseline-end", type=float, default=0.0)
    parser.add_argument("--bin-ms", type=float, default=10.0)
    parser.add_argument("--gaussian-sigma-ms", type=float, default=50.0)
    parser.add_argument("--ttest-point-alpha", type=float, default=0.01)
    parser.add_argument("--cluster-alpha", type=float, default=0.01)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.t_before <= 0 or args.t_after <= 0:
        raise ValueError("--t-before and --t-after must be positive")
    if args.bin_ms <= 0 or args.gaussian_sigma_ms <= 0:
        raise ValueError("--bin-ms and --gaussian-sigma-ms must be positive")
    if args.baseline_start >= args.baseline_end:
        raise ValueError("--baseline-start must precede --baseline-end")
    if args.baseline_start < -args.t_before or args.baseline_end > args.t_after:
        raise ValueError("baseline window must lie inside the analysis window")
    if not 0 < args.ttest_point_alpha < 1:
        raise ValueError("--ttest-point-alpha must lie between 0 and 1")
    if not 0 < args.cluster_alpha < 1:
        raise ValueError("--cluster-alpha must lie between 0 and 1")
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be at least 1")


def event_times_for_analysis(
    groups: population.EventGroups,
    analysis: str,
) -> np.ndarray:
    category_groups = getattr(groups, analysis)
    if not category_groups:
        return np.array([], dtype=np.float64)
    return np.sort(
        np.concatenate(
            [np.asarray(times, dtype=np.float64) for times in category_groups.values()]
        )
    )


def calculate_quality(spikes_s: np.ndarray) -> tuple[int, int, float, float]:
    spikes = np.sort(np.asarray(spikes_s, dtype=np.float64).ravel())
    n_spikes = int(spikes.size)
    if n_spikes < 2:
        return n_spikes, 0, float("inf"), float("inf")
    intervals = np.diff(spikes)
    violation_count = int(np.count_nonzero(intervals < REFRACTORY_PERIOD_S))
    violation_fraction = float(violation_count / intervals.size)
    median_isi_ms = float(np.median(intervals) * 1000.0)
    return n_spikes, violation_count, violation_fraction, median_isi_ms


def discover_candidates(
    sessions: Sequence[population.SessionMatch],
    args: argparse.Namespace,
) -> tuple[list[UnitCandidate], list[dict[str, Any]]]:
    candidates: list[UnitCandidate] = []
    session_rows: list[dict[str, Any]] = []
    for session in sessions:
        try:
            groups, trigger_qc = population.build_event_groups(session, args)
        except Exception as exc:  # noqa: BLE001
            session_rows.append(
                {
                    "subject": session.subject,
                    "session_key": session.key,
                    "status": "event_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        event_map = {
            analysis: event_times_for_analysis(groups, analysis)
            for analysis in ANALYSES
        }
        session_rows.append(
            {
                "subject": session.subject,
                "session_key": session.key,
                "status": "ok",
                "syllable_events": int(event_map["syllable"].size),
                "word_events": int(event_map["word"].size),
                "switch_events": int(event_map["switch"].size),
                **trigger_qc,
            }
        )
        for region, bombcell_path in session.region_bombcell_paths.items():
            try:
                units, _ = population.load_bombcell_units(
                    bombcell_path, accepted_labels={"good"}
                )
            except Exception as exc:  # noqa: BLE001
                session_rows.append(
                    {
                        "subject": session.subject,
                        "session_key": session.key,
                        "region": region,
                        "status": "unit_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            for analysis, event_times in event_map.items():
                if event_times.size == 0:
                    continue
                for unit_id, spikes in units.items():
                    quality = calculate_quality(spikes)
                    candidates.append(
                        UnitCandidate(
                            analysis=analysis,
                            region=region,
                            subject=session.subject,
                            session_key=session.key,
                            sorting_dir=session.sorting_dir,
                            unit_id=unit_id,
                            spikes_s=np.asarray(spikes, dtype=np.float64),
                            events_s=event_times,
                            n_spikes=quality[0],
                            isi_violation_count=quality[1],
                            isi_violation_fraction=quality[2],
                            median_isi_ms=quality[3],
                        )
                    )
    return candidates, session_rows


def is_pareto_optimal(candidate: UnitCandidate, group: Sequence[UnitCandidate]) -> bool:
    for other in group:
        at_least_as_many_spikes = other.n_spikes >= candidate.n_spikes
        no_more_violations = (
            other.isi_violation_fraction <= candidate.isi_violation_fraction
        )
        strictly_better = (
            other.n_spikes > candidate.n_spikes
            or other.isi_violation_fraction < candidate.isi_violation_fraction
        )
        if at_least_as_many_spikes and no_more_violations and strictly_better:
            return False
    return True


def select_representative_units(
    candidates: list[UnitCandidate],
) -> dict[tuple[str, str], UnitCandidate]:
    selected: dict[tuple[str, str], UnitCandidate] = {}
    for analysis in ANALYSES:
        for region in REGIONS:
            group = [
                candidate
                for candidate in candidates
                if candidate.analysis == analysis and candidate.region == region
            ]
            if not group:
                continue
            spike_ranks = rankdata(
                [-candidate.n_spikes for candidate in group], method="average"
            )
            isi_ranks = rankdata(
                [candidate.isi_violation_fraction for candidate in group],
                method="average",
            )
            denominator = max(len(group) - 1, 1)
            for candidate, spike_rank, isi_rank in zip(
                group, spike_ranks, isi_ranks
            ):
                candidate.spike_count_rank = float(spike_rank)
                candidate.isi_rank = float(isi_rank)
                candidate.quality_score = float(
                    0.5 * (spike_rank - 1) / denominator
                    + 0.5 * (isi_rank - 1) / denominator
                )
                candidate.pareto_optimal = is_pareto_optimal(candidate, group)
            pareto_front = [candidate for candidate in group if candidate.pareto_optimal]
            winner = min(
                pareto_front,
                key=lambda candidate: (
                    candidate.quality_score,
                    candidate.isi_violation_fraction,
                    -candidate.n_spikes,
                    candidate.subject,
                    candidate.session_key,
                    candidate.unit_id,
                ),
            )
            winner.selected = True
            selected[(analysis, region)] = winner
    return selected


def aligned_trial_rates(
    spikes_s: np.ndarray,
    events_s: np.ndarray,
    edges: np.ndarray,
    sigma_bins: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    n_bins = edges.size - 1
    bin_s = float(np.diff(edges)[0])
    pad_bins = max(1, int(math.ceil(4.0 * sigma_bins)))
    extended_edges = edges[0] + np.arange(
        -pad_bins, n_bins + pad_bins + 1, dtype=np.float64
    ) * bin_s
    spikes = np.sort(np.asarray(spikes_s, dtype=np.float64).ravel())
    trial_rates: list[np.ndarray] = []
    raster_rows: list[np.ndarray] = []
    for event in events_s:
        left = int(np.searchsorted(spikes, event + extended_edges[0], side="left"))
        right = int(np.searchsorted(spikes, event + extended_edges[-1], side="right"))
        relative = spikes[left:right] - event
        counts = np.histogram(relative, bins=extended_edges)[0].astype(np.float64)
        smoothed = gaussian_filter1d(
            counts / bin_s, sigma=sigma_bins, mode="constant", truncate=4.0
        )
        trial_rates.append(smoothed[pad_bins : pad_bins + n_bins])
        raster_rows.append(
            relative[(relative >= edges[0]) & (relative <= edges[-1])]
        )
    if not trial_rates:
        return np.empty((0, n_bins), dtype=np.float64), []
    return np.vstack(trial_rates), raster_rows


def normalize_and_sort_trials(
    trial_rates: np.ndarray,
    centers: np.ndarray,
    baseline_start_s: float,
    baseline_end_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_mask = (centers >= baseline_start_s) & (centers < baseline_end_s)
    baseline = np.mean(trial_rates[:, baseline_mask], axis=1, keepdims=True)
    changes = trial_rates - baseline
    scale = np.max(np.abs(changes), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    normalized = changes / scale
    post_indices = np.flatnonzero(centers >= 0)
    peak_indices = post_indices[np.argmax(normalized[:, post_indices], axis=1)]
    order = np.argsort(centers[peak_indices], kind="stable")
    return normalized[order], order, centers[peak_indices][order]


def trial_cluster_test(
    trial_rates: np.ndarray,
    centers: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    n_trials = trial_rates.shape[0]
    if n_trials < 2:
        return {
            "status": "not_estimable",
            "reason": "one-sample trial test requires at least two events",
            "n_trials": n_trials,
            "clusters": [],
        }
    baseline_mask = (centers >= args.baseline_start) & (
        centers < args.baseline_end
    )
    baseline = np.mean(trial_rates[:, baseline_mask], axis=1, keepdims=True)
    changes = trial_rates - baseline
    t_values, p_values = population.one_sample_t(changes)
    post_mask = centers >= args.baseline_end
    cluster_p = np.where(post_mask, p_values, 1.0)
    observed = population.temporal_clusters(
        t_values, cluster_p, args.ttest_point_alpha
    )
    rng = np.random.default_rng(seed)
    null_maxima = np.zeros(args.n_permutations, dtype=np.float64)
    for permutation_index in range(args.n_permutations):
        signs = rng.choice((-1.0, 1.0), size=(n_trials, 1))
        permuted_t, permuted_p = population.one_sample_t(changes * signs)
        clusters = population.temporal_clusters(
            permuted_t,
            np.where(post_mask, permuted_p, 1.0),
            args.ttest_point_alpha,
        )
        null_maxima[permutation_index] = max(
            (cluster[2] for cluster in clusters), default=0.0
        )
    cluster_results: list[dict[str, Any]] = []
    for start, end, statistic in observed:
        permutation_p = (
            1 + np.count_nonzero(null_maxima >= statistic)
        ) / (args.n_permutations + 1)
        cluster_results.append(
            {
                "start_index": start,
                "end_index": end,
                "t_abs_sum": statistic,
                "permutation_p": float(permutation_p),
                "significant": bool(permutation_p <= args.cluster_alpha),
                "direction": (
                    "increase"
                    if np.mean(t_values[start : end + 1]) > 0
                    else "decrease"
                ),
            }
        )
    return {
        "status": "ok",
        "reason": "",
        "n_trials": n_trials,
        "clusters": cluster_results,
    }


def analysis_title(analysis: str) -> str:
    return {
        "syllable": "All stimulus onsets (syllable)",
        "word": "Every second stimulus onset (word)",
        "switch": "Firing rate when word percepts switch (sub4 session03)",
    }[analysis]


def plot_candidate(
    output_path: Path,
    candidate: UnitCandidate | None,
    analysis: str,
    region: str,
    edges: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, (raster_ax, heatmap_ax, rate_ax) = plt.subplots(
        3,
        1,
        figsize=(11, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.7, 1.4]},
        constrained_layout=True,
    )
    if candidate is None:
        for axis in (raster_ax, heatmap_ax, rate_ax):
            axis.text(
                0.5,
                0.5,
                "No eligible Bombcell good unit with matching events",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )
        raster_ax.set_title(f"{analysis_title(analysis)} — {region}")
        rate_ax.set_xlabel("Time from onset (s)")
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        return {
            "status": "not_estimable",
            "reason": "no eligible good unit with matching events",
            "n_trials": 0,
            "clusters": [],
        }

    trial_rates, raster_rows = aligned_trial_rates(
        candidate.spikes_s,
        candidate.events_s,
        edges,
        sigma_bins=args.gaussian_sigma_ms / args.bin_ms,
    )
    heatmap, order, _ = normalize_and_sort_trials(
        trial_rates,
        centers,
        args.baseline_start,
        args.baseline_end,
    )
    sorted_rasters = [raster_rows[index] for index in order]
    for row, spikes in enumerate(sorted_rasters):
        if spikes.size:
            raster_ax.scatter(
                spikes,
                np.full(spikes.size, row),
                s=4,
                color="tab:blue",
                alpha=0.8,
                linewidths=0,
                rasterized=True,
            )
    raster_ax.set_ylim(max(len(sorted_rasters) - 0.5, 0.5), -0.5)
    raster_ax.set_ylabel("Trials\nsorted by peak latency")

    image = heatmap_ax.imshow(
        heatmap,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=[edges[0], edges[-1], heatmap.shape[0] - 0.5, -0.5],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        rasterized=True,
    )
    colorbar = fig.colorbar(image, ax=heatmap_ax, pad=0.01)
    colorbar.set_label("Normalized ΔFR")
    heatmap_ax.set_ylabel("Trials\nsorted by peak latency")
    heatmap_ax.set_yticks([])

    mean_rate = np.mean(trial_rates, axis=0)
    sem_rate = (
        np.std(trial_rates, axis=0, ddof=1) / math.sqrt(trial_rates.shape[0])
        if trial_rates.shape[0] > 1
        else np.zeros_like(mean_rate)
    )
    rate_ax.plot(centers, mean_rate, color="tab:blue", linewidth=2)
    rate_ax.fill_between(
        centers,
        mean_rate - sem_rate,
        mean_rate + sem_rate,
        color="tab:blue",
        alpha=0.2,
        linewidth=0,
    )
    rate_ax.set_ylabel("Firing rate (Hz)\nmean ± SEM across trials")
    rate_ax.set_xlabel("Time from onset (s)")

    seed_text = (
        f"{analysis}|{region}|{candidate.subject}|{candidate.session_key}|"
        f"{candidate.unit_id}"
    )
    seed = int.from_bytes(
        hashlib.blake2b(seed_text.encode(), digest_size=4).digest(), "big"
    )
    statistics = trial_cluster_test(trial_rates, centers, args, seed)
    direction_colors = {"increase": "#b2182b", "decrease": "#2166ac"}
    labeled: set[str] = set()
    for cluster in statistics["clusters"]:
        if not cluster["significant"]:
            continue
        start = edges[cluster["start_index"]]
        end = edges[cluster["end_index"] + 1]
        direction = cluster["direction"]
        for axis in (raster_ax, heatmap_ax, rate_ax):
            axis.axvspan(start, end, color=direction_colors[direction], alpha=0.07)
        rate_ax.plot(
            [start, end],
            [0.98, 0.98],
            transform=rate_ax.get_xaxis_transform(),
            color=direction_colors[direction],
            linewidth=4,
            label=(
                f"significant {direction} vs baseline"
                if direction not in labeled
                else None
            ),
        )
        labeled.add(direction)

    for axis in (raster_ax, heatmap_ax, rate_ax):
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.axvspan(0, 0.57556, color="0.5", alpha=0.07)
    raster_ax.set_title(
        f"{analysis_title(analysis)} — {region}\n"
        f"{candidate.subject} {candidate.session_key}, unit {candidate.unit_id}; "
        f"quality-only selection"
    )
    rate_ax.text(
        0.01,
        0.96,
        (
            f"spikes={candidate.n_spikes}; ISI<2 ms="
            f"{candidate.isi_violation_fraction:.4%} "
            f"({candidate.isi_violation_count}); events={candidate.events_s.size}"
        ),
        transform=rate_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.3",
    )
    if labeled:
        rate_ax.legend(frameon=False, fontsize=8, loc="upper right")
    rate_ax.set_xlim(edges[0], edges[-1])
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return statistics


def candidate_row(candidate: UnitCandidate) -> dict[str, Any]:
    return {
        "analysis": candidate.analysis,
        "region": candidate.region,
        "selected": candidate.selected,
        "pareto_optimal": candidate.pareto_optimal,
        "quality_score_lower_is_better": f"{candidate.quality_score:.8g}",
        "spike_count_rank": f"{candidate.spike_count_rank:.8g}",
        "isi_rank": f"{candidate.isi_rank:.8g}",
        "n_spikes": candidate.n_spikes,
        "isi_violation_count_lt_2ms": candidate.isi_violation_count,
        "isi_violation_fraction_lt_2ms": f"{candidate.isi_violation_fraction:.8g}",
        "median_isi_ms": f"{candidate.median_isi_ms:.8g}",
        "subject": candidate.subject,
        "session_key": candidate.session_key,
        "unit_id": candidate.unit_id,
        "sorting_dir": str(candidate.sorting_dir),
        "n_events": int(candidate.events_s.size),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_roots = args.raw_root if args.raw_root is not None else population.DEFAULT_RAW_ROOTS
        sessions, discovery_rows = population.discover_sessions(
            args.sorting_root.expanduser().resolve(),
            args.log_root.expanduser().resolve(),
            [path.expanduser().resolve() for path in raw_roots],
        )
        write_csv(output_dir / "session_discovery.csv", discovery_rows)
        if not sessions:
            raise RuntimeError(
                "No complete sorting/CSV/trigger matches; inspect session_discovery.csv"
            )
        candidates, session_rows = discover_candidates(sessions, args)
        write_csv(output_dir / "representative_session_qc.csv", session_rows)
        selected = select_representative_units(candidates)
        candidate_rows = [candidate_row(candidate) for candidate in candidates]
        write_csv(output_dir / "representative_unit_candidates.csv", candidate_rows)
        write_csv(
            output_dir / "selected_representative_units.csv",
            [candidate_row(candidate) for candidate in selected.values()],
        )

        edges = population.make_edges(args)
        figure_paths: list[Path] = []
        statistic_rows: list[dict[str, Any]] = []
        for analysis in ANALYSES:
            for region in REGIONS:
                candidate = selected.get((analysis, region))
                output_path = output_dir / f"{analysis}_{region}_representative_unit.png"
                statistics = plot_candidate(
                    output_path, candidate, analysis, region, edges, args
                )
                figure_paths.append(output_path)
                clusters = statistics.get("clusters", [])
                statistic_rows.append(
                    {
                        "analysis": analysis,
                        "region": region,
                        "status": statistics.get("status", "not_estimable"),
                        "reason": statistics.get("reason", ""),
                        "n_trials": statistics.get("n_trials", 0),
                        "candidate_cluster_count": len(clusters),
                        "significant_cluster_count": sum(
                            bool(cluster["significant"]) for cluster in clusters
                        ),
                    }
                )
                for cluster_id, cluster in enumerate(clusters, start=1):
                    statistic_rows.append(
                        {
                            "analysis": analysis,
                            "region": region,
                            "status": "cluster",
                            "reason": "",
                            "n_trials": statistics["n_trials"],
                            "candidate_cluster_count": len(clusters),
                            "significant_cluster_count": sum(
                                bool(item["significant"]) for item in clusters
                            ),
                            "cluster_id": cluster_id,
                            "direction": cluster["direction"],
                            "start_s": f"{edges[cluster['start_index']]:.6f}",
                            "end_s": f"{edges[cluster['end_index'] + 1]:.6f}",
                            "permutation_p": f"{cluster['permutation_p']:.8g}",
                            "significant": cluster["significant"],
                        }
                    )
        write_csv(output_dir / "representative_unit_statistics.csv", statistic_rows)
        run_info = {
            "figure_count": len(figure_paths),
            "figures": [str(path) for path in figure_paths],
            "selection_population": "Bombcell label good only",
            "selection_method": (
                "Pareto front on maximum total spike count and minimum adjacent "
                "ISI<2 ms fraction; winner minimizes equal-weight normalized rank sum. "
                "Task response magnitude is not used for selection."
            ),
            "selection_caveat": (
                "Total spike count can favor longer recording sessions; session and "
                "all candidate metrics are retained in representative_unit_candidates.csv."
            ),
            "smoothing_sigma_ms": args.gaussian_sigma_ms,
            "heatmap": (
                "Trial-wise baseline-subtracted firing rate, normalized by each trial's "
                "maximum absolute change and sorted by post-onset peak latency."
            ),
        }
        (output_dir / "representative_unit_run_info.json").write_text(
            json.dumps(run_info, indent=2), encoding="utf-8"
        )
        print(f"[DONE] Wrote {len(figure_paths)} representative-unit figures to {output_dir}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
