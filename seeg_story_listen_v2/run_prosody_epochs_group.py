#!/usr/bin/env python3
"""Subject-level group inference for syllable-offset prosody epochs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import preprocess_seeg as prep
import prosody_epoch_stats as epoch_stats
import run_strf


DEFAULT_INDIVIDUAL = run_strf.DEFAULT_PREPROCESSED / "prosody_epochs"
DEFAULT_OUTPUT = run_strf.DEFAULT_PREPROCESSED / "prosody_epochs_group"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine individual syllable-offset epoch data using subjects as "
            "independent units and electrodes nested within subjects."
        )
    )
    parser.add_argument(
        "--individual-epoch-dir", type=Path, default=DEFAULT_INDIVIDUAL
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis-start", type=float, default=-0.3)
    parser.add_argument("--analysis-end", type=float, default=0.3)
    parser.add_argument("--fdr-q", type=float, default=0.01)
    parser.add_argument("--minimum-cluster-samples", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def subject_id(recording_id: str) -> str:
    number = prep.source_subject_number(Path(recording_id))
    return f"sub{number:03d}" if number is not None else recording_id.split("/")[0]


def statistic_function(kind: str):
    return (
        epoch_stats.boundary_f_stat
        if kind == "boundary_strength"
        else epoch_stats.depth_f_stat
    )


def subject_statistic(
    datasets: Sequence[epoch_stats.EpochDataset],
    *,
    source_name: str,
    predictor_name: str,
    analysis_mask: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average electrode statistics within one subject."""
    recording_statistics: list[np.ndarray] = []
    recording_effects: list[np.ndarray] = []
    function = statistic_function(predictor_name)
    for dataset in datasets:
        if source_name == "observed":
            source = dataset.observed
        elif source_name == "m2_residual":
            source = dataset.residual_m2
        elif source_name.startswith("prediction_"):
            model_name = source_name.removeprefix("prediction_")
            source = dataset.predictions[
                :, :, dataset.model_names.index(model_name), :
            ]
        else:
            raise ValueError(f"unknown epoch source: {source_name}")
        values = (
            dataset.boundary_strength
            if predictor_name == "boundary_strength"
            else dataset.struc_depth
        )
        if rng is not None:
            values = epoch_stats.permute_within_stimulus(
                values, dataset.stimulus_ids, rng
            )
        electrode_statistics: list[np.ndarray] = []
        electrode_effects: list[np.ndarray] = []
        for channel in range(source.shape[-1]):
            statistic, effect = function(
                source[:, analysis_mask, channel], values
            )
            electrode_statistics.append(statistic)
            electrode_effects.append(effect)
        recording_statistics.append(np.mean(electrode_statistics, axis=0))
        recording_effects.append(np.mean(electrode_effects, axis=0))
    return (
        np.mean(recording_statistics, axis=0),
        np.mean(recording_effects, axis=0),
    )


def group_test(
    by_subject: dict[str, list[epoch_stats.EpochDataset]],
    *,
    source_name: str,
    predictor_name: str,
    analysis_mask: np.ndarray,
    times: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subject_statistics: list[np.ndarray] = []
    subject_effects: list[np.ndarray] = []
    for datasets in by_subject.values():
        statistic, effect = subject_statistic(
            datasets,
            source_name=source_name,
            predictor_name=predictor_name,
            analysis_mask=analysis_mask,
        )
        subject_statistics.append(statistic)
        subject_effects.append(effect)
    observed = np.mean(subject_statistics, axis=0)
    effect = np.mean(subject_effects, axis=0)
    exceedances = np.ones(len(times), dtype=int)
    for permutation in range(args.n_permutations):
        subject_null: list[np.ndarray] = []
        for subject, datasets in by_subject.items():
            rng = np.random.default_rng(
                epoch_stats.stable_seed(
                    args.seed,
                    source_name,
                    predictor_name,
                    subject,
                    permutation,
                )
            )
            statistic, _ = subject_statistic(
                datasets,
                source_name=source_name,
                predictor_name=predictor_name,
                analysis_mask=analysis_mask,
                rng=rng,
            )
            subject_null.append(statistic)
        null = np.mean(subject_null, axis=0)
        exceedances += null >= observed
    pvalues = exceedances / (args.n_permutations + 1.0)
    retained, adjusted, clusters = epoch_stats.fdr_clusters(
        times,
        observed,
        pvalues,
        q=args.fdr_q,
        minimum_samples=args.minimum_cluster_samples,
    )
    rows = [
        {
            "source": source_name,
            "predictor": predictor_name,
            "time_s": float(time),
            "mean_subject_f_statistic": float(statistic),
            "mean_subject_effect_size": float(effect_size),
            "permutation_p_value": float(pvalue),
            "fdr_p_value": float(adjusted_p),
            "passes_fdr_and_cluster": bool(significant),
            "n_subjects": len(by_subject),
            "baseline_correction": "none",
        }
        for time, statistic, effect_size, pvalue, adjusted_p, significant in zip(
            times, observed, effect, pvalues, adjusted, retained
        )
    ]
    cluster_rows = [
        {
            "source": source_name,
            "predictor": predictor_name,
            "n_subjects": len(by_subject),
            "baseline_correction": "none",
            **cluster,
        }
        for cluster in clusters
    ]
    return rows, cluster_rows


def plot_group_statistics(
    output_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sources = sorted({str(row["source"]) for row in rows})
    for source in sources:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for predictor in ("boundary_strength", "struc_depth"):
            selected = [
                row
                for row in rows
                if row["source"] == source and row["predictor"] == predictor
            ]
            times = np.asarray([float(row["time_s"]) for row in selected])
            statistic = np.asarray(
                [float(row["mean_subject_f_statistic"]) for row in selected]
            )
            significant = np.asarray(
                [bool(row["passes_fdr_and_cluster"]) for row in selected]
            )
            line = ax.plot(times, statistic, label=predictor)[0]
            if np.any(significant):
                ax.fill_between(
                    times,
                    0,
                    statistic,
                    where=significant,
                    color=line.get_color(),
                    alpha=0.22,
                )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            xlabel="Time from syllable offset (s)",
            ylabel="Mean subject F statistic",
            title=f"Group prosody separability: {source} (no baseline correction)",
        )
        ax.legend()
        fig.savefig(figures / f"group_{source}_separability.png", dpi=160)
        plt.close(fig)


def plot_subject_waveforms(
    output_dir: Path,
    by_subject: dict[str, list[epoch_stats.EpochDataset]],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    reference = next(iter(by_subject.values()))[0]
    times = reference.times
    for predictor_name in ("boundary_strength", "struc_depth"):
        subject_by_level: dict[float, list[np.ndarray]] = defaultdict(list)
        all_levels: set[float] = set()
        for datasets in by_subject.values():
            recording_level_waves: dict[float, list[np.ndarray]] = defaultdict(list)
            for dataset in datasets:
                labels = (
                    epoch_stats.quantile_bins(dataset.boundary_strength)
                    if predictor_name == "boundary_strength"
                    else dataset.struc_depth
                )
                for level in np.unique(labels):
                    wave = dataset.observed[labels == level].mean(axis=(0, 2))
                    recording_level_waves[float(level)].append(wave)
                    all_levels.add(float(level))
            for level, waves in recording_level_waves.items():
                subject_by_level[level].append(np.mean(waves, axis=0))
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for level in sorted(all_levels):
            waves = np.asarray(subject_by_level[level])
            if not len(waves):
                continue
            mean = waves.mean(axis=0)
            sem = waves.std(axis=0, ddof=1) / math.sqrt(len(waves))
            ax.plot(times, mean, label=f"{predictor_name} {level:g}")
            ax.fill_between(times, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.18)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            xlabel="Time from syllable offset (s)",
            ylabel="Subject-mean high-gamma response (z)",
            title=f"Group ER-Hγ by {predictor_name} (no baseline correction)",
        )
        ax.legend()
        fig.savefig(figures / f"group_{predictor_name}_epochs.png", dpi=160)
        plt.close(fig)


def run(args: argparse.Namespace) -> int:
    args.individual_epoch_dir = args.individual_epoch_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    paths = sorted(
        (args.individual_epoch_dir / "recordings").glob("*/epoch_data.npz")
    )
    if not paths:
        raise FileNotFoundError("no individual epoch_data.npz files were found")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty (use --overwrite): {args.output_dir}"
        )
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_subject: dict[str, list[epoch_stats.EpochDataset]] = defaultdict(list)
    for path in paths:
        dataset = epoch_stats.load_epoch_dataset(path)
        by_subject[subject_id(dataset.recording_id)].append(dataset)
    if len(by_subject) < 2:
        raise ValueError("group inference requires at least two subjects")
    reference = next(iter(by_subject.values()))[0]
    for datasets in by_subject.values():
        for dataset in datasets:
            if not np.array_equal(dataset.times, reference.times):
                raise ValueError(
                    f"epoch time grid differs for {dataset.recording_id}"
                )
    analysis_mask = (reference.times >= args.analysis_start) & (
        reference.times <= args.analysis_end
    )
    times = reference.times[analysis_mask]
    rows: list[dict[str, object]] = []
    clusters: list[dict[str, object]] = []
    sources = [
        "observed",
        "m2_residual",
        *(f"prediction_{model}" for model in reference.model_names),
    ]
    for source in sources:
        for predictor in ("boundary_strength", "struc_depth"):
            result_rows, result_clusters = group_test(
                by_subject,
                source_name=source,
                predictor_name=predictor,
                analysis_mask=analysis_mask,
                times=times,
                args=args,
            )
            rows.extend(result_rows)
            clusters.extend(result_clusters)
    epoch_stats.write_csv(args.output_dir / "group_timepoint_statistics.csv", rows)
    epoch_stats.write_csv(
        args.output_dir / "group_significant_clusters.csv", clusters
    )
    membership_rows = [
        {
            "subject": subject,
            "recording_id": dataset.recording_id,
            "n_electrodes": len(dataset.channel_names),
            "n_epochs": len(dataset.observed),
        }
        for subject, datasets in by_subject.items()
        for dataset in datasets
    ]
    epoch_stats.write_csv(args.output_dir / "group_membership.csv", membership_rows)
    plot_group_statistics(args.output_dir, rows)
    plot_subject_waveforms(args.output_dir, by_subject)
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration.update(
        {
            "baseline_correction": "none",
            "independent_unit": "subject",
            "electrode_handling": "average statistics within subject",
            "event_alignment": "syllable offset / prosody TSV end",
        }
    )
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    print(f"OK group epoch inference: {len(by_subject)} subjects")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
