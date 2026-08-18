#!/usr/bin/env python3
"""Individual-electrode syllable-offset prosody separability analysis."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import prosody_epoch_stats as epoch_stats
import run_strf


DEFAULT_STRF = run_strf.DEFAULT_PREPROCESSED / "strf"
DEFAULT_OUTPUT = run_strf.DEFAULT_PREPROCESSED / "prosody_epochs"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract syllable-offset epochs from held-out STRF predictions and "
            "test boundary strength and structure-depth separability without "
            "baseline correction."
        )
    )
    parser.add_argument("--strf-dir", type=Path, default=DEFAULT_STRF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sfreq", type=float, default=128.0)
    parser.add_argument("--epoch-start", type=float, default=-0.5)
    parser.add_argument("--epoch-end", type=float, default=0.5)
    parser.add_argument("--analysis-start", type=float, default=-0.3)
    parser.add_argument("--analysis-end", type=float, default=0.3)
    parser.add_argument("--fdr-q", type=float, default=0.01)
    parser.add_argument("--minimum-cluster-samples", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.strf_dir.is_dir():
        raise FileNotFoundError(f"STRF directory does not exist: {args.strf_dir}")
    if not args.epoch_start < args.analysis_start < args.analysis_end < args.epoch_end:
        raise ValueError(
            "epoch and analysis windows must satisfy epoch_start < analysis_start "
            "< analysis_end < epoch_end"
        )
    if args.sfreq <= 0 or args.n_permutations < 1:
        raise ValueError("sampling rate and permutation count must be positive")
    if not 0 < args.fdr_q < 1 or args.minimum_cluster_samples < 1:
        raise ValueError("FDR q and minimum cluster size are invalid")


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x**2) * np.sum(y**2))
    return float(np.sum(x * y) / max(denominator, np.finfo(float).eps))


def epoch_accuracy_rows(
    dataset: epoch_stats.EpochDataset,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stratifications = {
        "structure_depth": dataset.struc_depth,
        "boundary_quartile": epoch_stats.quantile_bins(dataset.boundary_strength),
    }
    for channel, channel_name in enumerate(dataset.channel_names):
        for stratification, labels in stratifications.items():
            levels = np.unique(labels)
            observed_average = np.concatenate(
                [
                    dataset.observed[labels == level, :, channel].mean(axis=0)
                    for level in levels
                ]
            )
            for model, model_name in enumerate(dataset.model_names):
                predicted_average = np.concatenate(
                    [
                        dataset.predictions[
                            labels == level, :, model, channel
                        ].mean(axis=0)
                        for level in levels
                    ]
                )
                rows.append(
                    {
                        "recording_id": dataset.recording_id,
                        "channel": channel_name,
                        "stratification": stratification,
                        "model": model_name,
                        "pearson_r_observed_vs_predicted_er_hg": correlation(
                            observed_average, predicted_average
                        ),
                        "n_epochs": len(labels),
                        "baseline_correction": "none",
                    }
                )
    return rows


def plot_epoch_means(
    output_dir: Path, dataset: epoch_stats.EpochDataset
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    boundary_bins = epoch_stats.quantile_bins(dataset.boundary_strength)
    for channel, channel_name in enumerate(dataset.channel_names):
        safe_channel = run_strf.safe_name(channel_name)
        for labels, label_name, filename in (
            (dataset.struc_depth, "Structure depth", "structure_depth"),
            (boundary_bins, "Boundary-strength quartile", "boundary_strength"),
        ):
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            for level in np.unique(labels):
                values = dataset.observed[labels == level, :, channel]
                mean = values.mean(axis=0)
                sem = values.std(axis=0, ddof=1) / math.sqrt(len(values))
                ax.plot(dataset.times, mean, label=f"{label_name} {level:g}")
                ax.fill_between(
                    dataset.times, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.18
                )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
            ax.set(
                xlabel="Time from syllable offset (s)",
                ylabel="High-gamma response (z)",
                title=f"{channel_name}: observed ER-Hγ (no baseline correction)",
            )
            ax.legend()
            fig.savefig(figures / f"{safe_channel}_{filename}_epochs.png", dpi=160)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        observed_mean = dataset.observed[:, :, channel].mean(axis=0)
        observed_sem = dataset.observed[:, :, channel].std(
            axis=0, ddof=1
        ) / math.sqrt(len(dataset.observed))
        ax.plot(dataset.times, observed_mean, color="black", linewidth=2, label="Observed")
        ax.fill_between(
            dataset.times,
            observed_mean - 1.96 * observed_sem,
            observed_mean + 1.96 * observed_sem,
            color="black",
            alpha=0.12,
        )
        for model, model_name in enumerate(dataset.model_names):
            prediction = dataset.predictions[:, :, model, channel]
            mean = prediction.mean(axis=0)
            sem = prediction.std(axis=0, ddof=1) / math.sqrt(len(prediction))
            ax.plot(dataset.times, mean, label=model_name)
            ax.fill_between(
                dataset.times, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.10
            )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            xlabel="Time from syllable offset (s)",
            ylabel="High-gamma response (z)",
            title=f"{channel_name}: observed and held-out predicted ER-Hγ",
        )
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(figures / f"{safe_channel}_observed_predicted_epochs.png", dpi=160)
        plt.close(fig)


def plot_statistics(
    output_dir: Path,
    rows: list[dict[str, object]],
    channel_names: Sequence[str],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for channel_name in channel_names:
        for source in ("observed", "m2_residual"):
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            for predictor in ("boundary_strength", "struc_depth"):
                selected = [
                    row
                    for row in rows
                    if row["channel"] == channel_name
                    and row["source"] == source
                    and row["predictor"] == predictor
                ]
                times = np.asarray([float(row["time_s"]) for row in selected])
                statistic = np.asarray(
                    [float(row["f_statistic"]) for row in selected]
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
                ylabel="F statistic",
                title=f"{channel_name}: {source} prosody separability",
            )
            ax.legend()
            fig.savefig(
                figures
                / (
                    f"{run_strf.safe_name(channel_name)}_{source}"
                    "_separability.png"
                ),
                dpi=160,
            )
            plt.close(fig)


def analyze_recording(
    strf_dir: Path,
    recording_name: str,
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    output_dir = output_root / "recordings" / recording_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = epoch_stats.extract_recording_epochs(
        strf_dir,
        recording_name,
        sfreq=args.sfreq,
        epoch_start=args.epoch_start,
        epoch_end=args.epoch_end,
    )
    epoch_stats.save_epoch_dataset(output_dir / "epoch_data.npz", dataset)
    analysis_mask = (dataset.times >= args.analysis_start) & (
        dataset.times <= args.analysis_end
    )
    analysis_times = dataset.times[analysis_mask]
    timepoint_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    sources = {
        "observed": dataset.observed,
        "m2_residual": dataset.residual_m2,
    }
    sources.update(
        {
            f"prediction_{model_name}": dataset.predictions[:, :, model, :]
            for model, model_name in enumerate(dataset.model_names)
        }
    )
    predictors = {
        "boundary_strength": dataset.boundary_strength,
        "struc_depth": dataset.struc_depth,
    }
    for source_name, source_epochs in sources.items():
        for channel, channel_name in enumerate(dataset.channel_names):
            epochs = source_epochs[:, analysis_mask, channel]
            for predictor_name, predictor in predictors.items():
                rows, clusters = epoch_stats.time_resolved_test(
                    epochs,
                    predictor,
                    dataset.stimulus_ids,
                    analysis_times,
                    kind=predictor_name,
                    n_permutations=args.n_permutations,
                    seed=epoch_stats.stable_seed(
                        args.seed,
                        dataset.recording_id,
                        source_name,
                        channel_name,
                        predictor_name,
                    ),
                    fdr_q=args.fdr_q,
                    minimum_cluster_samples=args.minimum_cluster_samples,
                )
                timepoint_rows.extend(
                    {
                        "recording_id": dataset.recording_id,
                        "channel": channel_name,
                        "source": source_name,
                        "predictor": predictor_name,
                        "n_epochs": len(epochs),
                        "baseline_correction": "none",
                        **row,
                    }
                    for row in rows
                )
                cluster_rows.extend(
                    {
                        "recording_id": dataset.recording_id,
                        "channel": channel_name,
                        "source": source_name,
                        "predictor": predictor_name,
                        "baseline_correction": "none",
                        **cluster,
                    }
                    for cluster in clusters
                )
    epoch_stats.write_csv(output_dir / "timepoint_statistics.csv", timepoint_rows)
    epoch_stats.write_csv(output_dir / "significant_clusters.csv", cluster_rows)
    epoch_stats.write_csv(
        output_dir / "model_epoch_accuracy.csv", epoch_accuracy_rows(dataset)
    )
    plot_epoch_means(output_dir, dataset)
    plot_statistics(output_dir, timepoint_rows, dataset.channel_names)


def run(args: argparse.Namespace) -> int:
    args.strf_dir = args.strf_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty (use --overwrite): {args.output_dir}"
        )
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recording_dirs = sorted((args.strf_dir / "recordings").iterdir())
    recording_dirs = [path for path in recording_dirs if path.is_dir()]
    if args.max_recordings is not None:
        recording_dirs = recording_dirs[: args.max_recordings]
    if not recording_dirs:
        raise FileNotFoundError("no STRF recording outputs were found")
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration["baseline_correction"] = "none"
    configuration["event_alignment"] = "syllable offset / prosody TSV end"
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    failures = 0
    for recording_dir in recording_dirs:
        try:
            analyze_recording(
                args.strf_dir, recording_dir.name, args.output_dir, args
            )
            print(f"OK {recording_dir.name}")
        except Exception as error:
            failures += 1
            print(f"ERROR {recording_dir.name}: {error}", file=sys.stderr)
    print(
        f"Finished epoch separability: {len(recording_dirs) - failures} succeeded, "
        f"{failures} failed"
    )
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
