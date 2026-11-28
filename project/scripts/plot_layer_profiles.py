#!/usr/bin/env python3
"""Plot recording-averaged revised conditional effects across model layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from speech_strf.revised_plots import (
    FAMILY_COLORS,
    FAMILY_LABELS,
    collect_layer_profile_series,
    sha256_file,
    write_json,
)
from speech_strf.revised_trf import PRIMARY_REDUCTIONS, RevisedTRFRunner


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _individual_figure(series, model: str, stem: Path) -> None:
    selected = series.loc[series["model"] == model]
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, constrained_layout=True
    )
    panels = [("acoustic",), tuple(PRIMARY_REDUCTIONS[1:])]
    for axis, families in zip(axes, panels):
        for family in families:
            rows = selected.loc[
                selected["feature_family"] == family
            ].sort_values("native_layer_index")
            x = rows["native_layer_index"].to_numpy()
            mean = rows["equal_recording_mean_delta_r2"].to_numpy()
            axis.plot(
                x,
                mean,
                marker="o",
                linewidth=1.8,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
            axis.fill_between(
                x,
                rows["bootstrap_ci_low"].to_numpy(),
                rows["bootstrap_ci_high"].to_numpy(),
                color=FAMILY_COLORS[family],
                alpha=0.18,
                linewidth=0,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel(r"Conditional $\Delta R^2$")
        axis.legend(frameon=False, ncol=2)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Native layer index (input representation = 0)")
    fig.suptitle(f"{model}: held-out recording-level conditional effects")
    _save(fig, stem)


def _overview_figure(series, models: list[str], stem: Path) -> None:
    fig, axes = plt.subplots(
        len(PRIMARY_REDUCTIONS),
        1,
        figsize=(10, 13),
        sharex=True,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("tab10")
    for axis, family in zip(axes, PRIMARY_REDUCTIONS):
        for model_index, model in enumerate(models):
            rows = series.loc[
                (series["model"] == model)
                & (series["feature_family"] == family)
            ].sort_values("relative_depth")
            x = rows["relative_depth"].to_numpy()
            color = cmap(model_index % 10)
            axis.plot(
                x,
                rows["equal_recording_mean_delta_r2"].to_numpy(),
                marker="o",
                markersize=2.8,
                linewidth=1.2,
                color=color,
                label=model,
            )
            axis.fill_between(
                x,
                rows["bootstrap_ci_low"].to_numpy(),
                rows["bootstrap_ci_high"].to_numpy(),
                color=color,
                alpha=0.07,
                linewidth=0,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_ylabel(
            f"{FAMILY_LABELS[family]}\n" + r"$\Delta R^2$"
        )
        axis.grid(alpha=0.2)
    axes[0].legend(
        frameon=False, ncol=3, fontsize=8, loc="upper center",
        bbox_to_anchor=(0.5, 1.55)
    )
    axes[-1].set_xlabel("Relative depth (input = 0, final encoder layer = 1)")
    fig.suptitle(f"Revised conditional-effect profiles ({len(models)} checkpoints)", y=1.01)
    _save(fig, stem)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/revised_trf.yaml")
    parser.add_argument(
        "--output",
        default="outputs/stage4_revision/revised_trf/figures/layer_profiles",
    )
    parser.add_argument("--individual-model", action="append")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Checkpoint to include; repeatable. Defaults to all nine.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 10000:
        raise SystemExit("--bootstrap-samples must be at least 10000")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be strictly between 0 and 1")

    runner = RevisedTRFRunner(args.config)
    models = list(dict.fromkeys(args.models or runner.models))
    unknown = sorted(set(models) - set(runner.models))
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")
    individual_models = list(dict.fromkeys(args.individual_model or [
        "hubert_base" if "hubert_base" in models else models[0]
    ]))
    unknown_individual = sorted(set(individual_models) - set(models))
    if unknown_individual:
        parser.error(f"Individual models must be included in --model: {unknown_individual}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    series, sources = collect_layer_profile_series(
        runner,
        models=models,
        n_bootstrap=args.bootstrap_samples,
        confidence=args.confidence,
        seed=int(runner.config["random_seed"]),
    )
    metrics = ["equal_recording_mean_delta_r2", "bootstrap_ci_low", "bootstrap_ci_high"]
    if series.empty or not np.isfinite(series[metrics].to_numpy()).all():
        raise SystemExit("Layer profiles contain missing or nonfinite estimates")
    if series.duplicated(["model", "native_layer_index", "feature_family"]).any():
        raise SystemExit("Duplicate model/layer/family estimates in layer profiles")
    for model in models:
        layers = series.loc[series["model"].eq(model)].groupby("native_layer_index")
        layer_indices = sorted(layers.groups)
        if (
            not layer_indices
            or layer_indices != list(range(len(layer_indices)))
            or any(set(rows["feature_family"]) != set(PRIMARY_REDUCTIONS) for _, rows in layers)
        ):
            raise SystemExit(f"Incomplete layer profile for {model}")

    series_path = output / "layer_profile_series.csv"
    series.to_csv(series_path, index=False)
    metadata_paths = []
    common_metadata = {
        "reported_score": (
            "equal-recording arithmetic mean held-out conditional delta R2 "
            "(full R2 minus matched reduced R2)"
        ),
        "uncertainty": {
            "method": "recording bootstrap percentile interval",
            "samples": args.bootstrap_samples,
            "confidence": args.confidence,
            "seed": int(runner.config["random_seed"]),
        },
        "negative_values": "shown without clipping; horizontal zero line included",
        "source_sha256": sources,
    }
    for model in individual_models:
        stem = output / f"layer_profiles_{model}"
        _individual_figure(series, model, stem)
        individual_series = series.loc[series["model"] == model]
        individual_series_path = stem.with_name(f"{stem.name}.series.csv")
        individual_series.to_csv(individual_series_path, index=False)
        metadata_path = stem.with_name(f"{stem.name}.metadata.json")
        write_json(
            metadata_path,
            {
                **common_metadata,
                "figure_selection": {
                    "fit_variant": "primary",
                    "outer_mode": "grouped",
                    "models": [model],
                    "families": list(PRIMARY_REDUCTIONS),
                    "layer_rule": (
                        "all saved native layers in HDF5 order, including layer 0"
                    ),
                    "x_axis": "native_layer_index",
                },
                "series_file": str(individual_series_path),
                "series_sha256": sha256_file(individual_series_path),
            },
        )
        metadata_paths.append(str(metadata_path))
    overview_stem = output / (
        "layer_profiles_nine_checkpoints"
        if set(models) == set(runner.models) and len(models) == 9
        else "layer_profiles_selected_checkpoints"
    )
    _overview_figure(
        series, models, overview_stem
    )
    overview_series_path = overview_stem.with_name(
        f"{overview_stem.name}.series.csv"
    )
    series.to_csv(overview_series_path, index=False)
    overview_metadata = {
        **common_metadata,
        "figure_selection": {
            "fit_variant": "primary",
            "outer_mode": "grouped",
            "models": models,
            "families": list(PRIMARY_REDUCTIONS),
            "layer_rule": "all saved layers in HDF5 order, including layer 0",
            "x_axis": "native_layer_index / (n_layers - 1)",
        },
        "series_file": str(overview_series_path),
        "series_sha256": sha256_file(overview_series_path),
    }
    overview_metadata_path = overview_stem.with_name(
        f"{overview_stem.name}.metadata.json"
    )
    write_json(overview_metadata_path, overview_metadata)
    metadata_paths.append(str(overview_metadata_path))
    result = {
        "state": "complete",
        "output": str(output),
        "series_rows": len(series),
        "metadata": metadata_paths,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
