#!/usr/bin/env python3
"""Plot benchmark five-lag ridge coefficients for one named predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from speech_strf.revised_plots import (
    load_kernel_series,
    primary_unit,
    sha256_file,
    write_json,
)
from speech_strf.revised_trf import RevisedTRFRunner


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/revised_trf.yaml")
    parser.add_argument("--model", default="hubert_base")
    parser.add_argument("--layer", required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--predictor", default="broadband_envelope")
    parser.add_argument(
        "--target-component",
        type=int,
        default=1,
        help="One-based outer-training target PC number (default: PC1)",
    )
    parser.add_argument(
        "--output",
        default="outputs/stage4_revision/revised_trf/figures/trf_kernels",
    )
    args = parser.parse_args(argv)
    if args.target_component < 1:
        raise SystemExit("--target-component is one-based and must be positive")

    runner = RevisedTRFRunner(args.config)
    unit = primary_unit(runner, args.model, args.layer)
    series, kernel_metadata, sources = load_kernel_series(
        unit,
        args.recording_id,
        args.predictor,
        args.target_component - 1,
    )
    expected_lags = [-200, -100, 0, 100, 200]
    if series["lag_ms"].tolist() != expected_lags:
        raise SystemExit(
            f"Benchmark kernel requires exact lags {expected_lags}; found "
            f"{series['lag_ms'].tolist()}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = (
        output
        / f"{args.model}_{args.layer}_{args.recording_id}_"
        f"{args.predictor}_pc{args.target_component}"
    )
    series_path = stem.with_name(f"{stem.name}.series.csv")
    series.to_csv(series_path, index=False)

    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(
        series["lag_ms"],
        series["coefficient_standardized_predictor"],
        marker="o",
        markersize=6,
        linewidth=1.6,
        color="#1f77b4",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="0.6", linewidth=0.6)
    axis.set_xticks(expected_lags)
    axis.set_xlabel("Predictor lag (ms)")
    axis.set_ylabel(
        f"PC{args.target_component} score units per predictor SD"
    )
    axis.set_title(
        f"{args.model}, {args.layer}: {args.predictor} TRF kernel"
    )
    axis.grid(alpha=0.2)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure_selection": {
            "model": args.model,
            "layer": args.layer,
            "held_out_recording_defining_outer_fold": args.recording_id,
            "predictor": args.predictor,
            "target_component_one_based": args.target_component,
            "lags_ms": expected_lags,
            "lag_fit": "published benchmark five-lag fit",
        },
        "kernel_contract": kernel_metadata,
        "dense_visualization_fit": {
            "implemented": False,
            "reason": (
                "No 20 ms visualization refit was requested; benchmark "
                "statistics and coefficients remain separate from any future "
                "dense visualization fit."
            ),
        },
        "series": {
            "file": str(series_path),
            "sha256": sha256_file(series_path),
            "contains": (
                "standardized and original-unit coefficients, predictor/lag "
                "mapping, training scaler mean/scale, and target PCA sign"
            ),
        },
        "source_sha256": sources,
    }
    metadata_path = stem.with_name(f"{stem.name}.metadata.json")
    write_json(metadata_path, metadata)
    result = {
        "state": "complete",
        "points": len(series),
        "series": str(series_path),
        "metadata": str(metadata_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
