#!/usr/bin/env python3
"""Plot held-out observed, full, and lexical-omitted activation trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from speech_strf.revised_plots import (
    layer_selection_table,
    load_trajectory,
    primary_unit,
    sha256_file,
    trajectory_scores,
    write_json,
)
from speech_strf.revised_trf import RevisedTRFRunner


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/revised_trf.yaml")
    parser.add_argument("--model", default="hubert_base")
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--start-seconds", required=True, type=float)
    parser.add_argument("--end-seconds", required=True, type=float)
    parser.add_argument(
        "--selection",
        choices=("named", "best-lexical"),
        default="named",
    )
    parser.add_argument("--layer", help="Required when --selection=named")
    parser.add_argument(
        "--target-component",
        type=int,
        default=1,
        help="One-based outer-training target PC number (default: PC1)",
    )
    parser.add_argument(
        "--output",
        default="outputs/stage4_revision/revised_trf/figures/predicted_activations",
    )
    args = parser.parse_args(argv)
    if (
        not np.isfinite([args.start_seconds, args.end_seconds]).all()
        or not 0 <= args.start_seconds < args.end_seconds
    ):
        parser.error("Excerpt bounds must be finite, nonnegative, and increasing")
    if args.target_component < 1:
        parser.error("--target-component is one-based and must be positive")
    if args.selection == "named" and args.layer is None:
        parser.error("--layer is required when --selection=named")
    if args.selection == "best-lexical" and args.layer is not None:
        parser.error("Use --layer with --selection=named")

    runner = RevisedTRFRunner(args.config)
    if args.selection == "best-lexical":
        selection, selection_sources = layer_selection_table(
            runner, args.model, args.recording_id
        )
        if selection.empty or not np.isfinite(
            selection["inner_validation_lexical_delta_r2"].to_numpy()
        ).all():
            raise SystemExit("Best-layer selection requires finite inner-validation scores")
        selected = selection.sort_values(
            ["inner_validation_lexical_delta_r2", "native_layer_index"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        layer = str(selected["layer"])
        selection_criterion = (
            "maximum inner-validation full-minus-lexical-omitted R2 "
            "within the outer fold; native-layer-index tie break"
        )
    else:
        layer = args.layer
        selection = pd.DataFrame([{"layer": layer}])
        selection_sources = {}
        selection_criterion = "prespecified named layer"

    unit = primary_unit(runner, args.model, layer)
    target_index = args.target_component - 1
    trajectory, trajectory_metadata, trajectory_sources = load_trajectory(
        unit, args.recording_id, target_index
    )
    times = trajectory["time_seconds"].to_numpy()
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise SystemExit("Saved trajectory timestamps must be finite and strictly increasing")
    components = ["observed_target_pc", "full_prediction", "lexical_omitted_prediction"]
    if not np.isfinite(trajectory[components].to_numpy()).all():
        raise SystemExit("Saved activation traces contain nonfinite values")
    if args.selection == "best-lexical" and "outer_fold" in selected.index:
        if int(selected["outer_fold"]) != int(trajectory_metadata["outer_fold"]):
            raise SystemExit("Layer selection and held-out trajectory came from different folds")
    excerpt_mask = (
        (trajectory["time_seconds"] >= args.start_seconds)
        & (trajectory["time_seconds"] <= args.end_seconds)
    ).to_numpy()
    if excerpt_mask.sum() < 2:
        raise SystemExit("Configured excerpt contains fewer than two saved frames")
    trajectory["in_excerpt"] = excerpt_mask
    excerpt = trajectory.loc[excerpt_mask]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = (
        output
        / f"{args.model}_{layer}_{args.recording_id}_pc{args.target_component}"
    )
    series_path = stem.with_name(f"{stem.name}.series.csv")
    trajectory.to_csv(series_path, index=False)
    selection_path = stem.with_name(f"{stem.name}.layer_selection.csv")
    selection.to_csv(selection_path, index=False)
    score_rows = [
        trajectory_scores(
            trajectory, np.ones(len(trajectory), dtype=bool), "whole_recording"
        ),
        trajectory_scores(trajectory, excerpt_mask, "configured_excerpt"),
    ]
    scores = pd.DataFrame(score_rows)
    scores_path = stem.with_name(f"{stem.name}.scores.csv")
    scores.to_csv(scores_path, index=False)

    fig, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    axis.plot(
        excerpt["time_seconds"],
        excerpt["observed_target_pc"],
        color="black",
        linewidth=1.7,
        label=f"Observed PC{args.target_component}",
    )
    axis.plot(
        excerpt["time_seconds"],
        excerpt["full_prediction"],
        color="#1f77b4",
        linewidth=1.4,
        label="Full-model prediction",
    )
    axis.plot(
        excerpt["time_seconds"],
        excerpt["lexical_omitted_prediction"],
        color="#d62728",
        linewidth=1.2,
        label="Lexical-omitted prediction",
    )
    axis.axhline(0, color="0.6", linewidth=0.6)
    axis.set_xlabel("Time in held-out recording (seconds)")
    axis.set_ylabel(f"Sign-fixed target PC{args.target_component} score")
    label = "best lexical layer" if args.selection == "best-lexical" else "named layer"
    axis.set_title(
        f"{args.model}, {layer} ({label}), held out {args.recording_id}"
    )
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.18)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure_selection": {
            "model": args.model,
            "recording_id": args.recording_id,
            "selection_mode": args.selection,
            "selection_criterion": selection_criterion,
            "selected_layer": layer,
            "excerpt_start_seconds_inclusive": args.start_seconds,
            "excerpt_end_seconds_inclusive": args.end_seconds,
            "target_component_one_based": args.target_component,
            "default_target_rule": "PC1 from outer-training target PCA",
            "held_out_trajectory_used_for_layer_selection": False,
        },
        "coordinate_contract": trajectory_metadata,
        "reported_scores": {
            "definition": (
                "single displayed target-component R2 on exact saved timestamps; "
                "lexical delta R2 = full minus lexical-omitted"
            ),
            "file": str(scores_path),
            "sha256": sha256_file(scores_path),
            "values": score_rows,
        },
        "series": {
            "file": str(series_path),
            "sha256": sha256_file(series_path),
            "timestamps": "exact canonical timestamps stored by the fit unit",
        },
        "layer_selection_series": {
            "file": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "source_sha256": {**selection_sources, **trajectory_sources},
    }
    metadata_path = stem.with_name(f"{stem.name}.metadata.json")
    write_json(metadata_path, metadata)
    result = {
        "state": "complete",
        "selected_layer": layer,
        "outer_fold": trajectory_metadata["outer_fold"],
        "series": str(series_path),
        "scores": str(scores_path),
        "metadata": str(metadata_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
