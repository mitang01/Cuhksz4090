#!/usr/bin/env python3
"""Run primary revised TRF fits and HuBERT Base LORO sensitivity."""

from __future__ import annotations

import argparse
import json

import numpy as np

from speech_strf.revised_trf import (
    FitSpec,
    RevisedTRFRunner,
    fit_revised_encoding,
)


def _synthetic_smoke() -> dict:
    generator = np.random.default_rng(17)
    records = {}
    groups = {}
    for index in range(6):
        recording_id = f"recording_{index:02d}"
        frames = 48
        dense = generator.normal(size=(frames, 7))
        lexical_vectors = generator.normal(size=(8, 6))
        lexical_onsets = np.arange(2, 42, 5)
        lexical_grid = np.zeros((frames, 2))
        lexical_grid[lexical_onsets] = lexical_vectors[:, :2]
        targets = np.column_stack(
            [
                dense[:, 0] + lexical_grid[:, 0],
                dense[:, 1] - lexical_grid[:, 1],
                dense[:, 3],
                dense[:, 4] + dense[:, 5],
            ]
        )
        records[recording_id] = {
            "dense": dense,
            "families": [
                "acoustic",
                "acoustic",
                "prosodic",
                "phonetic",
                "word",
                "word",
                "word",
            ],
            "blocks": [
                "acoustic",
                "acoustic",
                "prosodic",
                "phonetic",
                "word_timing",
                "word_timing",
                "word_timing",
            ],
            "lexical_vectors": lexical_vectors,
            "lexical_onset_indices": lexical_onsets,
            "targets": targets,
            "duration_seconds": frames / 50,
        }
        groups[recording_id] = recording_id
    result = fit_revised_encoding(
        records,
        groups,
        outer_mode="grouped",
        outer_folds=5,
        inner_folds=4,
        alphas=[1.0],
        lags_seconds=[-0.02, 0.0, 0.02],
        rate_hz=50,
        lexical_components=2,
        target_components=2,
        reductions=("word", "lexical"),
        seed=17,
    )
    return {
        "state": "PASS",
        "recordings": len(records),
        "outer_folds": len(result["split_reports"]),
        "score_rows": len(result["scores"]),
        "families": sorted(result["scores"]["feature_family"].unique()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/revised_trf.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "manifest", help="Resolve all HDF5 layers and write exact unit manifests"
    )
    fit = subparsers.add_parser("fit", help="Fit one resumable model × layer unit")
    fit.add_argument("model")
    fit.add_argument("layer")
    fit.add_argument(
        "--outer-mode",
        choices=("grouped", "loro"),
        default="grouped",
        help="LORO is accepted only for HuBERT Base",
    )
    subparsers.add_parser("summarize", help="Validate coverage and summarize panels")
    subparsers.add_parser(
        "local-smoke",
        help="Run a small synthetic nested-CV smoke without server data",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.command == "local-smoke":
        result = _synthetic_smoke()
    else:
        runner = RevisedTRFRunner(args.config)
        if args.command == "manifest":
            result = runner.build_manifests()
        elif args.command == "fit":
            result = runner.fit(
                FitSpec(
                    model=args.model,
                    layer=args.layer,
                    variant="primary",
                    outer_mode=args.outer_mode,
                )
            )
        else:
            result = runner.summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
