#!/usr/bin/env python3
"""Command-line entry point for the immutable-input Stage 4 revision."""

from __future__ import annotations

import argparse
import json
from typing import Any

from speech_strf.stage4_runner import Stage4Runner, VARIANTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage4_revision.yaml", help="Stage 4 YAML config"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="Fit observed encoding models")
    fit.add_argument("model")
    fit.add_argument("variant", choices=VARIANTS)
    fit.add_argument("--layer", help="Fit one HDF5 layer; default discovers all")

    null = subparsers.add_parser("null", help="Fit one circular-shift null index")
    null.add_argument("model")
    null.add_argument("null_index", type=int)
    count = null.add_mutually_exclusive_group(required=True)
    count.add_argument(
        "--dry-null-count",
        type=int,
        metavar="N",
        help="Nonfinal dry ensemble size (at least 20)",
    )
    count.add_argument(
        "--full-null-count",
        type=int,
        metavar="N",
        help="Final ensemble size (must be exactly 100)",
    )
    null.add_argument("--layer", help="Fit one HDF5 layer; default discovers all")

    subparsers.add_parser("summarize", help="Summarize valid completed units")
    subparsers.add_parser("audit", help="Run the full immutable-input audit")
    subparsers.add_parser("synthetic-test", help="Run deterministic synthetic E2E")
    return parser


def main(argv: list[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    runner = Stage4Runner(args.config)
    if args.command == "fit":
        result = runner.fit(args.model, args.variant, args.layer)
    elif args.command == "null":
        dry = args.dry_null_count is not None
        count = args.dry_null_count if dry else args.full_null_count
        result = runner.null(
            args.model,
            args.null_index,
            dry_run=dry,
            null_count=count,
            layer=args.layer,
        )
    elif args.command == "summarize":
        result = runner.summarize()
    elif args.command == "audit":
        result = runner.audit()
    else:
        result = runner.synthetic_test()
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
