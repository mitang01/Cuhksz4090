#!/usr/bin/env python3
"""Run selected-depth revised controls and observed-minus-shifted null units."""

from __future__ import annotations

import argparse
import json

from speech_strf.revised_trf import (
    CONTROL_VARIANTS,
    FitSpec,
    RevisedTRFRunner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/revised_trf.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "manifest", help="Resolve layers and write primary/control/null manifests"
    )
    fit = subparsers.add_parser(
        "fit", help="Fit one selected-depth model × layer control"
    )
    fit.add_argument("variant", choices=CONTROL_VARIANTS)
    fit.add_argument("model")
    fit.add_argument("layer")

    null = subparsers.add_parser(
        "null", help="Fit one resumable model × middle-layer × shift unit"
    )
    null.add_argument("model")
    null.add_argument("shift", type=int)
    null.add_argument(
        "--layer",
        help="Resolved middle layer; omitted by default and validated if supplied",
    )
    subparsers.add_parser("summarize", help="Validate and summarize all panels")
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    runner = RevisedTRFRunner(args.config)
    if args.command == "manifest":
        result = runner.build_manifests()
    elif args.command == "fit":
        resolved = runner.resolved_depths(args.model)
        if args.layer not in set(resolved.values()):
            raise SystemExit(
                f"{args.layer!r} is not a resolved input/middle/final layer: "
                f"{resolved}"
            )
        result = runner.fit(
            FitSpec(
                model=args.model,
                layer=args.layer,
                variant=args.variant,
            )
        )
    elif args.command == "null":
        middle = runner.resolved_depths(args.model)["middle"]
        if args.layer is not None and args.layer != middle:
            raise SystemExit(
                f"Null layer {args.layer!r} differs from resolved middle {middle!r}"
            )
        result = runner.fit(
            FitSpec(
                model=args.model,
                layer=middle,
                variant="null",
                shift=args.shift,
            )
        )
    else:
        result = runner.summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
