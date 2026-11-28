#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from speech_strf.alignments import parse_textgrid
from speech_strf.lexical_predictors import (
    build_revised_recording,
    load_lexical_recording,
    sha256_file,
    verify_words_against_alignment,
    write_csv,
    write_npz_atomic,
)


def check_acoustic_control(original: Path, rich: Path) -> None:
    """Verify that richer acoustics extends the existing dense feature design."""
    with np.load(original, allow_pickle=False) as base, np.load(rich, allow_pickle=False) as extended:
        if not np.array_equal(base["times"], extended["times"]):
            raise ValueError(f"Original/rich time grids differ for {original.stem}")
        original_names = base["names"].tolist()
        rich_names = extended["names"].tolist()
        rich_indices = {name: index for index, name in enumerate(rich_names)}
        if len(rich_indices) != len(rich_names) or len(set(original_names)) != len(original_names):
            raise ValueError(f"Duplicate feature names for {original.stem}")
        if not set(original_names).issubset(rich_indices):
            raise ValueError(f"Richer acoustics is missing original columns for {original.stem}")
        original_indices = [rich_indices[name] for name in original_names]
        if not np.array_equal(base["families"], extended["families"][original_indices]):
            raise ValueError(f"Shared feature families differ for {original.stem}")
        added = sorted(set(rich_names) - set(original_names))
        if not added or any(extended["families"][rich_indices[name]] != "acoustic" for name in added):
            raise ValueError(f"Richer acoustics must add only acoustic columns for {original.stem}")
        if not np.allclose(
            base["matrix"], extended["matrix"][:, original_indices], rtol=1e-6, atol=1e-8
        ):
            raise ValueError(f"Original predictor values differ between designs for {original.stem}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build original/rich predictor archives with sparse lexical events "
            "for fold-specific reduction."
        )
    )
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--original-features", default="outputs/features")
    parser.add_argument(
        "--rich-features", default="outputs/stage4_revision/features_rich"
    )
    parser.add_argument(
        "--lexical-root", default="outputs/stage4_revision/lexical_embeddings"
    )
    parser.add_argument(
        "--activation-store",
        action="append",
        default=None,
        help=(
            "Activation HDF5 grid to validate (repeatable). "
            "Defaults to outputs/hubert_base/activations.h5."
        ),
    )
    parser.add_argument(
        "--output", default="outputs/stage4_revision/revised_predictors"
    )
    parser.add_argument("--analysis-rate-hz", type=float, default=50.0)
    parser.add_argument("--recording-id", action="append", dest="recording_ids")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    activation_stores = args.activation_store or [
        "outputs/hubert_base/activations.h5"
    ]
    for path in activation_stores:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Activation store does not exist: {path}")

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    required = {"recording_id", "alignment_path"}
    if not required.issubset(manifest.columns):
        raise SystemExit(f"Manifest must contain columns {sorted(required)}")
    ids = manifest["recording_id"].astype(str)
    if manifest["recording_id"].isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise SystemExit("Manifest recording IDs must be nonempty and unique")
    if args.recording_ids:
        requested = set(args.recording_ids)
        unknown = requested - set(manifest["recording_id"].astype(str))
        if unknown:
            raise SystemExit(f"Unknown recording IDs: {sorted(unknown)}")
        manifest = manifest[
            manifest["recording_id"].astype(str).isin(requested)
        ]
    if manifest.empty:
        raise SystemExit("No recordings selected")

    designs = {
        "original": Path(args.original_features),
        "rich": Path(args.rich_features),
    }
    output = Path(args.output)
    lexical_recordings = Path(args.lexical_root) / "recordings"
    lexical_manifest = Path(args.lexical_root) / "run_manifest.json"
    lexical_manifest_sha256 = sha256_file(lexical_manifest)
    lexical_run = json.loads(lexical_manifest.read_text(encoding="utf-8"))
    selected_ids = set(manifest["recording_id"].astype(str))
    if args.recording_ids:
        existing_ids = {
            path.stem for design in designs for path in (output / design).glob("*.npz")
        }
        if existing_ids - selected_ids:
            parser.error("Use a separate --output for a recording subset")
    report_rows: list[dict] = []
    output_rows: list[dict] = []

    for row in manifest.to_dict("records"):
        recording_id = str(row["recording_id"])
        alignment_path = Path(str(row["alignment_path"]))
        if not alignment_path.is_file():
            raise FileNotFoundError(
                f"Alignment for {recording_id} does not exist: {alignment_path}"
            )
        lexical_archive = lexical_recordings / f"{recording_id}.npz"
        if not lexical_archive.is_file():
            raise FileNotFoundError(
                f"Lexical archive for {recording_id} does not exist: "
                f"{lexical_archive}"
            )
        lexical = load_lexical_recording(lexical_archive)
        for key in ("model_id", "resolved_revision"):
            if lexical["metadata"].get(key) != lexical_run[key]:
                raise ValueError(f"{lexical_archive} disagrees with the lexical run on {key}")
        verify_words_against_alignment(lexical, parse_textgrid(alignment_path))

        sources = {
            name: root / f"{recording_id}.npz" for name, root in designs.items()
        }
        for design_name, feature_archive in sources.items():
            if not feature_archive.is_file():
                raise FileNotFoundError(
                    f"{design_name} features for {recording_id} do not exist: "
                    f"{feature_archive}"
                )
        check_acoustic_control(sources["original"], sources["rich"])

        for design_name, feature_archive in sources.items():
            arrays, report = build_revised_recording(
                feature_archive,
                lexical_archive,
                recording_id=recording_id,
                design_name=design_name,
                activation_stores=activation_stores,
                analysis_rate_hz=args.analysis_rate_hz,
            )
            destination = output / design_name / f"{recording_id}.npz"
            write_npz_atomic(
                destination,
                arrays,
                compressed=False,
                overwrite=args.overwrite,
            )
            report_rows.append(report)
            output_rows.append(
                {
                    "recording_id": recording_id,
                    "design": design_name,
                    "archive": str(destination),
                    "archive_sha256": sha256_file(destination),
                    "frames": report["frames"],
                    "dense_dimensions": report["revised_dense_dimensions"],
                    "lexical_events": report["lexical_events"],
                    "lexical_dimensions": report["lexical_dimensions"],
                    "collision_bins": report["collision_bins"],
                }
            )
            print(
                f"{recording_id}/{design_name}: frames={report['frames']} "
                f"dense={report['source_dense_dimensions']}->"
                f"{report['revised_dense_dimensions']} "
                f"lexical={report['lexical_events']}x"
                f"{report['lexical_dimensions']} "
                f"collision_bins={report['collision_bins']} "
                f"grid={report['grid_status']}"
            )

    write_csv(output / "build_report.csv", report_rows)
    write_csv(output / "archive_manifest.csv", output_rows)
    run_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "lexical_root": args.lexical_root,
        "lexical_run_manifest_sha256": lexical_manifest_sha256,
        "design_feature_roots": {
            key: str(value) for key, value in designs.items()
        },
        "activation_stores": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in activation_stores
        ],
        "analysis_rate_hz": args.analysis_rate_hz,
        "recording_count": len(manifest),
        "archive_count": len(output_rows),
        "lexical_rasterization": (
            "fold-specific dimensionality reduction on event vectors, then "
            "nearest-frame additive rasterization"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(output_rows)} revised predictor archives to {output}")


if __name__ == "__main__":
    main()
