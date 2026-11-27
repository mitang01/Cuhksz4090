#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from speech_strf.alignments import parse_textgrid
from speech_strf.lexical_predictors import (
    extract_recording_embeddings,
    ordered_word_intervals,
    resolve_checkpoint_revision,
    save_lexical_recording,
    sha256_file,
    write_csv,
)


DEFAULT_MODEL = "hfl/chinese-roberta-wwm-ext"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _load_checkpoint(args: argparse.Namespace):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    model_path = Path(args.model)
    local_directory = model_path if model_path.is_dir() else None
    # A flat uploaded snapshot has no Hub revision lookup. Passing the commit
    # back into from_pretrained would send that lookup to the network.
    load_revision = None if local_directory is not None else args.revision
    config = AutoConfig.from_pretrained(
        args.model,
        revision=load_revision,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=load_revision,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model,
        revision=load_revision,
        config=config,
        local_files_only=args.local_files_only,
    )
    try:
        resolved, source = resolve_checkpoint_revision(
            object_revisions=[
                getattr(config, "_commit_hash", None),
                getattr(getattr(model, "config", None), "_commit_hash", None),
                getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            ],
            requested_revision=args.revision,
            local_directory=local_directory,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    return tokenizer, model, resolved, source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract reusable, contextual lexical word embeddings."
    )
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--output", default="outputs/stage4_revision/lexical_embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Immutable 40-character commit SHA. Required for an uploaded local "
            "directory that has no Hugging Face snapshot metadata."
        ),
    )
    parser.add_argument("--overlap-words", type=int, default=32)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--recording-id", action="append", dest="recording_ids")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = pd.read_csv(manifest_path)
    required = {"recording_id", "alignment_path"}
    if not required.issubset(rows.columns):
        raise SystemExit(f"Manifest must contain columns {sorted(required)}")
    if args.recording_ids:
        requested = set(args.recording_ids)
        unknown = requested - set(rows["recording_id"].astype(str))
        if unknown:
            raise SystemExit(f"Unknown recording IDs: {sorted(unknown)}")
        rows = rows[rows["recording_id"].astype(str).isin(requested)]
    if rows.empty:
        raise SystemExit("No recordings selected")

    tokenizer, model, resolved_revision, revision_source = _load_checkpoint(args)
    output = Path(args.output)
    recordings_dir = output / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    word_rows: list[dict] = []

    for row in rows.to_dict("records"):
        recording_id = str(row["recording_id"])
        alignment_path = Path(str(row["alignment_path"]))
        if not alignment_path.is_file():
            raise FileNotFoundError(
                f"Alignment for {recording_id} does not exist: {alignment_path}"
            )
        words = ordered_word_intervals(parse_textgrid(alignment_path))
        archive = recordings_dir / f"{recording_id}.npz"
        result = extract_recording_embeddings(
            words,
            tokenizer,
            model,
            overlap_words=args.overlap_words,
            max_tokens=args.max_tokens,
            device=args.device,
        )
        save_lexical_recording(
            archive,
            result,
            recording_id=recording_id,
            model_id=args.model,
            requested_revision=args.revision,
            resolved_revision=resolved_revision,
            overwrite=args.overwrite,
        )
        coverage = (
            result["mapped_word_count"] / result["word_count"]
            if result["word_count"]
            else 1.0
        )
        summary = {
            "recording_id": recording_id,
            "word_count": result["word_count"],
            "mapped_word_count": result["mapped_word_count"],
            "tokenization_coverage": coverage,
            "missing_mapping_count": len(result["missing_word_indexes"]),
            "unknown_word_count": result["unknown_word_count"],
            "unknown_piece_count": int(result["unknown_piece_counts"].sum()),
            "context_window_count": len(result["windows"]),
            "embedding_dimensions": result["vectors"].shape[1],
            "archive": str(archive),
            "archive_sha256": sha256_file(archive),
        }
        summary_rows.append(summary)
        for index, word in enumerate(words):
            word_rows.append(
                {
                    "recording_id": recording_id,
                    "word_index": index,
                    "word": word.label.strip(),
                    "start_seconds": f"{word.start:.12g}",
                    "end_seconds": f"{word.end:.12g}",
                    "mapped_piece_count": int(result["piece_counts"][index]),
                    "unknown_piece_count": int(
                        result["unknown_piece_counts"][index]
                    ),
                    "owner_window": int(result["owner_window"][index]),
                    "archive": str(archive),
                    "vector_row": index,
                }
            )
        print(
            f"{recording_id}: words={result['word_count']} "
            f"mapped={result['mapped_word_count']} coverage={coverage:.6f} "
            f"unknown_words={result['unknown_word_count']} "
            f"windows={len(result['windows'])} "
            f"dimensions={result['vectors'].shape[1]}"
        )
        for window in result["windows"]:
            print(
                f"  window={window['window_index']} "
                f"words=[{window['start_word']},{window['end_word']}) "
                f"emit=[{window['emit_start_word']},{window['emit_end_word']}) "
                f"tokens={window['token_count_with_special_tokens']}"
            )
        if result["missing_word_indexes"]:
            print(f"  WARN missing mappings: {result['missing_word_indexes']}")

    write_csv(output / "recording_summary.csv", summary_rows)
    write_csv(output / "word_manifest.csv", word_rows)
    run_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "revision_source": revision_source,
        "frozen_eval_inference_mode": True,
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "overlap_words": args.overlap_words,
        "max_tokens_override": args.max_tokens,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "recording_count": len(summary_rows),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Resolved immutable checkpoint revision: {resolved_revision} "
        f"({revision_source})\n"
        f"Wrote {len(summary_rows)} recordings to {output}"
    )


if __name__ == "__main__":
    main()
