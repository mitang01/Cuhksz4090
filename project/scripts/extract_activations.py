#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from speech_strf.extract_activations import estimate_storage_bytes, extract_recording
from speech_strf.model_registry import HubertAdapter, ModelSpec, get_model_entry
from speech_strf.provenance import write_run_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--validation-report", default="outputs/validation_report.json")
    parser.add_argument("--confirm-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--recording-id",
        action="append",
        help="Extract only this recording (repeatable); completed compatible records resume",
    )
    args = parser.parse_args()
    report = json.loads(Path(args.validation_report).read_text())
    if not report["valid"]:
        raise SystemExit("Validation report is invalid; extraction refused")
    entry = get_model_entry(args.config, args.model)
    if entry.values.get("locked_reference"):
        raise SystemExit(
            "The completed HuBERT output is locked. Use run_model_pipeline.py with "
            "--output outputs/hubert_large_refactor_smoke for a regression rerun."
        )
    if entry.adapter != "generic_speech":
        raise SystemExit(
            f"Legacy extraction supports generic_speech only; use "
            f"run_model_pipeline.py for {entry.key}"
        )
    config = {
        "checkpoint": entry.model_id,
        "revision": entry.revision,
        "sample_rate_hz": entry.sample_rate_hz,
        "device": entry.device,
        "dtype": entry.dtype,
        "batch_seconds": entry.batch_seconds,
        "chunk_overlap_seconds": entry.chunk_overlap_seconds,
        "store": "outputs/activations.h5",
        "expected_frame_rate_hz": entry.values.get("expected_frame_rate_hz", 50.0),
        "frame_rate_tolerance_hz": entry.values.get("frame_rate_tolerance_hz", 1.0),
    }
    manifest = pd.read_csv(args.manifest)
    if args.recording_id:
        requested = set(args.recording_id)
        available = set(manifest["recording_id"])
        if missing := requested - available:
            raise SystemExit(f"Unknown recording IDs: {sorted(missing)}")
        manifest = manifest[manifest["recording_id"].isin(requested)]
    report_records = {row["recording_id"]: row for row in report["records"]}
    durations = [
        report_records[recording_id]["audio_metadata"]["duration_seconds"]
        for recording_id in manifest["recording_id"]
    ]
    model_config = None
    try:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(
            config["checkpoint"], revision=config["revision"], local_files_only=True
        )
    except Exception:
        if not args.confirm_download:
            raise SystemExit("Checkpoint is not cached; rerun with --confirm-download after approval")
        model_config = AutoConfig.from_pretrained(
            config["checkpoint"], revision=config["revision"]
        )
    width = int(model_config.hidden_size)
    layers = int(model_config.num_hidden_layers) + 1
    estimate = estimate_storage_bytes(
        durations,
        config["expected_frame_rate_hz"],
        layers,
        width,
        np.dtype(config["dtype"]).itemsize,
    )
    print(
        f"Estimated uncompressed {config['dtype']} activation storage: "
        f"{estimate / 2**30:.2f} GiB",
        flush=True,
    )
    adapter = HubertAdapter(
        ModelSpec(config["checkpoint"], config["revision"], config["sample_rate_hz"]),
        config["device"],
        config["dtype"],
        local_files_only=not args.confirm_download,
        loading_class=entry.loading_class,
    )
    observed = {}
    for row in manifest.to_dict("records"):
        print(f"Extracting {row['recording_id']}...", flush=True)
        details = extract_recording(
            adapter,
            row["audio_path"],
            row["recording_id"],
            config["store"],
            batch_seconds=float(config["batch_seconds"]),
            overlap_seconds=float(config["chunk_overlap_seconds"]),
            overwrite=args.overwrite,
        )
        observed_rate = details["model"]["observed_frame_rate_hz"]
        if abs(observed_rate - config["expected_frame_rate_hz"]) > config[
            "frame_rate_tolerance_hz"
        ]:
            raise RuntimeError(
                f"{row['recording_id']} observed frame rate {observed_rate:.3f} Hz "
                f"outside configured tolerance"
            )
        observed[row["recording_id"]] = details
        print(
            f"{row['recording_id']}: {details['status']}, "
            f"{details['model']['frame_count']} frames, "
            f"{len(details['layers'])} representations",
            flush=True,
        )
    write_run_manifest(
        args.config,
        Path(config["store"]).parent,
        input_manifest_path=args.manifest,
        model_revision=config["revision"],
        extra={"observed_model_details": observed},
    )
    print(f"Activation extraction complete: {config['store']}", flush=True)


if __name__ == "__main__":
    main()

