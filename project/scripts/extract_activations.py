#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from speech_strf.extract_activations import estimate_storage_bytes, extract_recording
from speech_strf.model_registry import HubertAdapter, ModelSpec
from speech_strf.provenance import load_config, write_run_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--validation-report", default="outputs/validation_report.json")
    parser.add_argument("--confirm-download", action="store_true")
    args = parser.parse_args()
    report = json.loads(Path(args.validation_report).read_text())
    if not report["valid"]:
        raise SystemExit("Validation report is invalid; extraction refused")
    config = load_config(args.config)["model"]
    manifest = pd.read_csv(args.manifest)
    durations = [row["audio_metadata"]["duration_seconds"] for row in report["records"]]
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
        durations, config["expected_frame_rate_hz"], layers, width
    )
    print(f"Estimated uncompressed activation storage: {estimate / 2**30:.2f} GiB")
    adapter = HubertAdapter(
        ModelSpec(config["checkpoint"], config["revision"], config["sample_rate_hz"]),
        config["device"],
    )
    observed = {}
    for row in manifest.to_dict("records"):
        observed[row["recording_id"]] = extract_recording(
            adapter, row["audio_path"], row["recording_id"], config["store"]
        )
    write_run_manifest(
        args.config,
        Path(config["store"]).parent,
        input_manifest_path=args.manifest,
        model_revision=config["revision"],
        extra={"observed_model_details": observed},
    )


if __name__ == "__main__":
    main()

