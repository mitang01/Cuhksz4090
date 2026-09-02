#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from speech_strf.alignments import Interval
from speech_strf.design_matrix import lagged_design
from speech_strf.extract_features import extract_features
from speech_strf.figures import make_result_figures
from speech_strf.fit_encoding import nested_group_encoding
from speech_strf.provenance import load_config, write_run_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    rng = np.random.default_rng(config["random_seed"])
    sample_rate, xs, ys, groups, feature_families = 16000, [], [], [], None
    duration = float(config["smoke"]["duration_seconds"])
    feature_config = load_config(Path(args.config).with_name("features.yaml"))
    for group_index in range(int(config["smoke"]["n_groups"])):
        time = np.arange(int(sample_rate * duration)) / sample_rate
        audio = (
            .15 * np.sin(2 * np.pi * (140 + 10 * group_index) * time)
            + .03 * rng.standard_normal(len(time))
        ).astype(np.float32)
        intervals = [
            Interval("words", 1, 2, "你好"),
            Interval("words", 3, 4, "世界"),
            Interval("phones", 1, 1.4, "n"),
            Interval("phones", 1.4, 2, "i"),
        ]
        feature = extract_features(audio, sample_rate, duration, intervals, feature_config)
        x = feature["matrix"]
        weights = rng.normal(size=(x.shape[1], int(config["smoke"]["n_targets"])))
        y = x @ weights + rng.normal(scale=.15, size=(len(x), weights.shape[1]))
        xs.append(x)
        ys.append(y)
        groups.extend([f"story_{group_index}"] * len(x))
        feature_families = feature["families"]
    design, lagged_families = lagged_design(
        np.vstack(xs),
        np.asarray(groups),
        feature_families,
        config["analysis"]["lags_seconds"],
        config["analysis_rate_hz"],
    )
    results, kernels = nested_group_encoding(
        design, np.vstack(ys), np.asarray(groups), lagged_families, config
    )
    output = Path(config["smoke"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "results.csv", index=False)
    (output / "synthetic_validation_report.json").write_text(
        json.dumps({"valid": True, "source": "generated synthetic fixture"}, indent=2)
    )
    make_result_figures(results, kernels, output / "figures")
    write_run_manifest(args.config, output, extra={"data_scope": "synthetic_only"})
    print(results.groupby("feature_family")["conditional_delta_r2"].mean().to_string())


if __name__ == "__main__":
    main()

