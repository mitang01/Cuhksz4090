"""Temporary synthetic reproduction for selected-depth ProcessPool failures."""

from __future__ import annotations

import json

import h5py
import pandas as pd

from scripts.run_stage4_revision import main
from speech_strf.stage4_compute_scope import (
    ALL_SCOPE_MODELS,
    publish_compute_scope_manifests,
)
from speech_strf.stage4_runner import Stage4Runner
from test_stage4_runner import _config, _inputs


def test_selected_depth_manifest_round_trips_through_process_workers(tmp_path):
    config = _config(tmp_path)
    _inputs(tmp_path)
    model_layers = {
        model: [
            "layer_00_input",
            f"layer_01_{model}",
            f"layer_02_{model}",
        ]
        for model in ALL_SCOPE_MODELS
    }
    for model in ALL_SCOPE_MODELS:
        with h5py.File(tmp_path / "outputs" / model / "activations.h5", "r+") as store:
            for group in store.values():
                values = group["canonical/layer_00_input"][...]
                for layer in model_layers[model][1:]:
                    group["native"].create_dataset(layer, data=values)
                    group["canonical"].create_dataset(layer, data=values)
                group.attrs["layer_names_json"] = json.dumps(model_layers[model])

    manifest_dir = tmp_path / "scope"
    publish_compute_scope_manifests(
        manifest_dir,
        model_layers=model_layers,
        hubert_original_units={
            layer: f"/preserved/{layer}" for layer in model_layers["hubert_base"]
        },
    )
    tasks = pd.read_csv(manifest_dir / "selected_depth_controls.tsv", sep="\t")
    for task in tasks.itertuples(index=False):
        assert [task.input, task.middle, task.final] == model_layers[task.model]
    task = tasks.loc[tasks["model"] == "wavlm_large"].iloc[0]
    Stage4Runner(config)._audit()

    result = main(
        [
            "--config",
            str(config),
            "fit",
            str(task["model"]),
            str(task["variant"]),
            "--layers",
            str(task["input"]),
            str(task["middle"]),
            str(task["final"]),
            "--layer-workers",
            "2",
        ]
    )

    assert len(result) == 3
    assert all(value["state"] == "computed" for value in result)
