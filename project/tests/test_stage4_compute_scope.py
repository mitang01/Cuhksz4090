from __future__ import annotations

import json

import pandas as pd
import pytest

from speech_strf.stage4_compute_scope import (
    ALL_SCOPE_MODELS,
    FAMILIES,
    load_legacy_fixed_alphas,
    publish_compute_scope_manifests,
    resolve_depth_layers,
)


def test_middle_depth_resolution_is_deterministic_and_lower_tie_wins():
    layers = ["layer_00_input", *[f"layer_{index:02d}" for index in range(1, 13)]]

    assert resolve_depth_layers(layers) == {
        "input": "layer_00_input",
        "middle": "layer_06",
        "final": "layer_12",
    }
    assert resolve_depth_layers(["input", "encoder_1", "encoder_2", "encoder_3"]) == {
        "input": "input",
        "middle": "encoder_1",
        "final": "encoder_3",
    }


def test_legacy_alpha_loader_requires_one_complete_row_per_model_and_fold(tmp_path):
    path = tmp_path / "results.csv"
    rows = []
    for fold in range(5):
        for family in ("full", *FAMILIES):
            rows.append(
                {
                    "layer": "layer_00_input",
                    "outer_fold": fold,
                    "feature_family": family,
                    "alpha": fold + 0.1,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)

    result = load_legacy_fixed_alphas(path, "layer_00_input")

    assert set(result) == set(range(5))
    assert result[3]["full"] == 3.1
    assert set(result[0]) == {"full", *FAMILIES}


def test_compute_scope_manifests_have_exact_deadline_task_counts(tmp_path):
    layers = ["input", *[f"encoder_{index}" for index in range(1, 13)]]
    model_layers = {model: layers for model in ALL_SCOPE_MODELS}
    destination = tmp_path / "deadline_scope"

    payload = publish_compute_scope_manifests(
        destination,
        model_layers=model_layers,
        hubert_original_units={layer: f"/preserved/{layer}" for layer in layers},
    )

    assert payload["task_counts"] == {
        "primary_fast": 8,
        "controls": 21,
        "nulls": 80,
    }
    assert payload["resolved_depth_layers"]["hubert_base"]["middle"] == "encoder_6"
    primary = pd.read_csv(destination / "primary_fast_refits.tsv", sep="\t")
    controls = pd.read_csv(destination / "selected_depth_controls.tsv", sep="\t")
    nulls = pd.read_csv(destination / "structured_nulls.tsv", sep="\t")
    assert len(primary) == 8
    assert len(controls) == 21
    assert len(nulls) == 80
    assert set(nulls["layer"]) == {"encoder_6"}
    assert set(nulls["shift"]) == set(range(20))
    assert publish_compute_scope_manifests(
        destination,
        model_layers=model_layers,
        hubert_original_units={layer: f"/preserved/{layer}" for layer in layers},
    ) == payload
    saved = json.loads((destination / "resolved_model_layers.json").read_text())
    assert saved["hubert_base_original"]["action"].endswith("do_not_rerun")

    changed = dict(model_layers)
    changed["hubert_base"] = ["input", "different_middle", "different_final"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_compute_scope_manifests(
            destination,
            model_layers=changed,
            hubert_original_units={"input": "/preserved/input"},
        )
