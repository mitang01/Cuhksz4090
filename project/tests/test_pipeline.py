import subprocess
import sys
from pathlib import Path

from speech_strf.model_registry import get_model_entry
from speech_strf.pipeline import output_identity_mismatches
from speech_strf.provenance import sha256_file


PROJECT = Path(__file__).parents[1]
REGISTRY = PROJECT / "configs" / "models.yaml"
FEATURE_CONFIG = PROJECT / "configs" / "features.yaml"
ANALYSIS_CONFIG = PROJECT / "configs" / "analysis.yaml"


def _metadata(entry, key_field):
    return {
        "model": {
            key_field: entry.key,
            "model_id": entry.model_id,
            "revision": entry.revision,
        },
        "feature_config_sha256": sha256_file(FEATURE_CONFIG),
        "analysis_config_sha256": sha256_file(ANALYSIS_CONFIG),
    }


def test_output_identity_accepts_metadata_written_by_pipeline():
    entry = get_model_entry(REGISTRY, "hubert_large_reference")
    assert not output_identity_mismatches(
        entry,
        _metadata(entry, "key"),
        FEATURE_CONFIG,
        ANALYSIS_CONFIG,
    )


def test_output_identity_accepts_legacy_model_key_alias():
    entry = get_model_entry(REGISTRY, "hubert_large_reference")
    assert not output_identity_mismatches(
        entry,
        _metadata(entry, "model_key"),
        FEATURE_CONFIG,
        ANALYSIS_CONFIG,
    )


def test_fit_without_extraction_reports_actionable_error(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_model_pipeline.py",
            "--model",
            "hubert_large_reference",
            "--output",
            str(tmp_path / "rerun"),
            "--stage",
            "fit",
        ],
        cwd=PROJECT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Run this model and output with --stage extract first" in completed.stderr
