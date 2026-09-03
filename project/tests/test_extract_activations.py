from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import HubertConfig, HubertModel

from speech_strf.extract_activations import (
    estimate_storage_bytes,
    extract_chunked,
    extract_recording,
)
from speech_strf.model_registry import HubertAdapter, ModelSpec


class FakeAdapter:
    spec = SimpleNamespace(sample_rate_hz=16000)

    def frame_timing_samples(self):
        return 1600, 0.0

    def extract(self, audio):
        states = {
            "layer_00_input": np.ones((10, 4), dtype=np.float32),
            "layer_01_transformer": np.full((10, 4), 2, dtype=np.float32),
        }
        return states, {
            "frame_count": 10,
            "representation_count": 2,
            "hidden_dimension": 4,
            "model_config": {"hidden_size": 4},
        }


class ChunkAdapter:
    spec = SimpleNamespace(sample_rate_hz=10)

    def frame_timing_samples(self):
        return 2, 0.0

    def extract(self, audio):
        values = np.asarray(audio[::2, None], dtype=np.float32)
        return {"layer_00_input": values}, {
            "frame_count": len(values),
            "representation_count": 1,
            "hidden_dimension": 1,
            "model_config": {},
        }


class ArrayProcessor:
    def __call__(self, audio, **kwargs):
        return SimpleNamespace(input_values=torch.tensor(audio[None, :]))


def test_real_hubert_class_returns_explicit_input_and_transformer_layers():
    config = HubertConfig(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        conv_dim=(8,),
        conv_stride=(2,),
        conv_kernel=(4,),
        num_conv_pos_embedding_groups=2,
        num_conv_pos_embeddings=4,
    )
    adapter = HubertAdapter(
        ModelSpec("local-random-test-model", "unversioned", 16000),
        processor=ArrayProcessor(),
        model=HubertModel(config),
    )
    states, metadata = adapter.extract(np.zeros(64, dtype=np.float32))
    assert list(states) == [
        "layer_00_input",
        "layer_01_transformer",
        "layer_02_transformer",
    ]
    assert metadata["representation_count"] == 3
    assert metadata["hidden_dimension"] == 8
    assert adapter.frame_timing_samples() == (2, 1.5)


def test_device_typo_is_rejected_before_model_loading():
    with pytest.raises(ValueError, match="device must be"):
        HubertAdapter(ModelSpec("unused", "unused", 16000), device="coda")


def test_overlapping_chunks_stitch_once_on_global_frame_grid():
    audio = np.arange(100, dtype=np.float32)
    states, times, metadata = extract_chunked(
        ChunkAdapter(), audio, batch_seconds=4, overlap_seconds=1
    )
    np.testing.assert_array_equal(states["layer_00_input"][:, 0], np.arange(0, 100, 2))
    np.testing.assert_allclose(times, np.arange(0, 100, 2) / 10)
    assert metadata["chunk_count"] == 3
    assert metadata["stitching"].startswith("retain_frame_centers")


def test_storage_estimate_uses_configured_dtype_width():
    assert estimate_storage_bytes([10], 50, 3, 4, bytes_per_value=2) == 12_000


def test_layer_extraction_preserves_frames_and_metadata(tmp_path):
    wav = tmp_path / "tiny.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    store = tmp_path / "activations.h5"
    metadata = extract_recording(FakeAdapter(), str(wav), "tiny", str(store))
    assert metadata["status"] == "written"
    assert metadata["layers"][0] == "layer_00_input"
    with h5py.File(store) as handle:
        assert handle["tiny/layer_01_transformer"].shape == (10, 4)
        assert handle["tiny"].attrs["complete"]
        assert handle["tiny/_frame_times_seconds"].shape == (10,)
    resumed = extract_recording(FakeAdapter(), str(wav), "tiny", str(store))
    assert resumed["status"] == "skipped_complete"

