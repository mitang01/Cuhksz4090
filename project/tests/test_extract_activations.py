from types import SimpleNamespace

import h5py
import numpy as np
import soundfile as sf

from speech_strf.extract_activations import extract_recording


class FakeAdapter:
    spec = SimpleNamespace(sample_rate_hz=16000)

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


def test_layer_extraction_preserves_frames_and_metadata(tmp_path):
    wav = tmp_path / "tiny.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    store = tmp_path / "activations.h5"
    metadata = extract_recording(FakeAdapter(), str(wav), "tiny", str(store))
    assert metadata["layers"][0] == "layer_00_input"
    with h5py.File(store) as handle:
        assert handle["tiny/layer_01_transformer"].shape == (10, 4)
        assert "frame_times_seconds" in handle["tiny"].attrs

