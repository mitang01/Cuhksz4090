from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .audio import load_standardized


def estimate_storage_bytes(
    durations_seconds: list[float], frame_rate_hz: float, n_layers: int, width: int
) -> int:
    return int(sum(durations_seconds) * frame_rate_hz * n_layers * width * 4)


def extract_recording(adapter, audio_path: str, recording_id: str, store_path: str) -> dict:
    audio, audio_metadata = load_standardized(audio_path, adapter.spec.sample_rate_hz)
    states, model_metadata = adapter.extract(audio)
    frame_count = model_metadata["frame_count"]
    duration = audio_metadata["original_duration_seconds"]
    observed_rate = frame_count / duration
    model_metadata["observed_frame_rate_hz"] = observed_rate
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(store_path, "a") as store:
        if recording_id in store:
            del store[recording_id]
        group = store.create_group(recording_id)
        group.attrs["audio_metadata_json"] = json.dumps(audio_metadata)
        group.attrs["model_metadata_json"] = json.dumps(model_metadata)
        group.attrs["frame_times_seconds"] = json.dumps(
            (np.arange(frame_count) / observed_rate).tolist()
        )
        for name, values in states.items():
            group.create_dataset(
                name,
                data=values,
                chunks=(min(512, len(values)), values.shape[1]),
                compression="gzip",
                shuffle=True,
            )
    return {"audio": audio_metadata, "model": model_metadata, "layers": list(states)}

