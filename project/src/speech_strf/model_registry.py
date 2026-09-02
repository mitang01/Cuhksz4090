from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    revision: str
    sample_rate_hz: int


class HubertAdapter:
    """Lazy Hugging Face adapter; constructing it does not download a model."""

    def __init__(
        self, spec: ModelSpec, device: str = "cpu", processor=None, model=None
    ) -> None:
        self.spec, self.device = spec, device
        if processor is None or model is None:
            from transformers import AutoFeatureExtractor, HubertModel

            processor = AutoFeatureExtractor.from_pretrained(
                spec.checkpoint, revision=spec.revision
            )
            model = HubertModel.from_pretrained(spec.checkpoint, revision=spec.revision)
        self.processor = processor
        self.model = model.to(device).eval()

    def extract(self, audio):
        import torch

        inputs = self.processor(
            audio,
            sampling_rate=self.spec.sample_rate_hz,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self.model(
                input_values=inputs.input_values.to(self.device),
                output_hidden_states=True,
                return_dict=True,
            )
        states = output.hidden_states
        if not states:
            raise RuntimeError("Model returned no hidden states")
        arrays = [state[0].detach().cpu().float().numpy() for state in states]
        dimensions = {array.shape[1] for array in arrays}
        lengths = {array.shape[0] for array in arrays}
        if len(dimensions) != 1 or len(lengths) != 1:
            raise RuntimeError("Hidden-state shapes are inconsistent across layers")
        names = ["layer_00_input"] + [
            f"layer_{index:02d}_transformer" for index in range(1, len(arrays))
        ]
        config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else {}
        return dict(zip(names, arrays)), {
            "checkpoint": self.spec.checkpoint,
            "requested_revision": self.spec.revision,
            "resolved_commit_hash": getattr(self.model.config, "_commit_hash", None),
            "representation_count": len(arrays),
            "hidden_dimension": next(iter(dimensions)),
            "frame_count": next(iter(lengths)),
            "model_config": config,
        }

