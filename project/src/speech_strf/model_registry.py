from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    revision: str
    sample_rate_hz: int


class HubertAdapter:
    """Hugging Face HuBERT adapter with explicit device and precision controls."""

    def __init__(
        self,
        spec: ModelSpec,
        device: str = "cpu",
        dtype: str = "float32",
        local_files_only: bool = False,
        processor=None,
        model=None,
    ) -> None:
        import torch

        if dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be float16 or float32")
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("device must be cpu, cuda, or a CUDA device such as cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if not device.startswith("cuda") and dtype == "float16":
            raise ValueError("float16 inference is supported only on CUDA")
        self.spec, self.device, self.dtype = spec, device, dtype
        self.torch_dtype = getattr(torch, dtype)
        self.numpy_dtype = np.dtype(dtype)
        if processor is None or model is None:
            from transformers import AutoFeatureExtractor, HubertModel

            processor = AutoFeatureExtractor.from_pretrained(
                spec.checkpoint,
                revision=spec.revision,
                local_files_only=local_files_only,
            )
            model = HubertModel.from_pretrained(
                spec.checkpoint,
                revision=spec.revision,
                local_files_only=local_files_only,
            )
        self.processor = processor
        self.model = model.to(device=device, dtype=self.torch_dtype).eval()

    def frame_timing_samples(self) -> tuple[int, float]:
        """Return convolutional frame stride and receptive-field center in samples."""
        kernels = tuple(self.model.config.conv_kernel)
        strides = tuple(self.model.config.conv_stride)
        if len(kernels) != len(strides):
            raise RuntimeError("HuBERT convolution kernel and stride lengths differ")
        jump, receptive_field = 1, 1
        for kernel, stride in zip(kernels, strides):
            receptive_field += (int(kernel) - 1) * jump
            jump *= int(stride)
        return jump, (receptive_field - 1) / 2

    def prepare_audio(self, audio: np.ndarray) -> tuple[np.ndarray, str]:
        """Apply feature-extractor normalization once before audio is chunked."""
        audio = np.asarray(audio, dtype=np.float32)
        if getattr(self.processor, "do_normalize", False):
            variance = np.var(audio)
            audio = (audio - np.mean(audio)) / np.sqrt(variance + 1e-7)
            return audio.astype(np.float32, copy=False), "global_zero_mean_unit_variance"
        return audio, "none"

    def extract(self, audio):
        inputs = self.processor(
            audio,
            sampling_rate=self.spec.sample_rate_hz,
            return_tensors="pt",
        )
        return self._forward(inputs.input_values)

    def extract_prepared(self, audio):
        """Extract already globally normalized audio without per-chunk renormalization."""
        import torch

        values = torch.as_tensor(np.asarray(audio), dtype=torch.float32).unsqueeze(0)
        return self._forward(values)

    def _forward(self, input_values):
        import torch

        autocast = (
            torch.autocast(device_type="cuda", dtype=self.torch_dtype)
            if self.device.startswith("cuda") and self.dtype == "float16"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            output = self.model(
                input_values=input_values.to(self.device),
                output_hidden_states=True,
                return_dict=True,
            )
        states = output.hidden_states
        if not states:
            raise RuntimeError("Model returned no hidden states")
        arrays = [
            state[0].detach().cpu().numpy().astype(self.numpy_dtype, copy=False)
            for state in states
        ]
        dimensions = {array.shape[1] for array in arrays}
        lengths = {array.shape[0] for array in arrays}
        if len(dimensions) != 1 or len(lengths) != 1:
            raise RuntimeError("Hidden-state shapes are inconsistent across layers")
        expected_count = getattr(self.model.config, "num_hidden_layers", None)
        if expected_count is not None and len(arrays) != expected_count + 1:
            raise RuntimeError(
                f"Expected {expected_count + 1} representations from runtime config, "
                f"received {len(arrays)}"
            )
        expected_width = getattr(self.model.config, "hidden_size", None)
        if expected_width is not None and next(iter(dimensions)) != expected_width:
            raise RuntimeError(
                f"Expected hidden width {expected_width}, received {next(iter(dimensions))}"
            )
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
            "inference_dtype": self.dtype,
            "model_config": config,
        }

