"""Lazy MobileNetV3-Small inference for trained RealWaste weights."""

from __future__ import annotations

import os
from pathlib import Path

from .constants import CLASS_NAMES
from .gpu import gpu_task
from .schemas import Prediction


class ModelNotConfiguredError(RuntimeError):
    pass


class VisionClassifier:
    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self._model = None
        self._transform = None
        self._device = None

    def _resolve_weights(self) -> Path:
        configured = self.weights_path or (
            Path(os.environ["BINWISE_WEIGHTS_PATH"])
            if os.getenv("BINWISE_WEIGHTS_PATH")
            else None
        )
        if configured and configured.exists():
            return configured
        repository = os.getenv("BINWISE_MODEL_REPO")
        if repository:
            from huggingface_hub import hf_hub_download

            return Path(
                hf_hub_download(
                    repo_id=repository,
                    filename=os.getenv(
                        "BINWISE_MODEL_FILENAME", "mobilenet_v3_small.safetensors"
                    ),
                )
            )
        raise ModelNotConfiguredError(
            "No trained model is configured. Set BINWISE_WEIGHTS_PATH or "
            "BINWISE_MODEL_REPO after completing the Colab training notebook."
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from safetensors.torch import load_file
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, len(CLASS_NAMES)
        )
        state = load_file(str(self._resolve_weights()), device=str(self._device))
        model.load_state_dict(state, strict=True)
        self._model = model.to(self._device).eval()
        self._transform = weights.transforms()

    def predict(self, image) -> list[Prediction]:
        # Loading happens in the app process; only tensor computation uses a
        # ZeroGPU worker. The LangGraph checkpointer never leaves the app.
        self._load()
        return self._predict_loaded(image)

    @gpu_task(duration=15)
    def _predict_loaded(self, image) -> list[Prediction]:
        import torch

        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            probabilities = self._model(tensor).softmax(dim=1)[0]
            scores, indices = probabilities.topk(3)
        return [
            Prediction(label=CLASS_NAMES[index], confidence=float(score))
            for score, index in zip(scores.cpu(), indices.cpu(), strict=True)
        ]

