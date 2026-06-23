from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import timm
from pathlib import Path
from app.config import MODEL_NAME, WEIGHTS_PATH
from app.labels import LABEL_COLS

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DISPLAY_PATH = BASE_DIR / "weights" / "best_convnext_model.pth"


class ConvNextModel(nn.Module):
    def __init__(self, num_classes: int = 14, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=pretrained)
        self.backbone.reset_classifier(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@dataclass(slots=True)
class LoadedModel:
    model: nn.Module | None
    model_loaded: bool


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        return checkpoint
    raise TypeError("Unsupported checkpoint format")


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value
    return cleaned_state_dict


def load_model(device: str) -> LoadedModel:
    model = ConvNextModel(num_classes=len(LABEL_COLS), pretrained=False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    logger.info("Checkpoint path: %s", CHECKPOINT_DISPLAY_PATH)
    logger.info("Parameters: %s", f"{parameter_count:,}")
    logger.info("Device: %s", device)

    if not WEIGHTS_PATH.exists():
        logger.info("model_loaded=false")
        return LoadedModel(model=None, model_loaded=False)

    try:
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
        state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        logger.info("Loaded checkpoint: %s", CHECKPOINT_DISPLAY_PATH)
        logger.info("Model loaded successfully")
        logger.info("model_loaded=true")
        return LoadedModel(model=model, model_loaded=True)
    except Exception:
        logger.exception("Failed to load checkpoint: backend/weights/best_convnext_model.pth")
        logger.info("model_loaded=false")
        return LoadedModel(model=None, model_loaded=False)

