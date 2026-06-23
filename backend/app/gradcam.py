from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def resolve_gradcam_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last spatial feature map layer before the classifier.

    For timm ConvNeXtV2, the last stage in `backbone.stages` still emits a 4D
    feature map. The classifier consumes pooled features after that, so this is
    the correct hook target for Grad-CAM.
    """

    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "stages"):
        stages = getattr(backbone, "stages")
        if isinstance(stages, torch.nn.Sequential) and len(stages) > 0:
            return stages[-1]

    if hasattr(model, "stages"):
        stages = getattr(model, "stages")
        if isinstance(stages, torch.nn.Sequential) and len(stages) > 0:
            return stages[-1]

    raise ValueError("Unable to resolve Grad-CAM target layer")


@dataclass(slots=True)
class _HookState:
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None


class GradCAMGenerator:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = resolve_gradcam_layer(model)
        self.state = _HookState()
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _forward_hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor):
            self.state.activations = output.detach()
            return

        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            self.state.activations = output[0].detach()
            return

        raise TypeError("Grad-CAM forward hook did not capture a tensor output")

    def _backward_hook(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[Any, ...],
        grad_output: tuple[Any, ...],
    ) -> None:
        if grad_output and isinstance(grad_output[0], torch.Tensor):
            self.state.gradients = grad_output[0].detach()
            return

        raise TypeError("Grad-CAM backward hook did not capture tensor gradients")

    def _build_heatmap(self, original_size: tuple[int, int]) -> np.ndarray:
        if self.state.activations is None or self.state.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")

        activations = self.state.activations
        gradients = self.state.gradients

        if activations.ndim != 4 or gradients.ndim != 4:
            raise ValueError("Expected 4D tensors for Grad-CAM")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = torch.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        cam = cam[0]
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value

        return cam.detach().cpu().numpy()

    def generate_base64_png(
        self,
        image_tensor: torch.Tensor,
        target_class_idx: int,
        file_bytes: bytes,
        device: str,
    ) -> str:
        with Image.open(io.BytesIO(file_bytes)) as image:
            original_width, original_height = image.convert("RGB").size

        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            outputs = self.model(image_tensor.to(device))
            target_score = outputs[0, target_class_idx]
            target_score.backward()

        heatmap = self._build_heatmap((original_height, original_width))
        heatmap_uint8 = np.uint8(np.clip(heatmap * 255.0, 0, 255))
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        buffer = io.BytesIO()
        Image.fromarray(colored_heatmap).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_gradcam_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    file_bytes: bytes,
    target_class_idx: int,
    device: str,
) -> str:
    generator = GradCAMGenerator(model)
    try:
        return generator.generate_base64_png(image_tensor, target_class_idx, file_bytes, device)
    finally:
        generator.close()
