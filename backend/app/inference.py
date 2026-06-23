from __future__ import annotations

from collections.abc import Iterable

import torch

from app.labels import LABEL_COLS


def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def _build_predictions(probabilities: Iterable[float]) -> list[dict[str, float | str]]:
    pairs = list(zip(LABEL_COLS, probabilities, strict=True))
    pairs.sort(key=lambda item: item[1], reverse=True)

    predictions: list[dict[str, float | str]] = []
    for label, probability in pairs:
        numeric_probability = float(probability)
        predictions.append(
            {
                "label": label,
                "probability": round(numeric_probability, 4),
            }
        )

    return predictions


def predict_image(model, image_tensor: torch.Tensor, device: str) -> list[dict[str, float | str]]:
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).detach().cpu().tolist()

    return _build_predictions(probabilities)

