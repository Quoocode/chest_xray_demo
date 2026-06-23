from __future__ import annotations

import io

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image

from app.config import IMAGE_SIZE


val_transform = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
)


def preprocess_image(file_bytes: bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image_np = np.array(image)
    transformed = val_transform(image=image_np)
    return transformed["image"].unsqueeze(0)

