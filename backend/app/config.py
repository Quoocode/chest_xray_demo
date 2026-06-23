import os
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = BASE_DIR / "weights" / "best_convnext_model.pth"
MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
IMAGE_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_ORIGINS = [
	origin.strip()
	for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
	if origin.strip()
]
