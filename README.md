# Chest X-ray AI Assistant

A full-stack AI demo for multi-label chest X-ray screening with a PyTorch ConvNeXtV2 model and Grad-CAM visualization.

This project was built for research, educational demonstration, and portfolio/NCKH purposes.

## Project Overview

Chest X-ray AI Assistant allows users to upload a chest radiograph image and receive:

- Predicted probabilities for 14 CheXpert-style findings
- Ranked top findings
- Risk-level tags based on probability thresholds
- Grad-CAM attention heatmap overlay to highlight regions influencing the prediction

System architecture:

- Backend: FastAPI inference service
- Frontend: Next.js dashboard UI
- Deployment: Docker Compose (backend + frontend)

## Demo Workflow

1. User uploads a JPG/PNG chest X-ray image.
2. Frontend sends multipart request to backend `/predict`.
3. Backend preprocesses image, runs model inference, computes Grad-CAM heatmap.
4. Frontend renders:
	 - Original X-ray
	 - Heatmap overlay with opacity control
	 - Top findings
	 - Full prediction table
	 - Rule-based AI summary

## Model Architecture

- Backbone: `convnextv2_tiny.fcmae_ft_in22k_in1k_384` (timm)
- Framework: PyTorch
- Head: `reset_classifier(num_classes=14)`
- Task type: Multi-label classification
- Inference activation: `sigmoid(logits)`
- Input size: `384 x 384`
- Checkpoint path (default): `backend/weights/best_convnext_model.pth`
- Runtime model path via env: `MODEL_PATH`

## Dataset / Training Summary

The model follows a CheXpert-style 14-label setup:

1. No Finding
2. Enlarged Cardiomediastinum
3. Cardiomegaly
4. Lung Opacity
5. Lung Lesion
6. Edema
7. Consolidation
8. Pneumonia
9. Atelectasis
10. Pneumothorax
11. Pleural Effusion
12. Pleural Other
13. Fracture
14. Support Devices

Training/inference assumptions:

- Loss used in training: BCEWithLogitsLoss (multi-label)
- Validation/inference preprocessing is aligned:
	- PIL image read, RGB conversion
	- CLAHE
	- LongestMaxSize(384)
	- PadIfNeeded to 384x384
	- ImageNet normalization
	- ToTensorV2

## Features

# Chest X-ray AI Assistant

Multi-label chest X-ray screening demo built for research and educational presentation (NCKH style). The system uses a ConvNeXtV2 model to estimate probabilities for 14 CheXpert findings, then visualizes results in an interactive web dashboard with Grad-CAM overlays.

## Project Overview

This project demonstrates an end-to-end AI workflow:

- Chest X-ray image upload from browser
- FastAPI inference service with PyTorch + timm
- Multi-label probability output for 14 findings
- Rule-based summary and risk stratification
- Grad-CAM heatmap generation for model interpretability

The goal is technical demonstration and reproducible deployment, not clinical deployment.

## Demo Workflow

1. User uploads a JPG or PNG chest X-ray image.
2. Frontend sends multipart request to backend endpoint POST /predict.
3. Backend preprocesses image with the validation pipeline.
4. Model predicts 14 label probabilities using sigmoid(logits).
5. Backend computes Grad-CAM heatmap (base64 PNG) from final ConvNeXtV2 stage.
6. Frontend displays:
	 - Top findings
	 - Full 14-label table
	 - AI summary text
	 - Original image + Grad-CAM overlay with opacity slider

## Model Architecture

- Backbone: convnextv2_tiny.fcmae_ft_in22k_in1k_384 (timm)
- Framework: PyTorch
- Task: Multi-label classification
- Number of classes: 14
- Head: reset_classifier(num_classes=14)
- Inference activation: sigmoid
- Input size: 384 x 384

## Dataset and Training Summary

- Label schema follows CheXpert-style 14 findings:
	- No Finding
	- Enlarged Cardiomediastinum
	- Cardiomegaly
	- Lung Opacity
	- Lung Lesion
	- Edema
	- Consolidation
	- Pneumonia
	- Atelectasis
	- Pneumothorax
	- Pleural Effusion
	- Pleural Other
	- Fracture
	- Support Devices
- Checkpoint file location:
	- backend/weights/best_convnext_model.pth

Note: This repository focuses on inference and deployment demo. Full training scripts and experiment logs are out of scope.

## Features

- Upload and preview chest X-ray image
- Real-time inference API integration
- Top-5 findings and full 14-label ranking
- Risk badges (High / Moderate / Low)
- Rule-based AI summary card
- Grad-CAM heatmap overlay
- Heatmap show/hide and opacity control
- Docker-ready backend and frontend services
- Health endpoint with model_loaded status

## Tech Stack

Backend

- Python 3.10
- FastAPI + Uvicorn
- PyTorch + torchvision (CPU build)
- timm
- albumentations + OpenCV + Pillow

Frontend

- Next.js 14
- TypeScript
- Tailwind CSS

Deployment

- Docker
- Docker Compose

## Local Setup

### 1) Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3) Open app

```txt
http://localhost:3000
```

## Docker Setup

Build and run full stack:

```bash
docker compose up --build
```

Environment variables used in Docker:

```txt
NEXT_PUBLIC_API_URL=http://localhost:8000
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
MODEL_PATH=/app/weights/best_convnext_model.pth
```

Important deployment notes:

- Backend image is CPU-focused for PyTorch runtime.
- Model checkpoint is mounted from host via backend/weights volume.
- If checkpoint is missing:
	- GET /health returns model_loaded=false
	- POST /predict returns 503

## API Documentation

### GET /

Response:

```json
{
	"status": "healthy",
	"model_loaded": true
}
```

### GET /health

Response:

```json
{
	"status": "healthy",
	"model_loaded": true
}
```

### POST /predict

Request:

- Content-Type: multipart/form-data
- Field name: file
- Accepted types: image/jpeg, image/jpg, image/png

Response:

```json
{
	"filename": "sample_xray.png",
	"predictions": [
		{
			"label": "Pleural Effusion",
			"probability": 0.82
		}
	],
	"heatmap": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## Screenshots (Placeholder)

Add your screenshots to docs/images and update links below.

- Dashboard overview:
	- ![Dashboard](docs/images/dashboard-overview.png)
- Prediction results + Grad-CAM:
	- ![Prediction](docs/images/prediction-gradcam.png)
- Docker run and health check:
	- ![Docker](docs/images/docker-health.png)

## Limitations

- Demo is not calibrated for clinical risk scoring.
- No patient metadata integration.
- Single-image interactive inference only.
- Explainability is Grad-CAM heuristic and should not be treated as ground-truth lesion localization.
- CPU deployment is slower than GPU inference for large-scale usage.

## Medical Disclaimer

This is a research demo and must not be used as a medical diagnosis.
All outputs require review by qualified radiologists or physicians.

## Future Work

- Add confidence calibration and threshold tuning per label.
- Add report export (PDF/JSON).
- Add batch inference mode.
- Add audit logging and model versioning.
- Add stronger deployment hardening (monitoring, tracing, CI/CD).
- Extend to DICOM workflow and PACS-style integration mock.

## Author

- Name: Your Name
- Organization/Lab: Your Lab or University
- Project type: Research Demo (NCKH)
- Contact: your-email@example.com

### Health Check

```http
GET /health
```

Example response:

```json
{
	"status": "healthy",
	"model_loaded": true
}
```

### Prediction

```http
POST /predict
Content-Type: multipart/form-data
field: file
```

Example response:

```json
{
	"filename": "sample.png",
	"predictions": [
		{
			"label": "Pleural Effusion",
			"probability": 0.82
		}
	],
	"heatmap": "<base64_png_string>"
}
```

## Screenshots (Placeholders)

Add real screenshots to this section when publishing:

- Dashboard overview
- Upload card
- Prediction results table
- Grad-CAM overlay view

Example markdown placeholders:

```md
![Dashboard](docs/screenshots/dashboard.png)
![Prediction Results](docs/screenshots/results.png)
![Grad-CAM Overlay](docs/screenshots/gradcam.png)
```

## Limitations

- Demo scope only; not optimized for hospital-grade throughput or PACS integration.
- Prediction quality depends on training data quality and domain shift.
- Grad-CAM is interpretability support, not causal proof.
- Not validated as a medical device.

## Medical Disclaimer

This project is for research and demonstration purposes only.

It is not a clinical diagnostic system and must not be used to make medical decisions without qualified radiologist/physician review.

## Future Work

- DICOM ingest and metadata handling
- Better report generation and export
- Batch inference mode
- User authentication and study history
- Clinical workflow simulation (PACS/RIS mock integration)
- Extended evaluation metrics and calibration analysis

## Author

Project: Chest X-ray AI Assistant

Maintainer: NCKH project owner

Feel free to open issues or submit improvements for research/demo usage.