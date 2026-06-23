# Chest X-ray AI Assistant

A demo web application for multi-label chest X-ray classification using a ConvNeXtV2 model trained on CheXpert-style labels.

## Features

- Upload chest X-ray image
- Prepare AI inference with FastAPI backend
- Display top findings and full 14-label table
- Show probability and risk level for each finding
- Medical disclaimer included

## Model

- Architecture: ConvNeXtV2 Tiny
- Input size: 384x384
- Task: Multi-label classification
- Output: 14 CheXpert labels

## Run backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```txt
http://localhost:3000
```

## API

```txt
POST /predict
Content-Type: multipart/form-data
field: file
```

## Disclaimer

This project is for demonstration and research purposes only. It is not intended for clinical diagnosis.