# Chest X-ray AI Demo Web App — Vibe Coding Spec

## 1. Mục tiêu dự án

Xây dựng một web demo cho mô hình AI phân tích ảnh X-quang ngực đã được train bằng PyTorch trên dataset CheXpert.

Ứng dụng cho phép người dùng upload ảnh X-quang ngực, model sẽ trả về xác suất cho 14 nhãn bệnh/phát hiện, sau đó hiển thị kết quả theo dạng dễ hiểu: bảng xác suất, mức nguy cơ, top findings, và cảnh báo y khoa.

> Lưu ý: Đây là web demo phục vụ nghiên cứu/portfolio/sản phẩm mẫu. Không dùng thay thế bác sĩ hoặc hệ thống chẩn đoán lâm sàng.

---

## 2. Model hiện có làm được gì?

Model là bài toán **multi-label chest X-ray classification**.

Input:

- Ảnh X-quang ngực định dạng `.jpg`, `.jpeg`, `.png`
- Ảnh được convert sang RGB
- Preprocess về kích thước `384x384`
- Normalize theo ImageNet mean/std

Output:

- 14 xác suất tương ứng với 14 nhãn CheXpert
- Mỗi nhãn độc lập, dùng sigmoid thay vì softmax
- Một ảnh có thể có nhiều bệnh/phát hiện cùng lúc

Danh sách nhãn:

```python
LABEL_COLS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]
```

Model architecture:

- Backbone: `convnextv2_tiny.fcmae_ft_in22k_in1k_384`
- Library: `timm`
- Framework: PyTorch
- Classifier head: 14 outputs
- Loss khi train: `BCEWithLogitsLoss`
- Inference: `torch.sigmoid(logits)`

File weight dự kiến:

```txt
weights/best_convnext_model.pth
```

---

## 3. Tính năng MVP cần làm

### 3.1 Upload ảnh

Người dùng có thể:

- Kéo thả ảnh X-quang
- Chọn ảnh từ máy
- Xem preview ảnh vừa upload
- Bấm nút `Analyze X-ray`

### 3.2 Inference

Backend cần:

- Nhận file ảnh
- Kiểm tra định dạng file
- Preprocess ảnh giống validation pipeline lúc train
- Load model ConvNeXtV2
- Predict 14 xác suất
- Trả JSON cho frontend

### 3.3 Hiển thị kết quả

Frontend cần hiển thị:

- Ảnh gốc người dùng upload
- Top 3 findings có xác suất cao nhất
- Bảng đầy đủ 14 labels
- Probability dạng phần trăm
- Risk level:
  - `High`: probability >= 0.70
  - `Medium`: 0.40 <= probability < 0.70
  - `Low`: probability < 0.40

### 3.4 Medical disclaimer

Luôn hiển thị cảnh báo:

```txt
This AI result is for demonstration and research purposes only. It is not a medical diagnosis and must be reviewed by a qualified radiologist or physician.
```

---

## 4. Tech stack đề xuất

Làm bản demo theo hướng production-lite:

### Backend

- Python 3.10+
- FastAPI
- PyTorch
- timm
- albumentations
- OpenCV
- Pillow
- Uvicorn

### Frontend

- Next.js 14+
- TypeScript
- Tailwind CSS
- shadcn/ui nếu muốn UI đẹp nhanh

### Deploy

Ưu tiên theo thứ tự:

1. Local Docker demo
2. Hugging Face Spaces nếu muốn dễ share
3. Render/Fly.io/VPS nếu muốn backend API riêng

---

## 5. Cấu trúc repo chuẩn

```txt
chest_xray_demo/
├── README.md
├── .gitignore
├── docker-compose.yml
├── backend/
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── labels.py
│   │   ├── model.py
│   │   ├── preprocess.py
│   │   ├── schemas.py
│   │   └── inference.py
│   └── weights/
│       └── best_convnext_model.pth
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── app/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── UploadBox.tsx
│   │   ├── ResultTable.tsx
│   │   ├── TopFindings.tsx
│   │   └── Disclaimer.tsx
│   └── lib/
│       └── api.ts
└── docs/
    ├── model-card.md
    └── demo-flow.md
```

---

## 6. Backend implementation spec

### 6.1 `backend/app/labels.py`

```python
LABEL_COLS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]
```

### 6.2 `backend/app/config.py`

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = BASE_DIR / "weights" / "best_convnext_model.pth"
MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
IMAGE_SIZE = 384
DEVICE = "cuda"  # fallback to cpu in code if cuda unavailable
```

### 6.3 `backend/app/model.py`

```python
import torch
import torch.nn as nn
import timm
from app.config import MODEL_NAME, WEIGHTS_PATH
from app.labels import LABEL_COLS


class ConvNextModel(nn.Module):
    def __init__(self, num_classes: int = 14, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=pretrained)
        self.backbone.reset_classifier(num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_model(device: str):
    model = ConvNextModel(num_classes=len(LABEL_COLS), pretrained=False)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
```

### 6.4 `backend/app/preprocess.py`

Validation pipeline phải giống lúc train:

```python
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from app.config import IMAGE_SIZE


val_transform = A.Compose([
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
])


def preprocess_image(file_bytes: bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image_np = np.array(image)
    transformed = val_transform(image=image_np)
    tensor = transformed["image"].unsqueeze(0)
    return tensor
```

Sửa lỗi cần nhớ: file này phải import `io`.

```python
import io
```

### 6.5 `backend/app/inference.py`

```python
import torch
from app.labels import LABEL_COLS


def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def predict_image(model, image_tensor, device: str):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    results = []
    for label, prob in zip(LABEL_COLS, probs):
        probability = float(prob)
        results.append({
            "label": label,
            "probability": probability,
            "percentage": round(probability * 100, 2),
            "risk": get_risk_level(probability),
        })

    results = sorted(results, key=lambda x: x["probability"], reverse=True)

    return {
        "top_findings": results[:3],
        "all_findings": results,
        "disclaimer": "This AI result is for demonstration and research purposes only. It is not a medical diagnosis and must be reviewed by a qualified radiologist or physician."
    }
```

### 6.6 `backend/app/main.py`

```python
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import DEVICE
from app.model import load_model
from app.preprocess import preprocess_image
from app.inference import predict_image

app = FastAPI(title="Chest X-ray AI Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime_device = "cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
model = load_model(runtime_device)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": runtime_device,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")

    file_bytes = await file.read()

    try:
        image_tensor = preprocess_image(file_bytes)
        result = predict_image(model, image_tensor, runtime_device)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 7. Frontend implementation spec

### 7.1 UI style

Phong cách:

- Medical clean UI
- Nền sáng
- Card trắng
- Màu nhấn xanh dương hoặc cyan
- Có icon y tế nếu muốn
- Không làm quá màu mè

### 7.2 Main page behavior

Trang chính cần có:

- Header: `Chest X-ray AI Assistant`
- Subtext: `Upload a chest X-ray image to detect possible radiological findings.`
- Upload box
- Preview image
- Analyze button
- Loading state
- Error state
- Result section

### 7.3 API client `frontend/lib/api.ts`

```typescript
export type Finding = {
  label: string;
  probability: number;
  percentage: number;
  risk: "High" | "Medium" | "Low";
};

export type PredictionResponse = {
  top_findings: Finding[];
  all_findings: Finding[];
  disclaimer: string;
};

export async function predictXray(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Prediction failed");
  }

  return response.json();
}
```

### 7.4 Risk badge logic

```typescript
export function getRiskClass(risk: string) {
  if (risk === "High") return "bg-red-100 text-red-700 border-red-200";
  if (risk === "Medium") return "bg-yellow-100 text-yellow-700 border-yellow-200";
  return "bg-green-100 text-green-700 border-green-200";
}
```

### 7.5 Result table

Bảng gồm:

| Finding | Probability | Risk |
|---|---:|---|
| Pleural Effusion | 82.13% | High |

Sort theo probability giảm dần.

---

## 8. README.md cần có trong repo

Nội dung README chính nên gồm:

```md
# Chest X-ray AI Assistant

A demo web application for multi-label chest X-ray classification using a ConvNeXtV2 model trained on CheXpert-style labels.

## Features

- Upload chest X-ray image
- Run AI inference with PyTorch model
- Display top findings
- Show probability and risk level for 14 radiological labels
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
```

---

## 9. `backend/requirements.txt`

```txt
fastapi
uvicorn[standard]
python-multipart
pillow
numpy
opencv-python-headless
albumentations
scikit-learn
torch
torchvision
timm
```

Nếu deploy CPU, chú ý PyTorch package có thể cần cài riêng theo môi trường deploy.

---

## 10. `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/
.env

# Node
node_modules/
.next/
out/

# Model weights
*.pth
*.pt
*.onnx

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

Lưu ý: Không commit file weight lớn lên GitHub nếu repo public. Có thể dùng Git LFS hoặc upload weight riêng.

---

## 11. Docker backend

### `backend/Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY weights ./weights

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`

```yaml
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/weights:/app/weights

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
```

---

## 12. Các lỗi dễ gặp và cách tránh

### 12.1 Sai preprocess

Nếu preprocess lúc inference khác preprocess lúc validation, kết quả sẽ lệch.

Phải giữ:

- CLAHE
- LongestMaxSize 384
- PadIfNeeded 384
- Normalize ImageNet
- ToTensorV2

### 12.2 Sai activation

Không dùng softmax.

Phải dùng:

```python
torch.sigmoid(logits)
```

Vì đây là multi-label classification.

### 12.3 Không load đúng model architecture

Khi load weight `.pth`, architecture phải giống lúc train:

```python
timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384')
reset_classifier(14)
```

### 12.4 File weight quá lớn

Không nên push trực tiếp lên GitHub public.

Giải pháp:

- Git LFS
- Hugging Face Model Hub
- Google Drive private link
- S3/R2 storage

### 12.5 Medical risk

Không dùng wording như:

```txt
You have pneumonia.
```

Nên dùng:

```txt
The model detected a high probability of Pneumonia-like finding. Please consult a qualified physician.
```

---

## 13. Roadmap nâng cấp sau MVP

### Phase 1: MVP

- Upload ảnh
- Predict 14 labels
- Hiển thị bảng kết quả
- Deploy local hoặc Hugging Face Spaces

### Phase 2: Better demo

- Thêm Grad-CAM heatmap
- Cho phép download report PDF
- Lưu lịch sử inference local
- Thêm sample X-ray images

### Phase 3: Product-like demo

- Login user
- Dashboard bệnh nhân
- Report management
- DICOM support
- PACS/RIS integration mock
- Role: technician, radiologist, admin

---

## 14. Prompt dùng để vibe code trong Cursor/Claude Code

Copy prompt dưới đây vào công cụ vibe coding:

```txt
You are a senior full-stack AI engineer. Build a complete demo web app for chest X-ray AI inference.

Context:
I have a PyTorch ConvNeXtV2 Tiny model trained for multi-label chest X-ray classification with 14 CheXpert labels. The model weight file will be placed at backend/weights/best_convnext_model.pth.

Model details:
- Architecture: timm model `convnextv2_tiny.fcmae_ft_in22k_in1k_384`
- Classifier output: 14 labels
- Inference activation: sigmoid, not softmax
- Input image size: 384x384
- Preprocessing must match validation pipeline:
  - Convert image to RGB
  - CLAHE clip_limit=2.0, tile_grid_size=(8,8)
  - LongestMaxSize(max_size=384)
  - PadIfNeeded(min_height=384, min_width=384, border_mode=cv2.BORDER_CONSTANT, fill=0)
  - Normalize mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)
  - ToTensorV2

Labels:
[
  'No Finding',
  'Enlarged Cardiomediastinum',
  'Cardiomegaly',
  'Lung Opacity',
  'Lung Lesion',
  'Edema',
  'Consolidation',
  'Pneumonia',
  'Atelectasis',
  'Pneumothorax',
  'Pleural Effusion',
  'Pleural Other',
  'Fracture',
  'Support Devices'
]

Build:
1. Backend with FastAPI:
   - POST /predict accepts JPG/PNG image as multipart form-data field `file`
   - Load the PyTorch model once on startup
   - Use cuda if available, otherwise cpu
   - Return JSON with top_findings, all_findings, probability, percentage, risk level
   - Risk rules: High >= 0.70, Medium >= 0.40, Low < 0.40
   - Add /health endpoint
   - Add CORS for frontend local dev

2. Frontend with Next.js + TypeScript + Tailwind:
   - Clean medical-style landing page
   - Upload box with drag and drop
   - Image preview
   - Analyze button
   - Loading state
   - Error state
   - Top 3 findings cards
   - Full result table sorted by probability descending
   - Medical disclaimer always visible

3. Repo structure:
   - backend/app/main.py
   - backend/app/model.py
   - backend/app/preprocess.py
   - backend/app/inference.py
   - backend/app/labels.py
   - backend/app/config.py
   - frontend/app/page.tsx
   - frontend/components/UploadBox.tsx
   - frontend/components/ResultTable.tsx
   - frontend/components/TopFindings.tsx
   - frontend/components/Disclaimer.tsx
   - frontend/lib/api.ts

4. Include:
   - backend/requirements.txt
   - .gitignore
   - README.md with setup instructions

Important:
- Do not use softmax.
- Do not claim medical diagnosis.
- Make code runnable locally.
- Keep UI simple, polished, and demo-ready.
```

---

## 15. Acceptance criteria

Dự án được xem là hoàn chỉnh khi:

- Chạy backend không lỗi
- Chạy frontend không lỗi
- Upload ảnh JPG/PNG thành công
- API trả về đủ 14 labels
- Xác suất nằm trong khoảng 0 đến 1
- Frontend hiển thị top findings
- Frontend hiển thị bảng đầy đủ
- Có disclaimer y khoa
- README đủ hướng dẫn setup

---

## 16. Câu lệnh khởi tạo repo

```bash
mkdir chest-xray-ai-demo
cd chest-xray-ai-demo

git init
mkdir -p backend/app backend/weights frontend docs

touch README.md .gitignore docker-compose.yml

touch backend/requirements.txt backend/Dockerfile

touch backend/app/__init__.py \
      backend/app/main.py \
      backend/app/config.py \
      backend/app/labels.py \
      backend/app/model.py \
      backend/app/preprocess.py \
      backend/app/inference.py \
      backend/app/schemas.py
```

Sau đó tạo frontend:

```bash
npx create-next-app@latest frontend --typescript --tailwind --eslint --app
```

Copy file weight vào:

```bash
cp best_convnext_model.pth backend/weights/best_convnext_model.pth
```

Chạy backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Chạy frontend:

```bash
cd frontend
npm install
npm run dev
```

Mở:

```txt
http://localhost:3000
```
