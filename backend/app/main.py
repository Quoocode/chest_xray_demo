from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

from app.config import ALLOWED_ORIGINS, DEVICE
from app.gradcam import generate_gradcam_heatmap
from app.inference import predict_image
from app.labels import LABEL_COLS
from app.model import load_model
from app.preprocess import preprocess_image
from app.schemas import HealthResponse, PredictionItem, PredictionResponse

app = FastAPI(title="Chest X-ray AI Demo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime_device = DEVICE
runtime_model_bundle = load_model(runtime_device)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(status="healthy", model_loaded=runtime_model_bundle.model_loaded)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", model_loaded=runtime_model_bundle.model_loaded)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not runtime_model_bundle.model_loaded or runtime_model_bundle.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image_tensor = preprocess_image(file_bytes)
        predictions = predict_image(runtime_model_bundle.model, image_tensor, runtime_device)
        target_label = predictions[0]["label"]
        target_class_idx = LABEL_COLS.index(target_label)
        heatmap = generate_gradcam_heatmap(
            runtime_model_bundle.model,
            image_tensor,
            file_bytes,
            target_class_idx,
            runtime_device,
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        filename=file.filename or "upload",
        predictions=[PredictionItem(**prediction) for prediction in predictions],
        heatmap=heatmap,
    )
