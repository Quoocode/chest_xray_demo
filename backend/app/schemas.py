from pydantic import BaseModel


class PredictionItem(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    filename: str
    predictions: list[PredictionItem]
    heatmap: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

