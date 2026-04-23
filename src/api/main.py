from pydantic import BaseModel, field_validator
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib
import mlflow.pytorch
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SensorWindow(BaseModel):
    sensor_readings: list[list[float]]
    
    @field_validator("sensor_readings")
    def check_shape(cls, v):
        arr = np.array(v)
        if arr.shape != (30, 45):
            raise ValueError(f"Expected shape (30, 45), got {arr.shape}")
        return v
    

class PredictionResponse(BaseModel):
    predicted_rul: float
    failure_probability: float

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
ml_models = {}
feature_length = 30

@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs on startup
    ml_models["model"] = mlflow.pytorch.load_model("models:/rul-predictor/Production")
    ml_models["model"].to("cpu")
    ml_models["model"].eval()
    ml_models["scaler"] = joblib.load(PROJECT_ROOT / "artifacts" / f"scaler_{feature_length}.joblib")
    #ml_models["fe"] = FeatureEngineer.load(PROJECT_ROOT / "artifacts" / f"feature_engineer_{feature_length}.joblib")

    yield
    # runs on shutdown (cleanup)
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict(window: SensorWindow):
    data = np.array(window.sensor_readings)
    #features = ml_models["fe"].transform(data)
    data_scaled = ml_models["scaler"].transform(data).reshape(1, 30, 45)
    tensor = torch.tensor(data_scaled, dtype=torch.float32)
    with torch.no_grad():
        predicted_rul = ml_models["model"](tensor).item()

    failure_probability = float(np.exp(-predicted_rul / 50))
    return PredictionResponse(predicted_rul=predicted_rul, failure_probability=failure_probability)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "rul-predictor",
        "stage": "Production",
        "feature_length": feature_length
    }