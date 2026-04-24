from pydantic import BaseModel, field_validator
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib
#import mlflow.pytorch
from pathlib import Path
import torch
import os
import sys

current = os.getcwd()
parent = os.path.dirname(current)
sys.path.append(parent)


from src.training.model import RULPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))



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

#mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow.db"))
ml_models = {}
feature_length = 30

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = RULPredictor(input_size=45, hidden_size=128, num_layers=2, dropout=0.2788)
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "best_model.pt", map_location="cpu"))
    model.eval()
    ml_models["model"] = model
    ml_models["scaler"] = joblib.load(ARTIFACTS_DIR / "scaler_25.joblib")
    yield
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