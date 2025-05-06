from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import torch

from src.app.services.checkpoint_loader import load_model_from_checkpoint

router = APIRouter()


class PredictRequest(BaseModel):
    series: List[float] = Field(..., example=[100.0, 120.5, 130.2])


class PredictResponse(BaseModel):
    predicted_sales: float


# Load model once at startup
model = load_model_from_checkpoint("./checkpoints/model.ckpt")


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        x = torch.tensor(request.series).float().unsqueeze(0)  # [1, T]
        y_hat = model(x)
        return PredictResponse(predicted_sales=y_hat.item())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
