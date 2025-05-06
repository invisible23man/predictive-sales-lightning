from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.app.services.forecaster import SalesForecaster
from src.config.utils import load_config

router = APIRouter()

cfg = load_config()


class PredictRequest(BaseModel):
    series: List[float] = Field(
        ..., example=[float(i * 10) for i in range(cfg.data.window_size)]
    )

    @field_validator("series")
    @classmethod
    def check_length(cls, v: List[float]) -> List[float]:
        expected = cfg.data.window_size
        if len(v) != expected:
            raise ValueError(f"Series must contain exactly {expected} values")
        return v


class PredictResponse(BaseModel):
    predicted_sales: float


forecaster = SalesForecaster(cfg)


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        forecast = forecaster.forecast(request.series)
        return PredictResponse(predicted_sales=forecast)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/{category}", response_model=PredictResponse)
def predict_by_category(category: str, request: PredictRequest):
    try:
        forecaster = SalesForecaster.from_category(category)
        forecast = forecaster.forecast(request.series)
        return PredictResponse(predicted_sales=forecast)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model or stats for category '{category}' not found.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
