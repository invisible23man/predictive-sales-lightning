from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from src.ml.train.train_category import train_model_for_category

router = APIRouter()


class TrainRequest(BaseModel):
    category: Literal["Beauty", "Clothing", "Electronics"]
    experiment_name: str = "sales-forecast"


@router.post("/train")
def train_endpoint(request: TrainRequest):
    train_model_for_category(
        category=request.category, experiment=request.experiment_name
    )
    return {
        "status": "success",
        "message": f"Training completed for {request.category}",
    }
