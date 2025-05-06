from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from src.config.utils import load_config
from src.ml.train.train_category import train_model_for_category

router = APIRouter()
cfg = load_config()
CATEGORIES = cfg.data.available_categories


class TrainRequest(BaseModel):
    categories: List[str]
    experiment_name: str = "sales-forecast"


@router.post("/train")
def train_endpoint(request: TrainRequest):
    for category in request.categories:
        if category not in CATEGORIES:
            return {"status": "error", "message": f"Invalid category: {category}"}
        train_model_for_category(category=category, experiment=request.experiment_name)
    return {"status": "success", "trained": request.categories}


@router.post("/train_all")
def train_all_endpoint():
    categories = load_config().data.available_categories  # dynamic from config
    trained = []
    for category in categories:
        train_model_for_category(category)
        trained.append(category)
    return {"status": "success", "trained": trained}
