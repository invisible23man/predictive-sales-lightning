import json
import os
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig

from src.ml.models.model import CNNLSTMForecastModel


class SalesForecaster:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Load trained model
        self.model = CNNLSTMForecastModel.load_from_checkpoint(
            cfg.train.checkpoint_path, cfg=cfg
        )
        self.model.eval()

        # Load normalization stats
        norm_path = cfg.train.normalization_path
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Normalization stats not found at {norm_path}")
        with open(norm_path, "r") as f:
            stats = json.load(f)

        self.series_mean = stats["mean"]
        self.series_std = stats["std"]

    def forecast(self, series: List[float]) -> float:
        expected_length = self.cfg.data.window_size
        if len(series) != expected_length:
            raise ValueError(
                f"Expected input of length {expected_length}, got {len(series)}"
            )

        # Normalize input
        arr = np.array(series)
        norm_series = (arr - self.series_mean) / (self.series_std + 1e-6)

        x = torch.tensor(norm_series).float().unsqueeze(0)  # [1, T]
        with torch.no_grad():
            y_hat = self.model(x)

        # Denormalize output
        return float(y_hat.item() * self.series_std + self.series_mean)
