from typing import List

import numpy as np
import torch
from omegaconf import DictConfig

from src.ml.models.model import CNNLSTMForecastModel


class SalesForecaster:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = CNNLSTMForecastModel.load_from_checkpoint(
            cfg.train.checkpoint_path, cfg=cfg
        )
        self.model.eval()

        # These should ideally be stored and loaded, not hardcoded
        self.series_mean = 0.0
        self.series_std = 1.0

    def set_normalization_stats(self, mean: float, std: float):
        self.series_mean = mean
        self.series_std = std

    def forecast(self, series: List[float]) -> float:
        if len(series) != self.cfg.data.window_size:
            raise ValueError(
                f"Expected input of length {self.cfg.data.window_size}, "
                "got {len(series)}"
            )

        # Normalize input
        arr = np.array(series)
        norm_series = (arr - self.series_mean) / (self.series_std + 1e-6)

        x = torch.tensor(norm_series).float().unsqueeze(0)  # [1, T]
        with torch.no_grad():
            y_hat = self.model(x)

        # Denormalize output
        return float(y_hat.item() * self.series_std + self.series_mean)
