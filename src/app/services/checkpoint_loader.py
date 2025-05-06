import os
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from src.ml.models.model import CNNLSTMForecastModel


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str = "config/config.yaml"
) -> pl.LightningModule:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    cfg: DictConfig = OmegaConf.load(config_path)
    model = CNNLSTMForecastModel(cfg)
    model = model.load_from_checkpoint(checkpoint_path, cfg=cfg)

    model.eval()
    return model
