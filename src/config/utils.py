import os

from omegaconf import OmegaConf


def load_config(path="src/config/config.yaml"):
    env = os.getenv("MLFLOW_ENV", "local")
    base_cfg = OmegaConf.load(path)

    if env == "docker":
        base_cfg.train.mlflow_tracking_uri = "http://mlflow:5000"
    else:
        base_cfg.train.mlflow_tracking_uri = "http://localhost:5000"

    return base_cfg
