import os

from omegaconf import OmegaConf


def load_config(path="src/config/config.yaml"):
    base_cfg = OmegaConf.load(path)

    mlflow_env_uri = {"docker": "http://mlflow:5005", "local": "http://localhost:5005"}

    env = os.getenv("MLFLOW_ENV", "local")
    override_uri = mlflow_env_uri.get(env, base_cfg.train.mlflow_tracking_uri)

    base_cfg.train.mlflow_tracking_uri = override_uri
    return base_cfg
