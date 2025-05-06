import os

from omegaconf import OmegaConf


def load_config(path="src/config/config.yaml"):
    base_cfg = OmegaConf.load(path)

    env = os.getenv("MLFLOW_ENV", "local")
    base_cfg.env = env  # optionally store env in config

    # Map environment to URIs
    mlflow_env_uri = {
        "docker": "http://mlflow:5005",
        "local": "http://localhost:5005",
    }

    api_env_uri = {
        "docker": "http://api:8000/api",
        "local": "http://localhost:8000/api",
    }

    # Override based on environment
    base_cfg.train.mlflow_tracking_uri = mlflow_env_uri.get(
        env, base_cfg.train.mlflow_tracking_uri
    )
    base_cfg.app.api_url = api_env_uri.get(env, base_cfg.app.api_url)

    return base_cfg
