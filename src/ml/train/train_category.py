import json
import os
import tempfile
from typing import Optional

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.loggers import MLFlowLogger

from src.config.utils import load_config
from src.ml.data.module import SalesDataModule
from src.ml.models.model import CNNLSTMForecastModel
from src.ml.train.eval import evaluate_model


def train_model_for_category(category: str, experiment: Optional[str] = None):
    cfg = load_config()

    # Update config dynamically
    cfg.data.item_id = category
    checkpoint_name = f"model_{category.lower()}.ckpt"
    norm_name = f"normalization_{category.lower()}.json"

    cfg.train.checkpoint_path = f"checkpoints/{checkpoint_name}"
    cfg.train.normalization_path = f"checkpoints/{norm_name}"
    if experiment:
        cfg.train.experiment_name = experiment

    logger.info(f"📦 Starting training for category: {category}")

    # Setup components
    datamodule = SalesDataModule(cfg)
    model = CNNLSTMForecastModel(cfg)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.train.experiment_name,
        tracking_uri=cfg.train.mlflow_tracking_uri,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="gpu" if cfg.train.gpus else "cpu",
        devices=cfg.train.gpus or 1,
        deterministic=True,
        logger=mlflow_logger,
    )

    trainer.fit(model, datamodule)
    logger.success(f"✅ Training complete for {category}")

    # Evaluation and logging
    metrics = evaluate_model(
        model, datamodule.val_dataloader(), plot=True, category=category
    )
    mlflow_logger.log_metrics(metrics)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        from matplotlib import pyplot as plt

        plt.savefig(tmp.name)
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, tmp.name)

    # Save model and stats
    os.makedirs(os.path.dirname(cfg.train.checkpoint_path), exist_ok=True)
    trainer.save_checkpoint(cfg.train.checkpoint_path)
    logger.success(f"📍 Saved checkpoint: {cfg.train.checkpoint_path}")

    norm_stats = {
        "mean": datamodule.series_mean,
        "std": datamodule.series_std,
    }
    with open(cfg.train.normalization_path, "w") as f:
        json.dump(norm_stats, f)

    logger.success(f"📈 Saved normalization stats: {cfg.train.normalization_path}")
