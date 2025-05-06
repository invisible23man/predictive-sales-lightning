import json
import os
import tempfile
from typing import Optional

import pytorch_lightning as pl
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from src.config.utils import load_config
from src.ml.data.module import SalesDataModule
from src.ml.models.model import CNNLSTMForecastModel
from src.ml.train.eval import evaluate_model


def log_mlflow_artifacts(cfg, mlflow_logger, metrics, category, trainer, datamodule):
    """Log metrics, plot, checkpoint, normalization stats, and config to MLflow."""
    mlflow_logger.log_metrics(metrics)

    # Save and log plot
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, tmp.name)

    # Save and log checkpoint
    os.makedirs(os.path.dirname(cfg.train.checkpoint_path), exist_ok=True)
    trainer_ckpt = cfg.train.checkpoint_path
    trainer.save_checkpoint(trainer_ckpt)
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, trainer_ckpt)

    # Save and log normalization stats
    norm_stats = {
        "mean": datamodule.series_mean,
        "std": datamodule.series_std,
    }
    with open(cfg.train.normalization_path, "w") as f:
        json.dump(norm_stats, f)
    mlflow_logger.experiment.log_artifact(
        mlflow_logger.run_id, cfg.train.normalization_path
    )

    # Save and log config
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        OmegaConf.save(cfg, tmp.name)
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, tmp.name)


def train_model_for_category(category: str, experiment: Optional[str] = None):
    cfg = load_config()

    # Update config dynamically
    cfg.data.item_id = category
    cfg.train.checkpoint_path = f"checkpoints/model_{category.lower()}.ckpt"
    cfg.train.normalization_path = f"checkpoints/normalization_{category.lower()}.json"
    if experiment:
        cfg.train.experiment_name = experiment

    logger.info(f"📦 Starting training for category: {category}")

    # Setup
    datamodule = SalesDataModule(cfg)
    model = CNNLSTMForecastModel(cfg)

    run_name = f"{cfg.model.architecture or 'CNNLSTM'}-{category.lower()}"
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.train.experiment_name,
        tracking_uri=cfg.train.mlflow_tracking_uri,
        run_name=run_name,
    )
    mlflow_logger.experiment.set_tags(
        mlflow_logger.run_id,
        {
            "item_id": category,
            "model": cfg.model.architecture or "CNNLSTM",
            "dataset": os.path.basename(cfg.data.csv_path),
            "window_size": str(cfg.data.window_size),
            "learning_rate": str(cfg.model.lr),
        },
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="gpu" if cfg.train.gpus else "cpu",
        devices=cfg.train.gpus or 1,
        deterministic=True,
        logger=mlflow_logger,
    )
    trainer.fit(model, datamodule)
    logger.success(f"✅ Training complete for {category}")

    # Eval & Log
    metrics = evaluate_model(
        model, datamodule.val_dataloader(), plot=True, category=category
    )
    log_mlflow_artifacts(cfg, mlflow_logger, metrics, category, trainer, datamodule)
