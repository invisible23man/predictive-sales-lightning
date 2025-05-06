import json
import os
import tempfile

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.loggers import MLFlowLogger

from src.config.utils import load_config
from src.ml.data.module import SalesDataModule
from src.ml.models.model import CNNLSTMForecastModel
from src.ml.train.eval import evaluate_model


def main():
    """
    Main training script for the CNN-LSTM time-series forecaster.
    - Loads configuration
    - Trains the model with PyTorch Lightning
    - Logs metrics to MLflow
    - Saves normalization stats and model checkpoint
    """
    cfg = load_config()
    logger.info("✅ Loaded configuration")

    # Initialize components
    datamodule = SalesDataModule(cfg)
    model = CNNLSTMForecastModel(cfg)

    # MLflow Logger
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.train.experiment_name,
        tracking_uri=cfg.train.mlflow_tracking_uri,
    )

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="gpu" if cfg.train.gpus else "cpu",
        devices=cfg.train.gpus or 1,
        deterministic=True,
        logger=mlflow_logger,
    )

    logger.info("🚀 Starting training...")
    trainer.fit(model, datamodule)
    logger.success("🏁 Training complete")

    # Evaluate and log metrics
    metrics = evaluate_model(model, datamodule.val_dataloader(), plot=True)
    mlflow_logger.log_metrics(metrics)

    # Save prediction plot to MLflow
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        from matplotlib import pyplot as plt

        plt.savefig(tmp.name)
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, tmp.name)
        logger.info(f"📊 Saved evaluation plot to MLflow: {tmp.name}")

    # Save checkpoint
    os.makedirs(os.path.dirname(cfg.train.checkpoint_path), exist_ok=True)
    trainer.save_checkpoint(cfg.train.checkpoint_path)
    logger.success(f"✅ Model checkpoint saved: {cfg.train.checkpoint_path}")

    # Save normalization stats
    norm_stats = {
        "mean": datamodule.series_mean,
        "std": datamodule.series_std,
    }
    norm_path = cfg.train.normalization_path
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)

    with open(norm_path, "w") as f:
        json.dump(norm_stats, f)
    logger.success(f"📈 Normalization stats saved to {norm_path}")


if __name__ == "__main__":
    main()
