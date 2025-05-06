import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from src.config.utils import load_config
from src.ml.data.module import SalesDataModule
from src.ml.models.model import CNNLSTMForecastModel


def main():
    cfg = load_config()

    datamodule = SalesDataModule(cfg)
    model = CNNLSTMForecastModel(cfg)

    # Optional MLflow logging
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


if __name__ == "__main__":
    main()
