import pytest
from omegaconf import OmegaConf
from src.ml.data.module import SalesDataModule
from src.ml.models.model import CNNLSTMForecastModel
import pytorch_lightning as pl


@pytest.mark.slow
def test_training_loop_runs():
    # Fast config override for test
    cfg = OmegaConf.create({
        "data": {
            "csv_path": "data/raw/sales_data.csv",
            "item_id": "Beauty",
            "window_size": 14,
            "batch_size": 4,
        },
        "model": {
            "input_size": 1,
            "conv_channels": 16,
            "lstm_hidden": 32,
            "lstm_layers": 1,
            "lr": 0.001,
        },
        "train": {
            "max_epochs": 1,
            "gpus": 0,
            "mlflow_tracking_uri": None,
            "experiment_name": None,
        }
    })

    model = CNNLSTMForecastModel(cfg)
    datamodule = SalesDataModule(cfg)

    trainer = pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, datamodule)
