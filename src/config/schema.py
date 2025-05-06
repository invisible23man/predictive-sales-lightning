from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    csv_path: str
    item_id: str
    window_size: int
    batch_size: int


@dataclass
class ModelConfig:
    input_size: int = 1
    conv_channels: int = 32
    lstm_hidden: int = 64
    lstm_layers: int = 1
    lr: float = 0.001


@dataclass
class TrainConfig:
    max_epochs: int
    gpus: int
    seed: int
    mlflow_tracking_uri: Optional[str]
    experiment_name: Optional[str]
    checkpoint_path: str
