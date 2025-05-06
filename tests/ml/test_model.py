import torch
from omegaconf import OmegaConf

from src.config.schema import ModelConfig
from src.ml.models.model import CNNLSTMForecastModel


def test_cnn_lstm_forward_pass():
    # Create fake config
    cfg = OmegaConf.create(
        {
            "model": ModelConfig(
                input_size=1, conv_channels=32, lstm_hidden=64, lstm_layers=1, lr=0.001
            )
        }
    )

    model = CNNLSTMForecastModel(cfg)
    dummy_input = torch.randn(4, 14)  # batch_size = 4, window_size = 14
    output = model(dummy_input)
    assert output.shape == (4,), f"Expected output shape (4,), got {output.shape}"
