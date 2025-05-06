import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.config.schema import ModelConfig


class CNNLSTMForecastModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg: ModelConfig = ModelConfig(**cfg.model)

        self.conv1 = nn.Conv1d(
            in_channels=self.cfg.input_size,
            out_channels=self.cfg.conv_channels,
            kernel_size=3,
        )
        self.lstm = nn.LSTM(
            input_size=self.cfg.conv_channels,
            hidden_size=self.cfg.lstm_hidden,
            num_layers=self.cfg.lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(self.cfg.lstm_hidden, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B, 1, T]
        x = torch.relu(self.conv1(x))  # [B, C, T']
        x = x.permute(0, 2, 1)  # [B, T', C]
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # [B, 1]
        return out.squeeze(1)  # [B]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
