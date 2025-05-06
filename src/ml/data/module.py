from typing import Optional
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf
from src.ml.data.dataset import SalesDataset
from src.config.schema import DataConfig


class SalesDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg: DataConfig = OmegaConf.to_object(cfg.data)  # Cast to dataclass type

    def setup(self, stage: Optional[str] = None) -> None:
        df = pd.read_csv(self.cfg.csv_path, parse_dates=["Date"])
        df = df[df["Product Category"] == self.cfg.item_id]

        df = df.groupby("Date")["Total Amount"].sum().reset_index()
        df = df.set_index("Date").resample("D").sum().fillna(0)
        series = df["Total Amount"].values

        # Normalize
        self.series_mean = series.mean()
        self.series_std = series.std()
        series = (series - self.series_mean) / (self.series_std + 1e-6)

        # Split
        train_series, val_series = train_test_split(series, test_size=0.2, shuffle=False)
        self.train_dataset = SalesDataset(train_series, self.cfg.window_size)
        self.val_dataset = SalesDataset(val_series, self.cfg.window_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size)
