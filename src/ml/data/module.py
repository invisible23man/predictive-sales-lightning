import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.ml.data.dataset import SalesDataset
from sklearn.model_selection import train_test_split


class SalesDataModule(LightningDataModule):
    def __init__(self, csv_path, item_id='Beauty', window_size=14, batch_size=32):
        super().__init__()
        self.csv_path = csv_path
        self.item_id = item_id
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path, parse_dates=['Date'])

        # Group and aggregate
        df = df[df['Product Category'] == self.item_id]
        df = df.groupby('Date')['Total Amount'].sum().reset_index()
        df = df.set_index('Date').resample('D').sum().fillna(0)
        series = df['Total Amount'].values

        # Normalize (optional: can improve LSTM training)
        self.series_mean = series.mean()
        self.series_std = series.std()
        series = (series - self.series_mean) / (self.series_std + 1e-6)

        # Train-test split
        train_series, val_series = train_test_split(series, test_size=0.2, shuffle=False)

        self.train_dataset = SalesDataset(train_series, self.window_size)
        self.val_dataset = SalesDataset(val_series, self.window_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
