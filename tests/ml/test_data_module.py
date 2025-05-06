import os
from src.ml.data.module import SalesDataModule


def test_sales_dataloader_shapes():
    path = "data/raw/sales_data.csv"
    assert os.path.exists(path), "Missing sales_data.csv under data/raw"

    module = SalesDataModule(csv_path=path, item_id="Beauty", window_size=14, batch_size=4)
    module.setup()

    batch = next(iter(module.train_dataloader()))
    x, y = batch

    assert x.shape[1] == 14
    assert x.shape[0] == 4
    assert y.shape[0] == 4
