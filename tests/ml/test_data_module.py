import os

import pytest
from omegaconf import OmegaConf

from src.config.schema import DataConfig
from src.ml.data.module import SalesDataModule


@pytest.mark.skipif(
    not os.path.exists("data/raw/sales_data.csv"),
    reason="Missing data/raw/sales_data.csv",
)
def test_sales_dataloader_shapes():
    data_cfg = DataConfig(
        available_categories=["Beauty"],
        csv_path="data/raw/sales_data.csv",
        item_id="Beauty",
        window_size=14,
        batch_size=4,
    )
    cfg = OmegaConf.create({"data": data_cfg})

    module = SalesDataModule(cfg)
    module.setup()

    batch = next(iter(module.train_dataloader()))
    x, y = batch

    assert x.shape[1] == 14
    assert x.shape[0] == 4
    assert y.shape[0] == 4
