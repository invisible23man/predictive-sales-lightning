from dataclasses import dataclass

@dataclass
class DataConfig:
    csv_path: str
    item_id: str
    window_size: int
    batch_size: int
