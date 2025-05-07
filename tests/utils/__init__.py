import tempfile
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.utils.data_preview import get_last_series_for_category


@pytest.fixture
def sample_csv():
    """Create a temporary CSV with 30 days of data for 2 categories."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")

    dates = [datetime.today() - timedelta(days=i) for i in range(30)]
    data = []

    for date in dates:
        data.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "Product Category": "Beauty",
                "Total Amount": 100 + date.day,
            }
        )
        data.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "Product Category": "Clothing",
                "Total Amount": 200 + date.day,
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(temp_file.name, index=False)
    return temp_file.name


def test_get_last_series_for_category(sample_csv):
    series = get_last_series_for_category(sample_csv, category="Beauty", window_size=14)
    assert isinstance(series, list)
    assert len(series) == 14
    assert all(isinstance(val, (int, float)) for val in series)

    series2 = get_last_series_for_category(
        sample_csv, category="Clothing", window_size=7
    )
    assert len(series2) == 7
    assert all(val > 200 for val in series2)
