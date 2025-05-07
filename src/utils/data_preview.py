import pandas as pd


def get_last_series_for_category(
    csv_path: str, category: str, window_size: int = 14
) -> list[float]:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df[df["Product Category"] == category]
    df = df.groupby("Date")["Total Amount"].sum().reset_index()
    df = df.set_index("Date").resample("D").sum().fillna(0)
    series = df["Total Amount"].values
    return series[-window_size:].tolist()
