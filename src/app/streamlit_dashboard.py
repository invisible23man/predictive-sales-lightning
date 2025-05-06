import json

import streamlit as st

from src.app.services.forecaster import SalesForecaster
from src.config.utils import load_config

# Load config
cfg = load_config()
forecaster = SalesForecaster(cfg)

# Load normalization stats
with open(cfg.train.normalization_path, "r") as f:
    stats = json.load(f)

# App title
st.title("📈 Predictive Sales Forecasting")

# Instructions
st.markdown(
    "Enter the last **14 days** of sales data below to forecast tomorrow's sales."
)

# Sales input
input_series = st.text_area(
    "Sales (comma-separated):",
    "105.5, 110.2, 98.4, 112.0, 111.5, 115.0, "
    "109.8, 108.5, 102.1, 107.3, 104.0, 100.0, 103.2, 99.9",
)

# Parse input
try:
    series = [float(x.strip()) for x in input_series.split(",")]
    if len(series) != cfg.data.window_size:
        st.error(f"Please enter exactly {cfg.data.window_size} values.")
    else:
        prediction = forecaster.forecast(series)
        st.success(f"📊 Predicted next-day sales: **{prediction:.2f}**")
except Exception as e:
    st.error(f"Error parsing input: {e}")
