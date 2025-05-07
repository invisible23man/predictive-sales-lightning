import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.app.services.forecaster import SalesForecaster
from src.utils.data_preview import get_last_series_for_category  # NEW


def show_prediction_page():
    st.title("📊 Forecast Tomorrow's Sales")

    # --- Category selector
    category = st.selectbox(
        "Select Product Category", ["Beauty", "Clothing", "Electronics"]
    )

    try:
        forecaster = SalesForecaster.from_category(category)
    except FileNotFoundError:
        st.error(f"Model for category '{category}' not found.")
        return

    expected = forecaster.cfg.data.window_size

    # --- Load recent real sales data
    try:
        default_series = get_last_series_for_category(
            csv_path=forecaster.cfg.data.csv_path,
            category=category,
            window_size=expected,
        )
    except Exception as e:
        st.error(f"Failed to load recent sales data: {e}")
        return

    # --- Show editable table
    df = pd.DataFrame(
        {"Day": [f"T-{i}" for i in range(expected, 0, -1)], "Sales": default_series}
    )
    edited_df = st.data_editor(df, num_rows="fixed", use_container_width=True)
    series = edited_df["Sales"].tolist()

    if len(series) != expected:
        st.warning(f"Please provide exactly {expected} values.")
        return

    try:
        prediction = forecaster.forecast(series)
        st.success(
            f"✅ Predicted next-day sales for **{category}**: **{prediction:.2f}**"
        )

        # --- Plot historical + prediction
        fig, ax = plt.subplots(figsize=(8, 4))
        full_series = series + [prediction]  # extend historical with prediction
        ax.plot(
            range(len(full_series)),
            full_series,
            label="Forecast",
            marker="o",
            color="red",
        )
        ax.axvline(
            x=len(series) - 1, linestyle="--", color="gray", alpha=0.5
        )  # Optional: mark cutoff
        ax.set_title(f"Sales Forecast – {category}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❗ Forecast error: {e}")
