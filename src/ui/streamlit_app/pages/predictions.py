import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.app.services.forecaster import SalesForecaster


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

    # --- Editable input table
    default_series = [105.5 + i for i in range(expected)]
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
        ax.plot(range(expected), series, label="Historical", marker="o")
        ax.plot(expected, prediction, label="Prediction", marker="x", color="red")
        ax.set_title(f"Sales Forecast – {category}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❗ Forecast error: {e}")
