import streamlit as st

from src.app.services.forecaster import SalesForecaster


def show_prediction_page():
    st.title("📊 Forecast Tomorrow's Sales")

    # --- Category dropdown
    category = st.selectbox(
        "Select Product Category", ["Beauty", "Clothing", "Electronics"]
    )

    # --- Load category-specific forecaster
    try:
        forecaster = SalesForecaster.from_category(category)
    except FileNotFoundError:
        st.error(f"Model for category '{category}' not found.")
        return

    # --- Input area
    st.markdown(
        f"Enter the last {forecaster.cfg.data.window_size} "
        f"days of sales for **{category}**:"
    )

    default = (
        "105.5, 110.2, 98.4, 112.0, 111.5, 115.0, 109.8, "
        "108.5, 102.1, 107.3, 104.0, 100.0, 103.2, 99.9"
    )
    input_series = st.text_area("Sales Series", default)

    try:
        series = [float(x.strip()) for x in input_series.split(",")]
        expected = forecaster.cfg.data.window_size
        if len(series) != expected:
            st.warning(f"Please provide exactly {expected} values.")
        else:
            prediction = forecaster.forecast(series)
            st.success(
                f"✅ Predicted next-day sales for **{category}**: **{prediction:.2f}**"
            )
    except Exception as e:
        st.error(f"❗ Parsing error: {e}")
