import pandas as pd
import plotly.express as px
import streamlit as st

from src.config.utils import load_config


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    return df


def show_explorer_page():
    st.title("📊 Sales Data Explorer")

    cfg = load_config()
    csv_path = cfg.data.csv_path
    df = load_data(csv_path)

    # Sidebar filters
    categories = sorted(df["Product Category"].unique())
    selected = st.sidebar.selectbox(
        "Select Product Category", categories, index=categories.index(cfg.data.item_id)
    )

    df_cat = df[df["Product Category"] == selected]

    # Optional date filter
    min_date, max_date = df_cat["Date"].min(), df_cat["Date"].max()
    date_range = st.sidebar.date_input(
        "Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
    )

    if len(date_range) == 2:
        df_cat = df_cat[
            (df_cat["Date"] >= pd.to_datetime(date_range[0]))
            & (df_cat["Date"] <= pd.to_datetime(date_range[1]))  # noqa: W503
        ]

    # Daily aggregation
    df_daily = df_cat.groupby("Date")["Total Amount"].sum().reset_index()

    # Rolling mean
    window = st.sidebar.slider("Rolling Avg Window", 1, 30, 7)
    df_daily["Rolling Mean"] = df_daily["Total Amount"].rolling(window).mean()

    # Plot
    fig = px.line(
        df_daily,
        x="Date",
        y=["Total Amount", "Rolling Mean"],
        title=f"📈 Sales Over Time – {selected}",
        labels={"value": "Sales", "Date": "Date"},
    )
    st.plotly_chart(fig, use_container_width=True)
