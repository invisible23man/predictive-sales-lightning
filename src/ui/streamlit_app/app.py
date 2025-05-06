import streamlit as st

from src.ui.streamlit_app.pages.explorer import show_explorer_page
from src.ui.streamlit_app.pages.predictions import show_prediction_page
from src.ui.streamlit_app.pages.train_models import show_training_page

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="centered")
st.sidebar.title("🔍 Navigation")

page = st.sidebar.selectbox(
    "Choose a page", ["📊 Data Explorer", "📈 Predict Sales", "🛠️ Train Models"]
)

if page == "📊 Data Explorer":
    show_explorer_page()
elif page == "📈 Predict Sales":
    show_prediction_page()
elif page == "🛠️ Train Models":
    show_training_page()
