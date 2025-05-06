import streamlit as st

from src.ui.streamlit_app.pages.predictions import show_prediction_page

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="centered")
st.sidebar.title("🔍 Navigation")

page = st.sidebar.selectbox("Choose a page", ["📈 Predict Sales"])

if page == "📈 Predict Sales":
    show_prediction_page()
