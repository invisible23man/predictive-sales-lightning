import requests
import streamlit as st
from omegaconf import OmegaConf

from src.config.utils import load_config

cfg = load_config()
API_URL = cfg.app.api_url


def show_training_page():
    st.title("🛠️ Train Sales Forecasting Models")

    st.markdown("Select categories to train:")

    categories = OmegaConf.to_container(cfg.data.available_categories, resolve=True)
    selected = st.multiselect("Product Categories", categories, default=categories)

    experiment_name = st.text_input("Experiment Name", value="sales-forecast")

    if st.button("🚀 Start Training"):
        with st.spinner("Training in progress..."):
            response = requests.post(
                f"{API_URL}/train",
                json={"categories": selected, "experiment_name": experiment_name},
            )
            if response.status_code == 200:
                st.success(f"✅ Trained: {', '.join(response.json()['trained'])}")
            else:
                st.error(f"❌ Error: {response.text}")
