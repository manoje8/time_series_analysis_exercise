from datetime import datetime, timedelta

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from models import batch_forecast_catboost_model, batch_forecast_lightBGM_model
from upload import upload_data

st.set_page_config(
    page_title="Energy Consumption Forecaster",
    layout="wide",
    page_icon="?"
)
st.title("Energy Consumption Forecasting App")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_energy_modal():
    try:
        light_lgm_model = joblib.load("output/energy_model_lightgbm.pkl")
        catboost_model = joblib.load("output/energy_model_catboost.pkl")
        return light_lgm_model, catboost_model
    except FileNotFoundError:
        st.error("Energy model not found.")
        return None




def main():

    with st.expander("Model Performance Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MAE", "64.01 kWh")
        with col2:
            st.metric("RMSE", "83.27 kWh")
        with col3:
            st.metric("MAPE", "0.76 %")

        st.info("LightGBM model significantly outperformed LSTM (MAPE: 65.51 %) and CatBoost (MAPE: 2.40%)")

    energy_model, catboost_model = load_energy_modal()
    if energy_model is None or catboost_model is None:
        st.error("Energy model not found.")

    st.sidebar.title("Navigation")

    mode = st.sidebar.radio(
        "Input Method",
        ["Single Prediction", "Batch Forecast", "Upload Historical Data"]
    )

    option = st.selectbox("Select Model Type", ["LightGBM", "CatBoost"])

    if mode == "Single Prediction":
        st.header("Single Prediction")
    elif mode == "Batch Forecast":

        if option == "LightGBM":
            batch_forecast_lightBGM_model(energy_model)
        elif option == "CatBoost":
            batch_forecast_catboost_model(catboost_model)
    else:
        upload_data(energy_model)



if __name__ == "__main__":
    main()
