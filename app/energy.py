from datetime import datetime, timedelta

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px


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
        model = joblib.load("output/energy_model_lightgbm.pkl")
        return model
    except FileNotFoundError:
        st.error("Energy model not found.")
        return None


def create_features(df):
    df = df.copy()

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hours'] = df.index.hour
    df['weekday'] = df.index.day_name()
    df['weekday_numeric'] = df.index.weekday
    df['is_weekend'] = df.index.weekday > 4
    df['is_peak_hour'] = ((df.index.hour > 8) & (df.index.hour < 22)).astype(int)

    df['hour_sin'] = np.sin( 2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['month_sin'] = np.sin( 2 * np.pi *df['month'] / 12)
    df['month_cos'] = np.cos( 2 * np.pi * df['month'] / 12)

    return df

def manual_input(forecast_horizon):
    st.header("Manual Data Input")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date", value=datetime(2018, 1, 1))
        start_hour = st.selectbox("Start hour", options=range(24))

    with col2:
        base_datetime = datetime.combine(start_date, datetime.min.time()) + timedelta(hours=start_hour)

        hours_to_generate = forecast_horizon + 168

        dates = [base_datetime + timedelta(hours=i) for i in range(hours_to_generate)]

        base_consumption = 1000

        df_input = pd.DataFrame({
            "datetime": dates,
            "y": base_consumption + 500 * np.sin(np.arange(hours_to_generate) * 2  * np.pi / 24),
        })

        df_input.set_index("datetime", inplace=True)

        st.info(f"Generated {len(df_input)} hours of sample date starting from {base_datetime}")



def batch_forecast_model(model):
    st.header("Batch Forecast (24 hours)")

    st.write("Generate prediction for the next 24 hours based on recent consumption pattern")

    st.subheader("Recent historical data")

    col1, col2, col3 = st.columns(3)

    with col1:
        base_consumption = st.number_input("Base Consumption level (kWh)", value=8000, step=100)
    with col2:
        start_date = st.date_input("Forecast start date", value=datetime.now())
    with col3:
        start_hour = st.slider("Forecast start hour", 0, 23, 0)

    if st.button("Generate 24-hour forecast"):
        forecast_start = datetime.combine(start_date, datetime.min.time()) + timedelta(hours=start_hour)

        predictions = []
        timestamps = []

        for i in range(24):
            forecast_time = forecast_start + timedelta(hours=i)
            timestamps.append(forecast_time)

            hour = forecast_start.hour
            weekday = forecast_time.weekday()

            hour_factor = 1.2 if 7 <= hour <= 22 else 0.8
            week_factor = 0.9 if weekday >= 5 else 1.0

            estimated_lag = base_consumption * hour_factor * week_factor

            input_data = pd.DataFrame({
                'lag_1h': [estimated_lag],
                'lag_2h': [estimated_lag],
                'lag_3h': [estimated_lag],
                'lag_24h': [estimated_lag],
                'lag_48h': [estimated_lag],
                'lag_168h': [estimated_lag],
                'rolling_mean_24h': [base_consumption],
                'rolling_mean_168h': [base_consumption],
                'year': [forecast_time.year],
                'month': [forecast_time.month],
                'day': [forecast_time.day],
                'hour': [forecast_time.hour],
                'weekday_numeric': [weekday],
                'is_weekend': [1 if weekday >= 5 else 0],
                'is_peak_hour': [1 if 7 <= hour <= 22 else 0],
                'hour_sin': [np.sin(2 * np.pi * hour / 24)],
                'hour_cos': [np.cos(2 * np.pi * hour / 24)],
                'month_sin': [np.sin(2 * np.pi * forecast_time.month / 12)],
                'month_cos': [np.cos(2 * np.pi * forecast_time.month / 12)]
            })

            prediction = model.predict(input_data)[0]
            predictions.append(prediction)

        result_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Predicted Consumption (kWh)': predictions,
        })

        st.subheader('Forecast results')

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=result_df['Timestamp'],
            y=result_df['Predicted Consumption (kWh)'],
            mode='lines+markers',
            name="Prediction consumption",
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="24-Hour Energy Consumption Forecast",
            xaxis_title="Time",
            yaxis_title="Predicted Consumption (kWh)",
            hovermode="x unified",
            height=500,
        )


        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric('Average', f"{result_df['Predicted Consumption (kWh)'].mean():,.2f} kWh")
        with col2:
            st.metric("Peak", f"{result_df['Predicted Consumption (kWh)'].max():,.2f} kWh")
        with col3:
            st.metric("Minimum", f"{result_df['Predicted Consumption (kWh)'].min():,.2f} kWh")
        with col4:
            st.metric("Total (24h)", f"{result_df['Predicted Consumption (kWh)'].sum():,.2f} kWh")


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

    energy_model = load_energy_modal()
    if energy_model is None:
        st.error("Energy model not found.")

    st.sidebar.title("Navigation")

    mode = st.sidebar.radio(
        "Input Method",
        ["Single Prediction", "Batch Forecast", "Upload Historical Data"]
    )

    if mode == "Single Prediction":
        st.header("Single Prediction")
    elif mode == "Batch Forecast":
        batch_forecast_model(energy_model)
    else:
        st.info("Upload Historical Data")





if __name__ == "__main__":
    main()
