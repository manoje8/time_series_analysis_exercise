import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import plotly.graph_objects as go


def batch_forecast_lightBGM_model(model):
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

        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Energy forecast",
            data=csv,
            file_name=f"energy_forecast_{start_hour}.csv",
            mime="text/csv",
        )


def batch_forecast_catboost_model(model):
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

            hour = forecast_time.hour
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

            input_data_copy = input_data.copy()

            for col in input_data_copy.columns:
                input_data_copy[col] = input_data_copy[col].astype(str)


            prediction = model.predict(input_data_copy)
            predictions.append(prediction[0])

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

        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Energy forecast",
            data=csv,
            file_name=f"energy_forecast_{start_hour}.csv",
            mime="text/csv",
        )

