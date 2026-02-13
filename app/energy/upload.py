from datetime import timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit import title


def create_features(df):
    df = df.copy()

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    # df['weekday'] = df.index.day_name()
    df['weekday_numeric'] = df.index.weekday
    df['is_weekend'] = df.index.weekday > 4
    df['is_peak_hour'] = ((df.index.hour > 8) & (df.index.hour < 22)).astype(int)

    df['hour_sin'] = np.sin( 2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['month_sin'] = np.sin( 2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos( 2 * np.pi * df['month'] / 12)

    return df

def prepare_input_for_prediction(historical_data, forecast_date):

    forecast_df = pd.DataFrame(index=[forecast_date])

    # calculate lag feature from historical data

    try:
        forecast_df['lag_1h'] = historical_data.loc[forecast_date - timedelta(hours=1), "y"]
        forecast_df['lag_2h'] = historical_data.loc[forecast_date - timedelta(hours=2), "y"]
        forecast_df['lag_3h'] = historical_data.loc[forecast_date - timedelta(hours=3), "y"]
        forecast_df['lag_24h'] = historical_data.loc[forecast_date - timedelta(hours=24), "y"]
        forecast_df['lag_48h'] = historical_data.loc[forecast_date - timedelta(hours=48), "y"]
        forecast_df['lag_168h'] = historical_data.loc[forecast_date - timedelta(hours=168), "y"]

        last_24h = historical_data.loc[forecast_date - timedelta(hours=24):forecast_date - timedelta(hours=1), 'y']
        last_168h = historical_data.loc[forecast_date - timedelta(hours=168):forecast_date - timedelta(hours=1), 'y']


        forecast_df['rolling_mean_24h'] = last_24h.mean()
        forecast_df['rolling_mean_168h'] = last_168h.mean()


    except KeyError:
        st.warning("Some historical points are missing. using estimated value")
        average_consumption = historical_data["y"].mean()
        forecast_df['lag_1h'] = average_consumption
        forecast_df['lag_2h'] = average_consumption
        forecast_df['lag_3h'] = average_consumption
        forecast_df['lag_24h'] = average_consumption
        forecast_df['lag_48h'] = average_consumption
        forecast_df['lag_168h'] = average_consumption
        forecast_df['rolling_mean_24h'] = average_consumption
        forecast_df['rolling_mean_168h'] = average_consumption


    forecast_df = create_features(forecast_df)

    return forecast_df


def upload_data(model):

    st.header("Upload Data for prediction")

    st.write("""
    Upload historical energy consumption data (CSV format) with datetime a index and 'y' column.
    This app will generate predictions for future timestamps.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            st.success("File uploaded successfully")

            st.subheader("Data Overview")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total records", len(df))
            with col2:
                st.metric("Date range", f"{df.index.min().date()} to {df.index.max().date()}")
            with col3:
                st.metric("Average Consumption", f"{df['y'].mean():,.2f} kWh")

            st.write("**Sample Data**")
            st.dataframe(df.head(10))


            st.subheader("Generate predictions")

            hours_ahead = st.slider("Hours to forecast ahead", 1, 168, 24)

            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Generating predictions..."):
                    last_date = df.index.max()

                    predictions = []
                    timestamps = []

                    for i in range(1, hours_ahead + 1):
                        forecast_time = last_date + timedelta(hours=i)
                        timestamps.append(forecast_time)

                        try:
                            forecast_df = prepare_input_for_prediction(df, forecast_time)

                            prediction = model.predict(forecast_df)[0]

                            predictions.append(prediction)

                            df.loc[forecast_time, 'y'] = prediction
                        except Exception as e:
                            st.error(f"Error at {forecast_time}: {str(e)}")
                            break

                    result_df = pd.DataFrame({
                        "Timestamp": timestamps,
                        "Predicted Consumption kWh": predictions,
                    })

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=result_df["Timestamp"],
                        y=result_df["Predicted Consumption kWh"],
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#ff7f0e", width=2),
                    ))

                    fig.update_layout(
                        title=f"{hours_ahead} hours Energy Consumption Forecast",
                        xaxis_title="Time",
                        yaxis_title="Predicted Consumption kWh",
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    csv = result_df.to_csv(index=False)

                    st.download_button(
                        label="Download predictions",
                        data=csv,
                        file_name=f'predictions_{hours_ahead}h.csv',
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please ensure your CSV has a datetime index and a column named 'y' with consumption values.")


