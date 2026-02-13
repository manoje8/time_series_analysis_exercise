## Time Series Analysis Exercise

This repository demonstrates various forecasting techniques 
ranging from classical statistical methods to modern machine 
learning approaches, with a focus on energy consumption data.

### Exercise List
1. Air passenger
2. [Energy Forecasting](https://energy-consumption-forecasting-app.streamlit.app)


---

### [Air Passenger](rosetta_stone)

Classical time series methods for single-variable prediction:

| Model                        | MAE   | RMSE   | MAPE   |
| ---------------------------- | ----- | ------ | ------ |
| Simple Exponential Smoothing | 99.80 | 123.63 | 18.65% |
| Double Exponential Smoothing | 92.04 | 115.18 | 17.19% |
| Triple Exponential Smoothing | 10.61 | 16.44  | 2.26%  |
| ARIMA/SARIMA (Model 1)       | 40.18 | 55.65  | 7.76%  |
| ARIMA/SARIMA (Model 2)       | 81.85 | 93.27  | 15.93% |

### [Multivariate Energy Forecasting ](multivariate_energy_forecasting)

Machine learning approaches leveraging multiple features:

|Model|MAE|RMSE|MAPE|
|---|---|---|---|
|**LightGBM**|64.01|83.27|0.76%|
|**CatBoost**|199.51|260.77|2.40%|
|**LSTM**|5724.11|5907.37|65.51%|
