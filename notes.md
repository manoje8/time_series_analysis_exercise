### Basic Frequency String Aliases


```
'D'        # Calendar day
'B'        # Business day
'W'        # Weekly
'W-MON'    # Weekly on Monday
'W-TUE'    # Weekly on Tuesday
'MS'       # Month start
'M'        # Month end (deprecated, use 'ME')
'ME'       # Month end
'BMS'      # Business month start
'BM'       # Business month end (deprecated, use 'BME')
'BME'      # Business month end
'QS'       # Quarter start
'QE'       # Quarter end
'QS-JAN'   # Quarter starting in January
'AS'       # Year start
'AE'       # Year end (deprecated, use 'YE')
'YE'       # Year end
'H'        # Hourly
'T' or 'min'  # Minutely
'S'           # Secondly
'L' or 'ms'   # Milliseconds
'U' or 'us'   # Microseconds
'N'           # Nanoseconds
```

| Seasonality | Period            |
| ----------- | ----------------- |
| Daily       | `24`              |
| Weekly      | `24 * 7 = 168`    |
| Yearly      | `24 * 365 = 8760` |


**Use Prophet if:**

```
Data has missing months
You’ll later add holidays/events
You need business-friendly explainability
You’re building dashboards or reports
```

[rnn-vs-lstm-vs-gru-vs-transformers
](https://www.geeksforgeeks.org/deep-learning/rnn-vs-lstm-vs-gru-vs-transformers/)


**A Simple Moving Average (SMA)** is a common technical indicator that smooths price data 
by calculating the average closing price of an asset over a specific period (e.g., 10, 50, 200 days), 
making it easier to identify trends and generate buy/sell signals by spotting crossovers 
between short-term and long-term SMAs.

_Calculate a 4-day simple moving average (SMA)_

`
df['SMA_4'] = df['values'].rolling(window=4).mean()`