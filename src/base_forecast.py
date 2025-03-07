import numpy as np
import pandas as pd

def naive_forecast(df, forecast_column, date_column="date", steps=30):
    df_forecast = df.copy()
    
    df_forecast[date_column] = pd.to_datetime(df_forecast[date_column])
    
    df_forecast["naive_forecast"] = df_forecast[forecast_column].shift(1)
    df_forecast["is_future_forecast"] = False  # Mark historical data

    time_deltas = df_forecast[date_column].diff().dropna()
    most_common_delta = time_deltas.mode()[0] if not time_deltas.empty else pd.Timedelta(days=1)

    last_date = df_forecast[date_column].iloc[-1]
    future_dates = [last_date + most_common_delta * i for i in range(1, steps + 1)]

    future_forecast = pd.DataFrame(index=range(steps), columns=df.columns)
    future_forecast[date_column] = future_dates  # Assign new future dates
    future_forecast[forecast_column] = None  # No real values for the future
    future_forecast["naive_forecast"] = df_forecast[forecast_column].iloc[-1]  # Naïve prediction
    future_forecast["is_future_forecast"] = True  # Mark as future data

    df_final = pd.concat([df_forecast, future_forecast], ignore_index=True)

    return df_final