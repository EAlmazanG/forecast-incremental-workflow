import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

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

def random_walk_forecast(df, forecast_column, date_column="date", steps=30, drift=True):
    df_forecast = df.copy()
    df_forecast[date_column] = pd.to_datetime(df_forecast[date_column])

    df_forecast["random_walk_forecast"] = np.nan
    df_forecast["is_future_forecast"] = False  

    time_deltas = df_forecast[date_column].diff().dropna()
    most_common_delta = time_deltas.mode()[0] if not time_deltas.empty else pd.Timedelta(days=1)

    last_date = df_forecast[date_column].iloc[-1]
    future_dates = [last_date + most_common_delta * i for i in range(1, steps + 1)]

    std_dev = df_forecast[forecast_column].diff().std()
    mean_drift = df_forecast[forecast_column].diff().mean() if drift else 0

    df_forecast["random_walk_forecast"] = df_forecast[forecast_column].shift(1) + mean_drift + np.random.normal(0, std_dev, size=len(df_forecast))

    future_noise = np.random.normal(0, std_dev, size=steps)
    future_forecast = pd.DataFrame(index=range(steps), columns=df.columns)
    future_forecast[date_column] = future_dates  
    future_forecast[forecast_column] = None  
    future_forecast["random_walk_forecast"] = df_forecast[forecast_column].iloc[-1] + np.cumsum(mean_drift + future_noise)
    future_forecast["is_future_forecast"] = True  

    df_final = pd.concat([df_forecast, future_forecast], ignore_index=True)

    return df_final

def exponential_smoothing_forecast(transformed_df, forecast_column, steps=60):
    df = transformed_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    ts = df[forecast_column]
    model = SimpleExpSmoothing(ts, initialization_method="estimated").fit()

    fitted_values = model.fittedvalues
    df['exponential_smoothing_forecast'] = fitted_values
    df.iloc[0, df.columns.get_loc('exponential_smoothing_forecast')] = float('nan')

    df['is_future_forecast'] = False

    future_forecast = model.forecast(steps)
    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]

    future_df = pd.DataFrame({
        'date': future_dates,
        forecast_column: [pd.NA] * steps,
        'exponential_smoothing_forecast': future_forecast,
        'is_future_forecast': [True] * steps
    })

    for col in transformed_df.columns:
        if col not in future_df.columns:
            future_df[col] = pd.NA

    cols = list(transformed_df.columns) + ['exponential_smoothing_forecast', 'is_future_forecast']
    future_df = future_df[cols]
    result_df = pd.concat([df, future_df], ignore_index=True)
    return result_df


def holt_forecast(transformed_df, forecast_column, steps=60):
    df = transformed_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    ts = df[forecast_column]
    model = Holt(ts, initialization_method="estimated").fit()

    fitted_values = model.fittedvalues
    df['holt_forecast'] = fitted_values
    df.iloc[0, df.columns.get_loc('holt_forecast')] = float('nan')

    df['is_future_forecast'] = False

    future_forecast = model.forecast(steps)
    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]

    future_df = pd.DataFrame({
        'date': future_dates,
        forecast_column: [pd.NA] * steps,
        'holt_forecast': future_forecast,
        'is_future_forecast': [True] * steps
    })

    for col in transformed_df.columns:
        if col not in future_df.columns:
            future_df[col] = pd.NA

    cols = list(transformed_df.columns) + ['holt_forecast', 'is_future_forecast']
    future_df = future_df[cols]

    result_df = pd.concat([df, future_df], ignore_index=True)
    return result_df

def holt_winters_forecast(transformed_df, forecast_column, seasonal='add', trend='add', seasonal_periods= 30, steps=60):
    df = transformed_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    ts = df[forecast_column]
    model = ExponentialSmoothing(ts, seasonal=seasonal, trend=trend, seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
    fitted_values = model.fittedvalues
    df['holt_winters_forecast'] = fitted_values
    df.iloc[0, df.columns.get_loc('holt_winters_forecast')] = float('nan')
    df['is_future_forecast'] = False

    future_forecast = model.forecast(steps)
    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]
    future_df = pd.DataFrame({
        'date': future_dates,
        forecast_column: [pd.NA] * steps,
        'holt_winters_forecast': future_forecast,
        'is_future_forecast': [True] * steps
    })
    for col in transformed_df.columns:
        if col not in future_df.columns:
            future_df[col] = pd.NA

    cols = list(transformed_df.columns) + ['holt_winters_forecast', 'is_future_forecast']
    future_df = future_df[cols]
    result_df = pd.concat([df, future_df], ignore_index=True)
    return result_df

def moving_average_forecast(transformed_df, forecast_column, steps=60, window=7):
    df = transformed_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    df['moving_average_forecast'] = df[forecast_column].rolling(window=window).mean().shift(1)
    df.iloc[0, df.columns.get_loc('moving_average_forecast')] = float('nan')
    df['is_future_forecast'] = False

    # Iterative forecast: update window with forecasted values
    series = df[forecast_column].tolist()
    future_forecasts = []
    for _ in range(steps):
        forecast_val = sum(series[-window:]) / window
        future_forecasts.append(forecast_val)
        series.append(forecast_val)

    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]
    future_df = pd.DataFrame({
        'date': future_dates,
        forecast_column: [pd.NA] * steps,
        'moving_average_forecast': future_forecasts,
        'is_future_forecast': [True] * steps
    })

    for col in transformed_df.columns:
        if col not in future_df.columns:
            future_df[col] = pd.NA

    cols = list(transformed_df.columns) + ['moving_average_forecast', 'is_future_forecast']
    future_df = future_df[cols]
    result_df = pd.concat([df, future_df], ignore_index=True)
    return result_df