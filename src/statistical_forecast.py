
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def AR_forecast(transformed_df, to_forecast_column, steps=60, lags=7):
    df = transformed_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    ts = df[to_forecast_column]
    model = AutoReg(ts, lags=lags, old_names=False).fit()
    # Prepend NaNs for the initial lags
    fitted_values = [float('nan')] * lags + list(model.fittedvalues)
    df['AR_forecast'] = fitted_values
    df['is_future_forecast'] = False

    future_forecast = model.forecast(steps=steps)
    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]
    future_df = pd.DataFrame({
        'date': future_dates,
        to_forecast_column: [pd.NA] * steps,
        'AR_forecast': future_forecast,
        'is_future_forecast': [True] * steps
    })

    for col in transformed_df.columns:
        if col not in future_df.columns:
            future_df[col] = pd.NA

    cols = list(transformed_df.columns) + ['AR_forecast', 'is_future_forecast']
    future_df = future_df[cols]
    result_df = pd.concat([df, future_df], ignore_index=True)
    return result_df

def arima_forecast(transformed_df, to_forecast_column, steps=60, order=(1, 1, 1)):
    df = transformed_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    ts = df[to_forecast_column]
    model = ARIMA(ts, order=order).fit()
    fitted = model.predict(start=0, end=len(ts)-1)
    df['arima_forecast'] = np.nan
    df.loc[df.index[:len(fitted)], 'arima_forecast'] = fitted.values
    df['is_future_forecast'] = False

    future_forecast = model.forecast(steps=steps)
    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]
    future_df = pd.DataFrame({
        'date': future_dates,
        to_forecast_column: [np.nan] * steps,
        'arima_forecast': future_forecast,
        'is_future_forecast': [True] * steps
    })

    for col in transformed_df.columns:
        if col not in future_df.columns:
            future_df[col] = np.nan

    cols = list(transformed_df.columns) + ['arima_forecast', 'is_future_forecast']
    future_df = future_df[cols]
    result_df = pd.concat([df, future_df], ignore_index=True)
    return result_df

def sarima_forecast(transformed_df, to_forecast_column, steps=60, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    df = transformed_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    ts = df[to_forecast_column]
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order).fit(disp=False)
    fitted = model.predict(start=0, end=len(ts) - 1)
    df['sarima_forecast'] = np.nan
    df.loc[df.index[:len(fitted)], 'sarima_forecast'] = fitted.values
    df['is_future_forecast'] = False

    future_forecast = model.forecast(steps=steps)
    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]
    future_df = pd.DataFrame({
        'date': future_dates,
        to_forecast_column: [np.nan] * steps,
        'sarima_forecast': future_forecast,
        'is_future_forecast': [True] * steps
    })

    for col in transformed_df.columns:
        if col not in future_df.columns:
            future_df[col] = np.nan

    cols = list(transformed_df.columns) + ['sarima_forecast', 'is_future_forecast']
    future_df = future_df[cols]
    result_df = pd.concat([df, future_df], ignore_index=True)
    return result_df