
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.api import VAR
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

def vecm_forecast(selected_exog_df, transformed_endog_df, to_forecast_column, vecm_rank):
    endog_columns = transformed_endog_df.drop(columns=['date', to_forecast_column]).columns.tolist()
    selected_exog_df = selected_exog_df.merge(transformed_endog_df[to_forecast_column], left_index=True, right_index=True, how='inner').drop(columns = [to_forecast_column])
    vecm_model = VECM(transformed_endog_df[endog_columns + [to_forecast_column]], k_ar_diff=1, coint_rank= vecm_rank, exog=selected_exog_df)
    vecm_fit = vecm_model.fit()

    vecm_forecast = vecm_fit.predict(steps=len(transformed_endog_df), exog_fc=selected_exog_df)

    vecm_forecast_df = pd.DataFrame(vecm_forecast, columns=endog_columns + [to_forecast_column])
    vecm_forecast_df = transformed_endog_df.copy().merge(vecm_forecast_df[[to_forecast_column]].rename(columns={to_forecast_column:"vecm_forecast"}), left_index=True, right_index=True, how='inner')
    vecm_forecast_df['is_future_forecast'] = False
    return vecm_forecast_df

def var_forecast(selected_exog_df, transformed_endog_df, to_forecast_column, maxlags = 1):
    endog_columns = transformed_endog_df.drop(columns=['date', to_forecast_column]).columns.tolist()
    selected_exog_df = selected_exog_df.merge(transformed_endog_df[to_forecast_column], 
                                            left_index=True, right_index=True, how='inner').drop(columns=[to_forecast_column])
    varx_model = VAR(endog=transformed_endog_df[endog_columns + [to_forecast_column]], exog=selected_exog_df)
    varx_fit = varx_model.fit(maxlags=maxlags)
    varx_forecast = varx_fit.forecast(y=transformed_endog_df[endog_columns + [to_forecast_column]].values, 
                                    steps=len(transformed_endog_df), exog_future=selected_exog_df.values)

    varx_forecast_df = pd.DataFrame(varx_forecast, columns=endog_columns + [to_forecast_column])

    varx_forecast_df = transformed_endog_df.copy().merge(varx_forecast_df[[to_forecast_column]].rename(columns={to_forecast_column: "varx_forecast"}), 
                                                        left_index=True, right_index=True, how='inner')
    varx_forecast_df['is_future_forecast'] = False
    return varx_forecast_df

def sarimax_forecast(selected_exog_df, transformed_endog_df, to_forecast_column, order, seasonal_order):
    selected_exog_df = selected_exog_df.merge(transformed_endog_df[to_forecast_column], 
                                            left_index=True, right_index=True, how='inner').drop(columns=[to_forecast_column])

    sarimax_model = SARIMAX(transformed_endog_df[to_forecast_column], 
                            exog=selected_exog_df, 
                            order=order, 
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False, 
                            enforce_invertibility=False)

    sarimax_fit = sarimax_model.fit()
    sarimax_forecast = sarimax_fit.predict(start=0, 
                                        end=len(transformed_endog_df)-1, 
                                        exog=selected_exog_df)

    sarimax_forecast_df = transformed_endog_df.copy()
    sarimax_forecast_df["sarimax_forecast"] = sarimax_forecast
    sarimax_forecast_df['is_future_forecast'] = False
    return sarimax_forecast_df