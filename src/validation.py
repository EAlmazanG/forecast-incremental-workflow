
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import ttest_rel

from src import utils
from src.utils import print_title


def plot_time_series_forecast(df, time_series, p_alpha=0.9, p_linestyle="--", plot_ci = True, use_ci_scale_factor = None):
    fig, ax = plt.subplots(figsize=(20,6))
    
    colors = plt.get_cmap("tab10")(range(len(time_series)))
    future_mask = df["is_future_forecast"] == True

    if len(time_series) > 1:
        first_future_date = df.loc[future_mask, "date"].min()
    else:
        first_future_date = None
        
    for i, serie in enumerate(time_series):
        alpha = p_alpha if i > 0 else 1
        linestyle = p_linestyle if i > 0 else "-"

        ax.plot(df["date"], df[serie], label=serie, linewidth=2, color=colors[i], alpha=alpha, linestyle=linestyle)

        if i > 0 and plot_ci:
            mean_forecast = df.loc[future_mask, serie]

            past_errors = df.loc[~future_mask, forecast_column] - df.loc[~future_mask, serie]
            std_dev = past_errors.std()  
            
            forecast_horizon = np.arange(1, len(mean_forecast) + 1) 
            
            if use_ci_scale_factor == 'log':
                scale_factor = np.log1p(forecast_horizon)
            elif use_ci_scale_factor == 'sqrt':
                scale_factor = np.sqrt(forecast_horizon)
            else:
                scale_factor = 1

            lower_80 = mean_forecast - 1.28 * std_dev * scale_factor
            upper_80 = mean_forecast + 1.28 * std_dev * scale_factor
            lower_95 = mean_forecast - 1.96 * std_dev * scale_factor
            upper_95 = mean_forecast + 1.96 * std_dev * scale_factor
            
            ax.fill_between(df.loc[future_mask, "date"], lower_95, upper_95, color=colors[i], alpha=0.2, label=f"{serie} 95% CI")
            ax.fill_between(df.loc[future_mask, "date"], lower_80, upper_80, color=colors[i], alpha=0.4, label=f"{serie} 80% CI")


    if first_future_date:
        ax.axvline(first_future_date, color="black", linestyle="-", linewidth=2, alpha = 0.3,  label="Forecast Start")

    ax.set_title("Bikes Rented Forecast", fontsize=14, fontweight="bold")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="y", labelsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=12)

    plt.show()

def mase(y_true, y_pred, y_naive):
    naive_mae = mean_absolute_error(y_true, y_naive)
    model_mae = mean_absolute_error(y_true, y_pred)
    return model_mae / naive_mae if naive_mae != 0 else np.nan

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

def calculate_forecast_metrics(df_naive, df_forecast, to_forecast_column, forecasted_column):
    try:
        df_forecast_clean = df_forecast.drop(columns = 'naive_forecast')
    except:
        df_forecast_clean = df_forecast.copy()
    df_merged = df_forecast_clean.merge(df_naive[["date", 'naive_forecast']], on="date", how="left")
    valid_mask = df_merged[to_forecast_column].notna() & df_merged[forecasted_column].notna() & (df_merged["is_future_forecast"] == False)

    if valid_mask.sum() == 0:
        return {metric: np.nan for metric in ["MAE", "RMSE", "SMAPE", "MASE"]}

    y_true = df_merged.loc[valid_mask, to_forecast_column]
    y_pred = df_merged.loc[valid_mask, forecasted_column]
    y_naive = df_merged.loc[valid_mask, 'naive_forecast']

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "SMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_naive),
    }

    return metrics

def walk_forward_validation(df_naive, df_forecast, to_forecast_column, forecasted_column, steps=30, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    df_historical = df_forecast[df_forecast["is_future_forecast"] == False]

    for train_index, test_index in tscv.split(df_historical):
        train, test = df_historical.iloc[train_index], df_historical.iloc[test_index]

        metrics = calculate_forecast_metrics(df_naive, test, to_forecast_column, forecasted_column)
        results.append(metrics)

    return results

def expanding_window_validation(df_naive, df_forecast, to_forecast_column, forecasted_column, steps=30, initial_train_size=100):
    results = []
    df_historical = df_forecast[df_forecast["is_future_forecast"] == False]
    train_size = initial_train_size

    while train_size < len(df_historical) - steps:
        train, test = df_historical.iloc[:train_size], df_historical.iloc[train_size:train_size + steps]

        metrics = calculate_forecast_metrics(df_naive, test, to_forecast_column, forecasted_column)
        results.append(metrics)
        train_size += steps  # Expand window

    return results

def rolling_window_validation(df_naive, df_forecast, to_forecast_column, forecasted_column, steps=30, window_size=100):
    results = []
    df_historical = df_forecast[df_forecast["is_future_forecast"] == False]
    start = 0

    while start + window_size + steps < len(df_historical):
        train, test = df_historical.iloc[start:start + window_size], df_historical.iloc[start + window_size:start + window_size + steps]

        metrics = calculate_forecast_metrics(df_naive, test, to_forecast_column, forecasted_column)
        results.append(metrics)
        start += steps  # Move the window

    return results

def plot_validation_results(walk_results, expanding_results, rolling_results):
    metrics = ["MAE", "RMSE", "SMAPE", "MASE"]
    results = {
        "Walk-Forward": walk_results,
        "Expanding Window": expanding_results,
        "Rolling Window": rolling_results,
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    for i, (name, result) in enumerate(results.items()):
        df_results = pd.DataFrame(result)
        
        for metric in metrics:
            axes[i].plot(df_results.index, df_results[metric], marker="o", label=metric)
        
        axes[i].set_title(name, fontsize=14, fontweight="bold")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend(loc="upper left")
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)        
    axes[0].set_ylabel("Error Value")
    
    plt.tight_layout()
    plt.show()

def check_forecast_residuals(df_forecast, to_forecast_column, forecasted_column):
    df_residuals = df_forecast.copy()

    df_residuals["residuals"] = df_residuals[to_forecast_column] - df_residuals[forecasted_column]

    mean_residuals = df_residuals["residuals"].mean()
    std_residuals = df_residuals["residuals"].std()
    
    upper_bound = mean_residuals + 1.96 * std_residuals
    lower_bound = mean_residuals - 1.96 * std_residuals

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.plot(df_residuals["date"], df_residuals["residuals"], label="Residuals", color="tab:blue", linewidth=1)
    ax.axhline(0, color="black", linestyle="--", alpha=0.6, label="Zero Line")
    
    ax.axhline(upper_bound, color="red", linestyle="--", alpha=0.7, label="95% CI")
    ax.axhline(lower_bound, color="red", linestyle="--", alpha=0.7)

    ax.set_title("Residuals of Forecast", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    utils.plot_acf_and_pacf(df_residuals["residuals"].dropna(), additional_text=" - Residuals")

def diebold_mariano_test(y_true, y_model, y_baseline):
    errors_model = np.abs(y_true - y_model)
    errors_baseline = np.abs(y_true - y_baseline)
    
    _, p_value = ttest_rel(errors_model, errors_baseline)
    
    return p_value

def compare_forecast_models(df_naive, df_baseline, df_forecast, to_forecast_column, forecasted_column, baseline_column="naive_forecast"):
    model_metrics = calculate_forecast_metrics(df_naive, df_forecast, to_forecast_column, forecasted_column)
    baseline_metrics = calculate_forecast_metrics(df_naive, df_baseline, to_forecast_column, baseline_column)

    df_merged = df_forecast.drop(columns=baseline_column, errors="ignore").merge(
        df_baseline[["date", baseline_column]], on="date", how="left"
    )

    valid_mask = df_merged[to_forecast_column].notna() & df_merged[forecasted_column].notna() & (df_merged["is_future_forecast"] == False)

    if valid_mask.sum() == 0:
        return {metric: np.nan for metric in ["MAE", "RMSE", "SMAPE", "MASE"]}

    y_true = df_merged.loc[valid_mask, to_forecast_column]
    y_pred = df_merged.loc[valid_mask, forecasted_column]
    y_naive = df_merged.loc[valid_mask, baseline_column]

    dm_p_value = diebold_mariano_test(y_true, y_pred, y_naive)

    comparison = {
        metric: {
            "Model": model_metrics[metric],
            "Baseline": baseline_metrics[metric],
            "Improvement": (baseline_metrics[metric] - model_metrics[metric]) / baseline_metrics[metric] * 100 if baseline_metrics[metric] != 0 else np.nan
        }
        for metric in model_metrics
    }

    comparison["DM Test p-value"] = dm_p_value
    comparison["Statistical Significance"] = "Significant" if dm_p_value < 0.05 else "Not Significant"

    return comparison

def format_comparison_results(comparison_results):
    metrics = {k: v for k, v in comparison_results.items() if isinstance(v, dict)}
    
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.loc["DM Test p-value", :] = ["", "", str(comparison_results.get("DM Test p-value", "NaN"))]
    df_metrics.loc["Statistical Significance", :] = ["", "", str(comparison_results.get("Statistical Significance", "Unknown"))]

    return df_metrics

def validate_forecast(df_naive, df_forecast, df_baseline, to_forecast_column, forecasted_column):
    plot_time_series_forecast(df_forecast, [to_forecast_column, forecasted_column], 0.9, '--', True, 'sqrt')

    model_metrics = calculate_forecast_metrics(df_naive, df_forecast, to_forecast_column, forecasted_column)

    walk_results = walk_forward_validation(df_naive, df_forecast, to_forecast_column, forecasted_column, steps=30, n_splits=5)
    expanding_results = expanding_window_validation(df_naive, df_forecast, to_forecast_column, forecasted_column, steps=30, initial_train_size=100)
    rolling_results = rolling_window_validation(df_naive, df_forecast, to_forecast_column, forecasted_column, steps=30, window_size=100)

    plot_validation_results(walk_results, expanding_results, rolling_results)

    check_forecast_residuals(df_forecast, to_forecast_column, forecasted_column)

    comparison_results = compare_forecast_models(df_naive, df_baseline, df_forecast, to_forecast_column, forecasted_column)
    df_formatted = format_comparison_results(comparison_results)
    display(df_formatted)
    return model_metrics, comparison_results