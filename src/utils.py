
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

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


def print_title(title, line_length = 60, symbol = '-'):
    separator = symbol * ((line_length - len(title) - 2) // 2)
    print(f"{separator} {title} {separator}")

def format_columns(df, datetime_columns=[], int64_columns=[], float64_columns=[], str_columns=[]):
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in int64_columns:
        if col in df.columns:
            df[col] = df[col].astype('Int64')  

    for col in float64_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')

    for col in str_columns:
        if col in df.columns:
            df[col] = df[col].astype('str')
    
    return df

def detect_outliers(df, method="iqr", threshold=1.5):
    df_outliers = pd.DataFrame(index=df.index)
    outlier_columns = []
    
    if method in ["iqr", "both"]:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = (df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))
        df_outliers["iqr_outlier"] = iqr_outliers.any(axis=1)
        outlier_columns.append(iqr_outliers)
    else:
         df_outliers["iqr_outlier"] = False
    
    if method in ["zscore", "both"]:
        mean = df.mean()
        std = df.std()
        z_scores = (df - mean) / std
        zscore_outliers = (np.abs(z_scores) > threshold)
        df_outliers["zscore_outlier"] = zscore_outliers.any(axis=1)
        outlier_columns.append(zscore_outliers)
    else:
        df_outliers["zscore_outlier"] = False
    
    if outlier_columns:
        combined_outliers = pd.DataFrame(np.logical_or.reduce(outlier_columns), index=df.index, columns=df.columns)
        df_outliers["outlier_columns"] = combined_outliers.apply(lambda row: list(row.index[row]), axis=1)
        df_outliers["is_outlier"] = df_outliers.apply(lambda row: row['zscore_outlier'] or row['iqr_outlier'], axis=1)

    return df_outliers

def check_transformations(df_selection):
    numeric_vars = df_selection.select_dtypes(include=["number"]).columns
    transformations_needed = {}

    for col in numeric_vars:
        result = {}

        p_value = adfuller(df_selection[col].dropna())[1]
        result["stationary"] = p_value < 0.05  # True = Stationary, False = Not Stationary
        
        if not result["stationary"]:  # If not stationary
            if df_selection[col].min() > 0:  # Check if all values are positive
                boxcox_lambda = boxcox(df_selection[col] + 1)[1]  # +1 to avoid log(0) errors
                if abs(boxcox_lambda - 1) > 0.1:
                    result["recommended_transformation"] = "boxcox"
                else:
                    result["recommended_transformation"] = "log"
            else:
                result["recommended_transformation"] = "diff"
        else:
            result["recommended_transformation"] = "none"
        
        transformations_needed[col] = result

    transformations_df = pd.DataFrame(transformations_needed).T
    print(transformations_df)
    return transformations_df
        
def apply_transformations(df_selection, transformations):
    df_transformed = df_selection.copy()
    for col, row in transformations.iterrows():
        transformation = row["recommended_transformation"]
        
        if transformation == "diff":
            df_transformed[col] = df_selection[col].diff().dropna()
        elif transformation == "log":
            df_transformed[col] = np.log1p(df_selection[col])  # log1p avoids log(0) issues
        elif transformation == "boxcox":
            df_transformed[col], _ = boxcox(df_selection[col] + 1)  # +1 to handle zero values

    print("Transformations applied successfully.")
    display(df_transformed.head())
    return df_transformed

def plot_acf_and_pacf(input_df, additional_text = ''):
    fig, axes = plt.subplots(1, 2, figsize=(16, 3))

    plot_acf(input_df, ax=axes[0], lags=50)
    plot_pacf(input_df, ax=axes[1], lags=50)

    axes[0].set_title("ACF" + additional_text, fontsize=14, fontweight='bold')
    axes[1].set_title("PACF" + additional_text, fontsize=14, fontweight='bold')

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_time_series(df, time_series, p_alpha = 0.9, p_linestyle = "--"):
    fig, ax = plt.subplots(figsize=(20,6))
    
    colors = plt.get_cmap("tab10")(range(len(time_series)))
    for i, serie in enumerate(time_series):
        alpha = p_alpha if i > 0 else 1
        linestyle = p_linestyle if i > 0 else "-"
        
        ax.plot(df['date'], df[serie], label=serie, linewidth=2, color=colors[i], alpha=alpha, linestyle=linestyle)

    ax.set_title("Bikes rented", fontsize=14, fontweight='bold')

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis='y', labelsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=12)

    plt.show()

def check_transformations(df_selection):
    numeric_vars = df_selection.select_dtypes(include=["number"]).columns
    transformations_needed = {}

    for col in numeric_vars:
        result = {}

        p_value = adfuller(df_selection[col].dropna())[1]
        result["stationary"] = p_value < 0.05  # True = Stationary, False = Not Stationary
        
        if not result["stationary"]:  # If not stationary
            if df_selection[col].min() > 0:  # Check if all values are positive
                boxcox_lambda = boxcox(df_selection[col] + 1)[1]  # +1 to avoid log(0) errors
                if abs(boxcox_lambda - 1) > 0.1:
                    result["recommended_transformation"] = "boxcox"
                else:
                    result["recommended_transformation"] = "log"
            else:
                result["recommended_transformation"] = "diff"
        else:
            result["recommended_transformation"] = "none"
        
        transformations_needed[col] = result

    transformations_df = pd.DataFrame(transformations_needed).T
    print(transformations_df)
    return transformations_df
        

def apply_transformations(df_selection, transformations):
    df_transformed = df_selection.copy()
    for col, row in transformations.iterrows():
        transformation = row["recommended_transformation"]
        
        if transformation == "diff":
            df_transformed[col] = df_selection[col].diff().dropna()
        elif transformation == "log":
            df_transformed[col] = np.log1p(df_selection[col])  # log1p avoids log(0) issues
        elif transformation == "boxcox":
            df_transformed[col], _ = boxcox(df_selection[col] + 1)  # +1 to handle zero values

    print("Transformations applied successfully.")
    display(df_transformed.head())
    return df_transformed