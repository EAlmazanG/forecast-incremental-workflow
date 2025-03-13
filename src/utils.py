
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
from sklearn.feature_selection import VarianceThreshold

def print_title(title, line_length = 60, symbol = '-'):
    separator = symbol * ((line_length - len(title) - 2) // 2)
    print(f"{separator} {title} {separator}")

def format_columns(df, datetime_columns=[], int64_columns=[], float64_columns=[], str_columns=[]):
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in int64_columns:
        if col in df.columns:
            df[col] = df[col].astype('int64')  

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

        # Augmented Dickey-Fuller test for stationarity
        p_value = adfuller(df_selection[col].dropna())[1]
        result["stationary"] = p_value < 0.05  # True = Stationary, False = Not Stationary

        if not result["stationary"]:  # If not stationary
            if (df_selection[col] > 0).all():  # Ensure all values are positive for Box-Cox & Log
                _, boxcox_lambda = boxcox(df_selection[col] + 1)
                
                if abs(boxcox_lambda - 1) > 0.1:
                    result["recommended_transformation"] = "boxcox"
                else:
                    result["recommended_transformation"] = "log"
            else:
                result["recommended_transformation"] = "diff"  # Differencing for non-positive values
        else:
            result["recommended_transformation"] = "none"  # No transformation needed

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
    return df_transformed

def calculate_vif(df):
    return pd.Series([variance_inflation_factor(df.values, i) for i in range(df.shape[1])], index=df.columns)

def filter_relevant_features(df, threshold_corr=0.9, threshold_vif=10, threshold_var=1e-6):
    removed_features = {"low_variance": [], "high_correlation": [], "high_vif": []}

    # Remove low variance features
    selector = VarianceThreshold(threshold_var)
    selector.fit(df)
    low_var_cols = df.columns[~selector.get_support()]
    if low_var_cols.any():
        print(f"Removing low variance features: {list(low_var_cols)}")
        df = df.drop(columns=low_var_cols)
        removed_features["low_variance"].extend(low_var_cols)
    else:
        print("No low variance features removed.")
    
    # Remove highly correlated features
    corr_matrix = df.corr().abs()
    while True:
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        max_corr = upper_tri.max().max()

        if max_corr < threshold_corr:
            break
        most_corr_pair = np.where(upper_tri == max_corr)
        col1, col2 = upper_tri.columns[most_corr_pair[1][0]], upper_tri.index[most_corr_pair[0][0]]

        drop_col = col1 if corr_matrix[col1].mean() > corr_matrix[col2].mean() else col2
        
        print(f"Removing highly correlated feature: {drop_col}")
        df = df.drop(columns=[drop_col])
        removed_features["high_correlation"].append(drop_col)

        corr_matrix = df.corr().abs()

    # Remove high VIF features iteratively
    vif_data = calculate_vif(df)
    while vif_data.max() > threshold_vif:
        max_vif_col = vif_data.idxmax()
        print(f"Removing high VIF feature: {max_vif_col} (VIF={vif_data[max_vif_col]:.2f})")
        df = df.drop(columns=[max_vif_col])
        removed_features["high_vif"].append(max_vif_col)
        vif_data = calculate_vif(df)

    return df, removed_features

def add_lag_and_ma_features(df, lag_features, lags=None, ma_windows=None):
    transformed_df = df.copy()
    created_features = []

    if lags:
        for lag in lags:
            for feature in lag_features:
                col_name = f"{feature}_lag{lag}"
                transformed_df[col_name] = transformed_df[feature].shift(lag)
                created_features.append(col_name)

    if ma_windows:
        for ma in ma_windows:
            for feature in lag_features:
                col_name = f"{feature}_ma{ma}"
                transformed_df[col_name] = transformed_df[feature].shift(1).rolling(window=ma).mean()
                created_features.append(col_name)

    return transformed_df, created_features

def analyze_cointegration(trace_stat, critical_values, eigenvectors, columns, significance_level=0.05):
    significance_index = {0.10: 0, 0.05: 1, 0.01: 2}[significance_level]
    
    num_cointegrations = 0
    for i in range(len(trace_stat)):
        if trace_stat[i] > critical_values[i, significance_index]:
            num_cointegrations += 1
        else:
            break
    
    result = {
        "significance_level": significance_level,
        "num_cointegrations": num_cointegrations,
        "cointegration_detected": num_cointegrations > 0,
        "trace_statistic": trace_stat.tolist(),
        "critical_values": critical_values[:, significance_index].tolist(),
        "vecm_rank": num_cointegrations
    }

    print("\n===== Johansen Cointegration Test =====")
    print(f"Cointegration Relations Detected: {num_cointegrations}")
    print(f"Cointegration Present?: {'✅ Yes' if num_cointegrations > 0 else '❌ No'}\n")
    print("Detailed Results:")
    
    for i, (ts, cv) in enumerate(zip(trace_stat, critical_values[:, significance_index])):
        status = "✅ Reject H0 (Cointegration)" if ts > cv else "❌ Do not reject H0"
        print(f"r ≤ {i}: Trace Statistic = {ts:.2f}, Critical Value ({significance_level*100}%) = {cv:.2f} → {status}")

    if num_cointegrations > 0:
        # Get the eigenvectors corresponding to cointegration relationships
        cointegration_matrix = eigenvectors[:, :num_cointegrations]
        
        # Compute the absolute contributions of each feature to the cointegration vectors
        cointegration_contributions = pd.DataFrame(
            np.abs(cointegration_matrix),
            index=columns,
            columns=[f'Cointegration_{i+1}' for i in range(num_cointegrations)]
        )
        
        # Rank features by their mean contribution across all cointegrating relationships
        important_features = cointegration_contributions.mean(axis=1).sort_values(ascending=False)
        selected_features = important_features.index.tolist()

        print("\n🔹 Features contributing most to cointegration:")
        print(important_features)

        result["cointegrated_features"] = selected_features
        print("\n✅ Selected Cointegrated Features:", selected_features)
    else:
        result["cointegrated_features"] = []

    return result