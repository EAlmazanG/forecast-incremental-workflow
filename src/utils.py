
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import pandas as pd
import numpy as np
import math

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