
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split

def linear_regression_forecast(ml_features_df, to_forecast_column):
    linear_regression_forecast_df = ml_features_df.copy()

    X = linear_regression_forecast_df.drop(columns=[to_forecast_column, "date"])
    y = linear_regression_forecast_df[to_forecast_column]

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidate_fit_intercepts = [True, False]
    best_fit_intercept = candidate_fit_intercepts[0]
    min_error = float("inf")
    no_improve = 0

    for fi in candidate_fit_intercepts:
        lr_model = LinearRegression(fit_intercept=fi)
        lr_model.fit(X_train_scaled, y_train)
        error = mean_squared_error(y_test, lr_model.predict(X_test_scaled))
        if error < min_error:
            min_error = error
            best_fit_intercept = fi
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 1:
            break

    lr_model = LinearRegression(fit_intercept=best_fit_intercept)
    lr_model.fit(X_train_scaled, y_train)

    rmse_lr = np.sqrt(mean_squared_error(y_test, lr_model.predict(X_test_scaled)))
    print("rmse: ", rmse_lr)

    X_scaled = scaler.fit_transform(X)
    linear_regression_forecast_df["linear_regression_forecast"] = lr_model.predict(X_scaled)
    linear_regression_forecast_df["is_future_forecast"] = False

    return linear_regression_forecast_df

def lasso_regression_forecast(ml_features_df, to_forecast_column, alpha=0.1):
    lasso_regression_forecast_df = ml_features_df.copy()
    
    X = lasso_regression_forecast_df.drop(columns=[to_forecast_column, "date"])
    y = lasso_regression_forecast_df[to_forecast_column]

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidate_alphas = np.linspace(alpha/2, alpha*2, 11)
    best_alpha = candidate_alphas[0]
    min_error = float("inf")
    no_improve = 0

    for a in candidate_alphas:
        model = Lasso(alpha=a, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        error = mean_squared_error(y_test, model.predict(X_test_scaled))
        if error < min_error:
            min_error = error
            best_alpha = a
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 3:
            break

    lasso_model = Lasso(alpha=best_alpha, max_iter=10000)
    lasso_model.fit(X_train_scaled, y_train)
    X_scaled = scaler.fit_transform(X)

    lasso_regression_forecast_df["lasso_regression_forecast"] = lasso_model.predict(X_scaled)
    lasso_regression_forecast_df["is_future_forecast"] = False

    return lasso_regression_forecast_df

def decision_tree_forecast(ml_features_df, to_forecast_column, max_depth=4):
    decision_tree_forecast_df = ml_features_df.copy()

    X = decision_tree_forecast_df.drop(columns=[to_forecast_column, "date"])
    y = decision_tree_forecast_df[to_forecast_column]

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    best_depth = 1
    min_error = float("inf")
    no_improve = 0

    for depth in range(1, max_depth + 1):
        dt_model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt_model.fit(X_train, y_train)
        error = mean_squared_error(y_test, dt_model.predict(X_test))
        if error < min_error:
            min_error = error
            best_depth = depth
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 3:
            break

    dt_model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    dt_model.fit(X_train, y_train)

    decision_tree_forecast_df["decision_tree_forecast"] = dt_model.predict(X)
    decision_tree_forecast_df["is_future_forecast"] = False
    
    return decision_tree_forecast_df

def random_forest_forecast(ml_features_df, to_forecast_column, n_estimators=100, max_depth=8):
    random_forest_forecast_df = ml_features_df.copy()
    
    X = random_forest_forecast_df.drop(columns=[to_forecast_column, "date"])
    y = random_forest_forecast_df[to_forecast_column]

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    rf_model = RandomForestRegressor(
        n_estimators=1,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        warm_start=True
    )

    best_n_estimators = 1
    min_error = float("inf")
    no_improve = 0

    for i in range(1, n_estimators + 1):
        rf_model.n_estimators = i
        rf_model.fit(X_train, y_train)
        error = mean_squared_error(y_test, rf_model.predict(X_test))
        if error < min_error:
            min_error = error
            best_n_estimators = i
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 10:
            break

    rf_model = RandomForestRegressor(
        n_estimators=best_n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    random_forest_forecast_df["random_forest_forecast"] = rf_model.predict(X)
    random_forest_forecast_df["is_future_forecast"] = False

    return random_forest_forecast_df

def xgboost_forecast(ml_features_df, to_forecast_column, max_depth=4, learning_rate=0.05, n_estimators=200):
    xgboost_forecast_df = ml_features_df.copy()
    
    X = xgboost_forecast_df.drop(columns=[to_forecast_column, "date"])
    y = xgboost_forecast_df[to_forecast_column]

    split_index = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    xgb_model = xgb.XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        objective="reg:squarederror",
        colsample_bytree=0.8,
        subsample=0.8,
        reg_lambda=10,
        reg_alpha=2,
        random_state=42,
        eval_metric="rmse"
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    
    xgboost_forecast_df["xgboost_forecast"] = xgb_model.predict(X)
    xgboost_forecast_df["is_future_forecast"] = False

    return xgboost_forecast_df

def lightgbm_forecast(ml_features_df, to_forecast_column, num_leaves=31, learning_rate=0.1, n_estimators=500, max_depth=4):
    lightgbm_forecast_df = ml_features_df.copy()

    X = lightgbm_forecast_df.drop(columns=[to_forecast_column, "date"])
    y = lightgbm_forecast_df[to_forecast_column]

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    lgb_model = lgb.LGBMRegressor(
        num_leaves=num_leaves, 
        learning_rate=learning_rate, 
        n_estimators=n_estimators,
        max_depth=max_depth,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_lambda=10,
        reg_alpha=3,
        random_state=42
    )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50)]
    )

    lightgbm_forecast_df["lightgbm_forecast"] = lgb_model.predict(X)
    lightgbm_forecast_df['is_future_forecast'] = False

    return lightgbm_forecast_df

def catboost_forecast(ml_features_df, to_forecast_column, depth=3, learning_rate=0.1, iterations=500):
    catboost_forecast_df = ml_features_df.copy()

    X = catboost_forecast_df.drop(columns=[to_forecast_column, "date"])
    y = catboost_forecast_df[to_forecast_column]

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    catboost_model = cb.CatBoostRegressor(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        random_state=42,
        verbose=100,
        loss_function="RMSE"
    )

    catboost_model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )

    catboost_forecast_df["catboost_forecast"] = catboost_model.predict(X)
    catboost_forecast_df['is_future_forecast'] = False

    return catboost_forecast_df