import mlflow
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow.sklearn
import pandas as pd
import duckdb
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def cmapss_score(y_true, y_pred):
    d = y_pred - y_true  # positive = overestimate (dangerous)
    return np.sum(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1))

# add parent directory to import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.set_experiment("cmapss-rul-prediction")





con = duckdb.connect(str(PROJECT_ROOT / "database" / "cmapss.db"))
# query → pandas dataframe in one line
df = con.execute("SELECT * FROM features_fd001").df()
con.close()


with mlflow.start_run(run_name="xGBoost") as run:
    # log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("window", 30)
    
    # train model
    groups = df["unit number"]
    gkf = GroupKFold(n_splits=5)
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10)
    X = df.drop(columns=["RUL", "unit number", "time, in cycles"])
    y = df["RUL"]

    train_idx, val_idx = next(gkf.split(X, y, groups))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False)
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    mae = mean_absolute_error(y_val, y_pred)
    cmapss = cmapss_score(y_val, y_pred)
    
    # log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("cmapss", cmapss)

    # log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # log artifacts
    mlflow.log_artifact(str(PROJECT_ROOT / "artifacts" / "feature_engineer.joblib"))

    residuals = y_pred - y_val
    plt.hist(residuals, bins=50)
    plt.axvline(0, color='red')
    plt.xlabel("y_pred - y_true")
    plt.savefig(str(PROJECT_ROOT / "artifacts" / "residuals.png"))
    mlflow.log_artifact(str(PROJECT_ROOT / "artifacts" / "residuals.png"))
