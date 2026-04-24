"""
Baseline model training: Random Forest and XGBoost.
Logs all runs to the cmapss-rul-prediction MLflow experiment.

Usage:
    python -m src.training.train_baselines
"""

import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.training.utils import load_features_from_db, cmapss_score, compute_and_log_metrics

# ── config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH      = PROJECT_ROOT / "database" / "cmapss.db"
ARTIFACTS    = PROJECT_ROOT / "artifacts"
FEATURE_LEN  = 25          # rolling window used during ETL
TABLE        = f"features_{FEATURE_LEN}_fd001"

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.set_experiment("cmapss-rul-prediction")

# ── data ──────────────────────────────────────────────────────────────────────
df = load_features_from_db(DB_PATH, TABLE)

FEATURE_COLS = [c for c in df.columns if c not in ["unit number", "time, in cycles", "RUL"]]
X = df[FEATURE_COLS]
y = df["RUL"]
groups = df["unit number"]

gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, y, groups))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


def run_baseline(model, run_name: str, params: dict):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        compute_and_log_metrics(y_val.to_numpy(), y_pred, ARTIFACTS, run)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(ARTIFACTS / f"feature_engineer_{FEATURE_LEN}.joblib"))

        print(f"[{run_name}] done — run_id={run.info.run_id}")


# ── Random Forest ─────────────────────────────────────────────────────────────
run_baseline(
    RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    run_name="random-forest",
    params={"n_estimators": 100, "feature_window": FEATURE_LEN},
)

# ── XGBoost ───────────────────────────────────────────────────────────────────
xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
with mlflow.start_run(run_name="xgboost") as run:
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 10,
        "feature_window": FEATURE_LEN,
    }
    mlflow.log_params(params)

    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    mlflow.log_param("best_iteration", xgb.best_iteration)

    y_pred = xgb.predict(X_val)
    compute_and_log_metrics(y_val.to_numpy(), y_pred, ARTIFACTS, run)
    mlflow.sklearn.log_model(xgb, artifact_path="model")
    mlflow.log_artifact(str(ARTIFACTS / f"feature_engineer_{FEATURE_LEN}.joblib"))

    print(f"[xgboost] done — run_id={run.info.run_id}")
