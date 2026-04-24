"""
Final test set evaluation.
Trains the best LSTM architecture on all training data (no val split),
then evaluates once on the held-out test set.

Best hyperparameters are taken from the Optuna search run.
Epoch count is fixed from the best Optuna trial to avoid test-set leakage.

Usage:
    python -m src.training.evaluate
"""

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.training.models import RULPredictor
from src.training.utils import (
    load_features_from_db,
    make_sequences,
    get_last_window,
    compute_and_log_metrics,
)

# ── config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH      = PROJECT_ROOT / "database" / "cmapss.db"
ARTIFACTS    = PROJECT_ROOT / "artifacts"
FEATURE_LEN  = 25
SEQ_LEN      = 30
BATCH_SIZE   = 256

# Best hyperparameters from Optuna trial 15
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.4931543759316164
LR           = 0.0005943598811202427
N_EPOCHS     = 50          # from best trial's early stopping epoch

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.set_experiment("cmapss-rul-prediction")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── data ──────────────────────────────────────────────────────────────────────
df      = load_features_from_db(DB_PATH, f"features_{FEATURE_LEN}_fd001")
df_test = load_features_from_db(DB_PATH, f"features_{FEATURE_LEN}_fd001_test")

feature_cols = [c for c in df.columns if c not in ["unit number", "time, in cycles", "RUL"]]

# train on ALL training data — no val split needed
X, y, _ = make_sequences(df, window_length=SEQ_LEN, feature_cols=feature_cols)
n_train, seq_len, n_features = X.shape

scaler = joblib.load(ARTIFACTS / f"scaler_{FEATURE_LEN}.joblib")
X_sc = scaler.transform(X.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_sc, dtype=torch.float32),
                  torch.tensor(y, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True,
)

# test set — last window per engine, padded if necessary
X_test  = get_last_window(df_test, feature_cols, window_length=SEQ_LEN)
y_test  = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "RUL_FD001.txt",
                      header=None).values.flatten()

n_test  = X_test.shape[0]
X_test_sc = scaler.transform(X_test.reshape(-1, n_features)).reshape(n_test, seq_len, n_features)
X_test_t  = torch.tensor(X_test_sc, dtype=torch.float32)
y_test_t  = torch.tensor(y_test,    dtype=torch.float32)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# ── baseline test evaluation ──────────────────────────────────────────────────
# Tree models use a 2D input: (n_engines, n_features) — last window flattened
X_test_2d = X_test.reshape(n_test, -1)   # (100, seq_len * n_features)

# Use only the last timestep's features to match training input shape
X_train_2d = df[feature_cols].values     # all training rows, no sequence dim
y_train_2d = df["RUL"].values

for model_name, clf, params in [
    ("rf-test-evaluation",
     RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
     {"n_estimators": 100, "feature_window": FEATURE_LEN}),
    ("xgboost-test-evaluation",
     XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10,
                  subsample=0.8, colsample_bytree=0.8, random_state=42),
     {"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 10,
      "feature_window": FEATURE_LEN}),
]:
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        clf.fit(X_train_2d, y_train_2d)
        # for test, use only the last row per engine (equivalent to last window)
        X_test_last = get_last_window(df_test, feature_cols, window_length=1
                                      ).reshape(n_test, n_features)
        y_pred = clf.predict(X_test_last)
        compute_and_log_metrics(y_test, y_pred, ARTIFACTS, run)
        mlflow.sklearn.log_model(clf, artifact_path="model")
        print(f"[{model_name}] done")

# ── LSTM training ──────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="lstm-test-evaluation") as run:
    mlflow.log_params({
        "hidden_size":   HIDDEN_SIZE,
        "num_layers":    NUM_LAYERS,
        "dropout":       DROPOUT,
        "lr":            LR,
        "epochs":        N_EPOCHS,
        "feature_window": FEATURE_LEN,
    })

    model = RULPredictor(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        print(f"Epoch {epoch+1}/{N_EPOCHS}: train_loss={train_loss:.4f}")

    # ── evaluation (test set, touched once) ───────────────────────────────────
    model.eval()
    preds_list = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds_list.append(model(xb.to(device)).cpu().numpy())

    y_pred = np.concatenate(preds_list)
    compute_and_log_metrics(y_test, y_pred, ARTIFACTS, run)

    # save & register
    torch.save(model.state_dict(), ARTIFACTS / "best_model.pt")
    mlflow.pytorch.log_model(model, artifact_path="model")
    mlflow.log_artifact(str(ARTIFACTS / f"feature_engineer_{FEATURE_LEN}.joblib"))
    mlflow.log_artifact(str(ARTIFACTS / f"scaler_{FEATURE_LEN}.joblib"))

    # register as new Production version
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", "rul-predictor")
    from mlflow.tracking import MlflowClient
    MlflowClient().transition_model_version_stage(
        name="rul-predictor", version=mv.version, stage="Production"
    )
    print(f"\nRegistered model version {mv.version} → Production")
    print(f"run_id: {run.info.run_id}")
