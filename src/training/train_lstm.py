"""
LSTM training with Optuna hyperparameter search.
Each Optuna trial is logged as a nested MLflow run.

Usage:
    python -m src.training.train_lstm
"""

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.training.models import RULPredictor
from src.training.utils import (
    load_features_from_db,
    make_sequences,
    compute_and_log_metrics,
)

# ── config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH      = PROJECT_ROOT / "database" / "cmapss.db"
ARTIFACTS    = PROJECT_ROOT / "artifacts"
FEATURE_LEN  = 25
TABLE        = f"features_{FEATURE_LEN}_fd001"
SEQ_LEN      = 30
N_TRIALS     = 20
MAX_EPOCHS   = 200
PATIENCE     = 10
BATCH_SIZE   = 256

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.set_experiment("cmapss-rul-prediction")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── data ──────────────────────────────────────────────────────────────────────
df = load_features_from_db(DB_PATH, TABLE)
feature_cols = [c for c in df.columns if c not in ["unit number", "time, in cycles", "RUL"]]

X, y, groups = make_sequences(df, window_length=SEQ_LEN, feature_cols=feature_cols)

gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, y, groups))
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

n_train, seq_len, n_features = X_train.shape

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
X_val_sc   = scaler.transform(X_val.reshape(-1, n_features)).reshape(len(X_val), seq_len, n_features)

joblib.dump(scaler, ARTIFACTS / f"scaler_{FEATURE_LEN}.joblib")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train_sc, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val_sc, dtype=torch.float32),
                  torch.tensor(y_val, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=False,
)


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers  = trial.suggest_int("num_layers", 1, 3)
    dropout     = trial.suggest_float("dropout", 0.1, 0.5)
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    with mlflow.start_run(run_name=f"lstm-trial-{trial.number}", nested=True):
        mlflow.log_params(trial.params)

        model = RULPredictor(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_preds    = None
        patience_ctr  = 0

        for epoch in range(MAX_EPOCHS):
            # train
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

            # validate
            model.eval()
            val_loss  = 0.0
            preds_list = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    val_loss += criterion(preds, yb).item() * xb.size(0)
                    preds_list.append(preds.cpu().numpy())
            val_loss /= len(val_loader.dataset)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss",   val_loss,   step=epoch)
            print(f"  Trial {trial.number} | Epoch {epoch+1}: train={train_loss:.2f} val={val_loss:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds    = np.concatenate(preds_list)
                torch.save(model.state_dict(), ARTIFACTS / "best_model.pt")
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # log best checkpoint
        model.load_state_dict(torch.load(ARTIFACTS / "best_model.pt"))
        compute_and_log_metrics(y_val, best_preds, ARTIFACTS, None)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(ARTIFACTS / f"feature_engineer_{FEATURE_LEN}.joblib"))
        mlflow.log_artifact(str(ARTIFACTS / f"scaler_{FEATURE_LEN}.joblib"))

    return best_val_loss


# ── search ────────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="lstm-optuna-search"):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest trial:")
    print(f"  val_loss : {study.best_trial.value:.4f}")
    print(f"  params   : {study.best_trial.params}")
