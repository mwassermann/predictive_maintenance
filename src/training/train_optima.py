import mlflow
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import duckdb
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import optuna


class RULPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # take only the last timestep
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)
    


def make_sequences(df, window_length, feature_cols, target_col="RUL"):
    X_list = []
    y_list = []
    groups_list = []

    for unit_id, group in df.groupby("unit number"):
        data = group[feature_cols].to_numpy()
        target = group[target_col].to_numpy()

        if len(group) < window_length:
            continue

        # sliding windows
        X = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=(window_length, data.shape[1])
        ).squeeze(1)

        # targets aligned to last element in each window
        y = target[window_length - 1:]

        groups_list.append(np.full(len(y), unit_id))


        X_list.append(X)
        y_list.append(y)


    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)

    return X, y, groups


def cmapss_score(y_true, y_pred):
    d = y_pred - y_true  # positive = overestimate (dangerous)
    return np.sum(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1))

def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    with mlflow.start_run(run_name=f"lstm-trial-{trial.number}", nested=True):
        mlflow.log_params(trial.params)
        # train model with these params
        
        model = RULPredictor(input_size=X.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        n_epochs = 200
        best_val_loss = float('inf')
        patience, patience_counter = 10, 0
        
        for epoch in range(n_epochs):

            # ---- TRAIN ----
            model.train()
            train_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * xb.size(0)

            train_loss /= len(train_loader.dataset)


            # ---- VALIDATION ----
            model.eval()
            val_loss = 0.0
            preds_list = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)

                    preds = model(xb)
                    loss = criterion(preds, yb)

                    val_loss += loss.item() * xb.size(0)
                    preds_list.append(preds.cpu().numpy())

            val_loss /= len(val_loader.dataset)

            # log per-epoch metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), PROJECT_ROOT / "artifacts" / "best_model.pt")
                patience_counter = 0
                best_preds = np.concatenate(preds_list)

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break


        y_pred = best_preds
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        mae = mean_absolute_error(y_val, y_pred)
        cmapss = cmapss_score(y_val, y_pred)
        residuals = y_pred - y_val


        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("cmapss", cmapss)
        mlflow.log_metric("best_val_loss", best_val_loss)



        model.load_state_dict(torch.load(PROJECT_ROOT / "artifacts" / "best_model.pt"))
        mlflow.pytorch.log_model(model, artifact_path="model")


        # log artifacts
        mlflow.log_artifact(str(PROJECT_ROOT / "artifacts" / "feature_engineer_30.joblib"))

        plt.hist(residuals, bins=50)
        plt.axvline(0, color='red')
        plt.xlabel("y_pred - y_true")
        plt.savefig(str(PROJECT_ROOT / "artifacts" / "residuals.png"))
        mlflow.log_artifact(str(PROJECT_ROOT / "artifacts" / "residuals.png"))
        plt.close()
        joblib.dump(scaler, PROJECT_ROOT / "artifacts" / "scaler_30.joblib")
        mlflow.log_artifact(str(PROJECT_ROOT / "artifacts" / "scaler_30.joblib"))


    return best_val_loss




# add parent directory to import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.set_experiment("cmapss-rul-prediction")



con = duckdb.connect(str(PROJECT_ROOT / "database" / "cmapss.db"))
# query → pandas dataframe in one line
df = con.execute("SELECT * FROM features_30_fd001").df()
con.close()

feature_cols = [col for col in df.columns if col not in ["unit number", "time, in cycles", "RUL"]]

X, y, groups = make_sequences(df, window_length=30, feature_cols=feature_cols)

gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, y, groups))

X_train, X_val = X[train_idx], X[val_idx]   
y_train, y_val = y[train_idx], y[val_idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_train, seq_len, n_features = X_train.shape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)

# transform val with train statistics
n_val = X_val.shape[0]
X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(n_val, seq_len, n_features)
# convert to tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32)



train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)



with mlflow.start_run(run_name="lstm-optuna-search"):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    
    print("Best trial:")
    trial = study.best_trial 
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
