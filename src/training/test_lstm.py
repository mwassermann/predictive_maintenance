import mlflow
import numpy as np
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

def get_last_window(df_smoothed, feature_cols, window_length=30):
    X_list = []
    
    for unit_id, group in df_smoothed.groupby("unit number"):
        data = group[feature_cols].to_numpy()
        
        if len(data) >= window_length:
            window = data[-window_length:]
        else:
            # pad at the beginning by repeating first row
            pad_size = window_length - len(data)
            padding = np.repeat(data[[0]], pad_size, axis=0)
            window = np.vstack([padding, data])
        
        X_list.append(window)
    
    return np.array(X_list)

"""
def get_last_window(df_smoothed, feature_cols, y_test_all, window_length=30):
    X_list = []
    valid_indices = []
    
    for i, (unit_id, group) in enumerate(df_smoothed.groupby("unit number")):
        data = group[feature_cols].to_numpy()
        
        if len(data) < window_length:
            print(f"Dropping engine {unit_id} — only {len(data)} smoothed rows")
            continue
        
        X_list.append(data[-window_length:])
        valid_indices.append(i)
    
    return np.array(X_list), y_test_all[valid_indices]
"""


# add parent directory to import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.set_experiment("cmapss-rul-prediction")

feature_length = 25

con = duckdb.connect(str(PROJECT_ROOT / "database" / "cmapss.db"))
# query → pandas dataframe in one line
df = con.execute(f"SELECT * FROM features_{feature_length}_fd001").df()
df_test = con.execute(f"SELECT * FROM features_{feature_length}_fd001_test").df()
con.close()

feature_cols = [col for col in df.columns if col not in ["unit number", "time, in cycles", "RUL"]]

X, y, groups = make_sequences(df, window_length=30, feature_cols=feature_cols)

dropout = 0.4931543759316164
lr = 0.0005943598811202427
n_epochs = 50
hidden_size = 128
num_layers = 2


with mlflow.start_run(run_name="lstm-test-evaluation") as run:


    # log hyperparameters
    mlflow.log_param("hidden_size", hidden_size)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("dropout", dropout)
    mlflow.log_param("lr", lr)
    mlflow.log_param("epochs", n_epochs)

    
    # train data
    
    X_train = X
    y_train = y
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_train, seq_len, n_features = X_train.shape
    scaler = joblib.load(PROJECT_ROOT / "artifacts" / f"scaler_{feature_length}.joblib")
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)

    # convert to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

    X_test = get_last_window(df_test, feature_cols, window_length=30)
    y_test = pd.read_csv(str(PROJECT_ROOT / "data" / "raw" / "RUL_FD001.txt"), header=None).values.flatten()
    #X_test, y_test = get_last_window(df_test, feature_cols, y_test_all, window_length=30)
    n_test = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(n_test, seq_len, n_features)

    X_test_t   = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t   = torch.tensor(y_test, dtype=torch.float32)
    test_loader   = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=256, shuffle=False)

    model = RULPredictor(input_size=X.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #best_test_loss = float('inf')
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
        test_loss = 0.0
        preds_list = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)

                preds = model(xb)
                loss = criterion(preds, yb)

                test_loss += loss.item() * xb.size(0)
                preds_list.append(preds.cpu().numpy())

        test_loss /= len(test_loader.dataset)

        # log per-epoch metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")


    torch.save(model.state_dict(), PROJECT_ROOT / "artifacts" / "best_model.pt")
    y_pred = np.concatenate(preds_list)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    cmapss = cmapss_score(y_test, y_pred)
    residuals = y_pred - y_test


    # log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("cmapss", cmapss)
    #mlflow.log_metric("best_test_loss", best_test_loss)


    # log model
    mlflow.pytorch.log_model(model, artifact_path="model")

    # log artifacts
    mlflow.log_artifact(str(PROJECT_ROOT / "artifacts" / "feature_engineer.joblib"))

    plt.hist(residuals, bins=50)
    plt.axvline(0, color='red')
    plt.xlabel("y_pred - y_true")
    plt.savefig(str(PROJECT_ROOT / "artifacts" / "residuals.png"))
    mlflow.log_artifact(str(PROJECT_ROOT / "artifacts" / "residuals.png"))