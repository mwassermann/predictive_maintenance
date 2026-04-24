import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_features_from_db(db_path: Path, table_name: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    df = con.execute(f"SELECT * FROM {table_name}").df()
    con.close()
    return df


def make_sequences(df: pd.DataFrame, window_length: int, feature_cols: list, target_col: str = "RUL"):
    """
    Convert a per-row feature dataframe into sliding window sequences per engine.
    Returns X (n_samples, window_length, n_features), y (n_samples,), groups (n_samples,).
    """
    X_list, y_list, groups_list = [], [], []

    for unit_id, group in df.groupby("unit number"):
        data = group[feature_cols].to_numpy()
        target = group[target_col].to_numpy()

        if len(group) < window_length:
            continue

        X = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=(window_length, data.shape[1])
        ).squeeze(1)

        y = target[window_length - 1:]
        groups_list.append(np.full(len(y), unit_id))
        X_list.append(X)
        y_list.append(y)

    return (
        np.concatenate(X_list),
        np.concatenate(y_list),
        np.concatenate(groups_list),
    )


def get_last_window(df_smoothed: pd.DataFrame, feature_cols: list, window_length: int = 30) -> np.ndarray:
    """
    Extract the last window_length rows per engine for inference.
    Pads short engines by repeating the first available row.
    """
    X_list = []

    for _, group in df_smoothed.groupby("unit number"):
        data = group[feature_cols].to_numpy()

        if len(data) >= window_length:
            window = data[-window_length:]
        else:
            pad_size = window_length - len(data)
            padding = np.repeat(data[[0]], pad_size, axis=0)
            window = np.vstack([padding, data])

        X_list.append(window)

    return np.array(X_list)


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Asymmetric CMAPSS scoring function.
    Overestimates (d > 0) are penalised more steeply than underestimates.
    """
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def compute_and_log_metrics(y_true: np.ndarray, y_pred: np.ndarray, artifact_dir: Path, mlflow_run):
    """Compute RMSE, MAE, CMAPSS score and log residual plot as MLflow artifact."""
    import mlflow

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    score = cmapss_score(y_true, y_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("cmapss", score)

    residuals = y_pred - y_true
    plt.figure()
    plt.hist(residuals, bins=50)
    plt.axvline(0, color="red")
    plt.xlabel("y_pred - y_true")
    residual_path = artifact_dir / "residuals.png"
    plt.savefig(str(residual_path))
    plt.close()
    mlflow.log_artifact(str(residual_path))

    return rmse, mae, score
