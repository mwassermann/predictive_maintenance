from prefect import task, flow
import duckdb
import pandas as pd
import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# add parent directory to import path
current = os.getcwd()
parent = os.path.dirname(current)
sys.path.append(parent)

from src.features.feature_engineering import FeatureEngineer


@task
def extract(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep=r'\s+', names=["unit number", "time, in cycles","operational setting 1","operational setting 2","operational setting 3","sensor measurement  1",
                                                                                            "sensor measurement  2","sensor measurement  3", "sensor measurement  4","sensor measurement  5",
                                                                                            "sensor measurement  6","sensor measurement  7","sensor measurement  8","sensor measurement  9",
                                                                                            "sensor measurement  10","sensor measurement  11", "sensor measurement  12","sensor measurement  13",
                                                                                            "sensor measurement  14","sensor measurement  15", "sensor measurement  16","sensor measurement  17",
                                                                                            "sensor measurement  18","sensor measurement  19", "sensor measurement  20","sensor measurement  21"])
    df["RUL"] = df.groupby("unit number")["time, in cycles"].transform("max") - df["time, in cycles"]
    df["RUL"] = df["RUL"].clip(upper=125)
    return df


@task
def transform(df: pd.DataFrame, window: int = 25, fe: FeatureEngineer = None) -> tuple[pd.DataFrame, FeatureEngineer]:
    if fe is None:
        fe = FeatureEngineer(window)
        fe.fit(df)
    smoothed_df = fe.transform(df)
    return smoothed_df, fe

@task
def load(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    con = duckdb.connect(db_path)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

    con.close()

@flow
def etl_pipeline(raw_path: str, db_path: str, window: int = 25, 
                 table_name: str = "features_25_fd001",
                 fe: FeatureEngineer = None) -> None:
    
    df = extract(raw_path)
    smoothed_df, fe = transform(df, window, fe)
    load(smoothed_df, db_path, table_name)
    fe.save(str(PROJECT_ROOT / "artifacts" / f"feature_engineer_{window}.joblib"))

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    feature_length = 25
    # train
    etl_pipeline(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"),
        db_path=str(PROJECT_ROOT / "database" / "cmapss.db"),
        window=feature_length,
        table_name=f"features_{feature_length}_fd001",
    )
    
    # test
    fe = FeatureEngineer.load(PROJECT_ROOT / "artifacts" / f"feature_engineer_{feature_length}.joblib")
    etl_pipeline(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "test_FD001.txt"),
        db_path=str(PROJECT_ROOT / "database" / "cmapss.db"),
        window=feature_length,
        table_name=f"features_{feature_length}_fd001_test",
        fe=fe
    )
