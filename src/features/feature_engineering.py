import joblib
import pandas as pd


class FeatureEngineer:
    def __init__(self, window: int = 30):
        self.window = window
        self.sensor_cols = None  # filled in during fit()
    
    def fit(self, df: pd.DataFrame):
        # figure out which columns are valid sensors
        # save them to self.sensor_cols
        df = df.copy()

        base_drop = ["unit number", "time, in cycles", 
             "operational setting 1", "operational setting 2", 
             "operational setting 3"]
        base_drop = [c for c in base_drop if c in df.columns]
        if "RUL" in df.columns:
            base_drop.append("RUL")


        sensor_cols = df.select_dtypes(include="number").columns.drop(base_drop)
        self.cols_to_drop = sensor_cols

        # remove low variance once, before feature generation
        low_var = df[sensor_cols].std()
        self.sensor_cols = low_var[low_var >= 0.01].index
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # use self.sensor_cols to compute features
        # never recompute which columns to use
        df = df.copy()
        grouped = df.groupby("unit number")

        new_cols = []

        for col in self.sensor_cols:
            df[f"{col}_mean"] = grouped[col].rolling(self.window).mean().reset_index(level=0, drop=True)
            df[f"{col}_std"]  = grouped[col].rolling(self.window).std().reset_index(level=0, drop=True)
            df[f"{col}_diff"] = grouped[col].diff()

            new_cols += [f"{col}_mean", f"{col}_std", f"{col}_diff"]

        # ONLY drop rows where new features are incomplete
        df = df.drop(columns=self.cols_to_drop)
        return df.dropna(subset=new_cols)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
    
    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        return joblib.load(path)

