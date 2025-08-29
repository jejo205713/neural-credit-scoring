# preprocessing/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas.api.types as ptypes
import os

class DataPreprocessor:
    """
    Robust preprocessor that can work with either training or production datasets.
    - Auto-detects column name variants
    - Cleans, coerces types, fills missing values
    - Normalizes numeric features
    - Splits into behavioral features + graph data
    """

    _COLUMN_VARIANTS = {
        "username": ["Username", "username", "user_id", "User", "user"],
        "avg_call_count_weekly": ["Avg Weekly Calls", "avg_call_count_weekly"],
        "sms_to_call_ratio": ["SMS/Call Ratio", "sms_to_call_ratio"],
        "app_finance_hours_weekly": ["Finance App Hours", "app_finance_hours_weekly"],
        "recharge_freq_monthly": ["Recharge Freq", "recharge_freq_monthly"],
        "bill_pay_on_time_ratio": ["On-time Payments", "bill_pay_on_time_ratio"],
        "upi_txn_count_monthly": ["UPI Txn Count", "upi_txn_count_monthly"],
        "cart_size_avg": ["E-comm Spending", "cart_size_avg"],
        "behavioral_embedding": ["Behavioral Score", "behavioral_embedding"],
        "label": ["Label", "ground_truth_risk", "label"]
    }

    BEHAVIOR_COLS = [
        "avg_call_count_weekly", "sms_to_call_ratio", "app_finance_hours_weekly",
        "recharge_freq_monthly", "bill_pay_on_time_ratio", "upi_txn_count_monthly",
        "cart_size_avg", "behavioral_embedding"
    ]

    def __init__(self, csv_path=None, mode="train", drop_columns=None):
        """
        mode = "train" (training dataset) or "prod" (production/live dataset)
        csv_path = override path if not using defaults
        """
        self.mode = mode
        if csv_path:
            self.csv_path = csv_path
        else:
            self.csv_path = (
                "data/train/neural_credit_train.csv" if mode == "train"
                else "data/prod/neural_credit_live.csv"
            )

        self.drop_columns = drop_columns or []
        self.df = None
        self.behavioral_features = None
        self.graph_data = None

    def _detect_and_rename(self):
        found = {}
        cols = list(self.df.columns)
        cols_stripped = [c.strip() if isinstance(c, str) else c for c in cols]
        self.df.columns = cols_stripped

        for canonical, variants in self._COLUMN_VARIANTS.items():
            for v in variants:
                if v in self.df.columns:
                    if v != canonical:
                        self.df.rename(columns={v: canonical}, inplace=True)
                    found[canonical] = canonical
                    break
        return found

    def load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [c.strip() if isinstance(c, str) else c for c in self.df.columns]

        for c in self.drop_columns:
            if c in self.df.columns:
                self.df.drop(columns=[c], inplace=True)

        self._detect_and_rename()

        print(f"[INFO] Loaded dataset ({self.mode}): {self.df.shape[0]} rows, {self.df.shape[1]} cols")
        return self

    def clean_data(self):
        self.df.drop_duplicates(inplace=True)

        for col in list(self.df.columns):
            if col == "username":
                continue
            try:
                coerced = pd.to_numeric(self.df[col], errors="coerce")
                if coerced.notna().sum() > 0:
                    self.df[col] = coerced
            except Exception:
                pass

        for col in self.df.columns:
            if ptypes.is_numeric_dtype(self.df[col]):
                median_val = float(self.df[col].median()) if not self.df[col].dropna().empty else 0.0
                self.df[col] = self.df[col].fillna(median_val)
            else:
                self.df[col] = self.df[col].fillna("Unknown")

        return self

    def normalize_features(self):
        numeric_cols = [c for c in self.BEHAVIOR_COLS if c in self.df.columns]
        if not numeric_cols:
            numeric_cols = [c for c in self.df.columns if ptypes.is_numeric_dtype(self.df[c]) and c != "label"]
        if numeric_cols:
            scaler = MinMaxScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self

    def split_features(self):
        missing = [c for c in self.BEHAVIOR_COLS if c not in self.df.columns]
        if missing:
            raise KeyError(f"Missing expected behavioral columns in {self.csv_path}. Missing: {missing}")

        behavior_df = self.df[self.BEHAVIOR_COLS].astype(float)
        self.behavioral_features = behavior_df.values

        if "username" in self.df.columns:
            self.graph_data = self.df[["username"]].copy()
        else:
            self.graph_data = pd.DataFrame({"username": [f"user_{i}" for i in range(len(self.df))]})

        return self.behavioral_features, self.graph_data

    def get_labels(self):
        if "label" not in self.df.columns:
            if self.mode == "prod":
                print("[WARN] Production dataset has no labels, returning None")
                return None
            raise KeyError("Label column not found in CSV for training dataset.")
        return self.df["label"].astype(int).values

    def preprocess(self):
        """Full preprocessing pipeline -> returns (X, y) for training or (X, None) for prod"""
        self.load_data().clean_data().normalize_features()
        X, _ = self.split_features()
        y = self.get_labels() if self.mode == "train" else None
        return X, y


if __name__ == "__main__":
    # Example: training
    train_pre = DataPreprocessor(mode="train")
    X_train, y_train = train_pre.preprocess()
    print("[DEBUG] Train features shape:", X_train.shape, "Labels shape:", y_train.shape)

    # Example: production
    prod_pre = DataPreprocessor(mode="prod")
    X_prod, _ = prod_pre.preprocess()
    print("[DEBUG] Prod features shape:", X_prod.shape)

