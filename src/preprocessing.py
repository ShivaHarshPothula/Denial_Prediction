"""Feature engineering and the model preprocessor.

engineer() runs on both history and current claims so scored claims get the
same columns the model saw. The leakage columns (config.LEAKAGE_COLS) are never
used as inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four engineered denial-risk features."""
    df = df.copy()
    # "Required but not on file" gaps.
    df["prior_auth_gap"] = (
        (df["prior_auth_required"] == 1) & (df["has_prior_auth"] == 0)
    ).astype(int)
    df["referral_gap"] = (
        (df["referral_required"] == 1) & (df["referral_present"] == 0)
    ).astype(int)
    df["payment_ratio"] = df["expected_payment"] / df["total_billed"].replace(0, np.nan)
    df["payment_ratio"] = df["payment_ratio"].fillna(0)
    df["log_total_billed"] = np.log1p(df["total_billed"])
    return df


def build_preprocessor() -> ColumnTransformer:
    """One-hot categoricals, scale numerics, pass binary flags through."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
        ("num", StandardScaler(), config.NUMERIC + config.ENGINEERED),
        ("flag", "passthrough", config.BINARY_FLAGS),
    ])


def split_history(hist: pd.DataFrame):
    """Split engineered history on the provided split column."""
    hist_e = engineer(hist)
    train = hist_e[hist_e["split"] == "train"]
    val = hist_e[hist_e["split"] == "validation"]
    test = hist_e[hist_e["split"] == "test"]

    def xy(frame):
        return frame[config.FEATURES], frame[config.TARGET]

    return {
        "train": xy(train),
        "val": xy(val),
        "test": xy(test),
    }
