"""Feature engineering and the model preprocessor.

The exact same :func:`engineer` function is applied to both the history and the
current claims, so scored claims always get the columns the model was trained on.

**Leakage guard:** ``is_denied``, ``denial_reason`` and ``split`` are never used
as model inputs (see :data:`config.LEAKAGE_COLS`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four engineered features that encode the real denial patterns."""
    df = df.copy()
    # Risk gaps: "required but not on file".
    df["prior_auth_gap"] = (
        (df["prior_auth_required"] == 1) & (df["has_prior_auth"] == 0)
    ).astype(int)
    df["referral_gap"] = (
        (df["referral_required"] == 1) & (df["referral_present"] == 0)
    ).astype(int)
    # Contractual context.
    df["payment_ratio"] = df["expected_payment"] / df["total_billed"].replace(0, np.nan)
    df["payment_ratio"] = df["payment_ratio"].fillna(0)
    df["log_total_billed"] = np.log1p(df["total_billed"])
    return df


def build_preprocessor() -> ColumnTransformer:
    """One-hot the categoricals, scale the numerics, pass the binary flags through."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
        ("num", StandardScaler(), config.NUMERIC + config.ENGINEERED),
        ("flag", "passthrough", config.BINARY_FLAGS),
    ])


def split_history(hist: pd.DataFrame):
    """Split the engineered history using the provided ``split`` column (as-is)."""
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
