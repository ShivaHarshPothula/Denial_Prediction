"""Readers for the two source CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_history(path: Path | None = None) -> pd.DataFrame:
    """Labeled claims with the split and is_denied columns."""
    path = path or config.HISTORY_CSV
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    return pd.read_csv(path)


def load_current(path: Path | None = None) -> pd.DataFrame:
    """Unlabeled claims to score."""
    path = path or config.CURRENT_CSV
    if not path.exists():
        raise FileNotFoundError(f"Current-claims file not found: {path}")
    return pd.read_csv(path)
