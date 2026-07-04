"""Data loading — thin wrappers around the two source CSVs.

Kept deliberately small so the rest of the pipeline never reads from disk
directly and file locations stay centralised in :mod:`config`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_history(path: Path | None = None) -> pd.DataFrame:
    """Load the labelled training/validation/test claims (has ``split``, ``is_denied``)."""
    path = path or config.HISTORY_CSV
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    return pd.read_csv(path)


def load_current(path: Path | None = None) -> pd.DataFrame:
    """Load the unlabelled claims to be scored."""
    path = path or config.CURRENT_CSV
    if not path.exists():
        raise FileNotFoundError(f"Current-claims file not found: {path}")
    return pd.read_csv(path)
