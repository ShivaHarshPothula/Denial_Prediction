"""SHAP-based per-claim risk drivers.

SHAP runs on the fitted booster in the *transformed* space; one-hot contributions
are summed back onto the original feature, so the reported drivers name real claim
fields (e.g. ``prior_auth_gap``, ``missing_documentation_flag``) rather than
opaque encoded columns.
"""
from __future__ import annotations

import numpy as np
import shap
from sklearn.pipeline import Pipeline

from . import config

# Columns rendered as integers in the human-readable "field = value" string.
INT_COLS = set(config.BINARY_FLAGS) | {
    "prior_auth_gap", "referral_gap", "num_procedures", "num_diagnoses", "days_to_submit",
}


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def build_explainer(clf) -> shap.TreeExplainer:
    """Exact TreeExplainer on the fitted XGBoost booster."""
    return shap.TreeExplainer(clf)


def _orig_feature_map(feat_names):
    """Map each transformed column name back to its original feature."""
    def to_orig(feat):
        if feat.startswith("cat__"):
            rest = feat[len("cat__"):]
            for col in config.CATEGORICAL:          # longest known prefix wins
                if rest.startswith(col + "_"):
                    return col
            return rest
        for pfx in ("num__", "flag__", "remainder__"):
            if feat.startswith(pfx):
                return feat[len(pfx):]
        return feat

    return np.array([to_orig(f) for f in feat_names])


def _fmt_value(col, row):
    v = row[col]
    if col in config.CATEGORICAL:
        return f"{v}"
    if col in INT_COLS:
        return f"{int(v)}"
    return f"{v:,.2f}"


class ClaimExplainer:
    """Computes SHAP values for every claim and exposes per-claim drivers."""

    def __init__(self, pipe: Pipeline, explainer: shap.TreeExplainer, feat_names):
        self.pipe = pipe
        self.prep = pipe.named_steps["prep"]
        self.explainer = explainer
        self.feat_names = list(feat_names)
        self.orig_of = _orig_feature_map(self.feat_names)
        self.base_value = float(np.ravel(explainer.expected_value)[-1])
        self._sv = None
        self._engineered = None

    def fit_transform(self, engineered_df):
        """Compute SHAP values for all rows of an engineered current-claims frame."""
        self._engineered = engineered_df.reset_index(drop=True)
        Xt = _dense(self.prep.transform(self._engineered[config.FEATURES]))
        sv = self.explainer.shap_values(Xt)
        if isinstance(sv, list):                    # some shap versions return [neg, pos]
            sv = sv[1]
        self._sv = sv
        return self

    def _aggregate(self, pos: int) -> dict[str, float]:
        """Sum one-hot SHAP contributions back onto each original feature."""
        agg: dict[str, float] = {}
        for feat, val in zip(self.orig_of, self._sv[pos]):
            agg[feat] = agg.get(feat, 0.0) + float(val)
        return agg

    def drivers(self, pos: int, k: int = 3) -> list[dict]:
        """Top-k original-feature drivers for one claim, ranked by |SHAP|."""
        agg = self._aggregate(pos)
        ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
        row = self._engineered.iloc[pos]
        return [
            {
                "feature": f,
                "value": _fmt_value(f, row),
                "shap": round(s, 3),
                "direction": "raises" if s > 0 else "lowers",
            }
            for f, s in ranked
        ]

    def text_explanation(self, pos: int, prob: float, tier: str,
                         n_up: int = 3, n_down: int = 2) -> str:
        """Human-readable SHAP breakdown (top risk-raising / risk-lowering drivers)."""
        agg = self._aggregate(pos)
        drivers = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        up = [(f, s) for f, s in drivers if s > 0][:n_up]
        down = [(f, s) for f, s in drivers if s < 0][-n_down:][::-1]

        row = self._engineered.iloc[pos]
        cid = row[config.ID_COL]
        lines = [f"Claim {cid} — denial risk {prob:.0%} ({tier} tier)"]
        lines.append("  Raises risk:  " + "; ".join(
            f"{f} = {_fmt_value(f, row)} ({s:+.2f})" for f, s in up))
        if down:
            lines.append("  Lowers risk:  " + "; ".join(
                f"{f} = {_fmt_value(f, row)} ({s:+.2f})" for f, s in down))
        return "\n".join(lines)
