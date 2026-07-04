"""XGBoost tuning, evaluation, and capacity-aware metrics.

Everything is built around the review-team constraint: only the top 25% of
claims by risk get inspected, so we tune and threshold on recall@25% rather
than a 0.5 cutoff.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config
from .preprocessing import build_preprocessor

# Search space leans on regularization since the dataset is small.
# n_estimators is a cap; early stopping picks the actual number of trees.
XGB_SPACE = {
    "clf__n_estimators":     [1000],
    "clf__max_depth":        [2, 3, 4, 5],
    "clf__learning_rate":    [0.01, 0.02, 0.03, 0.05, 0.1],
    "clf__subsample":        [0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
    "clf__min_child_weight": [1, 3, 5],
    "clf__gamma":            [0, 0.5, 1.0],
    "clf__reg_alpha":        [0, 0.1, 1.0],
    "clf__reg_lambda":       [1.0, 3.0, 5.0],
}
XGB_N_ITER = 80
EARLY_STOP = 50


def capture_at_top_k(y_true, scores, k: float = config.REVIEW_CAPACITY) -> dict:
    """Recall and precision within the top-k fraction by score."""
    y_true = np.asarray(y_true)
    n_review = max(1, int(np.ceil(len(scores) * k)))
    order = np.argsort(scores)[::-1]
    top = order[:n_review]
    captured = y_true[top].sum()
    total_pos = y_true.sum()
    return {
        "reviewed": n_review,
        "recall_at_25": captured / total_pos if total_pos else 0.0,
        "precision_at_25": captured / n_review,
    }


def recall_at_25_loss(y_true, y_pred) -> float:
    """Custom XGBoost eval metric: 1 - recall@25% (minimized).

    Only depends on score ranking, so early stopping optimizes the metric we
    actually deploy on.
    """
    y_true = np.asarray(y_true)
    n_review = max(1, int(np.ceil(len(y_pred) * config.REVIEW_CAPACITY)))
    top = np.argsort(y_pred)[::-1][:n_review]
    total_pos = y_true.sum()
    recall = y_true[top].sum() / total_pos if total_pos else 0.0
    return 1.0 - recall


def _make_xgb(scale_pos_weight: float, **params) -> XGBClassifier:
    return XGBClassifier(
        eval_metric=recall_at_25_loss,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        early_stopping_rounds=EARLY_STOP,
        **params,
    )


def tune_xgboost(X_train, y_train, X_val, y_val,
                 n_iter: int = XGB_N_ITER) -> tuple[Pipeline, dict]:
    """Randomized search, selecting on (recall@25%, PR-AUC) on validation.

    Returns the fitted Pipeline (preprocessor + booster) and the best params,
    including the early-stopped best iteration.
    """
    pos = y_train.mean()
    scale_pos_weight = (1.0 - pos) / pos

    # Preprocessor doesn't depend on the booster params, so fit it once and
    # reuse the transformed eval_set across candidates.
    prep = build_preprocessor().fit(X_train, y_train)
    Xtr_t = prep.transform(X_train)
    Xval_t = prep.transform(X_val)

    candidates = list(ParameterSampler(
        XGB_SPACE, n_iter=n_iter, random_state=config.RANDOM_STATE))

    best_score = (-1.0, -1.0)
    best_pipe: Pipeline | None = None
    best_params: dict | None = None

    for cand in candidates:
        params = {k.replace("clf__", ""): v for k, v in cand.items()}
        clf = _make_xgb(scale_pos_weight, **params)
        clf.fit(Xtr_t, y_train, eval_set=[(Xval_t, y_val)], verbose=False)
        scores = clf.predict_proba(Xval_t)[:, 1]
        score = (capture_at_top_k(y_val, scores)["recall_at_25"],
                 average_precision_score(y_val, scores))
        if score > best_score:
            best_score = score
            best_pipe = Pipeline([("prep", prep), ("clf", clf)])
            best_params = dict(params, best_iteration=int(clf.best_iteration))

    return best_pipe, best_params


def evaluate(pipe: Pipeline, X, y, threshold: float, label: str) -> dict:
    """Operating-point metrics for one split."""
    scores = pipe.predict_proba(X)[:, 1]
    cap = capture_at_top_k(y, scores)
    pred = (scores >= threshold).astype(int)
    return {
        "split": label,
        "pr_auc": round(average_precision_score(y, scores), 3),
        "roc_auc": round(roc_auc_score(y, scores), 3),
        "recall@25%": round(cap["recall_at_25"], 3),
        "precision@25%": round(cap["precision_at_25"], 3),
        "precision@thr": round(precision_score(y, pred, zero_division=0), 3),
        "recall@thr": round(recall_score(y, pred), 3),
        "f1@thr": round(f1_score(y, pred), 3),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
