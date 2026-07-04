"""Part 1: train, threshold, run SHAP, and save artifacts.

    python src/train.py

Writes artifacts/model.joblib, artifacts/shap_explainer.joblib,
artifacts/metadata.joblib, and top10_shap_explanations.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import numpy as np

# Let `python src/train.py` work from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data_loader import load_history, load_current
from src.explain import ClaimExplainer, build_explainer
from src.model import capture_at_top_k, evaluate, tune_xgboost
from src.preprocessing import engineer, split_history


def risk_tier(prob: float, threshold: float, median_cut: float) -> str:
    if prob >= threshold:
        return "High"
    if prob >= median_cut:
        return "Medium"
    return "Low"


def main() -> None:
    # Load + split.
    hist = load_history()
    splits = split_history(hist)
    (X_train, y_train) = splits["train"]
    (X_val, y_val) = splits["val"]
    (X_test, y_test) = splits["test"]

    print(f"history: {hist.shape}")
    for name in ("train", "val", "test"):
        _, y = splits[name]
        print(f"  {name:5s} n={len(y):4d}  denial_rate={y.mean():.3f}")

    # Tune XGBoost.
    print("\nTuning XGBoost (early stop on 1 - recall@25%)...")
    model, params = tune_xgboost(X_train, y_train, X_val, y_val)
    print(f"  best iteration (trees): {params['best_iteration'] + 1}")
    print(f"  best params: {params}")

    # Pick the operating point on validation: top-25% cutoff for High,
    # median for Medium.
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = float(np.quantile(val_scores, 1 - config.REVIEW_CAPACITY))
    median_cut = float(np.quantile(val_scores, 0.50))
    print(f"\nOperating threshold (top-{config.REVIEW_CAPACITY:.0%} cutoff): {threshold:.3f}")
    print(f"Risk tiers: High >= {threshold:.3f} | Medium >= {median_cut:.3f} | Low below")

    # Report on validation and the locked test set.
    for label, X, y in (("validation", X_val, y_val), ("test", X_test, y_test)):
        rep = evaluate(model, X, y, threshold, label)
        print(f"\n=== {label} metrics ===")
        for k, v in rep.items():
            print(f"  {k:16s} {v}")

    # SHAP explainer + driver breakdown for the top-10 current claims.
    clf = model.named_steps["clf"]
    prep = model.named_steps["prep"]
    feat_names = list(prep.get_feature_names_out())
    explainer = build_explainer(clf)
    base_value = float(np.ravel(explainer.expected_value)[-1])

    curr_e = engineer(load_current())
    ce = ClaimExplainer(model, explainer, feat_names).fit_transform(curr_e)
    curr_scores = model.predict_proba(curr_e[config.FEATURES])[:, 1]
    top_pos = np.argsort(curr_scores)[::-1][:config.TOP_N_EXPLANATIONS]

    explanations = []
    print("\n=== Top-10 riskiest current claims (SHAP drivers) ===")
    for pos in top_pos:
        pos = int(pos)
        prob = float(curr_scores[pos])
        text = ce.text_explanation(pos, prob, risk_tier(prob, threshold, median_cut))
        explanations.append({
            config.ID_COL: curr_e.iloc[pos][config.ID_COL],
            "denial_prob": prob,
            "explanation": text,
        })
        print(text + "\n")

    with open(config.TOP_SHAP_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[config.ID_COL, "denial_prob", "explanation"])
        writer.writeheader()
        writer.writerows(explanations)
    print(f"Saved {len(explanations)} SHAP explanations -> {config.TOP_SHAP_CSV}")

    # Persist artifacts.
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    try:
        joblib.dump(explainer, config.EXPLAINER_PATH)
    except Exception as exc:  # noqa: BLE001 — score.py rebuilds it from the model
        print(f"Could not pickle explainer (score.py will rebuild it): {exc}")

    joblib.dump({
        "THRESHOLD": threshold, "MEDIAN_CUT": median_cut, "base_value": base_value,
        "feat_names": feat_names, "FEATURES": config.FEATURES,
        "CATEGORICAL": config.CATEGORICAL, "BINARY_FLAGS": config.BINARY_FLAGS,
        "NUMERIC": config.NUMERIC, "ENGINEERED": config.ENGINEERED,
    }, config.METADATA_PATH)

    print(f"\nSaved artifacts -> {config.ARTIFACTS_DIR}")
    print("  - model.joblib\n  - shap_explainer.joblib\n  - metadata.joblib")


if __name__ == "__main__":
    main()
