"""Part 2: load artifacts, score current claims, explain, write CSV.

Run after `python src/train.py`:

    python src/score.py

Writes predictions_current_claims.csv, sorted by denial_probability (desc),
with columns: claim_id, denial_probability, predicted_denial, risk_tier,
top_risk_factors, explanation.

The top-10 riskiest claims are explained by the LLM; the rest use the template.
Without OPENAI_API_KEY, every row uses the template.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import shap

# Let `python src/score.py` work from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, llm_explainer
from src.data_loader import load_current
from src.explain import ClaimExplainer
from src.preprocessing import engineer

OUTPUT_COLS = ["claim_id", "denial_probability", "predicted_denial",
               "risk_tier", "top_risk_factors", "explanation"]


def risk_tier(prob: float, threshold: float, median_cut: float) -> str:
    if prob >= threshold:
        return "High"
    if prob >= median_cut:
        return "Medium"
    return "Low"


def load_artifacts():
    """Load model + metadata; rebuild the explainer if its pickle is missing."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{config.MODEL_PATH} not found. Run `python src/train.py` first.")

    model = joblib.load(config.MODEL_PATH)
    meta = joblib.load(config.METADATA_PATH)
    try:
        explainer = joblib.load(config.EXPLAINER_PATH)
    except Exception as exc:  # noqa: BLE001 — rebuild from the fitted booster
        print(f"Rebuilding explainer from model: {exc}")
        explainer = shap.TreeExplainer(model.named_steps["clf"])
    return model, meta, explainer


def main() -> None:
    model, meta, explainer = load_artifacts()
    threshold, median_cut = meta["THRESHOLD"], meta["MEDIAN_CUT"]
    print(f"Loaded model + explainer.  THRESHOLD={threshold:.3f}  MEDIAN_CUT={median_cut:.3f}")

    # Score.
    curr_e = engineer(load_current())
    curr_X = curr_e[config.FEATURES]
    probs = model.predict_proba(curr_X)[:, 1]
    preds = (probs >= threshold).astype(int)
    print(f"Scored {len(curr_e)} claims.  flagged (High) = {int(preds.sum())} "
          f"({preds.mean():.0%})")

    # Per-claim SHAP drivers.
    ce = ClaimExplainer(model, explainer, meta["feat_names"]).fit_transform(curr_e)

    records = []
    for pos in range(len(curr_e)):
        drivers = ce.drivers(pos, k=3)
        trf = "; ".join(
            f"{d['feature']}={d['value']} ({d['direction']} risk)" for d in drivers)
        records.append({
            "claim_id": curr_e.iloc[pos][config.ID_COL],
            "denial_probability": round(float(probs[pos]), 4),
            "predicted_denial": int(preds[pos]),
            "risk_tier": risk_tier(float(probs[pos]), threshold, median_cut),
            "top_risk_factors": trf,
            "explanation": "",       # filled below
            "_drivers": drivers,     # internal, dropped before CSV
        })

    records.sort(key=lambda r: r["denial_probability"], reverse=True)

    # LLM explanations for the top-10, template for the rest.
    use_llm = config.has_llm_key()
    print(f"Generating explanations (LLM={'on' if use_llm else 'off'}, model={config.LLM_MODEL})")

    for i, rec in enumerate(records[:config.TOP_N_EXPLANATIONS], 1):
        rec["explanation"] = llm_explainer.explain(rec, use_llm)
        print(f"  [{i:2d}] {rec['claim_id']}  p={rec['denial_probability']:.0%}  {rec['risk_tier']}")
    for rec in records[config.TOP_N_EXPLANATIONS:]:
        rec["explanation"] = llm_explainer.template_explanation(rec)

    # Write CSV.
    with open(config.PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        for rec in records:
            writer.writerow({c: rec[c] for c in OUTPUT_COLS})

    tiers = ", ".join(
        f"{t}={sum(1 for r in records if r['risk_tier'] == t)}"
        for t in ("High", "Medium", "Low"))
    print(f"\nWrote {config.PREDICTIONS_CSV}")
    print(f"  {len(records)} rows, sorted by denial_probability (desc)")
    print(f"  tiers: {tiers}")


if __name__ == "__main__":
    main()
