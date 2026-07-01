# Claim Denial Prediction

A machine-learning pipeline that predicts the probability that a healthcare
insurance claim will be **denied**, ranks current claims by risk so a limited
review team can triage the highest-risk slice, and produces short, plain-English
explanations (grounded in SHAP drivers, written by an LLM) for the riskiest claims.

This repository is the solution to the *ML + basic Gen AI* hiring assessment.

---

## Problem

Given a history of adjudicated claims (`is_denied` label), train a binary
classifier and score a set of new, unadjudicated claims. The operating constraint
from the brief is that **the review team can only inspect the top 25% of claims by
risk**, so the threshold and metrics are chosen around that capacity — not around
a generic 0.5 cutoff.

Two deliverables:

- **Part 1 — Modeling.** Train on `claims_history.csv` using the provided `split`,
  pick a capacity-driven threshold, and score `current_claims.csv`.
- **Part 2 — Gen AI.** Write a short, non-alarmist explanation for each of the
  **top-10 highest-risk** current claims, grounded only in that claim's fields and
  the model's SHAP risk drivers.

---

## Repository layout

```
Denial_Prediction/
├── data/
│   ├── claims_history.csv        # labelled training/validation/test claims (has `split`, `is_denied`)
│   └── current_claims.csv        # unlabelled claims to score
├── notebooks/
│   ├── data_exploration_feature_engg.ipynb   # EDA + feature analysis
│   ├── model_training.ipynb                  # Part 1: FE → model → threshold → SHAP → save artifacts
│   └── scoring_and_explanations.ipynb        # Part 2: load artifacts → score → LLM explanations
├── artifacts/
│   ├── model.joblib              # fitted XGBoost pipeline (preprocessor + booster)
│   ├── shap_explainer.joblib     # SHAP explainer for the booster
│   └── metadata.joblib           # threshold, median cut, feature lists, base value
├── predictions_current_claims.csv    # final scored output (sorted highest → lowest risk)
├── top10_shap_explanations.csv       # raw SHAP driver breakdown for the top-10
├── Denial_Prediction_WriteUp.pdf     # narrative write-up
└── Denial_Prediction_WriteUp.docx
```

---

## Approach

### Feature engineering
One function is applied to **both** datasets so the current claims get exactly the
columns the model was trained on. On top of the raw fields it adds four engineered
features that encode the real denial patterns seen in the data:

- `prior_auth_gap` — prior auth required but not on file
- `referral_gap` — referral required but not present
- `payment_ratio` — `expected_payment / total_billed`
- `log_total_billed`

**Leakage guard:** `is_denied`, `denial_reason`, and `split` are never used as inputs.

### Model
- The provided `split` column is used **as-is** (no custom resampling).
- Preprocessing = one-hot for categoricals (`payer_id`, `payer_type`, `visit_type`),
  pass-through for numerics and binary flags.
- A FLAML AutoML pass is run as a sanity check, then **XGBoost** is selected and
  hyperparameter-tuned (early stopping on validation `recall@25%`).

### Threshold — capacity-driven
The deployed threshold (`capacity_top25`) is the 75th percentile of validation
scores, so it flags exactly the **top 25% by volume**, matching the review-team
constraint. `f1_optimal` and `youden_J` are computed for comparison. All thresholds
are chosen on **validation** and only *reported* on the locked test set.

Operating threshold: **0.579**.

### Risk tiers
| Tier   | Rule                                            |
|--------|-------------------------------------------------|
| High   | prob ≥ top-25% threshold (the reviewed slice)   |
| Medium | median ≤ prob < top-25% threshold               |
| Low    | prob < median                                   |

### Test-set performance (XGBoost)
| Metric          | Value |
|-----------------|-------|
| PR-AUC          | 0.484 |
| ROC-AUC         | 0.684 |
| recall@25%      | 0.457 |
| precision@25%   | 0.474 |
| F1 @ threshold  | 0.466 |

Confusion matrix (test) `[[tn fp][fn tp]]`: `[[322 77][74 66]]`.

---

## Part 2 — Explanations (SHAP + LLM)

For each current claim, SHAP runs on the fitted booster in the transformed space and
one-hot contributions are summed back to the original feature, so the reported
`top_risk_factors` name the real claim fields (e.g. `prior_auth_gap`,
`missing_documentation_flag`, `payment_ratio`). The top 2–3 drivers by absolute
contribution are kept.

The **top-10 highest-risk** claims are explained by an OpenAI LLM. The prompt pins
the rules from the brief — grounded **only** in the provided facts, plain language,
one recommended action, an explicit hedge that this is an estimate (not a guarantee),
2–3 sentences. The user message passes *only* the probability, tier, and SHAP
drivers, so there is nothing for the model to hallucinate from. If no API key is
present, the notebook falls back to a deterministic template so it still runs
end-to-end. Remaining claims get the deterministic template.

### Output — `predictions_current_claims.csv`
Sorted highest → lowest `denial_probability`, with columns:

`claim_id, denial_probability, predicted_denial, risk_tier, top_risk_factors, explanation`

---

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Set the OpenAI key **in the environment** before running Part 2 (do not paste it
into the notebook):

```bash
# Windows (persists; restart the kernel afterwards)
setx OPENAI_API_KEY "sk-..."
# macOS/Linux
export OPENAI_API_KEY="sk-..."
```

## How to run

1. **`notebooks/model_training.ipynb`** — feature engineering, model training,
   threshold selection, and SHAP. Saves `model.joblib`, `shap_explainer.joblib`, and
   `metadata.joblib` into `artifacts/`. Run this once so the `artifacts/` folder exists.
2. **`notebooks/scoring_and_explanations.ipynb`** — loads the saved artifacts, scores
   `current_claims.csv`, generates LLM explanations for the top-10, and writes
   `predictions_current_claims.csv`.

(`notebooks/data_exploration_feature_engg.ipynb` is the exploratory analysis and is
optional to reproduce the deliverables.)

---

## Dependencies
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `flaml`, `shap`,
`matplotlib`, `seaborn`, `joblib`, `openai`. See `requirements.txt`.
