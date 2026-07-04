# Claim Denial Prediction

A machine-learning pipeline that predicts the probability that a healthcare
insurance claim will be **denied**, ranks current claims by risk so a limited
review team can triage the highest-risk slice, and produces short, plain-English
explanations (grounded in SHAP drivers, written by an LLM) for the riskiest claims.

This repository is the solution to the *ML + basic Gen AI* hiring assessment. It
is packaged as a **modular Python project** that runs as scripts
(`python src/train.py`, `python src/score.py`) — the notebooks under `notebooks/`
are kept only as the exploratory analysis.

---

## Problem

Given a history of adjudicated claims (`is_denied` label), train a binary
classifier and score a set of new, unadjudicated claims. The operating constraint
from the brief is that **the review team can only inspect the top 25% of claims by
risk**, so the threshold and metrics are chosen around that capacity — not around
a generic 0.5 cutoff.

Two deliverables:

- **Part 1 — Modeling** (`src/train.py`). Train on `claims_history.csv` using the
  provided `split`, pick a capacity-driven threshold, run SHAP, and save artifacts.
- **Part 2 — Gen AI** (`src/score.py`). Score `current_claims.csv` and write a
  short, non-alarmist explanation for each of the **top-10 highest-risk** claims,
  grounded only in that claim's fields and the model's SHAP risk drivers.

---

## Repository layout

```
Denial_Prediction/
├── src/                              # modular pipeline (production code)
│   ├── config.py                     # paths, constants, feature schema, .env loading
│   ├── data_loader.py                # read the source CSVs
│   ├── preprocessing.py              # feature engineering + preprocessor + split
│   ├── model.py                      # capacity metrics + XGBoost tuning + evaluation
│   ├── explain.py                    # SHAP per-claim risk drivers
│   ├── llm_explainer.py              # LLM explanation step (+ deterministic fallback)
│   ├── train.py                      # ENTRY POINT — Part 1 (train → threshold → SHAP → save)
│   └── score.py                      # ENTRY POINT — Part 2 (load → score → explain → CSV)
├── notebooks/                        # exploration only (not the deliverable path)
│   ├── data_exploration_feature_engg.ipynb
│   ├── model_training.ipynb
│   └── scoring_and_explanations.ipynb
├── data/
│   ├── claims_history.csv            # labelled claims (has `split`, `is_denied`)
│   └── current_claims.csv            # unlabelled claims to score
├── artifacts/                        # written by train.py
│   ├── model.joblib                  # fitted XGBoost pipeline (preprocessor + booster)
│   ├── shap_explainer.joblib         # SHAP explainer for the booster
│   └── metadata.joblib               # threshold, median cut, feature lists, base value
├── predictions_current_claims.csv    # final scored output (written by score.py)
├── top10_shap_explanations.csv       # raw SHAP driver breakdown for the top-10
├── requirements.txt
├── .env.example                      # placeholder API key template
└── README.md
```

---

## Approach

### Feature engineering (`preprocessing.py`)
One function (`engineer`) is applied to **both** datasets so the current claims get
exactly the columns the model was trained on. On top of the raw fields it adds four
engineered features that encode the real denial patterns seen in the data:

- `prior_auth_gap` — prior auth required but not on file
- `referral_gap` — referral required but not present
- `payment_ratio` — `expected_payment / total_billed`
- `log_total_billed`

**Leakage guard:** `is_denied`, `denial_reason`, and `split` are never used as inputs.

### Model (`model.py`)
- The provided `split` column is used **as-is** (no custom resampling).
- Preprocessing = one-hot for categoricals (`payer_id`, `payer_type`, `visit_type`),
  standard-scale for numerics, pass-through for binary flags.
- **XGBoost** is hyperparameter-tuned via randomised search with early stopping on a
  custom `recall@25%` metric. (The notebook also runs a FLAML AutoML pass as a
  sanity check.)

### Threshold — capacity-driven
The deployed threshold is the 75th percentile of validation scores, so it flags
exactly the **top 25% by volume**, matching the review-team constraint. Thresholds
are chosen on **validation** and only *reported* on the locked test set.

Operating threshold: **~0.579**.

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

For each current claim, SHAP (`explain.py`) runs on the fitted booster in the
transformed space and one-hot contributions are summed back to the original
feature, so the reported `top_risk_factors` name the real claim fields (e.g.
`prior_auth_gap`, `missing_documentation_flag`, `payment_ratio`). The top 3 drivers
by absolute contribution are kept.

The **top-10 highest-risk** claims are explained by an LLM (`llm_explainer.py`). The
prompt pins the rules from the brief — grounded **only** in the provided facts,
plain language, one recommended action, an explicit hedge that this is an estimate
(not a guarantee), 2–3 sentences. The user message passes *only* the probability,
tier, and SHAP drivers, so there is nothing for the model to hallucinate from. If no
API key is present, a deterministic template is used so the pipeline still runs
end-to-end. Remaining claims always get the deterministic template.

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

Configure the API key for the LLM step by copying the template and filling in your
key (do **not** commit real credentials — `.env` is git-ignored):

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

Alternatively, export it in the shell instead of using `.env`:

```bash
# Windows
setx OPENAI_API_KEY "sk-..."
# macOS/Linux
export OPENAI_API_KEY="sk-..."
```

---

## How to run

Run both scripts from the project root, in order:

```bash
# Part 1 — train, threshold, SHAP, save artifacts/ + top10_shap_explanations.csv
python src/train.py

# Part 2 — load artifacts, score current claims, LLM explanations, write predictions
python src/score.py
```

`src/train.py` must be run once so the `artifacts/` folder exists before scoring.
`src/score.py` runs without an API key too — it just falls back to the template
explanations.

---

## Dependencies
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `flaml`, `shap`,
`matplotlib`, `seaborn`, `joblib`, `openai`, `python-dotenv`. See `requirements.txt`.
