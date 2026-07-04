"""Central configuration: paths, run constants, and the feature schema.

All other modules import from here so paths and column lists live in exactly one
place. Paths are resolved relative to the project root (the parent of ``src/``),
so the project runs from any machine without editing hard-coded absolute paths.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file (e.g. OPENAI_API_KEY) if present.
# Real secrets live in .env (git-ignored); see .env.example for the template.
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

HISTORY_CSV = DATA_DIR / "claims_history.csv"
CURRENT_CSV = DATA_DIR / "current_claims.csv"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.joblib"

PREDICTIONS_CSV = PROJECT_ROOT / "predictions_current_claims.csv"
TOP_SHAP_CSV = PROJECT_ROOT / "top10_shap_explanations.csv"

# ---------------------------------------------------------------------------
# Run constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
REVIEW_CAPACITY = 0.25          # review team can only inspect the top 25% by risk
TOP_N_EXPLANATIONS = 10         # brief asks for LLM explanations on the top-10

# ---------------------------------------------------------------------------
# Feature schema (single source of truth for both training and scoring)
# ---------------------------------------------------------------------------
TARGET = "is_denied"
ID_COL = "claim_id"

# Leakage columns — never used as model inputs.
LEAKAGE_COLS = ["is_denied", "denial_reason", "split"]

CATEGORICAL = ["payer_id", "payer_type", "visit_type"]
BINARY_FLAGS = [
    "prior_auth_required", "has_prior_auth", "is_in_network",
    "missing_documentation_flag", "eligibility_verified",
    "referral_required", "referral_present",
]
NUMERIC = ["total_billed", "expected_payment", "num_procedures",
           "num_diagnoses", "days_to_submit"]
ENGINEERED = ["prior_auth_gap", "referral_gap", "payment_ratio", "log_total_billed"]

FEATURES = CATEGORICAL + BINARY_FLAGS + NUMERIC + ENGINEERED

# ---------------------------------------------------------------------------
# LLM (Gen AI) settings — provider-agnostic env-driven config
# ---------------------------------------------------------------------------
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "160"))


def has_llm_key() -> bool:
    """True when an OpenAI API key is available in the environment."""
    return bool(os.environ.get("OPENAI_API_KEY"))
