"""Gen AI explanation step.

Turns a scored claim (probability, risk tier, SHAP drivers) into a short,
non-alarmist, plain-English explanation. The user message passes *only* the
model's own outputs, so there is nothing for the model to hallucinate from.

If no API key is configured, a deterministic template obeying the same content
rules is used, so the pipeline always runs end-to-end.
"""
from __future__ import annotations

from . import config

SYSTEM_PROMPT = (
    "You are a claims-denial analyst assistant. Write a short, plain-English risk "
    "explanation for ONE insurance claim. Follow every rule:\n"
    "- Use ONLY the facts provided (probability, tier, and risk drivers). Do NOT invent facts, "
    "codes, names, or dollar amounts that are not given.\n"
    "- Plain language an analyst can act on in seconds; no jargon or acronyms.\n"
    "- Exactly 2-3 sentences.\n"
    "- Include exactly ONE specific, concrete recommended action.\n"
    "- Make clear this is a model risk ESTIMATE, not a guarantee of denial."
)


def build_user_prompt(record: dict) -> str:
    """Compose the user message from a scored claim record and its SHAP drivers."""
    drivers = "\n".join(
        f"- {d['feature']} = {d['value']} ({d['direction']} denial risk)"
        for d in record["_drivers"]
    )
    return (
        f"Claim ID: {record['claim_id']}\n"
        f"Model denial probability: {record['denial_probability']:.0%}\n"
        f"Risk tier: {record['risk_tier']}\n"
        f"Top risk drivers (from the model):\n{drivers}\n\n"
        "Write the explanation now."
    )


def template_explanation(record: dict) -> str:
    """Deterministic fallback (no API needed) obeying the same content rules."""
    d0 = record["_drivers"][0]
    return (
        f"This claim has an estimated {record['denial_probability']:.0%} chance of denial "
        f"({record['risk_tier'].lower()} risk), driven mainly by {d0['feature']} = {d0['value']}. "
        f"Recommended action: verify this field before submission. "
        f"This is a model estimate, not a guaranteed outcome."
    )


def llm_explanation(record: dict) -> str:
    """Call the OpenAI chat API to write the explanation for one claim."""
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
        ],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def explain(record: dict, use_llm: bool) -> str:
    """Explain one claim, falling back to the template if the LLM is unavailable."""
    if not use_llm:
        return template_explanation(record)
    try:
        return llm_explanation(record)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully to the template
        print(f"  LLM failed for {record['claim_id']} ({exc}); using template.")
        return template_explanation(record)
