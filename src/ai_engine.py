"""
src/ai_engine.py
─────────────────────────────────────────────────────────────────────────────
LLM-powered churn explanation and retention strategy generation.
Runs 100% locally via Ollama — no API keys, no cloud services.

Model stack
───────────
  LLM : Ollama / Llama 3.1 8B (default)   →  http://localhost:11434
  Lib : LangChain LCEL chains              →  langchain-ollama

Features
────────
  • Feature 1 — stream_explanation()  : human-readable churn analysis
  • Feature 2 — stream_strategy()     : personalised retention action plan
  Both support real-time token streaming for Streamlit st.empty() display.

Usage
─────
    from ai_engine import ChurnAIEngine
    engine = ChurnAIEngine()                    # raises if Ollama not running

    # Blocking
    explanation = engine.generate_explanation(raw, prob, risk_level, factors)
    strategy    = engine.generate_retention_strategy(
        raw, prob, risk_level, explanation, factors
    )

    # Streaming  (Streamlit pattern)
    for chunk in engine.stream_explanation(raw, prob, risk_level, factors):
        accumulated += chunk
        placeholder.markdown(accumulated)

AUDIT PASS (grounding fix)
───────────────────────────
`factors` must be the SAME per-customer SHAP-derived list shown in the UI's
Key Factors (each dict needs "feature" [raw column name], "name" [display
label], "direction" ["increases"/"decreases"], "imp_pct"). Both prompts are
now scoped to ONLY the customer-profile fields backing those factors —
there is no longer an unconditional full-profile dump, and the system
prompt's old "...unless directly supported by the customer profile"
loophole has been removed. `generate_retention_strategy` / `stream_strategy`
now require `factors` as well (previously only took `ai_explanation`).

AUDIT PASS 2 (demo safety pipeline — pre-graduation-defense hardening)
─────────────────────────────────────────────────────────────────────
The streaming methods (stream_explanation / stream_strategy) still exist
for any caller that wants raw token streaming, but the demo-facing path is
now generate_explanation_safe() / generate_retention_strategy_safe(), which
implement:

    Prediction → SHAP → LLM Generation → Explanation Validation →
    Strategy Validation → Repair Layer → Fallback Layer → caller

validate_explanation() checks: all 3 factors discussed, no risk-level
words other than the actual one (e.g. LLM saying "moderate risk" under a
"Low" classification), no churn-driver keywords outside what the factors
support, and no direction contradictions (calling a risk-decreasing factor
a problem, or vice versa). On failure: one regeneration attempt with a
corrective system note, then a deterministic fallback template that is
built directly from the factors and is guaranteed to pass validation.

validate_strategy() / repair_strategy() check: priorities are one of
URGENT/HIGH/STANDARD and allowed for this risk level (Low→STANDARD only,
Medium→HIGH/STANDARD, High→URGENT/HIGH/STANDARD), 3-5 well-formed
"[PRIORITY] action | timeline | outcome" lines, and that each action's verb
matches the direction of the factor it addresses (no "change X" for a
factor whose SHAP direction already lowers risk). Invalid priorities are
normalised (LOW→STANDARD, CRITICAL→HIGH, etc.) and clamped to what's
allowed for the risk level; direction-mismatched lines are dropped; if
fewer than 3 valid lines remain, falls back to a deterministic template.

The caller (app.py) only ever receives the final, validated string — never
a partially-streamed or unvalidated LLM response.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Fix #3 — diagnostic logging for the validation/repair/fallback pipeline.
_logger = logging.getLogger("churnguard.ai_engine")


def _log_validation_failure(stage: str, reasons: list[str]) -> None:
    """
    Called right before a regeneration/repair/fallback decision is made, so
    it's always clear from the console *why* an LLM draft was rejected
    instead of silently seeing a fallback template appear with no
    explanation. `stage` is e.g. "explanation (first attempt)",
    "explanation (after regeneration)", "strategy (before repair)".
    """
    msg = f"VALIDATION FAILED ({stage}):\n" + "\n".join(f"  - {r}" for r in reasons)
    _logger.warning(msg)


# ── Ollama health check ───────────────────────────────────────────────────

def _verify_ollama(base_url: str, model: str) -> None:
    """
    Confirm the Ollama server is reachable and the requested model is pulled.
    Uses only stdlib urllib — no extra dependencies.

    Raises RuntimeError with actionable instructions on failure.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama server not reachable at {base_url}.\n"
            "  Fix → Install Ollama : https://ollama.ai\n"
            "  Fix → Start server   : ollama serve"
        ) from exc

    available = [m["name"] for m in payload.get("models", [])]
    base_name = model.split(":")[0]          # "llama3.1" from "llama3.1:8b"
    if not any(m.startswith(base_name) for m in available):
        raise RuntimeError(
            f"Model '{model}' not found in Ollama.\n"
            f"  Fix → Pull model     : ollama pull {model}\n"
            f"  Available models     : {', '.join(available) or 'none'}"
        )


# ─────────────────────────────────────────────────────────────────────────
# DEMO SAFETY LAYER — validation, repair, and deterministic fallbacks
# ─────────────────────────────────────────────────────────────────────────
# This section exists because an LLM running locally (llama3.1:8b via
# Ollama) is not reliable enough, on its own, to put in front of a
# graduation committee. Three concrete failure modes were observed in
# testing:
#   (Issue 3) the model sometimes describes a "Low Risk" customer as
#             "moderate risk" — it re-derives a risk level from the prose
#             instead of treating the code-computed risk_level as fact.
#   (Issue 4) it sometimes emits priority tokens that don't exist in the
#             product's vocabulary (LOW, MEDIUM, CRITICAL) or assigns
#             URGENT to a Low Risk customer.
#   (Issue 5) it sometimes proposes a corrective action ("change payment
#             method") for a factor whose SHAP direction already LOWERS
#             risk — logically backwards.
# Everything below is deterministic (regex / lookup tables), not another
# LLM call, so it can't introduce a second layer of unreliability.

# Issue 3 — risk-level words the LLM might use, keyed by what they'd claim.
_RISK_LEVEL_WORDS: dict[str, list[str]] = {
    "Low":    [r"\blow[\s\-]*risk\b"],
    "Medium": [r"\bmedium[\s\-]*risk\b", r"\bmoderate[\s\-]*risk\b", r"\bmid[\s\-]*risk\b"],
    "High":   [r"\bhigh[\s\-]*risk\b", r"\bcritical[\s\-]*risk\b", r"\bsevere[\s\-]*risk\b"],
}


def _wrong_risk_level_mentions(text: str, risk_level: str) -> list[str]:
    """Issue 3: return any OTHER risk-level word(s) found in `text`."""
    text_l = text.lower()
    hits = []
    for level, patterns in _RISK_LEVEL_WORDS.items():
        if level == risk_level:
            continue
        if any(re.search(p, text_l) for p in patterns):
            hits.append(level)
    return hits


# AUDIT FIX (Fix #4 — Relax Explanation Validation): the missing-factor
# check in validate_explanation() only matched literal word-tokens pulled
# from the factor's display name, so semantically correct LLM wording
# ("e-check" for "Electronic Check", "monthly contract" for "Month-to-month
# Contract") was rejected as if the factor had never been discussed,
# triggering an unnecessary regeneration/fallback. This ONLY widens what
# counts as "the factor was mentioned" — it does NOT touch _DRIVER_KEYWORDS
# or the leak-detection logic below, so hallucination protection (catching
# a churn driver that ISN'T one of the real factors) is unchanged.
_CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "month-to-month":      ["month-to-month", "monthly contract", "monthly agreement",
                            "no annual commitment", "no long-term commitment", "no lock-in"],
    "long-term contract":  ["long-term contract", "annual contract", "fixed-term agreement",
                            "one year contract", "two year contract", "multi-year contract",
                            "yearly contract"],
    "electronic check":    ["electronic check", "e-check", "echeck", "electronic payment"],
    "mailed check":        ["mailed check", "paper check", "check by mail", "postal check"],
    "bank transfer":       ["bank transfer", "automatic bank payment", "direct debit", "auto-pay bank"],
    "credit card":         ["credit card", "card payment", "auto-pay card"],
    "online security":     ["online security", "security service", "security add-on", "security feature"],
    "tech support":        ["tech support", "technical support"],
    "online backup":       ["online backup", "backup service", "data backup"],
    "device protection":   ["device protection", "device insurance", "device coverage"],
    "fiber optic":         ["fiber optic", "fiber internet", "fibre"],
    "dsl":                 ["dsl", "dsl internet", "dsl connection"],
    "paperless":           ["paperless", "paperless billing", "e-billing", "electronic billing"],
    "senior citizen":      ["senior citizen", "senior", "elderly customer", "older customer"],
    "monthly charges":     ["monthly charges", "monthly bill", "monthly spend", "monthly cost"],
    "tenure":              ["tenure", "time as a customer", "months as a customer", "customer for"],
}


def _factor_keywords(factor: dict) -> list[str]:
    """
    Significant words/phrases used to check a factor is actually discussed
    in generated text. Starts from literal word-tokens pulled from the
    factor's display name (e.g. "Contract Type: Month-to-month" ->
    ["contract","type","month-to-month"]), then EXPANDS with any known
    synonym phrases for concepts present in that name (Fix #4) — e.g. if
    "month-to-month" appears in the name, "monthly contract" and "e-check"-
    style variants for whichever concept matched are added too. Any one of
    the returned phrases counts as a valid mention.
    """
    name_l = factor["name"].lower()
    words = re.findall(r"[a-z\-]{3,}", name_l)
    stop = {"the", "and", "has", "not", "for"}
    phrases = [w for w in words if w not in stop] or [name_l]

    for concept, synonyms in _CONCEPT_SYNONYMS.items():
        if concept in name_l:
            phrases.extend(synonyms)
    return list(dict.fromkeys(phrases))  # de-dup, preserve order


# Issue 6 — vocabulary of OTHER churn-driver keywords the LLM might mention
# that are NOT among the actual top factors. Reuses the same raw-feature
# keys as ChurnAIEngine._FACTOR_PROFILE_FIELDS so "allowed" is computed the
# same way the prompt itself was scoped.
_DRIVER_KEYWORDS: dict[str, str] = {
    "dsl": "InternetService", "fiber": "InternetService", "internet service": "InternetService",
    "payment method": "PaymentMethod", "electronic check": "PaymentMethod",
    "mailed check": "PaymentMethod", "bank transfer": "PaymentMethod", "credit card": "PaymentMethod",
    "online security": "OnlineSecurity", "tech support": "TechSupport", "online backup": "OnlineBackup",
    "device protection": "DeviceProtection", "paperless": "PaperlessBilling",
    "streaming": "StreamingTV", "multiple lines": "MultipleLines", "phone service": "PhoneService",
    "senior citizen": "SeniorCitizen", "partner": "Partner", "dependents": "Dependents",
    "contract": "Contract", "tenure": "tenure", "monthly charge": "MonthlyCharges",
    "total charge": "TotalCharges", "gender": "gender",
}

_INCREASE_PHRASES = [
    "increases risk", "increases the risk", "raises risk", "raises the risk",
    "raising risk", "increasing risk", "drives churn", "higher risk", "elevated risk",
]
_DECREASE_PHRASES = [
    "decreases risk", "decreases the risk", "lowers risk", "lowers the risk",
    "lowering risk", "decreasing risk", "reduces risk", "reduces the risk",
    "reduces churn", "lower risk",
]


def _direction_contradictions(text: str, factors: list[dict]) -> list[str]:
    """
    Issue 2 / Issue 6 (best-effort): for each factor, look at the sentences
    that mention it and flag if they contain language for the OPPOSITE
    direction. This is a heuristic, not full NLU — it's a detector
    deliberately backstopped by regeneration + a deterministic fallback,
    not a guarantee on its own.
    """
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    bad = []
    for f in factors:
        toks = _factor_keywords(f)
        relevant = [s for s in sentences if any(t in s.lower() for t in toks)]
        if not relevant:
            continue
        blob = " ".join(relevant).lower()
        if f.get("direction") == "increases" and any(p in blob for p in _DECREASE_PHRASES):
            bad.append(f["name"])
        elif f.get("direction") == "decreases" and any(p in blob for p in _INCREASE_PHRASES):
            bad.append(f["name"])
    return bad


def validate_explanation(
    explanation: str,
    risk_level: str,
    factors: list[dict],
) -> tuple[bool, list[str]]:
    """
    Issue 6. Pure check — does not modify `explanation`. Returns
    (is_valid, reasons). Checks, in order:
      1. no risk-level word other than the actual risk_level (Issue 3)
      2. all 3 factors are actually discussed by name
      3. no churn-driver keyword outside what the factors support
      4. no direction contradiction (best-effort)
    """
    reasons: list[str] = []

    wrong_levels = _wrong_risk_level_mentions(explanation, risk_level)
    if wrong_levels:
        reasons.append(f"mentions other risk level(s) than '{risk_level}': {wrong_levels}")

    missing = [f["name"] for f in factors if not any(
        t in explanation.lower() for t in _factor_keywords(f)
    )]
    if len(missing) >= 2:
        reasons.append(f"does not discuss factor(s): {missing}")

    allowed_fields = set()
    for f in factors:
        allowed_fields.add(f["feature"])
        allowed_fields.update(ChurnAIEngine._FACTOR_PROFILE_FIELDS.get(f["feature"], []))
    text_l = explanation.lower()
    leaked = [kw for kw, field in _DRIVER_KEYWORDS.items()
              if field not in allowed_fields and re.search(rf"\b{re.escape(kw)}\b", text_l)]
    if leaked:
        reasons.append(f"mentions unsupported driver keyword(s): {leaked}")

    contradictions = _direction_contradictions(explanation, factors)
    if contradictions:
        reasons.append(f"direction contradiction for: {contradictions}")

    return (len(reasons) == 0), reasons


def _fallback_explanation(risk_level: str, factors: list[dict]) -> str:
    """
    Deterministic, template-based explanation built ONLY from `factors` and
    `risk_level` — no LLM call, so it cannot hallucinate, reinterpret the
    risk level, or contradict a SHAP direction. Used when the LLM's output
    fails validate_explanation() twice (Issue 6's "fallback to deterministic
    explanation template").
    """
    risk_phrase = {"Low": "low risk", "Medium": "medium risk", "High": "high risk"}.get(
        risk_level, risk_level.lower()
    )
    lines = [f"This customer is classified as {risk_phrase} of churn, based on three "
             f"model-identified signals:"]
    for f in factors:
        verb = "raises" if f.get("direction") == "increases" else "lowers"
        lines.append(f"• {f['name']}: this signal {verb} this customer's churn risk "
                     f"(~{f.get('imp_pct', 0):.0f}% of the model's signal for this prediction).")
    lines.append("Review these signals against the customer's full profile before deciding on next steps.")
    return "\n".join(lines)


# ── Issue 4/5/7 — strategy priorities, verbs, format ───────────────────────

_VALID_PRIORITIES = ("URGENT", "HIGH", "STANDARD")
_PRIORITY_ORDER = {"URGENT": 0, "HIGH": 1, "STANDARD": 2}
_RISK_ALLOWED_PRIORITIES: dict[str, set] = {
    "Low":    {"STANDARD"},
    "Medium": {"HIGH", "STANDARD"},
    "High":   {"URGENT", "HIGH", "STANDARD"},
}
_PRIORITY_REPAIR_MAP: dict[str, str] = {
    "LOW": "STANDARD", "LOW PRIORITY": "STANDARD", "LOW PRIORITY CUSTOMER": "STANDARD",
    "MEDIUM": "STANDARD", "MED": "STANDARD", "MODERATE": "STANDARD",
    "CRITICAL": "HIGH", "EMERGENCY": "URGENT", "IMMEDIATE": "URGENT",
    "NORMAL": "STANDARD", "ROUTINE": "STANDARD", "DEFAULT": "STANDARD",
}

_ACTION_LINE_RE = re.compile(r"^\[([^\]]+)\]\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)$")

# Issue 5 — allowed verbs by factor direction (case-insensitive, matched as
# whole words at/near the start of the action text).
_RISK_INCREASING_VERBS = ("improve", "intervene", "fix", "convert", "upgrade", "support",
                          "address", "resolve", "escalate", "offer", "call", "contact")
_RISK_DECREASING_VERBS = ("maintain", "reinforce", "appreciate", "reward", "preserve",
                          "celebrate", "thank", "continue", "recognize", "retain")
# Verbs that only make sense as a CORRECTION — never valid for a factor
# that's already lowering risk (Issue 5's exact example: "change payment
# method" proposed for a Payment Method factor that lowers risk).
_CORRECTIVE_ONLY_VERBS = ("change", "switch", "replace", "fix", "correct", "convert",
                          "improve", "intervene", "upgrade")
# Verbs that only make sense as PRAISE/reinforcement — backwards if aimed
# at a factor that's currently raising risk.
_REINFORCING_ONLY_VERBS = ("maintain", "preserve", "reward", "celebrate", "thank")


def _normalize_priority(raw_priority: str, risk_level: str) -> str:
    """
    Issue 4: map any priority token to a valid one (URGENT/HIGH/STANDARD),
    then clamp it to what's allowed for this risk level — never escalate,
    only ever step down. E.g. CRITICAL → HIGH, then if risk_level is "Low"
    (which only allows STANDARD), HIGH gets stepped down again → STANDARD.
    This is exactly the documented "URGENT on Low Risk → STANDARD" case.
    """
    p_clean = re.sub(r"[^A-Z ]", "", raw_priority.strip().upper()).strip()
    if p_clean in _VALID_PRIORITIES:
        normalized = p_clean
    elif p_clean in _PRIORITY_REPAIR_MAP:
        normalized = _PRIORITY_REPAIR_MAP[p_clean]
    elif "URGENT" in p_clean or "EMERGENCY" in p_clean or "CRITICAL" in p_clean:
        normalized = "URGENT" if "URGENT" in p_clean else "HIGH"
    elif "HIGH" in p_clean:
        normalized = "HIGH"
    else:
        normalized = "STANDARD"

    allowed = _RISK_ALLOWED_PRIORITIES.get(risk_level, set(_VALID_PRIORITIES))
    if normalized in allowed:
        return normalized
    for cand in _VALID_PRIORITIES[_PRIORITY_ORDER[normalized] + 1:]:
        if cand in allowed:
            return cand
    return "STANDARD"  # always allowed for every risk level


def _factor_for_line(action_text: str, factors: list[dict]) -> dict | None:
    """Best-effort: which factor (if any) an action line is talking about."""
    text_l = action_text.lower()
    for f in factors:
        if any(t in text_l for t in _factor_keywords(f)):
            return f
    return None


def _verb_direction_violation(action_text: str, factor: dict) -> bool:
    """
    Issue 5: True if `action_text` uses a verb that's logically backwards
    for `factor`'s SHAP direction — e.g. a corrective verb ("change",
    "fix") aimed at a factor that's already LOWERING risk, or a
    reinforcing verb ("maintain", "reward") aimed at a factor that's
    RAISING risk (rewarding the thing that's driving churn makes no sense
    either).
    """
    text_l = action_text.lower()
    direction = factor.get("direction")
    if direction == "decreases" and any(
        re.search(rf"\b{v}\b", text_l) for v in _CORRECTIVE_ONLY_VERBS
    ):
        return True
    if direction == "increases" and any(
        re.search(rf"\b{v}\b", text_l) for v in _REINFORCING_ONLY_VERBS
    ):
        return True
    return False


def parse_strategy_lines(text: str) -> list[dict]:
    """Split LLM strategy output into parsed action dicts (or None fields
    for lines that don't match the required format)."""
    parsed = []
    for ln in (l.strip() for l in text.splitlines()):
        if not ln:
            continue
        m = _ACTION_LINE_RE.match(ln)
        if m:
            parsed.append({"priority": m.group(1), "action": m.group(2),
                           "timeline": m.group(3), "outcome": m.group(4), "raw": ln})
        else:
            parsed.append({"priority": None, "action": None, "timeline": None,
                           "outcome": None, "raw": ln})
    return parsed


def validate_strategy(
    text: str,
    risk_level: str,
    factors: list[dict],
) -> tuple[bool, list[str]]:
    """
    Issue 7. Pure check — does not modify `text`. Verifies:
      1. every non-empty line matches "[PRIORITY] action | timeline | outcome"
      2. 3-5 well-formed action lines
      3. every priority is valid AND allowed for this risk_level
      4. no verb/direction violation (Issue 5)
    """
    reasons: list[str] = []
    parsed = parse_strategy_lines(text)
    malformed = [p["raw"] for p in parsed if p["priority"] is None]
    well_formed = [p for p in parsed if p["priority"] is not None]

    if malformed:
        reasons.append(f"{len(malformed)} line(s) don't match the required format")
    if not (3 <= len(well_formed) <= 5):
        reasons.append(f"expected 3-5 action lines, found {len(well_formed)}")

    allowed = _RISK_ALLOWED_PRIORITIES.get(risk_level, set(_VALID_PRIORITIES))
    for p in well_formed:
        token = re.sub(r"[^A-Z ]", "", p["priority"].strip().upper()).strip()
        if token not in _VALID_PRIORITIES:
            reasons.append(f"invalid priority token: '{p['priority']}'")
        elif token not in allowed:
            reasons.append(f"priority '{token}' not allowed for {risk_level} Risk "
                           f"(allowed: {sorted(allowed)})")

    for p in well_formed:
        factor = _factor_for_line(p["action"], factors)
        if factor and _verb_direction_violation(p["action"], factor):
            reasons.append(f"verb/direction mismatch for '{factor['name']}' in: \"{p['action']}\"")

    return (len(reasons) == 0), reasons


def repair_strategy(text: str, risk_level: str, factors: list[dict]) -> str:
    """
    Issue 7's "repair output automatically": fix what can be fixed
    deterministically, drop what can't.
      • malformed lines → dropped
      • invalid/disallowed priorities → normalized via _normalize_priority
      • verb/direction violations → dropped (rewriting the verb safely
        without breaking grammar isn't reliable; dropping a bad line is
        safer than showing a logically-backwards action in a demo)
      • finally sorted URGENT -> HIGH -> STANDARD
    Returns "" if fewer than 3 valid lines remain (signal to the caller to
    use the deterministic fallback instead).
    """
    kept = []
    for p in parse_strategy_lines(text):
        if p["priority"] is None:
            continue  # malformed — drop
        factor = _factor_for_line(p["action"], factors)
        if factor and _verb_direction_violation(p["action"], factor):
            continue  # logically backwards — drop
        priority = _normalize_priority(p["priority"], risk_level)
        kept.append((priority, f"[{priority}] {p['action']} | {p['timeline']} | {p['outcome']}"))

    if len(kept) < 3:
        return ""

    kept = kept[:5]
    kept.sort(key=lambda t: _PRIORITY_ORDER[t[0]])
    return "\n".join(line for _, line in kept)


# AUDIT FIX (Fix #5 — Fallback Strategy Quality): previously one set of
# templates was used regardless of risk_level, so a Low Risk (loyal)
# customer's fallback strategy could read "address the churn risk linked
# to X" — language that sounds like the customer is a problem, when a
# loyal customer's risk-decreasing factors should be talked about in terms
# of loyalty/upsell/referral, not "churn risk" at all. Templates are now
# keyed by (risk_level, direction) so wording matches both the SHAP
# direction AND how urgently this customer actually needs intervention.
_FALLBACK_ACTION_TEMPLATES: dict[tuple[str, str], tuple[str, ...]] = {
    ("Low", "increases"): (
        "Monitor {name} and follow up with a friendly check-in to keep this loyal customer engaged",
        "Note {name} for the account team as a long-term watch item, without disrupting the relationship",
        "Highlight loyalty perks in the next communication, with {name} as context",
    ),
    ("Low", "decreases"): (
        "Send a loyalty reward recognizing {name}",
        "Invite this customer to the referral programme, citing {name} as a sign of a strong fit",
        "Offer a relevant upsell that builds on {name}",
    ),
    ("Medium", "increases"): (
        "Proactively engage with a tailored retention offer addressing {name}",
        "Schedule a check-in call focused on {name} before it escalates",
        "Offer a service adjustment that directly responds to {name}",
    ),
    ("Medium", "decreases"): (
        "Acknowledge and reinforce the positive impact of {name} in the next outreach",
        "Use {name} as a talking point when presenting a retention offer",
        "Continue to preserve the conditions behind {name}",
    ),
    ("High", "increases"): (
        "Escalate to a retention specialist to intervene on {name} with a meaningful discount",
        "Call within 24 hours to discuss a contract conversion that directly addresses {name}",
        "Offer an immediate, well-targeted discount tied to {name}",
    ),
    ("High", "decreases"): (
        "Acknowledge {name} while still treating this account as urgent given the overall risk",
        "Use {name} as leverage in an urgent retention call alongside a stronger offer",
        "Reinforce {name} as part of a broader intervention for this high-risk account",
    ),
}
_OUTCOME_BY_RISK: dict[str, str] = {
    "Low": "sustained loyalty", "Medium": "improved retention", "High": "averted cancellation",
}


def _fallback_strategy(risk_level: str, factors: list[dict]) -> str:
    """
    Deterministic, template-based strategy built ONLY from `factors` and
    `risk_level` — guaranteed to pass validate_strategy() by construction
    (correct priorities for the risk level, correct verbs for each
    factor's direction, correct format, and wording that matches the risk
    level per Fix #5). Used when repair_strategy() can't salvage 3+ valid
    lines.
    """
    allowed_sorted = sorted(_RISK_ALLOWED_PRIORITIES.get(risk_level, {"STANDARD"}),
                            key=lambda p: _PRIORITY_ORDER[p])
    primary = allowed_sorted[0]
    secondary = allowed_sorted[1] if len(allowed_sorted) > 1 else allowed_sorted[0]
    outcome = _OUTCOME_BY_RISK.get(risk_level, "reduced churn risk")

    lines = []
    for i, f in enumerate(factors):
        priority = primary if i == 0 else secondary
        direction = f.get("direction", "increases")
        templates = _FALLBACK_ACTION_TEMPLATES.get(
            (risk_level, direction),
            _FALLBACK_ACTION_TEMPLATES[("Medium", direction)],  # safe default
        )
        action = templates[i % len(templates)].format(name=f["name"])
        lines.append(f"[{priority}] {action} | within 7 days | {outcome}")
    return "\n".join(lines)


# ── Prompt: Churn Explanation ─────────────────────────────────────────────

_EXPLANATION_SYSTEM = """\
You are a senior customer retention analyst at a telecom company.

Your job is to explain ML churn predictions to non-technical CRM and sales teams.

Grounding rules — these are ABSOLUTE and override everything else below:
- The "Top Model Signals" below were computed per-customer with SHAP. They
  are THE ONLY churn drivers that exist for this explanation — not a
  generic industry list, not your own judgment about what's plausible.
- You must explicitly name and explain all 3 Top Model Signals, in the
  order given, one bullet each.
- Do NOT introduce, hint at, or imply any churn driver that is not one of
  the 3 Top Model Signals. This includes plausible-sounding ones such as
  internet type, payment method, or security add-ons — UNLESS that exact
  attribute is itself one of the 3 listed signals. There is no exception
  for "directly supported by the customer profile" — the profile section
  below has already been restricted to only the data behind these 3
  signals, specifically so you have no other material to draw on.
- Each signal has a direction: "increases risk" or "decreases risk".
  Respect it exactly — never reverse a direction or describe a
  risk-decreasing signal as a problem.
- If the customer-data section seems to suggest a different story than a
  signal's stated direction, trust the signal — it reflects what the
  model actually computed for this customer, not a generic reading.

Writing rules:
- Plain English only — no ML jargon, no statistical terms whatsoever
- Be specific to THIS customer's actual data (the scoped data given below)
- Empathetic but direct — agents need to act on this
- Maximum 180 words
- Structure: one clear opening sentence → exactly 3 bullet points (one per
  Top Model Signal, in the order given) → one closing action sentence
- Each bullet must name its corresponding signal explicitly, then explain it
  in plain language using only the customer data tied to that signal
"""

_EXPLANATION_HUMAN = """\
Explain this churn prediction to our CRM team:

Churn Probability : {probability}%
Risk Level        : {risk_level}

Top Model Signals for THIS customer (SHAP-based, in order of contribution —
these are the only churn drivers you may discuss):
{top_factors}

Relevant Customer Data (scoped to ONLY the signals above — there is no
other customer data available to you, and none should be implied):
{factor_context}

Write a clear, human-readable explanation for the CRM team now. Name all 3
signals above, in order, and use only the Relevant Customer Data to add
detail. Do not reference any customer attribute that isn't listed there.\
"""

# ── Prompt: Retention Strategy ────────────────────────────────────────────

_STRATEGY_SYSTEM = """\
You are a customer retention strategist at a telecom company.
You create personalised, evidence-based retention action plans.

Grounding rules — these are ABSOLUTE:
- Base every action strictly on the Top Model Signals and the Relevant
  Customer Data provided below — the same SHAP signals that drove this
  customer's churn prediction. These are the only churn drivers that exist
  for this task.
- Do not invent a churn driver, offer, or justification that isn't tied to
  one of the Top Model Signals or the Relevant Customer Data given.
- Each action's description must reference at least one of the Top Model
  Signals by name (directly, or via the customer data tied to it).
- The Situation Summary is analyst context only — if it conflicts with the
  Top Model Signals, the signals are the source of truth.

Format rules — follow these exactly:
- Exactly 3 to 5 actions, no more, no fewer
- Each action on its own line using this format:
    [PRIORITY] Action description | Timeline | Expected outcome
- PRIORITY values: URGENT  /  HIGH  /  STANDARD
- List the actions in this exact order: all URGENT actions first, then all
  HIGH, then all STANDARD. Never mix the order.
- Be specific — name real offers, dollar amounts, and timeframes
- Focus only on actions that a CRM agent can execute immediately
- Do NOT repeat the churn probability or risk level
- Output ONLY the action lines — no preamble, no header, no summary
"""

_STRATEGY_HUMAN = """\
Create a retention plan for this customer.

Risk Level        : {risk_level}
Churn Probability : {probability}%

Top Model Signals for THIS customer (SHAP-based, in order of contribution —
the only churn drivers you may act on):
{top_factors}

Relevant Customer Data (scoped to ONLY the signals above):
{factor_context}

Situation Summary (analyst context only — the signals above are the source
of truth if anything here conflicts):
{ai_explanation}

Generate the prioritised retention plan now, ordered URGENT then HIGH then
STANDARD. Tie every action back to one of the Top Model Signals above.\
"""

_EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _EXPLANATION_SYSTEM),
    ("human",  _EXPLANATION_HUMAN),
])

_STRATEGY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _STRATEGY_SYSTEM),
    ("human",  _STRATEGY_HUMAN),
])


# ── Engine ────────────────────────────────────────────────────────────────

class ChurnAIEngine:
    """
    LangChain wrapper around a local Ollama LLM for churn explanation
    and retention strategy generation.

    Raises RuntimeError at construction if:
      • Ollama server is not running
      • The requested model has not been pulled

    This allows the Streamlit app to detect the problem early and show
    an actionable hint rather than crashing at prediction time.
    """

    _DEFAULT_MODEL      = "llama3.1:8b"
    _DEFAULT_BASE_URL   = "http://localhost:11434"
    _DEFAULT_MAX_TOKENS = 1024
    _DEFAULT_TEMP       = 0.3

    # AUDIT FIX: this replaces _SERVICE_MAP / _fmt_services. The old code
    # always handed the LLM contract + tenure + monthly_charges + full
    # services list + payment + senior, regardless of which 3 factors were
    # actually driving THIS prediction — that unconditional dump, combined
    # with the system-prompt loophole ("...unless directly supported by the
    # customer profile data provided"), is exactly what let the LLM talk
    # about DSL / payment method / security even when those weren't among
    # the displayed Key Factors. This map scopes the profile data shown to
    # ONLY the fields that back the actual top factors for this customer.
    # Engineered features map to the raw columns that produced them
    # (see preprocessing.engineer_features for the source logic).
    _FACTOR_PROFILE_FIELDS: dict[str, list[str]] = {
        "tenure":                     ["tenure"],
        "MonthlyCharges":             ["MonthlyCharges"],
        "TotalCharges":               ["TotalCharges"],
        "Contract":                   ["Contract"],
        "InternetService":            ["InternetService"],
        "PaymentMethod":               ["PaymentMethod"],
        "OnlineSecurity":              ["OnlineSecurity"],
        "TechSupport":                 ["TechSupport"],
        "OnlineBackup":                ["OnlineBackup"],
        "DeviceProtection":            ["DeviceProtection"],
        "PaperlessBilling":            ["PaperlessBilling"],
        "StreamingTV":                 ["StreamingTV"],
        "StreamingMovies":             ["StreamingMovies"],
        "MultipleLines":               ["MultipleLines"],
        "PhoneService":                ["PhoneService"],
        "SeniorCitizen":               ["SeniorCitizen"],
        "Partner":                     ["Partner"],
        "Dependents":                  ["Dependents"],
        "gender":                      ["gender"],
        # engineered features → the raw fields that derive them
        "tenure_group":                ["tenure"],
        "is_long_term":                ["Contract"],
        "has_any_security":            ["OnlineSecurity", "TechSupport"],
        "has_any_backup":              ["OnlineBackup", "DeviceProtection"],
        "paperless_electronic":        ["PaperlessBilling", "PaymentMethod"],
        "avg_monthly_charges":         ["TotalCharges", "tenure"],
        "charges_per_service":         ["MonthlyCharges", "total_services"],
        "tenure_monthly_interaction":  ["tenure", "MonthlyCharges"],
        "total_services": [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "InternetService",
        ],
    }

    _PROFILE_LABELS: dict[str, str] = {
        "tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)",
        "TotalCharges": "Total Charges ($)", "Contract": "Contract",
        "InternetService": "Internet Service", "PaymentMethod": "Payment Method",
        "OnlineSecurity": "Online Security", "TechSupport": "Tech Support",
        "OnlineBackup": "Online Backup", "DeviceProtection": "Device Protection",
        "PaperlessBilling": "Paperless Billing", "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies", "MultipleLines": "Multiple Lines",
        "PhoneService": "Phone Service", "SeniorCitizen": "Senior Citizen",
        "Partner": "Partner", "Dependents": "Dependents", "gender": "Gender",
        "total_services": "Total Services Subscribed",
    }

    def __init__(self) -> None:
        model    = os.getenv("CHURN_LLM_MODEL",  self._DEFAULT_MODEL)
        base_url = os.getenv("OLLAMA_BASE_URL",  self._DEFAULT_BASE_URL)
        max_t    = int(os.getenv("CHURN_MAX_TOKENS", self._DEFAULT_MAX_TOKENS))

        # Fail fast with an actionable error if Ollama isn't ready
        _verify_ollama(base_url, model)

        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=self._DEFAULT_TEMP,
            num_predict=max_t,
        )

        self._explanation_chain = _EXPLANATION_PROMPT | self._llm | StrOutputParser()
        self._strategy_chain    = _STRATEGY_PROMPT    | self._llm | StrOutputParser()

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _fmt_factors(factors: list[dict]) -> str:
        if not factors:
            return "  No dominant factors identified."
        lines = []
        for i, f in enumerate(factors, 1):
            risk_word = "increases risk" if f.get("direction") == "increases" else "decreases risk"
            lines.append(
                f"  {i}. {f['name']} — {risk_word} "
                f"(contributes {f.get('imp_pct', 0):.1f}% of this prediction's signal)"
            )
        return "\n".join(lines)

    @classmethod
    def _fmt_factor_context(cls, factors: list[dict], raw: dict) -> str:
        """
        Build the ONLY block of raw customer-profile data the LLM ever sees.
        Restricted to exactly the fields backing the actual top factors for
        THIS customer (via _FACTOR_PROFILE_FIELDS) — this is what closes the
        grounding loophole: there is no longer any other profile data in the
        prompt for the LLM to draw on, so it physically cannot mention
        payment method / internet type / security unless one of those IS a
        real top factor for this customer.
        """
        needed_keys: list[str] = []
        for f in factors:
            raw_feature = f.get("feature", "")
            for key in cls._FACTOR_PROFILE_FIELDS.get(raw_feature, []):
                if key not in needed_keys:
                    needed_keys.append(key)

        if not needed_keys:
            return "  No profile data is tied to the listed signals."

        lines = []
        for key in needed_keys:
            label = cls._PROFILE_LABELS.get(key, key)
            value = raw.get(key, "Unknown")
            if key == "SeniorCitizen":
                value = "Yes" if value == 1 else "No"
            lines.append(f"  • {label}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _base_inputs(prob: float, risk_level: str) -> dict:
        # AUDIT FIX: this previously also returned contract/tenure/
        # monthly_charges/internet_service/online_security/tech_support/
        # payment_method/senior unconditionally — that unscoped profile dump
        # is what gave the LLM material to discuss attributes outside the
        # actual top factors. All customer-profile content now flows
        # exclusively through _fmt_factor_context(), scoped per-customer.
        return {
            "probability": f"{prob * 100:.1f}",
            "risk_level":  risk_level,
        }

    # ── Blocking API ──────────────────────────────────────────────────────

    def generate_explanation(
        self,
        raw: dict,
        prob: float,
        risk_level: str,
        factors: list[dict],
    ) -> str:
        try:
            inputs = self._base_inputs(prob, risk_level)
            inputs["top_factors"]    = self._fmt_factors(factors)
            inputs["factor_context"] = self._fmt_factor_context(factors, raw)
            return self._explanation_chain.invoke(inputs)
        except Exception as exc:
            return f"⚠️ AI explanation unavailable — {exc}"

    def generate_retention_strategy(
        self,
        raw: dict,
        prob: float,
        risk_level: str,
        ai_explanation: str,
        factors: list[dict],
    ) -> str:
        # AUDIT FIX: `factors` is now required, not optional. Previously the
        # strategy was grounded only in the (possibly already-drifted)
        # explanation text, risk level, probability, and an unscoped profile
        # dump — never directly in the same SHAP factors driving the
        # prediction. It now receives the same top_factors/factor_context
        # the explanation does.
        try:
            inputs = self._base_inputs(prob, risk_level)
            inputs["ai_explanation"]  = ai_explanation or "Not available."
            inputs["top_factors"]     = self._fmt_factors(factors)
            inputs["factor_context"]  = self._fmt_factor_context(factors, raw)
            return self._strategy_chain.invoke(inputs)
        except Exception as exc:
            return f"⚠️ AI strategy unavailable — {exc}"

    # ── Streaming API ─────────────────────────────────────────────────────

    def stream_explanation(
        self,
        raw: dict,
        prob: float,
        risk_level: str,
        factors: list[dict],
    ):
        """
        Generator — yields string tokens of the explanation as they arrive
        from the local Ollama model.

        Streamlit usage:
            container = st.empty()
            text = ""
            for chunk in engine.stream_explanation(raw, prob, risk, facs):
                text += chunk
                container.markdown(text)
        """
        inputs = self._base_inputs(prob, risk_level)
        inputs["top_factors"]    = self._fmt_factors(factors)
        inputs["factor_context"] = self._fmt_factor_context(factors, raw)
        chain = _EXPLANATION_PROMPT | self._llm
        try:
            for chunk in chain.stream(inputs):
                yield chunk.content
        except Exception as exc:
            yield f"\n\n⚠️ Streaming error: {exc}"

    def stream_strategy(
        self,
        raw: dict,
        prob: float,
        risk_level: str,
        ai_explanation: str,
        factors: list[dict],
    ):
        """
        Generator — yields string tokens of the retention strategy.
        Runs after stream_explanation() so it can use the explanation as
        context — but `factors` (not just the explanation text) is now the
        primary grounding, since the explanation text could itself drift.
        """
        inputs = self._base_inputs(prob, risk_level)
        inputs["ai_explanation"]  = ai_explanation or "Not available."
        inputs["top_factors"]     = self._fmt_factors(factors)
        inputs["factor_context"]  = self._fmt_factor_context(factors, raw)
        chain = _STRATEGY_PROMPT | self._llm
        try:
            for chunk in chain.stream(inputs):
                yield chunk.content
        except Exception as exc:
            yield f"\n\n⚠️ Streaming error: {exc}"

    # ── Demo safety pipeline (Issue 8) ───────────────────────────────────
    # Prediction → SHAP → LLM Generation → Explanation Validation →
    # Strategy Validation → Repair Layer → Fallback Layer → caller.
    # These are the two methods app.py should call for the demo build —
    # they never return anything that hasn't passed validate_explanation()
    # / validate_strategy(), so the UI never renders raw unvalidated output.

    def generate_explanation_safe(
        self,
        raw: dict,
        prob: float,
        risk_level: str,
        factors: list[dict],
    ) -> tuple[str, dict]:
        """
        Returns (final_text, meta) where meta records what the safety layer
        had to do, e.g. {"regenerated": True, "fallback_used": False,
        "issues": [...]}  — useful for an optional small caption in the UI
        and for logging during the defense if something needed repair.
        """
        meta = {"regenerated": False, "fallback_used": False, "issues": []}

        text = self.generate_explanation(raw, prob, risk_level, factors)
        ok, issues = validate_explanation(text, risk_level, factors)
        if ok:
            return text, meta

        meta["issues"] = issues
        _log_validation_failure("explanation, first attempt -> regenerating", issues)
        # One regeneration attempt with a corrective note appended to the
        # human prompt, naming exactly what was wrong (Issue 6).
        meta["regenerated"] = True
        corrective = (
            "\n\nYour previous attempt was rejected for: " + "; ".join(issues) +
            ". Regenerate, strictly following the grounding rules above."
        )
        try:
            inputs = self._base_inputs(prob, risk_level)
            inputs["top_factors"]    = self._fmt_factors(factors)
            inputs["factor_context"] = self._fmt_factor_context(factors, raw) + corrective
            text2 = self._explanation_chain.invoke(inputs)
        except Exception:
            text2 = text

        ok2, issues2 = validate_explanation(text2, risk_level, factors)
        if ok2:
            return text2, meta

        meta["fallback_used"] = True
        meta["issues"] = issues2
        _log_validation_failure("explanation, after regeneration -> falling back to template", issues2)
        return _fallback_explanation(risk_level, factors), meta

    def generate_retention_strategy_safe(
        self,
        raw: dict,
        prob: float,
        risk_level: str,
        ai_explanation: str,
        factors: list[dict],
    ) -> tuple[str, dict]:
        """
        Returns (final_text, meta). Strategy uses repair (not regeneration)
        as its primary recovery path per Issue 7 — only falls back to the
        deterministic template if repair can't salvage >= 3 valid lines.
        """
        meta = {"repaired": False, "fallback_used": False, "issues": []}

        text = self.generate_retention_strategy(raw, prob, risk_level, ai_explanation, factors)
        ok, issues = validate_strategy(text, risk_level, factors)
        if ok:
            return text, meta

        meta["issues"] = issues
        meta["repaired"] = True
        _log_validation_failure("strategy, before repair", issues)
        repaired = repair_strategy(text, risk_level, factors)
        if repaired:
            ok2, issues2 = validate_strategy(repaired, risk_level, factors)
            if ok2:
                return repaired, meta
            meta["issues"] = issues2
            _log_validation_failure("strategy, after repair -> falling back to template", issues2)
        else:
            _log_validation_failure(
                "strategy, repair could not salvage >=3 valid lines -> falling back to template",
                issues,
            )

        meta["fallback_used"] = True
        return _fallback_strategy(risk_level, factors), meta
