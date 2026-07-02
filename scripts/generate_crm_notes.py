#!/usr/bin/env python3
"""
scripts/generate_crm_notes.py
─────────────────────────────────────────────────────────────────────────────
Generates one business-style CRM intelligence note per customer, derived
entirely from data/Telco-Customer-Churn.csv (7,043 rows / 21 columns).

This replaces the demo knowledge base (sample_complaints.csv,
sample_crm_notes.csv, sample_support_tickets.csv) with a REAL,
dataset-grounded customer intelligence corpus for the RAG pipeline.

Pipeline position
──────────────────
  Telco-Customer-Churn.csv
      → generate_crm_notes.py   (THIS SCRIPT — rule-based, no LLM)
      → data/knowledge_base/generated_crm_notes.csv  (crm_id | customer_id | note)
      → data/knowledge_base/generated_crm_notes.txt  (delimited plain text)
      → scripts/ingest_knowledge.py   (embeds + builds FAISS index)
      → rag_engine.py / Streamlit Knowledge Base tab

IMPORTANT — fully deterministic, zero LLM calls
─────────────────────────────────────────────────
  Every note is built with plain string templates and if/else rules over
  the customer's actual column values. No Ollama, no Claude, no randomness.
  Running this script twice on the same CSV produces byte-identical output.

Usage
─────
    python scripts/generate_crm_notes.py

    # Custom paths
    python scripts/generate_crm_notes.py --input data/Telco-Customer-Churn.csv \\
                                          --output-dir data/knowledge_base

    # Skip the .txt export (CSV only)
    python scripts/generate_crm_notes.py --no-txt
"""

import argparse
import csv
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT  = ROOT / "data" / "Telco-Customer-Churn.csv"
DEFAULT_OUTDIR = ROOT / "data" / "knowledge_base"

TXT_DELIMITER = "=" * 70


# ── Safe parsing helpers ────────────────────────────────────────────────
# The raw CSV has known data-quality quirks (11 blank TotalCharges rows,
# "No internet service" / "No phone service" sentinel strings). These
# helpers normalise everything so rule logic never crashes or mis-fires.

def _safe_float(value: str, default: float = 0.0) -> float:
    """Parse a numeric string, tolerating blank/whitespace-only cells."""
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def _has_service(value: str) -> bool:
    """
    True only for an explicit 'Yes'. Handles the dataset's sentinel
    values ('No internet service', 'No phone service') as 'No' —
    the customer doesn't have the underlying service at all, so they
    can't have the add-on either.
    """
    return str(value).strip().lower() == "yes"


# ── Tenure bucketing ──────────────────────────────────────────────────────

def _tenure_label(tenure: int) -> str:
    """Human phrase for a tenure value in months."""
    if tenure <= 0:
        return "brand new (0 months)"
    if tenure == 1:
        return "1 month"
    return f"{tenure} months"


def _tenure_bucket(tenure: int) -> str:
    """Coarse lifecycle stage used for risk-indicator logic."""
    if tenure <= 6:
        return "new"
    if tenure <= 24:
        return "developing"
    if tenure <= 48:
        return "established"
    return "loyal"


# ── Risk indicator rules ──────────────────────────────────────────────────

def _build_risk_indicators(row: dict) -> list[str]:
    """
    Rule-based churn risk signals derived purely from the row's own values.
    Order reflects typical influence strength seen in churn analysis
    (contract flexibility and tenure dominate; service gaps are secondary).
    """
    indicators: list[str] = []

    contract = row["Contract"]
    tenure   = _safe_int(row["tenure"])
    internet = row["InternetService"]
    security = row["OnlineSecurity"]
    techsup  = row["TechSupport"]
    backup   = row["OnlineBackup"]
    device   = row["DeviceProtection"]
    payment  = row["PaymentMethod"]
    monthly  = _safe_float(row["MonthlyCharges"])
    senior   = _safe_int(row["SeniorCitizen"])
    partner  = row["Partner"]
    deps     = row["Dependents"]
    paperless = row["PaperlessBilling"]

    # Contract flexibility — strongest single churn driver in this dataset
    if contract == "Month-to-month":
        indicators.append("Flexible contract (month-to-month, no lock-in)")
    elif contract == "One year":
        indicators.append("Moderate commitment (one-year contract)")

    # Tenure — early-life customers are the highest-risk segment
    bucket = _tenure_bucket(tenure)
    if bucket == "new":
        indicators.append("Short tenure — early lifecycle, not yet loyal")
    elif bucket == "developing":
        indicators.append("Developing tenure — loyalty still forming")

    # Service attachment gaps (only meaningful if internet is active at all)
    if internet != "No":
        if not _has_service(security):
            indicators.append("No online security service")
        if not _has_service(techsup):
            indicators.append("No tech support service")
        if not _has_service(backup) and not _has_service(device):
            indicators.append("No backup or device protection services")
    else:
        indicators.append("No internet service — limited product attachment")

    if internet == "Fiber optic" and not _has_service(security):
        indicators.append("Fiber optic without security — high-value, high-risk profile")

    # Payment friction
    if payment == "Electronic check":
        indicators.append("Electronic check payment — historically higher churn segment")

    # Price sensitivity
    if monthly >= 80 and contract == "Month-to-month":
        indicators.append("High monthly spend on a flexible contract")

    # Household stability signals
    if partner == "No" and deps == "No":
        indicators.append("No partner or dependents — fewer household ties to the service")

    # Senior citizens skew slightly higher-touch / price-sensitive
    if senior == 1:
        indicators.append("Senior citizen — may value simplicity and support responsiveness")

    if paperless == "Yes" and payment == "Electronic check":
        indicators.append("Paperless billing + electronic check — combination linked to higher churn")

    if not indicators:
        indicators.append("No significant risk indicators identified")

    return indicators


# ── Retention recommendation rules ────────────────────────────────────────

def _build_retention_action(row: dict, churned: bool) -> str:
    """
    Single, specific retention recommendation derived from contract type,
    tenure, churn label, and service attachment — mirrors the same rule
    families used elsewhere in the project (get_recommendations in app.py),
    kept independent here so this script has zero dependency on app code.
    """
    contract = row["Contract"]
    tenure   = _safe_int(row["tenure"])
    internet = row["InternetService"]
    security = row["OnlineSecurity"]
    techsup  = row["TechSupport"]
    payment  = row["PaymentMethod"]

    has_security = _has_service(security)
    has_techsup  = _has_service(techsup)

    if churned:
        # Customer already left — note is for win-back / pattern analysis
        if contract == "Month-to-month":
            return (
                "Customer churned from a month-to-month contract. "
                "For win-back outreach, lead with a discounted annual plan "
                "and waived setup fees to address the lack of lock-in that "
                "likely contributed to the departure."
            )
        if not has_security and internet != "No":
            return (
                "Customer churned without an online security subscription. "
                "Win-back campaigns should foreground a free security trial, "
                "since unmet service-attachment needs may have driven the loss."
            )
        return (
            "Customer churned despite a longer-term contract or service "
            "attachment — investigate service-quality or pricing complaints "
            "before attempting win-back outreach."
        )

    # Active customer — forward-looking retention guidance
    actions: list[str] = []

    if contract == "Month-to-month":
        actions.append("offer an annual contract upgrade with a loyalty discount")
    if internet != "No" and not has_security:
        actions.append("present a free trial of the online security add-on")
    if internet != "No" and not has_techsup:
        actions.append("highlight the tech support bundle to reduce friction")
    if payment == "Electronic check":
        actions.append("encourage a switch to automatic bank transfer or card payment")
    if tenure <= 6:
        actions.append("enrol the customer in a structured onboarding journey")

    if not actions:
        return (
            "Customer profile shows strong retention fundamentals — "
            "prioritise as a candidate for referral or premium upsell programmes."
        )

    if len(actions) == 1:
        return f"Recommended action: {actions[0]}."

    *head, tail = actions
    return "Recommended actions: " + "; ".join(head) + f"; and {tail}."


# ── Note assembly ──────────────────────────────────────────────────────────

def _format_service(value: str) -> str:
    """Normalise sentinel strings into clean display labels."""
    v = str(value).strip()
    if v in ("No internet service", "No phone service"):
        return "N/A (no base service)"
    return v


def build_crm_note(row: dict) -> str:
    """
    Build one complete, human-readable CRM note for a single customer row.
    Pure rule-based string templating — deterministic, no LLM involved.
    """
    customer_id = row["customerID"]
    contract    = row["Contract"]
    tenure      = _safe_int(row["tenure"])
    monthly     = _safe_float(row["MonthlyCharges"])
    total       = _safe_float(row["TotalCharges"], default=monthly * tenure)
    internet    = row["InternetService"]
    security    = _format_service(row["OnlineSecurity"])
    techsup     = _format_service(row["TechSupport"])
    backup      = _format_service(row["OnlineBackup"])
    device      = _format_service(row["DeviceProtection"])
    payment     = row["PaymentMethod"]
    gender      = row["gender"]
    senior      = "Yes" if _safe_int(row["SeniorCitizen"]) == 1 else "No"
    partner     = row["Partner"]
    deps        = row["Dependents"]
    paperless   = row["PaperlessBilling"]
    churn_raw   = str(row["Churn"]).strip().lower()
    churned     = churn_raw == "yes"

    risk_indicators = _build_risk_indicators(row)
    retention_action = _build_retention_action(row, churned)

    risk_lines = "\n".join(f"  - {r}" for r in risk_indicators)

    outcome_line = (
        "Customer churned (left the service)."
        if churned else
        "Customer did not churn (active / retained)."
    )

    note = f"""Customer ID: {customer_id}

Customer Profile:
  - Gender: {gender}
  - Senior Citizen: {senior}
  - Partner: {partner}
  - Dependents: {deps}
  - Contract: {contract}
  - Tenure: {_tenure_label(tenure)}
  - Internet Service: {internet}
  - Online Security: {security}
  - Online Backup: {backup}
  - Device Protection: {device}
  - Tech Support: {techsup}
  - Payment Method: {payment}
  - Paperless Billing: {paperless}
  - Monthly Charges: ${monthly:.2f}
  - Total Charges: ${total:.2f}

Risk Indicators:
{risk_lines}

Outcome:
{outcome_line}

Suggested Retention Action:
{retention_action}"""

    return note


# ── Main generation pipeline ────────────────────────────────────────────

def generate_notes(input_path: Path) -> list[dict]:
    """
    Load the raw Telco CSV and generate one CRM note per row.
    Returns a list of {crm_id, customer_id, note} dicts, in file order.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_path}.\n"
            f"  Expected: data/Telco-Customer-Churn.csv at the project root."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    required_cols = {
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    }
    missing = required_cols - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {sorted(missing)}"
        )

    records: list[dict] = []
    for i, row in enumerate(rows, start=1):
        note = build_crm_note(row)
        records.append({
            "crm_id":      f"CRM-{i:05d}",
            "customer_id": row["customerID"],
            "note":        note,
        })

    return records


def write_csv(records: list[dict], output_path: Path) -> None:
    """Write crm_id | customer_id | note to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crm_id", "customer_id", "note"])
        writer.writeheader()
        writer.writerows(records)

# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rule-based CRM intelligence notes from the "
                     "Telco-Customer-Churn dataset (no LLM, fully deterministic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", metavar="CSV_PATH", default=str(DEFAULT_INPUT),
        help="Path to Telco-Customer-Churn.csv (default: data/Telco-Customer-Churn.csv)",
    )
    parser.add_argument(
        "--output-dir", metavar="DIR", default=str(DEFAULT_OUTDIR),
        help="Output directory for generated files (default: data/knowledge_base)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir     = Path(args.output_dir)
    csv_path   = outdir / "generated_crm_notes.csv"

    print("\n" + "═" * 62)
    print("  CRM INTELLIGENCE NOTE GENERATOR  (rule-based, no LLM)")
    print("═" * 62)

    print(f"\n  Input  : {input_path}")
    try:
        records = generate_notes(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ❌  {exc}")
        sys.exit(1)

    n_churned = sum(
        1 for r in records
        if "Customer churned" in r["note"].split("Outcome:\n")[1][:40]
    )

    print(f"  Loaded : {len(records):,} customer rows")

    write_csv(records, csv_path)
    print(f"\n  ✅  CSV written : {csv_path}")
    print(f"       Columns   : crm_id | customer_id | note")
    print(f"       Rows      : {len(records):,}")

    print(f"\n  Churn breakdown in generated notes:")
    print(f"       Churned     : {n_churned:,}")
    print(f"       Retained    : {len(records) - n_churned:,}")

    print("\n  Next step — rebuild the FAISS index:")
    print("    python scripts/ingest_knowledge.py --rebuild\n")


if __name__ == "__main__":
    main()