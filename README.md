# ChurnGuard AI

**AI-Powered Customer Churn Analysis & Retention Platform**
Digital Egypt Pioneers Initiative — Data Analytics Track (Graduation Project)

ChurnGuard AI is an end-to-end system that predicts which telecom customers are
likely to churn, explains *why* in plain language, and generates a
personalised retention plan — combining a trained ML model, a local LLM, and
a retrieval-augmented knowledge base, all served through a Streamlit app.
---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Architecture Overview](#architecture-overview)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Feature Engineering](#feature-engineering)
- [Explainability — SHAP](#explainability--shap)
- [AI Explanation & Retention Strategy Generation](#ai-explanation--retention-strategy-generation)
- [Local RAG — Knowledge Assistant](#local-rag--knowledge-assistant)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Model Results](#model-results)

---

## What This Project Does

1. Trains and compares several classifiers on the Telco Customer Churn dataset
   and automatically deploys the best production-eligible model.
2. Serves live predictions through a Streamlit app, with a per-customer risk
   score and SHAP-based explanation of the top churn drivers.
3. Uses a locally-hosted LLM (via Ollama) to turn those SHAP factors into a
   human-readable explanation and a concrete retention action plan — with a
   validation/repair/fallback safety layer so the demo never shows an
   ungrounded or contradictory AI response.
4. Lets the retention team ask natural-language questions over a
   dataset-derived CRM knowledge base using a local Retrieval-Augmented
   Generation (RAG) pipeline — no cloud APIs, no API keys.

---

## Architecture Overview

```
                         ┌──────────────────────────┐
                         │   Telco-Customer-Churn   │
                         │           .csv           │
                         └────────────┬─────────────┘
                                      │
                     ┌────────────────┼─────────────────┐
                     ▼                                  ▼
        ┌──────────────────────┐              ┌────────────────────────────┐
        │   ML TRAINING PATH   │              │   CRM KNOWLEDGE PATH       │
        │  preprocessing.py    │              │  generate_crm_notes.py     │
        │  eda.py              │              │  (rule-based, no LLM)      │
        │  model.py / train.py │              │  → generated_crm_notes.csv │
        │  → churn_model.pkl   │              │  → ingest_knowledge.py     │
        └──────────┬───────────┘              │  → FAISS vector index      │
                   │                          └─────────────┬──────────────┘
                   ▼                                        ▼
        ┌────────────────────────────────────────────────────────────────┐
        │                      app/app.py (Streamlit)                    │
        │                                                                │
        │   ⚡ Predict tab            📊 Model Insights tab             │
        │   prepare_input() → model → SHAP → risk score                  │
        │            │                                                   │
        │            ▼                                                   │
        │   ai_engine.py — ChurnAIEngine (Ollama / Llama 3.1)            │
        │   SHAP factors → validated AI explanation → validated strategy │
        │                                                                │
        │   📚 Knowledge Assistant tab                                  │ 
        │   rag_engine.py — ChurnRAGEngine (Ollama embeddings + FAISS)   │
        │   question → retrieve top-k CRM notes → grounded LLM answer    │
        └────────────────────────────────────────────────────────────────┘
```

Everything — the classifier, the LLM, and the embedding model — runs
locally. There is no dependency on a cloud AI provider or API key.

---

## Machine Learning Pipeline

**Files:** `src/preprocessing.py`, `src/eda.py`, `src/model.py`, `src/train.py`

- **Cleaning:** drops `customerID`, converts blank `TotalCharges` strings to
  `0` and casts to float, normalises `Churn` to a 0/1 target.
- **EDA:** five auto-generated figures (class balance, churn by contract
  type, churn by tenure, churn by service subscription, churn by charges)
  plus a console summary of the top univariate churn drivers.
- **Model comparison:** Random Forest, XGBoost,
  Gradient Boosting, and a soft-voting ensemble of the three, each evaluated
  with 5-fold stratified cross-validation on ROC-AUC.
- **Hyperparameter tuning:** the Random Forest candidate is additionally
  tuned with `RandomizedSearchCV` (30 iterations, 5-fold CV) as a stronger
  contender.
- **Class imbalance:** the dataset is ~26.5% churn. `SMOTE` is applied to the
  training split only, after a stratified train/test split — the test set
  stays untouched so evaluation reflects real-world class imbalance.
- **Automatic production-model selection:** `select_production_model()` in
  `model.py` picks the highest-ROC-AUC candidate that also exposes
  `.feature_importances_`, since the app's Key Factors panel and the Model
  Insights chart both depend on that attribute. This is why the
  VotingClassifier — a heterogeneous soft-voting ensemble with no single
  coherent importance vector — is excluded from deployment even when it
  scores well; the rule is criteria-based, not hardcoded to one model name,
  so a future retraining run re-evaluates and promotes whichever eligible
  model wins. In the current training run, **XGBoost** is the eligible
  candidate with the highest ROC-AUC and is the deployed production model
  (see [Model Results](#model-results)).
- **Output:** a single `models/churn_model.pkl` bundle containing the fitted
  model, per-column `LabelEncoder`s, the fitted `StandardScaler`, the ordered
  28-feature list, the deployed model's name, and the full metrics table for
  every candidate — so there is no separate scaler/encoder file and no risk
  of feature-order mismatch at inference time.

---

## Feature Engineering

`preprocessing.engineer_features()` is stateless (no fitted parameters), so
it produces identical results whether called on the full training set or a
single row at inference time. It adds 9 derived features on top of the 19
raw columns:

| Feature | Description |
|---|---|
| `tenure_group` | Bucketed tenure: 0–12 / 13–24 / 25–48 / 49–72 months |
| `total_services` | Count of subscribed services + internet service flag |
| `has_any_security` | 1 if `OnlineSecurity` or `TechSupport` = Yes |
| `has_any_backup` | 1 if `OnlineBackup` or `DeviceProtection` = Yes |
| `is_long_term` | 1 if contract is One year or Two year |
| `avg_monthly_charges` | `TotalCharges / (tenure + 1)` |
| `charges_per_service` | `MonthlyCharges / (total_services + 1)` |
| `paperless_electronic` | 1 if paperless billing **and** electronic check payment |
| `tenure_monthly_interaction` | `tenure × MonthlyCharges` |

16 categorical columns (including `tenure_group`) are label-encoded, and all
28 features are then scaled with a single `StandardScaler` fit on the full
encoded feature matrix — the same scaler used both at training time and by
`prepare_input()` at inference time, which guarantees consistency between
training and serving.

---

## Explainability — SHAP

The app computes per-customer (local) SHAP values with `shap.TreeExplainer`,
which gives exact, fast, signed attributions for the tree-based models used
here (currently deployed: XGBoost; also used for Random Forest / Gradient
Boosting candidates). This powers the
**Key Factors** panel on the Predict tab — it replaced an earlier heuristic
based on the model's global `feature_importances_`, which could not explain
an individual prediction, only the model as a whole.

---

## AI Explanation & Retention Strategy Generation

**File:** `src/ai_engine.py` — `ChurnAIEngine`, running 100% locally via
**Ollama + Llama 3.1 (8B)** through LangChain LCEL chains.

- `generate_explanation()` turns the customer's risk score and top SHAP
  factors into a plain-language explanation of why they're at risk.
- `generate_retention_strategy()` turns that explanation and the same SHAP
  factors into a concrete action plan.
- Both prompts are scoped **only** to the customer-profile fields backing the
  actual SHAP factors shown in the UI — there is no unconditional full
  profile dump, so the model can't invent a churn driver the data doesn't
  support.
- Streaming variants (`stream_explanation`, `stream_strategy`) yield tokens
  for a live Streamlit `st.empty()` display.

**Demo safety pipeline.** The app calls `generate_explanation_safe()` /
`generate_retention_strategy_safe()`, not the raw generation methods
directly:

```
Prediction → SHAP → LLM Generation → Explanation Validation →
Strategy Validation → Repair Layer → Fallback Layer → caller
```

This exists because a locally-hosted 8B model is not, on its own, reliable
enough to put in front of a graduation committee. Concretely, validation
checks for and corrects:

- the explanation mentioning a risk level other than the actual computed one
  (e.g. calling a "Low Risk" customer "moderate risk"),
- churn-driver claims not backed by the real SHAP factors,
- a retention-strategy priority token outside the allowed vocabulary for
  that risk level, or an action whose direction contradicts its factor's
  SHAP sign (e.g. proposing a "fix" for something that's already lowering
  risk).

On failure, the explanation gets one regeneration attempt with a corrective
note describing exactly what was wrong; the strategy instead goes through a
deterministic repair step (normalising priority labels, dropping
direction-mismatched lines). If recovery still fails, both fall back to a
deterministic template built directly from the SHAP factors, which is
guaranteed to pass validation. The UI only ever renders this final,
validated string.

---

## Local RAG — Knowledge Assistant

**Files:** `scripts/generate_crm_notes.py`, `scripts/ingest_knowledge.py`,
`src/rag_engine.py`

1. **`generate_crm_notes.py`** builds a dataset-grounded CRM knowledge base:
   one business-style note per customer, generated entirely with
   deterministic string templates and if/else rules over the real columns in
   `Telco-Customer-Churn.csv` — no LLM calls, byte-identical output on every
   run. It writes `data/knowledge_base/generated_crm_notes.csv`
   (`crm_id | customer_id | note`); there is no `.txt` export.
2. **`ingest_knowledge.py`** is the standalone CLI for building or rebuilding
   the FAISS index without launching Streamlit (`--dir`, `--file`,
   `--rebuild`).
3. **`rag_engine.ChurnRAGEngine`** loads and chunks the supported knowledge
   base files, embeds them locally with Ollama (`nomic-embed-text` by
   default) in batches of 100 chunks per call, stores the vectors in a
   **FAISS** index persisted to disk (no re-embedding on restart), retrieves
   the top-k relevant chunks for a question, and answers with **Llama 3.1**
   through a LangChain retrieval chain. The system prompt instructs the
   model to answer only from retrieved documents and to say so explicitly
   when the answer isn't in the knowledge base, rather than fabricating
   customer IDs, dates, or agent names.

Supported knowledge base file types: **CSV, PDF, Markdown** (`.csv`, `.pdf`,
`.md`). The index is currently populated from the generated CRM CSV only, so
there are no duplicated embeddings from redundant sample files.

The **📚 Knowledge Assistant** tab in the Streamlit app exposes this as a
chat interface over the CRM knowledge base.

---

## Project Structure

```
churn_project/
│
├── data/
│   ├── Telco-Customer-Churn.csv            ← Raw source dataset (7,043 rows)
│   ├── processed/
│   │   └── cleaned_feature_engineered_dataset.csv
│   └── knowledge_base/
│       └── generated_crm_notes.csv         ← RAG source corpus
│
├── src/
│   ├── preprocessing.py                    ← Single source of truth for all transforms
│   ├── eda.py                              ← EDA visualisation functions
│   ├── model.py                            ← Training, evaluation, tuning, PKL save/load
│   ├── train.py                            ← Master training entry point
│   ├── ai_engine.py                        ← LLM explanation + retention strategy generation
│   └── rag_engine.py                       ← Local RAG (Ollama + FAISS)
│
├── scripts/
│   ├── generate_crm_notes.py               ← Builds the CRM knowledge base from the dataset
│   └── ingest_knowledge.py                 ← Standalone FAISS ingestion CLI
│
├── models/
│   ├── churn_model.pkl                     ← Saved model bundle (auto-generated)
│   └── faiss_index/                        ← Persisted vector index (auto-generated)
│
├── app/
│   └── app.py                              ← Streamlit deployment app
│
├── reports/
│   ├── figures/                            ← Auto-generated EDA & model plots
│   └── retention_strategy.md               ← Business recommendations report
│
├── requirements.txt
├── .env                                    ← Local Ollama config
└── README.md
```

---

## Installation

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama
This project runs its LLM and embedding model 100% locally — no API keys.
```bash
# Install: https://ollama.ai
ollama serve

# Pull the required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3. Configure environment
Copy the provided `.env` (already set up for local Ollama) and adjust if
needed — e.g. to point `OLLAMA_BASE_URL` at a different host, or to switch
`CHURN_LLM_MODEL` to a smaller/larger Llama variant.

---

## Usage

### Train the model
Generates `churn_model.pkl`, the EDA/diagnostic figures, and the business
export CSV:
```bash
python src/train.py
```

### Build the CRM knowledge base
```bash
python scripts/generate_crm_notes.py
```

### Ingest the knowledge base into FAISS
```bash
# Default knowledge base directory
python scripts/ingest_knowledge.py

# Rebuild the index from scratch
python scripts/ingest_knowledge.py --rebuild
```

### Launch the app
```bash
streamlit run app/app.py
```

The app has three tabs: **⚡ Predict** (single-customer risk scoring, SHAP
Key Factors, AI explanation and retention strategy), **📚 Knowledge
Assistant** (RAG chat over the CRM knowledge base), and **📊 Model
Insights** (global feature importance and model comparison).

---

## Technologies Used

| Layer | Tools |
|---|---|
| Data / ML | Python, pandas, numpy, scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP (`TreeExplainer`) |
| Local LLM | Ollama, Llama 3.1 (8B) |
| Local embeddings | Ollama, `nomic-embed-text` |
| RAG orchestration | LangChain (LCEL chains, prompt templates, text splitters) |
| Vector store | FAISS (CPU) |
| App / visualisation | Streamlit, matplotlib, seaborn, plotly |

---

## Model Results

*From the current training run (`python src/train.py`):*

| Model | Accuracy | ROC-AUC | CV ROC-AUC |
|---|---|---|---|
| Voting Ensemble | 77.0% | 0.844\* | 0.915 ± 0.031 |
| **XGBoost (Production)** | **78.2%** | **0.840** | **0.939 ± 0.050** |
| Random Forest | 76.5% | 0.838 | 0.914 ± 0.023 |
| Gradient Boosting | 78.4% | 0.836 | 0.936 ± 0.053 |
| Random Forest (Tuned) | 77.2% | 0.826 | 0.932 (search CV score) |

\* The Voting Ensemble has the highest raw ROC-AUC but is **excluded from
production candidacy** — it doesn't expose `.feature_importances_`, which
the SHAP explanation pipeline and the Model Insights dashboard both require.
Among the remaining, feature-importance-exposing candidates, **XGBoost**
has the highest ROC-AUC (0.840) and is the deployed production model — see
[Automatic production-model selection](#machine-learning-pipeline).

### Top Churn Drivers

| Signal | Churn Rate |
|---|---|
| Month-to-month contract | 42.7% |
| Fiber optic + no security | 55%+ |
| Tenure 0–12 months | 47.4% |
| Electronic check payment | 45.0% |
| Two-year contract | 2.8% |


*Digital Egypt Pioneers Initiative — Data Analytics Specialist Track*
