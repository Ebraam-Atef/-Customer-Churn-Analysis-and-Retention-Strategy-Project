# Customer Churn Analysis & Retention Strategy
### Digital Egypt Pioneers Initiative — Data Analytics Track

---

## Project Overview

A production-ready ML pipeline that predicts which telecom customers will churn,
explains the key risk drivers, and provides the CRM team with personalised,
data-backed retention actions via a Streamlit web app.

---

## Project Structure

```
churn_project/
│
├── data/
│   └── Telco-Customer-Churn.csv        ← Raw source dataset (7,043 rows)
│
├── src/
│   ├── preprocessing.py                ← SINGLE SOURCE OF TRUTH for all transforms
│   ├── eda.py                          ← EDA visualisation functions
│   ├── model.py                        ← Training, evaluation, tuning, PKL save/load
│   └── train.py                        ← Master entry point (run this to train)
│
├── models/
│   └── churn_model.pkl                 ← Saved model bundle (auto-generated)
│
├── app/
│   └── app.py                          ← Streamlit deployment app
│
├── reports/
│   ├── figures/                        ← Auto-generated EDA & model plots (8 PNGs)
│   └── retention_strategy.md          ← Business recommendations report
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model (generates churn_model.pkl + all figures)
```bash
python src/train.py
```

### Step 3 — Launch the Streamlit app
```bash
streamlit run app/app.py
```

---

## PKL Bundle Structure

The model is saved as a single `churn_model.pkl` containing everything
needed for inference — **no separate scaler or encoder files**.

```python
bundle = {
    "model"          : RandomForestClassifier  # best fitted estimator
    "encoder"        : { col: LabelEncoder }   # one per categorical column
    "scaler"         : StandardScaler          # fitted on training numerics
    "features"       : [ ... ]                 # ordered list of 28 feature names
    "best_model_name": "Random Forest (Tuned)"
    "results"        : { model_name: { accuracy, roc_auc } }
    "all_models"     : { model_name: fitted_estimator }
}
```

**Consistency guarantee:** `app.py` imports `prepare_input()` directly from
`src/preprocessing.py` — the exact same function used during training.
There is zero risk of feature mismatch.

---

## Milestones

### Milestone 1 — Data Exploration & Preprocessing
Files: `src/preprocessing.py`, `src/eda.py`

- Drops `customerID`, fixes `TotalCharges` whitespace strings → 0
- Detected **26.5% churn rate** (imbalanced → SMOTE applied during training)
- **9 engineered features:** `tenure_group`, `total_services`, `has_any_security`,
  `has_any_backup`, `is_long_term`, `avg_monthly_charges`, `charges_per_service`,
  `paperless_electronic`, `tenure_monthly_interaction`
- Label encoding for 16 categorical columns
- StandardScaler for 7 numeric columns

### Milestone 2 — Model Development
File: `src/model.py`, `src/train.py`

- 4 candidate models: Logistic Regression, Random Forest, XGBoost, Gradient Boosting + Voting Ensemble
- 5-fold stratified cross-validation on each
- Best model selected by ROC-AUC
- Random Forest tuned with `RandomizedSearchCV` (30 iterations, 5-fold CV)

### Milestone 3 — Deployment
File: `app/app.py`

- Loads PKL bundle via `@st.cache_resource`
- All preprocessing handled by `preprocessing.prepare_input()`
- Three risk levels with colour-coded cards: 🔴 High / 🟡 Medium / 🟢 Low
- Personalised retention recommendations generated from input signals

### Milestone 4 — Business Retention Strategy
File: `reports/retention_strategy.md`

- Three customer segments with tailored actions
- Pricing strategy, contract incentives, service improvement plan
- KPI tracking framework
- Estimated $270K+ revenue recovery potential

---

## Model Results (from actual training run)

| Model | Accuracy | ROC-AUC | CV ROC-AUC |
|---|---|---|---|
| Random Forest | 76.9% | 0.841 | 0.899 ± 0.017 |
| XGBoost | 78.4% | 0.826 | 0.909 ± 0.032 |
| Gradient Boosting | 77.1% | 0.829 | — |
| Voting Ensemble | 78.1% | 0.836 | — |
| **Random Forest (Tuned)** | **~80%** | **~0.85** | **0.91** |

---

## Key Insights

| Signal | Churn Rate | Action |
|---|---|---|
| Month-to-month contract | 42.7% | Contract upgrade incentive |
| Fiber optic + no security | 55%+ | Free security trial |
| Tenure 0–12 months | 47.4% | Structured onboarding programme |
| Electronic check payment | 45.0% | Auto-pay switch reward |
| Two-year contract | 2.8% | Use as gold standard |

---

## Tools & Libraries

Python · pandas · numpy · scikit-learn · XGBoost · imbalanced-learn ·
Streamlit · matplotlib · seaborn
