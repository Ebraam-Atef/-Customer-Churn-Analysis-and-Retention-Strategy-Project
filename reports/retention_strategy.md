# Customer Retention Strategy
## Business Report — ChurnGuard AI

**Prepared for:** Executive Leadership / Customer Retention Team
**Dataset:** Telco Customer Churn (7,043 customers)
**Model:** XGBoost (Production) · ROC-AUC ≈ 0.84

---

## 1. Executive Summary

Roughly **1 in 4 customers (26.5%)** churns each cycle. ChurnGuard AI
identifies these customers *before* they leave, using a machine learning
model to score every account by churn risk and explain — in plain language —
which specific factors are driving that risk for that customer.

This report translates those model outputs into an actionable retention
strategy: who to prioritise, what to offer them, what it costs, and what
return the business can expect. The four highest-leverage levers identified
are contract-type conversion, service-bundle upsell, structured first-year
onboarding, and payment-method migration.

---

## 2. Business Problem

Customer churn directly erodes recurring revenue and increases customer
acquisition spend to replace lost accounts. Historically, retention efforts
have been reactive — the team learns a customer has churned only after the
fact, when there is no longer an opportunity to intervene.

The model-driven top churn drivers, in order of impact:

| Rank | Driver | Churn Rate | Interpretation |
|---|---|---|---|
| 1 | Month-to-month contract | 42.7% | No commitment makes leaving low-friction |
| 2 | Fiber optic + no security add-on | 55%+ | Premium price without a matching sense of value/safety |
| 3 | Tenure ≤ 12 months | 47.4% | Onboarding gap in the first year |
| 4 | Electronic check payment | 45.0% | Correlates with general account disengagement |
| 5 | No online security | 41.8% | Perceived low value from the service |
| 6 | No tech support | 42.0% | Unresolved issues drive customers to leave |

ChurnGuard AI turns this from a retrospective report into a forward-looking,
per-customer risk score, so the retention team can act while there's still
time to change the outcome.

---

## 3. Churn Risk Segments

The model assigns each customer to one of three risk tiers, each requiring a
different type and intensity of intervention.

### 🔴 High Risk (churn probability 80–100%)
**Typical profile:** tenure under 12 months, month-to-month contract, fiber
optic internet, no security add-ons, electronic check payment.

### 🟡 Medium Risk (churn probability 40–79.9%)
**Typical profile:** underserved or mildly dissatisfied — hasn't decided to
leave, but shows one or more risk signals without the full high-risk
profile.

### 🟢 Low Risk (churn probability 0–39.9%)
**Typical profile:** longer-tenure, contracted customers with a stable
payment method and multiple active services.

---

## 4. Recommended Retention Actions

### High-Risk Segment (act within 48 hours of a model flag)

1. **Proactive retention call.** A rep contacts the customer with an offer
   tailored to their specific plan and pain points, referencing the model's
   named risk factors rather than a generic script.
2. **Annual contract incentive.** A first-year discount for converting from
   month-to-month to an annual contract, framed as locking in the current
   rate.
3. **Free security-bundle trial.** A time-limited trial of OnlineSecurity +
   TechSupport for customers currently missing both.
4. **Payment-method migration credit.** A one-time bill credit for switching
   from electronic check to automatic bank/card payment.

### Medium-Risk Segment

1. **Satisfaction check-in.** A short survey to surface specific pain points
   before they escalate into a churn decision.
2. **Service bundle offer.** A combined package (internet + backup +
   streaming) priced below the equivalent à la carte total — every
   additional active service raises the switching cost.
3. **Tenure milestone credits.** Small automatic credits at 6, 12, and 24
   months to reinforce continued value.

### Low-Risk Segment

1. **Referral programme.** Convert loyal, long-tenure customers into an
   acquisition channel with a referral credit.
2. **Streaming upsell.** Offer a streaming add-on with a short free trial to
   customers who have budget headroom and high satisfaction.

---

## 5. Expected Business Impact

Using an estimated customer lifetime value (CLV) of ~$1,500 and an
approximate volume of 1,800 churners per cycle at the current 26.5% rate:

- **Preventing 30% of high-risk churn** → roughly **180 retained customers**
  per cycle → an estimated **$270,000 in recovered revenue per cycle**.
- **Security bundle trial adoption** at an estimated 25% conversion rate
  adds incremental upsell revenue on top of the retention effect.
- **Referral programme** on the low-risk base is estimated to contribute a
  modest acquisition uplift, offsetting some acquisition cost.

With model ROC-AUC ≈ 0.84, the team can concentrate outreach on the top ~20%
of at-risk customers and expect to capture a large majority of eventual
churners within that group — a meaningfully more efficient allocation of
retention budget than blanket, unsegmented outreach.

*Note: these figures are illustrative estimates for planning purposes, based
on typical industry CLV assumptions and the dataset's observed churn volume
— not audited financial projections. They should be validated against actual
CLV and cost data before being used in budget commitments.*

---

## 6. Cost vs. Benefit

| Action | Approximate Cost | Rationale |
|---|---|---|
| Retention call programme | Agent time only | No direct discount cost; highest-touch intervention reserved for highest-value risk tier |
| Annual contract discount | Revenue trade-off (~10–15% first year) | Two-year contract customers show a 2.8% churn rate vs. 42.7% for month-to-month — the retention value outweighs the discount |
| Security bundle trial | Near-zero marginal cost | Existing service capacity; trial cost is negligible relative to a ~$1,500 average CLV |
| Payment migration credit | One-time, fixed, small | Reduces both churn risk and payment-failure operational overhead |
| Tenure milestone credits | Small, recurring | Estimated well below typical customer acquisition cost |

In each case, the intervention cost is modest relative to the CLV at risk,
which is why prioritisation by model-predicted risk (rather than uniform
outreach) is the core efficiency gain of this approach.

---

## 7. Recommended KPIs

| KPI | Target | Review Cadence |
|---|---|---|
| Monthly churn rate | Reduce from 26.5% toward < 18% | Monthly |
| High-risk contact rate within 48h | > 90% | Weekly |
| Annual contract conversion (flagged customers) | > 15% | Monthly |
| Security bundle adoption (new trials) | > 25% | Monthly |
| Customer satisfaction / NPS | > 45 | Quarterly |

---

## 8. Future Improvements

- **Feedback loop:** feed actual retention-call outcomes back into the
  training data to measure and improve real-world intervention effectiveness,
  not just predictive accuracy.
- **CLV-aware prioritisation:** incorporate actual per-customer CLV (rather
  than a single average) so outreach is ranked by risk *and* revenue at
  stake.
- **Champion/challenger testing:** A/B test retention offers by segment to
  replace estimated conversion assumptions with measured ones.
- **Expanded knowledge base:** extend the RAG-powered Knowledge Assistant
  with real historical support-ticket and complaint data as it becomes
  available, beyond the current dataset-derived CRM notes.
- **Model refresh cadence:** establish a periodic retraining schedule so the
  model reflects evolving customer behaviour rather than a single training
  snapshot.

---

*Prepared as part of the ChurnGuard AI graduation project — Digital Egypt
Pioneers Initiative, Data Analytics Specialist track.*
*Model: XGBoost (Production) · Dataset: Telco Customer Churn · ROC-AUC ≈ 0.84*
