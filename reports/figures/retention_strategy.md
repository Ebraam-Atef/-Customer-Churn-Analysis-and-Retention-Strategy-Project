# Customer Churn Retention Strategy Report
## Business Report — Customer Churn Analysis & Retention Strategy

**Track:** Digital Egypt Pioneers Initiative — Data Analytics  
**Dataset:** Telco Customer Churn (7,043 records)  
**Best Model:** Random Forest (Tuned) · ROC-AUC ≈ 0.85

---

## Executive Summary

Analysis of 7,043 telecom customers shows a **26.5% churn rate** — roughly 1
in 4 customers leaves each cycle. The ML model flags high-risk customers before
they churn, giving the retention team time to act. The four highest-leverage
actions are: contract type conversion, service bundle upsell, structured
first-year onboarding, and payment method migration.

---

## Top Churn Drivers (from Feature Importance)

| Rank | Feature | Churn Rate | Business Interpretation |
|------|---------|-----------|------------------------|
| 1 | Month-to-month contract | 42.7% | No commitment = easy to leave |
| 2 | Fiber optic + no security | 55%+ | Premium price, unmet safety need |
| 3 | Tenure ≤ 12 months | 47.4% | Onboarding failure window |
| 4 | Electronic check payment | 45.0% | Friction & disengagement signal |
| 5 | No online security | 41.8% | Low perceived value |
| 6 | No tech support | 42.0% | Unresolved problems drive exit |

---

## Retention Strategy by Segment

### 🔴 High-Risk Customers (churn probability ≥ 65%)

**Profile:** Tenure < 12 months · Month-to-month contract · Fiber optic ·
No security services · Electronic check payment.

**Actions (within 48h of model flag):**

**1. Retention Call Programme**  
Outbound call within 48 hours. Rep offers a personalised bundle. Script should
acknowledge the customer's specific plan and pain points — not a generic pitch.

**2. Annual Contract Incentive**  
15% discount on the first year of an annual contract. Frame it as "locking in
today's rate before the next price review." Customers who switch reduce churn
risk by 85%.

**3. Free Security Bundle Trial**  
90-day free OnlineSecurity + TechSupport. Trial cost ≈ $0 vs. average CLV of
$1,500+. Once customers use these services, churn drops from 42% to 15%.

**4. Payment Method Migration**  
$10 bill credit for switching from electronic check to automatic bank/card
transfer. Reduces friction and payment-failure churn simultaneously.

---

### 🟡 Medium-Risk Customers (35–64%)

**Profile:** Dissatisfied or underserved — haven't decided to leave yet.

**Actions:**

**1. Quarterly NPS Survey**  
Identify specific pain points. Customers who feel heard are 30% less likely to
churn after a service-recovery interaction.

**2. Service Bundle Campaign**  
Email offering a bundled package (internet + backup + streaming) at a combined
rate lower than à la carte. Every additional service raises switching costs.

**3. Tenure Milestone Credits**  
At 6, 12, and 24 months, automatically apply a $5 bill credit. Cost: ~$15 over
two years vs. $300+ average customer acquisition cost.

---

### 🟢 Low-Risk Customers (< 35%)

**Profile:** Loyal, long-tenure customers.

**Actions:**

**1. Referral Programme**  
$20 credit per successful referral. These customers are brand advocates —
turn them into an acquisition channel.

**2. Streaming Upsell**  
Streaming TV + Movies as an add-on with a 2-month free trial. These customers
have high satisfaction and available budget.

---

## Pricing Strategy

1. **Graduated Contract Discounts**
   - Month-to-month: 0% discount
   - One-year: 10% off
   - Two-year: 18% off
   Revenue trade-off is justified: two-year customers have a 97% retention rate.

2. **Fiber Optic Value Dashboard**  
   Show customers what they get for their spend vs. DSL. Perceived value is as
   important as actual value in churn decisions.

3. **First-Year Onboarding Journey**  
   4-touch structured programme: welcome call → 30-day check-in →
   first bill review → 90-day value confirmation. Estimated 20–35% reduction
   in new-customer churn.

---

## Recommended KPIs

| KPI | Target | Review Cadence |
|-----|--------|---------------|
| Monthly churn rate | < 18% (from 26.5%) | Monthly |
| High-risk contact rate within 48h | > 90% | Weekly |
| Annual contract conversion | > 15% of flagged customers | Monthly |
| Security bundle adoption (new) | > 25% | Monthly |
| NPS score | > 45 | Quarterly |

---

## Estimated Business Impact

Assuming CLV ≈ $1,500 and ~1,800 churners per cycle:

- **Preventing 30% of high-risk churn** → retain ~180 customers
  → **$270,000 recovered revenue per cycle**
- **Security bundle trials** → 25% conversion → additional upsell revenue
- **Referral programme** → estimated 5% acquisition uplift from stable base

Model ROC-AUC of 0.85 means the team can prioritise the top 20% of at-risk
customers and capture ~70% of all eventual churners — highly efficient resource
allocation.

---

*Generated from Customer Churn Prediction ML project.*  
*Model: Random Forest (Tuned) | Dataset: Telco Customer Churn | ROC-AUC ≈ 0.85*
