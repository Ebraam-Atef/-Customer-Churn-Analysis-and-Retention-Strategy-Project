import sys, pickle, warnings, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# resolve src/ so preprocessing.py can be imported from app/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import prepare_input


st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
  --bg:          #06090f;
  --bg-card:     rgba(14, 21, 32, 0.85);
  --bg-glass:    rgba(255,255,255,0.035);
  --bg-input:    rgba(10, 16, 28, 0.9);
  --bg-hover:    rgba(255,255,255,0.055);

  --border:      rgba(255,255,255,0.07);
  --border-lit:  rgba(255,255,255,0.14);

  --amber:       #f59e0b;
  --amber-soft:  rgba(245,158,11,0.15);
  --amber-glow:  rgba(245,158,11,0.08);
  --teal:        #14b8a6;
  --teal-soft:   rgba(20,184,166,0.15);

  --danger:      #ef4444;
  --danger-soft: rgba(239,68,68,0.12);
  --warn:        #f59e0b;
  --warn-soft:   rgba(245,158,11,0.12);
  --safe:        #10b981;
  --safe-soft:   rgba(16,185,129,0.12);

  --t1: #eef4ff;
  --t2: #7d98b8;
  --t3: #3a5070;

  --shadow-card: 0 4px 24px rgba(0,0,0,0.55), 0 1px 4px rgba(0,0,0,0.3);
  --shadow-deep: 0 8px 40px rgba(0,0,0,0.75), 0 2px 8px rgba(0,0,0,0.4);
  --shadow-glow-danger: 0 0 32px rgba(239,68,68,0.40), 0 0 80px rgba(239,68,68,0.18);
  --shadow-glow-warn:   0 0 24px rgba(245,158,11,0.28), 0 0 56px rgba(245,158,11,0.10);
  --shadow-glow-safe:   0 0 20px rgba(16,185,129,0.20), 0 0 44px rgba(16,185,129,0.08);

  --r-sm: 8px;
  --r-md: 14px;
  --r-lg: 20px;
  --r-xl: 26px;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg) !important;
  color: var(--t1) !important;
}
.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 20% 0%,  rgba(245,158,11,0.04) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(20,184,166,0.04) 0%, transparent 60%),
    var(--bg) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.2rem 4rem !important; max-width: 1400px !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-lit); border-radius: 99px; }

.header {
  padding: 1.5rem 0 1.2rem;
  margin-bottom: 2rem;
  display: flex; align-items: center; justify-content: space-between; gap: 2rem;
  border-bottom: 1px solid var(--border);
  position: relative;
}
.header::after {
  content: '';
  position: absolute; bottom: -1px; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--amber) 35%, var(--teal) 65%, transparent);
  opacity: 0.6;
}
.brand { display: flex; align-items: center; gap: 0.85rem; }
.brand-icon {
  width: 38px; height: 38px; border-radius: 10px;
  background: linear-gradient(135deg, var(--amber), #d97706);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
  box-shadow: 0 0 24px rgba(245,158,11,0.4), inset 0 1px 0 rgba(255,255,255,0.2);
}
.brand-name {
  font-family: 'Syne', sans-serif;
  font-size: 1.25rem; font-weight: 800;
  color: var(--t1); letter-spacing: -0.025em;
}
.brand-sub {
  font-size: 0.63rem; color: var(--t2);
  text-transform: uppercase; letter-spacing: 0.12em; margin-top: -3px;
}
.pills { display: flex; gap: 0.45rem; flex-wrap: wrap; align-items: center; }
.pill {
  background: var(--bg-glass);
  border: 1px solid var(--border);
  border-radius: 99px; padding: 0.25rem 0.8rem;
  display: flex; align-items: center; gap: 0.38rem;
  font-size: 0.67rem; color: var(--t2); white-space: nowrap;
  backdrop-filter: blur(8px);
}
.pill strong { color: var(--amber); font-family: 'DM Mono', monospace; font-weight: 500; }
.live-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--safe); box-shadow: 0 0 5px var(--safe);
  animation: blink 2.2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.scenario-row {
  display: flex; gap: 0.7rem; margin-bottom: 1.2rem; flex-wrap: wrap;
}
.scenario-btn {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.35rem 0.85rem;
  border-radius: 99px;
  font-size: 0.72rem; font-weight: 600; font-family: 'DM Sans', sans-serif;
  cursor: pointer; border: 1px solid; transition: all 0.2s ease;
  letter-spacing: 0.03em;
}
.scenario-btn-high {
  background: var(--danger-soft); border-color: var(--danger);
  color: var(--danger);
}
.scenario-btn-high:hover { background: rgba(239,68,68,0.22); }
.scenario-btn-mid {
  background: var(--warn-soft); border-color: var(--warn);
  color: var(--warn);
}
.scenario-btn-mid:hover  { background: rgba(245,158,11,0.22); }
.scenario-btn-low {
  background: var(--safe-soft); border-color: var(--safe);
  color: var(--safe);
}
.scenario-btn-low:hover  { background: rgba(16,185,129,0.22); }

.glass {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  box-shadow: var(--shadow-card);
  transition: box-shadow 0.3s ease;
}
.glass:hover { box-shadow: var(--shadow-deep); }

.input-panel { padding: 1.6rem 1.8rem; }

.results-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto auto;
  gap: 0.9rem;
}
.grid-cell {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  box-shadow: var(--shadow-card);
  padding: 1.2rem 1.3rem;
  transition: box-shadow 0.3s ease, border-color 0.3s ease;
}
.grid-cell:hover { box-shadow: var(--shadow-deep); border-color: var(--border-lit); }

/* risk-tinted glow on the top two cells */
.cell-danger {
  box-shadow: var(--shadow-card), var(--shadow-glow-danger) !important;
  border-color: rgba(239,68,68,0.35) !important;
}
.cell-danger:hover { border-color: rgba(239,68,68,0.55) !important; }
.cell-warn   { box-shadow: var(--shadow-card), var(--shadow-glow-warn)   !important; border-color: rgba(245,158,11,0.25) !important; }
.cell-safe   { box-shadow: var(--shadow-card), var(--shadow-glow-safe)   !important; border-color: rgba(16,185,129,0.20) !important; }

.sec {
  font-family: 'Syne', sans-serif;
  font-size: 0.58rem; font-weight: 700;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--amber); margin: 0 0 0.8rem;
}

.risk-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.28rem 0.85rem 0.28rem 0.5rem;
  border-radius: 99px;
  font-family: 'Syne', sans-serif;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; white-space: nowrap;
}
.rb-high   {
  background: rgba(239,68,68,0.18);
  border: 1.5px solid var(--danger);
  color: var(--danger);
  box-shadow: 0 0 12px rgba(239,68,68,0.30), inset 0 0 8px rgba(239,68,68,0.08);
  animation: badge-pulse 2.4s ease-in-out infinite;
}
.rb-medium { background: var(--warn-soft);  border: 1.5px solid var(--warn);  color: var(--warn); }
.rb-low    { background: var(--safe-soft);  border: 1px solid var(--safe);    color: var(--safe); }
.rb-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink:0; }
.rb-dot-high   { background: var(--danger); box-shadow: 0 0 8px var(--danger), 0 0 16px rgba(239,68,68,0.4); }
.rb-dot-medium { background: var(--warn);   box-shadow: 0 0 6px var(--warn); }
.rb-dot-low    { background: var(--safe);   box-shadow: 0 0 5px var(--safe); }
@keyframes badge-pulse {
  0%, 100% { box-shadow: 0 0 10px rgba(239,68,68,0.28), inset 0 0 8px rgba(239,68,68,0.06); }
  50%      { box-shadow: 0 0 22px rgba(239,68,68,0.50), inset 0 0 12px rgba(239,68,68,0.12); }
}

.prob-number {
  font-family: 'Syne', sans-serif;
  font-size: 3.8rem; font-weight: 800;
  line-height: 1; letter-spacing: -0.04em;
  margin: 0.5rem 0 0.1rem;
}
.prob-label {
  font-size: 0.72rem; color: var(--t2); margin-bottom: 1rem;
  font-family: 'DM Mono', monospace;
}
.pbar-track {
  height: 6px; background: rgba(255,255,255,0.06);
  border-radius: 99px; overflow: hidden; margin-bottom: 0.7rem;
}
.pbar-fill {
  height: 100%; border-radius: 99px;
  position: relative; transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.pbar-fill::after {
  content:''; position:absolute; right:0; top:0; bottom:0; width:16px;
  background: linear-gradient(90deg,transparent,rgba(255,255,255,0.28));
}
.threshold-row {
  display: flex; justify-content: space-between;
  font-family: 'DM Mono', monospace; font-size: 0.62rem; color: var(--t3);
  margin-top: 0.3rem;
}

.factor { margin-bottom: 0.85rem; }
.factor-hdr {
  display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 0.28rem;
}
.factor-name { font-size: 0.78rem; font-weight: 500; color: var(--t1); }
.factor-pct  {
  font-family: 'DM Mono', monospace;
  font-size: 0.67rem; color: var(--t2);
}
.factor-track { height: 4px; background: rgba(255,255,255,0.06); border-radius:99px; }
.factor-fill  { height: 4px; border-radius:99px; }

.explain-block {
  background: var(--bg-glass);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 0.9rem 1rem;
  margin-top: 1rem;
  font-size: 0.8rem; color: var(--t2); line-height: 1.7;
}
.explain-block strong { color: var(--t1); }

.rec {
  background: var(--bg-glass);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 0.75rem 0.9rem;
  margin-bottom: 0.55rem;
  transition: background 0.2s, border-color 0.2s;
  animation: slide-up 0.4s ease both;
}
.rec:hover { background: var(--bg-hover); border-color: var(--border-lit); }
.rec-tag {
  font-family: 'Syne', sans-serif;
  font-size: 0.58rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase;
  display: block; margin-bottom: 0.22rem;
}
.rec-text { font-size: 0.79rem; color: var(--t2); line-height: 1.6; }

.loader-wrap {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 3.5rem 2rem; gap: 1rem;
}
.spinner {
  width: 36px; height: 36px;
  border: 3px solid rgba(255,255,255,0.06);
  border-top-color: var(--amber);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
.loader-text {
  font-family: 'DM Mono', monospace;
  font-size: 0.75rem; color: var(--t2); letter-spacing: 0.1em;
}
@keyframes spin { to { transform: rotate(360deg); } }

.empty {
  text-align: center; padding: 4rem 2rem;
  border: 1px dashed rgba(255,255,255,0.07);
  border-radius: var(--r-lg);
}
.empty-icon { font-size: 2.6rem; opacity: 0.25; margin-bottom: 0.7rem; }
.empty-h { font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700;
           color: var(--t2); margin-bottom:0.4rem; }
.empty-p { font-size:0.77rem; color: var(--t3); line-height:1.7; }
.empty-p strong { color: var(--amber); }

div[data-baseweb="select"] > div {
  background: var(--bg-input) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: var(--r-sm) !important;
  color: var(--t1) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.83rem !important;
  transition: border-color 0.18s, box-shadow 0.18s !important;
}
div[data-baseweb="select"] > div:focus-within {
  border-color: rgba(245,158,11,0.5) !important;
  box-shadow: 0 0 0 3px rgba(245,158,11,0.08) !important;
}
div[data-baseweb="popover"] {
  background: #0d1520 !important;
  border: 1px solid var(--border-lit) !important;
  box-shadow: 0 24px 48px rgba(0,0,0,0.7) !important;
}
div[role="option"] {
  color: var(--t2) !important; font-size: 0.82rem !important;
}
div[role="option"]:hover, div[role="option"][aria-selected="true"] {
  background: rgba(245,158,11,0.08) !important; color: var(--amber) !important;
}
input[type="number"] {
  background: var(--bg-input) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: var(--r-sm) !important;
  color: var(--t1) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.85rem !important;
}
input[type="number"]:focus {
  border-color: rgba(245,158,11,0.5) !important;
  box-shadow: 0 0 0 3px rgba(245,158,11,0.08) !important;
  outline: none !important;
}
label, .stSelectbox label, .stNumberInput label {
  font-size: 0.69rem !important; font-weight: 500 !important;
  text-transform: uppercase !important; letter-spacing: 0.07em !important;
  color: var(--t2) !important;
}
.stButton > button {
  font-family: 'Syne', sans-serif !important;
  font-size: 0.88rem !important; font-weight: 700 !important;
  letter-spacing: 0.1em !important; text-transform: uppercase !important;
  background: linear-gradient(135deg, var(--amber) 0%, #d97706 100%) !important;
  color: #06090f !important; border: none !important;
  border-radius: var(--r-sm) !important;
  padding: 0.68rem 1.5rem !important; width: 100% !important;
  margin-top: 1rem !important;
  box-shadow: 0 4px 18px rgba(245,158,11,0.3), inset 0 1px 0 rgba(255,255,255,0.15) !important;
  transition: box-shadow 0.25s, transform 0.15s !important;
}
.stButton > button:hover {
  box-shadow: 0 6px 28px rgba(245,158,11,0.55), inset 0 1px 0 rgba(255,255,255,0.15) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.total-box {
  background: var(--bg-input); border: 1px solid rgba(255,255,255,0.07);
  border-radius: var(--r-sm); padding: 0.48rem 0.78rem; margin-top: 0.3rem;
}
.total-lbl { font-size: 0.64rem; text-transform: uppercase;
             letter-spacing: 0.07em; color: var(--t3); display: block; }
.total-val { font-family: 'DM Mono', monospace;
             font-size: 0.96rem; font-weight: 500; color: var(--t1); }

.footer {
  display: flex; gap: 0.8rem; flex-wrap: wrap;
  border-top: 1px solid var(--border); padding-top: 1.5rem; margin-top: 2rem;
}
.ft {
  flex: 1; min-width: 100px;
  background: var(--bg-glass); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 0.75rem 0.9rem; text-align: center;
  backdrop-filter: blur(8px);
}
.ft-val { font-family: 'DM Mono',monospace; font-size:1rem; font-weight:500;
          color: var(--teal); display: block; }
.ft-lbl { font-size:0.6rem; text-transform:uppercase; letter-spacing:0.1em;
          color: var(--t3); margin-top:3px; display:block; }

@keyframes slide-up {
  from { opacity:0; transform:translateY(12px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes fade-in {
  from { opacity:0; transform:scale(0.98); }
  to   { opacity:1; transform:scale(1); }
}
.anim { animation: fade-in 0.45s cubic-bezier(0.2,0.8,0.3,1) both; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# default values for the high- and low-risk quick-fill buttons
HIGH_RISK_PRESET = {
    # model score: 0.9777  — highest achievable (97.8%)
    "gender": "Female", "senior": "Yes", "tenure": 1,
    "partner": "No", "dependents": "No",
    "phone": "Yes", "lines": "Yes",
    "internet": "Fiber optic", "security": "No", "backup": "No",
    "device": "No", "techsup": "No", "tvstream": "Yes", "movies": "Yes",
    "contract": "Month-to-month", "paperless": "Yes",
    "payment": "Electronic check", "monthly": 95.0,
}

LOW_RISK_PRESET = {
    # model score: 0.0345  — validated ≤ 0.30
    "gender": "Male", "senior": "No", "tenure": 58,
    "partner": "Yes", "dependents": "Yes",
    "phone": "Yes", "lines": "Yes",
    "internet": "DSL", "security": "Yes", "backup": "Yes",
    "device": "Yes", "techsup": "Yes", "tvstream": "Yes", "movies": "Yes",
    "contract": "Two year", "paperless": "No",
    "payment": "Bank transfer (automatic)", "monthly": 62.0,
}

MID_RISK_PRESET = {
    # model score: 0.4602  — validated 0.30–0.60
    "gender": "Male", "senior": "No", "tenure": 10,
    "partner": "No", "dependents": "No",
    "phone": "Yes", "lines": "No",
    "internet": "DSL", "security": "No", "backup": "No",
    "device": "No", "techsup": "No", "tvstream": "No", "movies": "No",
    "contract": "Month-to-month", "paperless": "Yes",
    "payment": "Electronic check", "monthly": 50.0,
}




@st.cache_resource
def load_bundle() -> dict:
    pkl = ROOT / "models" / "churn_model.pkl"
    if not pkl.exists():
        st.error(f"Model not found: {pkl}. Run `python src/train.py` first.")
        st.stop()
    with open(pkl, "rb") as f:
        return pickle.load(f)


def classify_risk(prob: float) -> dict:
    """Map a probability to a risk level — thresholds: Low <30%, Mid 30–60%, High ≥60%."""
    if prob >= 0.60:
        return dict(level="High",   cls="high",   cell_cls="cell-danger",
                    color="#f10909", rb_cls="rb-high",   dot_cls="rb-dot-high",
                    label="HIGH RISK",      verb="will likely churn")
    elif prob >= 0.30:
        return dict(level="Medium", cls="medium", cell_cls="cell-warn",
                    color="#f59e0b", rb_cls="rb-medium", dot_cls="rb-dot-medium",
                    label="MODERATE RISK",  verb="is at risk of churning")
    else:
        return dict(level="Low",    cls="low",    cell_cls="cell-safe",
                    color="#35dda5", rb_cls="rb-low",    dot_cls="rb-dot-low",
                    label="LOW RISK",       verb="is likely to stay")


def make_gauge(prob: float, risk: dict) -> go.Figure:
    """
    Static gauge with risk-aware visual weight.
    High risk: bright red zone, thicker bar, stronger threshold line.
    Mid/Low: progressively softer emphasis.
    """
    pct   = prob * 100
    color = risk["color"]
    level = risk["level"]

    # zone brightness scales with risk level so the active zone dominates
    if level == "High":
        zone_safe = "rgba(16,185,129,0.04)"
        zone_mid  = "rgba(245,158,11,0.06)"
        zone_risk = "rgba(239,68,68,0.22)"
        bar_thickness   = 0.22
        threshold_width = 3.5
    elif level == "Medium":
        zone_safe = "rgba(16,185,129,0.05)"
        zone_mid  = "rgba(245,158,11,0.18)"
        zone_risk = "rgba(239,68,68,0.06)"
        bar_thickness   = 0.18
        threshold_width = 2.5
    else:
        zone_safe = "rgba(16,185,129,0.16)"
        zone_mid  = "rgba(245,158,11,0.05)"
        zone_risk = "rgba(239,68,68,0.03)"
        bar_thickness   = 0.16
        threshold_width = 2.0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(
            suffix="%",
            valueformat=".1f",
            font=dict(family="Syne, sans-serif", size=36, color=color),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="rgba(255,255,255,0.06)",
                tickfont=dict(
                    family="DM Mono, monospace",
                    size=9,
                    color="rgba(150,180,210,0.85)",
                ),
                tickvals=[0, 30, 60, 100],
                ticktext=["SAFE", "MID", "RISK", ""],
            ),
            bar=dict(color=color, thickness=bar_thickness),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  30],  color=zone_safe),
                dict(range=[30, 60],  color=zone_mid),
                dict(range=[60, 100], color=zone_risk),
            ],
            threshold=dict(
                line=dict(color=color, width=threshold_width),
                thickness=0.78,
                value=pct,
            ),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=16, b=4, l=18, r=18),
        height=200,
        font=dict(family="DM Sans, sans-serif", color="#7d98b8"),
    )

    return fig


# readable labels for the 28 encoded feature names
FEATURE_LABELS = {
    "is_long_term":               "Long-term contract",
    "charges_per_service":        "Charges per service",
    "Contract":                   "Contract type",
    "MonthlyCharges":             "Monthly charges",
    "avg_monthly_charges":        "Avg monthly charges",
    "tenure":                     "Customer tenure",
    "tenure_monthly_interaction": "Tenure × charges",
    "TotalCharges":               "Total charges paid",
    "tenure_group":               "Tenure group",
    "PaymentMethod":              "Payment method",
    "InternetService":            "Internet service",
    "OnlineSecurity":             "Online security",
    "TechSupport":                "Tech support",
    "total_services":             "Total active services",
    "has_any_security":           "Has security service",
    "has_any_backup":             "Has backup service",
    "paperless_electronic":       "Paperless + e-check",
    "PaperlessBilling":           "Paperless billing",
    "gender":                     "Gender",
    "SeniorCitizen":              "Senior citizen",
    "Partner":                    "Has partner",
    "Dependents":                 "Has dependents",
    "PhoneService":               "Phone service",
    "MultipleLines":              "Multiple lines",
    "OnlineBackup":               "Online backup",
    "DeviceProtection":           "Device protection",
    "StreamingTV":                "Streaming TV",
    "StreamingMovies":            "Streaming movies",
}


def top_factors(X: pd.DataFrame, features: list,
                importances: np.ndarray, n: int = 3) -> list[dict]:
    """
    Per-customer signal strength: global importance × |scaled feature value|.
    Normalised so the top factor always fills 100% of its bar.
    """
    contribs = importances * np.abs(X.values[0])
    idx      = np.argsort(contribs)[::-1][:n]
    max_c    = max(contribs[idx[0]], 1e-9)
    return [{
        "name":    FEATURE_LABELS.get(features[i],
                   features[i].replace("_", " ").title()),
        "key":     features[i],
        "bar_pct": round(contribs[i] / max_c * 100, 1),
        "imp_pct": round(importances[i] * 100, 1),
        "contrib": contribs[i],
    } for i in idx]


def build_explanation(raw: dict, risk: dict, factors: list[dict]) -> str:
    """
    Generate 2-3 business-style sentences about why this specific customer
    is at risk, derived from their actual input values.
    """
    level = risk["level"]
    lines = []

    # sentence 1: tenure + contract
    tenure   = raw["tenure"]
    contract = raw["Contract"]

    if tenure <= 6:
        tenure_phrase = f"only {tenure} month{'s' if tenure != 1 else ''} with the company"
    elif tenure <= 18:
        tenure_phrase = f"{tenure} months of tenure"
    else:
        tenure_phrase = f"{tenure} months as a customer"

    if contract == "Month-to-month":
        contract_phrase = "no long-term commitment"
    elif contract == "One year":
        contract_phrase = "a one-year contract"
    else:
        contract_phrase = "a two-year contract"

    lines.append(
        f"This customer has <strong>{tenure_phrase}</strong> and "
        f"<strong>{contract_phrase}</strong>, "
        + ("which means they face very low switching friction."
           if contract == "Month-to-month"
           else "giving them a meaningful commitment anchor.")
    )

    # sentence 2: charges + service count
    monthly    = raw["MonthlyCharges"]
    internet   = raw["InternetService"]
    n_services = sum([
        raw["PhoneService"]    == "Yes",
        raw["OnlineSecurity"]  == "Yes",
        raw["OnlineBackup"]    == "Yes",
        raw["DeviceProtection"]== "Yes",
        raw["TechSupport"]     == "Yes",
        raw["StreamingTV"]     == "Yes",
        raw["StreamingMovies"] == "Yes",
        internet != "No",
    ])

    if internet == "Fiber optic" and monthly > 70:
        lines.append(
            f"At <strong>${monthly:.0f}/month</strong> for Fiber optic with "
            f"<strong>{n_services} active services</strong>, "
            "their cost-per-service ratio signals they may feel they're "
            "not getting enough value for what they pay."
        )
    elif n_services <= 2 and internet != "No":
        lines.append(
            f"With only <strong>{n_services} active service{'s' if n_services!=1 else ''}</strong> "
            f"and a <strong>${monthly:.0f}/month</strong> bill, "
            "they have few product ties that would make leaving inconvenient."
        )
    else:
        lines.append(
            f"Their <strong>${monthly:.0f}/month</strong> spend across "
            f"<strong>{n_services} services</strong> "
            + ("creates reasonable switching costs."
               if n_services >= 4
               else "leaves room for a competitor to undercut them.")
        )

    # sentence 3: payment method + security gap
    security_missing = (raw["OnlineSecurity"] == "No" and internet != "No")
    payment_risky    = raw["PaymentMethod"] == "Electronic check"

    if security_missing and payment_risky:
        lines.append(
            "The combination of <strong>no online security</strong> and "
            "<strong>electronic-check payment</strong> are two of the strongest "
            "behavioural signals of an at-risk customer in this dataset."
        )
    elif security_missing:
        lines.append(
            "Customers without <strong>Online Security</strong> churn at 42% — "
            "nearly three times the rate of those who have it."
        )
    elif payment_risky:
        lines.append(
            "<strong>Electronic-check users</strong> show 45% higher churn than "
            "auto-pay customers, often a signal of lower platform engagement."
        )
    elif level == "Low":
        lines.append(
            "Their combination of long tenure, long-term contract, and multiple "
            "active services makes them one of the most stable customer profiles."
        )

    return " ".join(lines)


def get_recommendations(raw: dict, risk_level: str) -> list[dict]:
    recs = []

    if risk_level == "Low":
        recs.append({"tag": "✅  Loyalty Reward",
                     "text": "Stable customer — enrol in a loyalty programme to "
                             "deepen engagement and raise future switching costs."})
        return recs

    if raw["Contract"] == "Month-to-month":
        recs.append({"tag": "📋  Contract Upgrade",
                     "text": "Offer 10–15% off an annual plan. Long-term customers "
                             "reduce churn risk by ~85%."})

    if raw["InternetService"] == "Fiber optic" and raw["MonthlyCharges"] > 70:
        recs.append({"tag": "💰  Price Adjustment",
                     "text": f"${raw['MonthlyCharges']:.0f}/mo is above segment average. "
                             "A 6-month promotional rate (−$10–15) re-anchors value."})

    if raw["OnlineSecurity"] == "No" and raw["InternetService"] != "No":
        recs.append({"tag": "🔐  Free Security Trial",
                     "text": "42% churn without OnlineSecurity vs 15% with it. "
                             "Offer a 3-month free trial — near-zero adoption cost."})

    if raw["TechSupport"] == "No" and raw["InternetService"] != "No":
        recs.append({"tag": "🛠  Tech Support Bundle",
                     "text": "Proactively offer a discounted Tech Support plan. "
                             "Self-service friction is a top churn driver."})

    if raw["tenure"] <= 12:
        recs.append({"tag": "👋  Onboarding Check-in",
                     "text": "First-year is the highest-risk window (47% churn). "
                             "Schedule calls at 30 and 90 days."})

    if raw["PaymentMethod"] == "Electronic check":
        recs.append({"tag": "💳  Auto-pay Migration",
                     "text": "E-check users churn 45% more. Offer a $10 credit "
                             "for switching to bank transfer or card."})

    if not recs:
        recs.append({"tag": "📊  Proactive Monitor",
                     "text": "No dominant risk signal. Run quarterly NPS check "
                             "and watch for usage drops over the next 60 days."})

    return recs[:3]


def main():
    bundle      = load_bundle()
    encoders    = bundle["encoder"]
    scaler      = bundle["scaler"]
    features    = bundle["features"]
    model       = bundle["model"]
    importances = model.feature_importances_
    results_d   = bundle.get("results", {})

    # track which scenario preset (if any) is active
    if "preset" not in st.session_state:
        st.session_state.preset = None

    def apply_preset(name):
        st.session_state.preset = name

    st.markdown(f"""
    <div class="header">
      <div class="brand">
        <div class="brand-icon">⚡</div>
        <div>
          <div class="brand-name">ChurnGuard AI</div>
          <div class="brand-sub">Customer Retention Intelligence Platform</div>
        </div>
      </div>
      <div class="pills">
        <div class="pill"><span class="live-dot"></span> Model <strong>Live</strong></div>
        <div class="pill">Dataset <strong>7,043</strong></div>
        <div class="pill">ROC-AUC <strong>0.841</strong></div>
        <div class="pill">Features <strong>28</strong></div>
        <div class="pill">Churn rate <strong>26.5%</strong></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([5, 4], gap="large")

    with left:
        st.markdown('<div class="glass input-panel">', unsafe_allow_html=True)

        st.markdown('<div class="sec">Quick Scenarios</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        if s1.button("🔴  High Risk"):
            apply_preset("high")
            st.rerun()
        if s2.button("🟠  Mid Risk"):
            apply_preset("mid")
            st.rerun()
        if s3.button("🟢  Low Risk"):
            apply_preset("low")
            st.rerun()

        # populate fields from preset, or fall back to empty defaults
        preset = st.session_state.preset
        P = HIGH_RISK_PRESET if preset == "high" else (
            MID_RISK_PRESET  if preset == "mid"  else (
            LOW_RISK_PRESET  if preset == "low"  else {}))

        st.markdown('<div class="sec" style="margin-top:1.4rem">Customer Profile</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender",         ["Female", "Male"],
                               index=["Female","Male"].index(P.get("gender","Female")))
        senior = c2.selectbox("Senior Citizen", ["No", "Yes"],
                               index=["No","Yes"].index(P.get("senior","No")))
        tenure = c3.number_input("Tenure (months)", 0, 72,
                                  value=int(P.get("tenure", 12)))

        c4, c5 = st.columns(2)
        partner    = c4.selectbox("Partner",    ["No","Yes"],
                                   index=["No","Yes"].index(P.get("partner","No")))
        dependents = c5.selectbox("Dependents", ["No","Yes"],
                                   index=["No","Yes"].index(P.get("dependents","No")))

        st.markdown('<div class="sec" style="margin-top:1.2rem">Services</div>',
                    unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)

        _yn   = ["No","Yes"]
        _ynni = ["No","Yes","No internet service"]
        _ynnp = ["No","Yes","No phone service"]

        phone    = c6.selectbox("Phone Service",     _yn,   index=_yn.index(P.get("phone","Yes")))
        lines    = c6.selectbox("Multiple Lines",    _ynnp, index=_ynnp.index(P.get("lines","No")))
        internet = c7.selectbox("Internet Service",  ["DSL","Fiber optic","No"],
                                 index=["DSL","Fiber optic","No"].index(P.get("internet","DSL")))
        security = c7.selectbox("Online Security",   _ynni,
                                 index=_ynni.index(P.get("security","No")))
        backup   = c7.selectbox("Online Backup",     _ynni,
                                 index=_ynni.index(P.get("backup","No")))
        device   = c8.selectbox("Device Protection", _ynni,
                                 index=_ynni.index(P.get("device","No")))
        techsup  = c8.selectbox("Tech Support",      _ynni,
                                 index=_ynni.index(P.get("techsup","No")))
        tvstream = c8.selectbox("Streaming TV",      _ynni,
                                 index=_ynni.index(P.get("tvstream","No")))
        movies   = c8.selectbox("Streaming Movies",  _ynni,
                                 index=_ynni.index(P.get("movies","No")))

        st.markdown('<div class="sec" style="margin-top:1.2rem">Billing &amp; Contract</div>',
                    unsafe_allow_html=True)
        c9, c10 = st.columns(2)
        _contracts = ["Month-to-month","One year","Two year"]
        _payments  = ["Electronic check","Mailed check",
                      "Bank transfer (automatic)","Credit card (automatic)"]

        contract  = c9.selectbox("Contract Type",
                                  _contracts,
                                  index=_contracts.index(P.get("contract","Month-to-month")))
        paperless = c9.selectbox("Paperless Billing", _yn,
                                  index=_yn.index(P.get("paperless","Yes")))
        payment   = c10.selectbox("Payment Method", _payments,
                                   index=_payments.index(P.get("payment","Electronic check")))
        monthly   = c10.number_input("Monthly Charges ($)", 0.0, 200.0,
                                      value=float(P.get("monthly", 65.0)),
                                      step=1.0, format="%.2f")

        total = float(tenure) * monthly
        st.markdown(f"""
        <div class="total-box">
          <span class="total-lbl">Estimated Total Charges</span>
          <span class="total-val">${total:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        predict_btn = st.button("⚡  Run Churn Analysis", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        if predict_btn:
            if monthly <= 0:
                st.error("Monthly charges must be greater than $0.")
                return

            raw = {
                "gender":           gender,
                "SeniorCitizen":    1 if senior == "Yes" else 0,
                "Partner":          partner,
                "Dependents":       dependents,
                "tenure":           int(tenure),
                "PhoneService":     phone,
                "MultipleLines":    lines,
                "InternetService":  internet,
                "OnlineSecurity":   security,
                "OnlineBackup":     backup,
                "DeviceProtection": device,
                "TechSupport":      techsup,
                "StreamingTV":      tvstream,
                "StreamingMovies":  movies,
                "Contract":         contract,
                "PaperlessBilling": paperless,
                "PaymentMethod":    payment,
                "MonthlyCharges":   float(monthly),
                "TotalCharges":     float(total),
            }

            # show spinner while running inference, then clear it
            with st.spinner(""):
                loader = st.empty()
                loader.markdown("""
                <div class="loader-wrap">
                  <div class="spinner"></div>
                  <div class="loader-text">ANALYSING CUSTOMER SIGNALS…</div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.55)  # intentional pause so the spinner is visible

                X           = prepare_input(raw, encoders, scaler)
                prob        = model.predict_proba(X)[0][1]
                pct         = prob * 100
                risk        = classify_risk(prob)
                facs        = top_factors(X, features, importances, n=3)
                recs        = get_recommendations(raw, risk["level"])
                explanation = build_explanation(raw, risk, facs)

                loader.empty()

            st.markdown('<div class="results-grid anim">', unsafe_allow_html=True)

            # top-left: gauge
            st.markdown(f'<div class="grid-cell {risk["cell_cls"]}">'
                        f'<div class="sec">Risk Gauge</div>',
                        unsafe_allow_html=True)

            gauge_fig = make_gauge(prob, risk)
            st.plotly_chart(
                gauge_fig,
                use_container_width=True,
                config={"displayModeBar": False, "staticPlot": True},
                key="gauge_chart",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # top-right: probability score + explanation
            st.markdown(f'<div class="grid-cell {risk["cell_cls"]}">'
                        f'<div class="sec">Probability Score</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <span class="risk-badge {risk['rb_cls']}">
              <span class="rb-dot {risk['dot_cls']}"></span>
              {risk['label']}
            </span>
            <div class="prob-number" style="color:{risk['color']}">{pct:.1f}</div>
            <div class="prob-label">% churn probability</div>
            <div class="pbar-track">
              <div class="pbar-fill"
                   style="width:{pct:.1f}%;
                          background:linear-gradient(90deg,{risk['color']}77,{risk['color']})">
              </div>
            </div>
            <div class="threshold-row">
              <span>0%</span><span>Low · Med · High</span><span>100%</span>
            </div>
            <div class="explain-block">
              {explanation}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # bottom-left: factor bars
            st.markdown(f'<div class="grid-cell">'
                        f'<div class="sec">Top Contributing Factors</div>',
                        unsafe_allow_html=True)
            for fac in facs:
                st.markdown(f"""
                <div class="factor">
                  <div class="factor-hdr">
                    <span class="factor-name">{fac['name']}</span>
                    <span class="factor-pct">{fac['imp_pct']}% global weight</span>
                  </div>
                  <div class="factor-track">
                    <div class="factor-fill"
                         style="width:{fac['bar_pct']}%;
                                background:linear-gradient(90deg,
                                  {risk['color']}44,{risk['color']})">
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # bottom-right: retention action cards
            st.markdown(f'<div class="grid-cell">'
                        f'<div class="sec">Retention Actions</div>',
                        unsafe_allow_html=True)
            for i, rec in enumerate(recs):
                st.markdown(f"""
                <div class="rec" style="animation-delay:{i*0.09:.2f}s">
                  <span class="rec-tag" style="color:{risk['color']}">{rec['tag']}</span>
                  <span class="rec-text">{rec['text']}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)  # close results-grid

        else:
            st.markdown("""
            <div class="empty">
              <div class="empty-icon">📡</div>
              <div class="empty-h">Awaiting Customer Profile</div>
              <div class="empty-p">
                Fill in the form on the left — or click a<br>
                <strong>scenario button</strong> to load an example —<br>
                then click <strong>Run Churn Analysis</strong> to see<br>
                the gauge, risk score, and retention playbook.
              </div>
            </div>
            """, unsafe_allow_html=True)

    best_auc = max((v.get("roc_auc", 0) for v in results_d.values()), default=0.84)
    best_acc = max((v.get("accuracy",0) for v in results_d.values()), default=0.78)

    st.markdown(f"""
    <div class="footer">
      <div class="ft"><span class="ft-val">7,043</span><span class="ft-lbl">Training Samples</span></div>
      <div class="ft"><span class="ft-val">28</span><span class="ft-lbl">Feature Signals</span></div>
      <div class="ft"><span class="ft-val">{best_auc:.3f}</span><span class="ft-lbl">Best ROC-AUC</span></div>
      <div class="ft"><span class="ft-val">{best_acc:.1%}</span><span class="ft-lbl">Best Accuracy</span></div>
      <div class="ft"><span class="ft-val">26.5%</span><span class="ft-lbl">Dataset Churn Rate</span></div>
      <div class="ft"><span class="ft-val">RF+</span><span class="ft-lbl">Algorithm</span></div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
