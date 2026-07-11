"""
app/app.py  —  ChurnGuard AI · AI-Powered Customer Retention Platform
─────────────────────────────────────────────────────────────────────────────
UI/UX REDESIGN — v4 (Premium AI SaaS pass)
  • [CSS] Richer 5-color system: Orange (brand) · Indigo/Violet (AI) ·
    Green/Gold/Red (risk tiers, now distinct from brand orange). Restored
    tasteful gradients, soft glows, glass effects, hover lift — without
    reverting to the earlier heavy-glow/multi-border "gamer" aesthetic.
  • [HERO] New premium hero band above the tabs: gradient wordmark, product
    statement, inline key-metric chips, status pills. First-open impression.
  • [ICONS] Consistent iconography across tabs and section titles
    (⚡ Predict · 📚 Knowledge Assistant · 📊 Model Insights · 🧠 Explanation
    · 🎯 Retention Strategy) for scanning/navigation.
  • [AI CARDS] Explanation + Retention Strategy now visually distinct
    "flagship" cards — indigo/violet gradient edge + soft glow, set apart
    from ordinary neutral cards. Still always visible, never in expanders.
  • [PREDICTION RESULT] Stronger centerpiece treatment — risk-colored glow,
    larger type, gradient accent — same merged Gauge+Score+Factors content.
  • [METRICS] Stripe/Vercel-style metric cards — icon + gradient number +
    colored top accent, used in hero and Model Insights.
  • Zero changes to: AI engine calls, RAG engine calls, preprocessing,
    model loading, preset values, or get_recommendations' body.
  • [AUDIT PASS] Key Factors were previously global feature_importances_
    weighted by |scaled value| — same top-2 factors for almost every
    customer (see compute_shap_factors() docstring). Replaced with real
    per-customer SHAP (Tree SHAP) contributions, signed by direction.
    build_explanation() removed (dead code, never called). classify_risk's
    stale comment (claimed 0.60/0.30 thresholds; code used 0.80/0.40 —
    which already matched the requested business spec) corrected, no
    threshold change. Added sort_retention_actions() to deterministically
    enforce URGENT->HIGH->STANDARD ordering on AI strategy output.
  • [AUDIT PASS 2 — pre-defense demo reliability] Preset profiles fixed:
    MID preset previously netted Low risk (~20-27%) because it stacked
    multiple protective traits against one risk trait; rebuilt and swept
    against the real model to land centered in-band (~62%, ±20pt margin).
    HIGH preset hardened from an 82% (2pt margin above the 80% boundary)
    to ~94%. New humanize_factor_label() makes Key Factor labels
    value-aware so a label can never read as the logical opposite of the
    customer's actual data (e.g. "Long-term Contract ▲ raises risk" shown
    for a month-to-month customer). AI explanation/strategy generation now
    goes through ai_engine.py's generate_explanation_safe() /
    generate_retention_strategy_safe() — validate -> regenerate/repair ->
    deterministic fallback — instead of raw token streaming, so the UI
    never displays unvalidated LLM output (see ai_engine.py's docstring
    for the full pipeline).
─────────────────────────────────────────────────────────────────────────────
"""

# ── Standard library & third-party ───────────────────────────────────────
import os
import re
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── SHAP — required for customer-specific (local) Key Factors ─────────────
# See compute_shap_factors() below for the full audit rationale: global
# feature_importances_ alone cannot answer "what drove THIS customer's
# prediction" — only a local attribution method like SHAP can.
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

load_dotenv()

# ── Project src/ imports ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import prepare_input

# ── AI engine imports — degrade gracefully if package missing ─────────────
try:
    from ai_engine import ChurnAIEngine as _ChurnAIEngine
    _AI_ENGINE_IMPORTABLE = True
except ImportError:
    _AI_ENGINE_IMPORTABLE = False

try:
    from rag_engine import ChurnRAGEngine as _ChurnRAGEngine
    _RAG_ENGINE_IMPORTABLE = True
except ImportError:
    _RAG_ENGINE_IMPORTABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────
# CSS  — Premium AI SaaS design system
# ─────────────────────────────────────────────────────────────────────────

CSS = """
<style>
/* ── Design tokens ─────────────────────────────────────────────────────── */
:root {
  --bg          : #0a0b0f;
  --surface     : #15171c;
  --surface-2   : #1a1d24;
  --border      : rgba(255,255,255,0.08);
  --border-md   : rgba(255,255,255,0.13);
  --t1          : #f5f6f8;
  --t2          : #9398a3;
  --t3          : #5c6170;

  --brand       : #f97316;
  --brand-2     : #fb923c;
  --brand-soft  : rgba(249,115,22,0.10);
  --brand-line  : rgba(249,115,22,0.35);

  --ai          : #6366f1;
  --ai-2        : #8b5cf6;
  --ai-soft     : rgba(99,102,241,0.08);
  --ai-line     : rgba(99,102,241,0.30);

  --danger      : #ef4444;
  --danger-soft : rgba(239,68,68,0.10);
  --gold        : #eab308;
  --gold-soft   : rgba(234,179,8,0.10);
  --safe        : #22c55e;
  --safe-soft   : rgba(34,197,94,0.10);

  --r-sm        : 7px;
  --r-md        : 11px;
  --r-lg        : 16px;
  --shadow      : 0 2px 10px rgba(0,0,0,0.35);
  --shadow-lift : 0 6px 24px rgba(0,0,0,0.45);
}

/* ── Global reset ──────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important; color: var(--t1);
}
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { max-width: 1300px; padding: 0 2rem 2.2rem !important; }
* { box-sizing: border-box; }
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
body, p, div, span, label, select, input, button { font-family: 'Inter', sans-serif; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* ── Hero band ──────────────────────────────────────────────────────────── */
.hero-wrap {
  position: relative; overflow: hidden;
  padding: 2.4rem 0 1.8rem; margin-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
}
.hero-glow {
  position: absolute; top: -120px; left: 50%; transform: translateX(-50%);
  width: 640px; height: 320px; pointer-events: none;
  background: radial-gradient(ellipse at center, rgba(249,115,22,0.16) 0%, rgba(99,102,241,0.07) 45%, transparent 75%);
  filter: blur(10px);
}
.hero-top { position: relative; display: flex; justify-content: space-between; align-items: flex-start; }
.hero-brand-row { display: flex; align-items: center; gap: 0.7rem; }
.hero-icon {
  width: 40px; height: 40px; border-radius: var(--r-md);
  background: linear-gradient(135deg, var(--brand), var(--brand-2));
  display: flex; align-items: center; justify-content: center;
  font-size: 1.15rem; box-shadow: 0 4px 18px rgba(249,115,22,0.30);
}
.hero-title {
  font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(90deg, #ffffff 30%, var(--brand-2) 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  line-height: 1.1;
}
.hero-subtitle { font-size: 0.85rem; color: var(--t2); margin-top: 0.15rem; font-weight: 500; }
.hero-statement { font-size: 0.82rem; color: var(--t3); margin-top: 0.6rem; max-width: 460px; line-height: 1.6; }
.hero-status-col { display: flex; flex-direction: column; gap: 0.45rem; align-items: flex-end; }
.status-pill {
  display: flex; align-items: center; gap: 5px;
  background: rgba(255,255,255,0.04); border: 1px solid var(--border);
  backdrop-filter: blur(6px);
  border-radius: 99px; padding: 0.3rem 0.75rem;
  font-size: 0.70rem; color: var(--t2);
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; }
.status-dot.on  { background: var(--safe); box-shadow: 0 0 6px rgba(34,197,94,0.6); }
.status-dot.off { background: var(--t3); }

.hero-metrics { position: relative; display: flex; gap: 0.7rem; margin-top: 1.4rem; flex-wrap: wrap; }
.hero-metric {
  flex: 1; min-width: 130px;
  background: rgba(255,255,255,0.035); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 0.75rem 1rem;
  backdrop-filter: blur(6px);
  transition: border-color 0.18s ease, transform 0.18s ease;
}
.hero-metric:hover { border-color: var(--brand-line); transform: translateY(-2px); }
.hero-metric-val {
  font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700;
  background: linear-gradient(90deg, var(--t1), var(--brand-2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-metric-lbl { font-size: 0.66rem; color: var(--t3); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.3rem;
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  padding-bottom: 0 !important;
  margin-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  color: var(--t3) !important;
  font-size: 0.86rem !important;
  font-weight: 500 !important;
  padding: 0.7rem 0.2rem !important;
  margin-right: 1.5rem !important;
  transition: color 0.15s ease, border-color 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
  color: var(--t1) !important;
  border-bottom: 2px solid var(--brand) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding-top: 1.7rem !important;
  background: transparent !important;
}

/* ── Section label ─────────────────────────────────────────────────────── */
.sec {
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--t3); margin-bottom: 0.75rem;
  display: flex; align-items: center; gap: 0.4rem;
}

/* ── Standard cards ─────────────────────────────────────────────────────── */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: 1.3rem 1.5rem;
  box-shadow: var(--shadow);
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
.card:hover { border-color: var(--border-md); }
.card + .card { margin-top: 0.9rem; }

/* ── Input panel ───────────────────────────────────────────────────────── */
.input-panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: 1.4rem 1.5rem;
}

/* ── Total box ─────────────────────────────────────────────────────────── */
.total-box {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--r-sm); padding: 0.6rem 1rem;
  display: flex; justify-content: space-between; align-items: center;
  margin-top: 0.7rem;
}
.total-lbl { font-size: 0.72rem; color: var(--t3); }
.total-val { font-family: 'JetBrains Mono', monospace; font-size: 0.95rem;
             color: var(--brand-2); font-weight: 600; }

/* ── Prediction Result — the centerpiece ────────────────────────────────── */
.result-card {
  position: relative; overflow: hidden;
  background: linear-gradient(160deg, var(--surface) 0%, var(--surface-2) 100%);
  border: 1px solid var(--border-md);
  border-radius: var(--r-lg); padding: 1.8rem 2rem;
  box-shadow: var(--shadow-lift);
}
.result-card.risk-high   { border-color: rgba(239,68,68,0.30);  box-shadow: 0 8px 36px -10px rgba(239,68,68,0.35),  var(--shadow); }
.result-card.risk-medium { border-color: rgba(234,179,8,0.30);  box-shadow: 0 8px 36px -10px rgba(234,179,8,0.30),  var(--shadow); }
.result-card.risk-low    { border-color: rgba(34,197,94,0.30);  box-shadow: 0 8px 36px -10px rgba(34,197,94,0.30),  var(--shadow); }
.result-card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.result-card.risk-high::before   { background: linear-gradient(90deg, var(--danger), #f87171); }
.result-card.risk-medium::before { background: linear-gradient(90deg, var(--gold), #fde047); }
.result-card.risk-low::before    { background: linear-gradient(90deg, var(--safe), #4ade80); }

.result-title { font-size: 0.78rem; font-weight: 600; color: var(--t3); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.9rem; }

.risk-badge {
  display: inline-flex; align-items: center; gap: 6px;
  border-radius: 99px; padding: 0.3rem 0.9rem;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
  border: 1px solid;
}
.rb-high   { background: var(--danger-soft); border-color: rgba(239,68,68,0.35); color: #f87171; }
.rb-medium { background: var(--gold-soft);    border-color: rgba(234,179,8,0.35); color: var(--gold); }
.rb-low    { background: var(--safe-soft);    border-color: rgba(34,197,94,0.35); color: #4ade80; }
.rb-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

.prob-number { font-family: 'JetBrains Mono', monospace; font-size: 3rem;
                font-weight: 700; line-height: 1; }
.prob-label { font-size: 0.76rem; color: var(--t3); margin-top: 0.25rem; }

.factor-row { margin-bottom: 0.65rem; }
.factor-row:last-child { margin-bottom: 0; }
.factor-hdr { display: flex; justify-content: space-between; margin-bottom: 4px; }
.factor-name { font-size: 0.80rem; color: var(--t1); }
.factor-pct  { font-size: 0.69rem; color: var(--t3); font-family: 'JetBrains Mono', monospace; }
.factor-track { height: 5px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
.factor-fill  { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--brand), var(--brand-2)); }

/* ── AI flagship cards - visually distinct from ordinary cards ─────────── */
.ai-card {
  position: relative; overflow: hidden;
  background: linear-gradient(160deg, rgba(99,102,241,0.06) 0%, var(--surface) 55%);
  border: 1px solid var(--ai-line);
  border-radius: var(--r-lg); padding: 1.4rem 1.6rem;
  box-shadow: 0 6px 28px -10px rgba(99,102,241,0.20), var(--shadow);
}
.ai-card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--ai), var(--ai-2));
}
.ai-card + .ai-card { margin-top: 0.9rem; }
.ai-card-title {
  display: flex; align-items: center; gap: 0.55rem;
  font-size: 0.95rem; font-weight: 700; color: var(--t1); margin-bottom: 1rem;
}
.ai-badge {
  font-size: 0.60rem; font-weight: 700; letter-spacing: 0.04em;
  color: var(--ai-2); background: var(--ai-soft); border: 1px solid var(--ai-line);
  border-radius: 99px; padding: 0.15rem 0.55rem; text-transform: uppercase;
}
.ai-body {
  font-size: 0.85rem; line-height: 1.8; color: var(--t2);
  white-space: pre-wrap; word-wrap: break-word;
}
.ai-body strong, .ai-body b { color: var(--t1); }

/* ── Quiet supplementary chips ("Also consider") ────────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.9rem; }
.chip {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 99px; padding: 0.32rem 0.8rem;
  font-size: 0.73rem; color: var(--t2);
}
.chip strong { color: var(--t1); font-weight: 500; }

/* ── Unavailable / empty states ────────────────────────────────────────── */
.notice {
  background: var(--surface); border: 1px dashed var(--border-md);
  border-radius: var(--r-md); padding: 0.9rem 1.15rem;
  font-size: 0.79rem; color: var(--t3); text-align: center;
}
.notice code {
  font-family: 'JetBrains Mono', monospace; color: var(--brand-2);
  background: var(--brand-soft); padding: 0.1rem 0.4rem; border-radius: 4px;
}
.empty {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 3.6rem 2rem; text-align: center;
}
.empty-h { font-size: 0.98rem; font-weight: 700; color: var(--t2); margin-bottom: 0.5rem; }
.empty-p { font-size: 0.80rem; color: var(--t3); line-height: 1.7; max-width: 380px; }
.empty-p strong { color: var(--brand-2); font-weight: 600; }

/* ── Streamlit widget overrides ────────────────────────────────────────── */
.stButton > button {
  background: var(--surface-2); border: 1px solid var(--border-md);
  color: var(--t1); border-radius: var(--r-sm);
  font-size: 0.81rem; font-weight: 500; padding: 0.52rem 1rem;
  transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
  box-shadow: none;
}
.stButton > button:hover {
  border-color: var(--brand-line); color: var(--brand-2);
  transform: translateY(-1px); box-shadow: 0 4px 14px rgba(249,115,22,0.12);
}
.stButton > button:active { transform: translateY(0); }
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, var(--brand), #ea580c) !important;
  border: 1px solid var(--brand) !important;
  color: #1a0e00 !important; font-weight: 700 !important;
  box-shadow: 0 4px 18px rgba(249,115,22,0.30) !important;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 22px rgba(249,115,22,0.40) !important;
  color: #1a0e00 !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
.stSelectbox select {
  background: var(--surface-2) !important; border: 1px solid var(--border) !important;
  color: var(--t1) !important; border-radius: var(--r-sm) !important;
  font-size: 0.82rem !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
  font-size: 0.72rem !important; color: var(--t3) !important;
}
.stProgress > div > div { background: linear-gradient(90deg, var(--brand), var(--brand-2)) !important; }
[data-testid="stInfo"] {
  background: var(--brand-soft) !important; border-color: var(--brand-line) !important;
  border-radius: var(--r-sm) !important; font-size: 0.78rem !important;
}
[data-testid="stFileUploader"] {
  background: var(--surface) !important; border: 1px dashed var(--border-md) !important;
  border-radius: var(--r-md) !important;
}
.input-panel .stTabs [data-baseweb="tab"] { font-size: 0.79rem !important; padding: 0.45rem 0.1rem !important; margin-right: 1.1rem !important; }

/* ── Knowledge Assistant ────────────────────────────────────────────────── */
.kb-status-line { font-size: 0.79rem; color: var(--t3); margin-bottom: 1rem; }
.kb-status-line strong { color: var(--t1); font-weight: 600; }
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1rem; }
.stat-card {
  flex: 1; min-width: 100px;
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 0.75rem 0.9rem; text-align: center;
  transition: transform 0.15s ease;
}
.stat-card:hover { transform: translateY(-1px); }
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 0.95rem;
            font-weight: 700; color: var(--t1); display: block; margin-bottom: 2px; }
.stat-lbl { font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--t3); }

[data-testid="stChatMessage"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important; margin-bottom: 0.55rem !important;
}
[data-testid="stChatMessage"] p { font-size: 0.84rem !important; line-height: 1.7 !important; color: var(--t2) !important; }
[data-testid="stChatInput"] textarea {
  background: var(--surface-2) !important; border: 1px solid var(--border-md) !important;
  border-radius: var(--r-md) !important; color: var(--t1) !important; font-size: 0.84rem !important;
}
.chat-source-row { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.5rem; }
.chat-source-chip {
  background: var(--ai-soft); border: 1px solid var(--ai-line);
  border-radius: 99px; padding: 0.12rem 0.6rem;
  font-size: 0.64rem; font-family: 'JetBrains Mono', monospace; color: var(--ai-2);
}
.quick-q-btn button {
  font-size: 0.73rem !important; padding: 0.42rem 0.7rem !important;
  white-space: normal !important; text-align: left !important; height: auto !important;
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  color: var(--t2) !important;
}
.quick-q-btn button:hover { color: var(--ai-2) !important; border-color: var(--ai-line) !important; }

/* ── Model Insights ─────────────────────────────────────────────────────── */
.metric-card {
  position: relative; overflow: hidden; text-align: center;
  transition: transform 0.18s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--brand), var(--brand-2));
}
.metric-icon { font-size: 1.1rem; margin-bottom: 0.3rem; opacity: 0.85; }
.metric-val {
  font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700;
  background: linear-gradient(90deg, var(--t1), var(--brand-2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.metric-lbl { font-size: 0.72rem; color: var(--t3); margin-top: 0.35rem; }
.cmp-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.cmp-table th {
  text-align: left; padding: 0.65rem 0.85rem; color: var(--t3);
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em;
  border-bottom: 1px solid var(--border-md);
}
.cmp-table td { padding: 0.7rem 0.85rem; border-bottom: 1px solid var(--border); color: var(--t2); }
.cmp-table tr.deployed td { color: var(--t1); }
.cmp-table tr.deployed { background: var(--brand-soft); }
.deployed-tag {
  font-size: 0.6rem; font-weight: 700; color: var(--brand-2);
  background: var(--brand-soft); border: 1px solid var(--brand-line);
  border-radius: 4px; padding: 0.08rem 0.4rem; margin-left: 0.5rem;
}

/* ── Modern fade separator ──────────────────────────────────────────────── */
.divider-fade {
  height: 1px; margin: 1.6rem 0;
  background: linear-gradient(90deg, transparent, var(--border-md) 30%, var(--border-md) 70%, transparent);
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# PRESET PROFILES
# AUDIT FIX (Issue 1): the MID preset combined ONE risk signal (month-to-
# month, tenure=10) with FOUR protective signals (OnlineSecurity=Yes,
# OnlineBackup=Yes, DSL — the lowest-risk internet type — and Mailed check
# instead of Electronic check). Net effect verified against the real fitted
# model: ~20-27% probability, landing in Low, not Medium. Rebuilt using a
# profile with genuinely medium-strength signals (Fiber optic + no
# OnlineSecurity, the EDA's strongest combined driver, offset by one
# protective trait so it doesn't read as "obviously high risk"), then
# empirically swept tenure/charges/backup combinations against the real
# model until landing centered in-band with comfortable margin from both
# the 40% and 80% boundaries (~62%, +22pts/-18pts margin), not just barely
# clearing 40%.
# HIGH preset also hardened: the original (82.0%) had only a 2-point
# margin above the 80% boundary — too fragile for a live demo against any
# model-to-model variance. Adding active Multiple Lines + Streaming TV/
# Movies (still zero protective services) pushed it to ~94% (+14pt margin)
# without changing any of the core risk attributes that made it "high risk"
# in the first place.
# LOW preset (4.8%, ~35pt margin below 40%) was already robust — unchanged.
# ─────────────────────────────────────────────────────────────────────────

HIGH_RISK_PRESET = dict(
    gender="Male", senior="Yes", tenure=1, partner="No", dependents="No",
    phone="Yes", lines="Yes", internet="Fiber optic", security="No",
    backup="No", device="No", techsup="No", tvstream="Yes", movies="Yes",
    contract="Month-to-month", paperless="Yes", payment="Electronic check",
    monthly=95.0,
)
MID_RISK_PRESET = dict(
    gender="Female", senior="No", tenure=20, partner="Yes", dependents="No",
    phone="Yes", lines="Yes", internet="Fiber optic", security="No",
    backup="Yes", device="No", techsup="No", tvstream="Yes", movies="No",
    contract="Month-to-month", paperless="Yes", payment="Electronic check",
    monthly=78.0,
)
LOW_RISK_PRESET = dict(
    gender="Female", senior="No", tenure=60, partner="Yes", dependents="Yes",
    phone="Yes", lines="Yes", internet="Fiber optic", security="Yes",
    backup="Yes", device="Yes", techsup="Yes", tvstream="Yes", movies="Yes",
    contract="Two year", paperless="No", payment="Bank transfer (automatic)",
    monthly=110.0,
)

# ─────────────────────────────────────────────────────────────────────────
# CACHE FUNCTIONS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_bundle() -> dict:
    path = ROOT / "models" / "churn_model.pkl"
    if not path.exists():
        st.error(f"Model file not found: {path}. Run `python src/train.py` first.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_ai_engine():
    """Load the LLM AI engine. Returns None if Ollama isn't running/pulled."""
    if not _AI_ENGINE_IMPORTABLE:
        return None
    try:
        return _ChurnAIEngine()
    except Exception:
        return None


@st.cache_resource
def load_rag_engine():
    """Load the RAG engine. Returns None if unconfigured."""
    if not _RAG_ENGINE_IMPORTABLE:
        return None
    try:
        return _ChurnRAGEngine()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# AUDIT FIX (comment only, no logic change): this comment previously said
# thresholds were "0.60 / 0.30", which was stale/incorrect — the code below
# has always branched at 0.80 / 0.40. Verified against the requested
# business spec (Low 0–39.9% / Medium 40–79.9% / High 80–100%): the
# existing >= 0.80 / >= 0.40 branches already implement that spec exactly,
# so no behavioral change was needed here — only this misleading comment.
# classify_risk's color values: "Medium" maps to gold (#eab308) instead of
# orange, since orange is reserved exclusively for brand identity in the
# richer 5-color system (Orange=brand, Indigo=AI, Green/Gold/Red=risk tiers).
# ─────────────────────────────────────────────────────────────────────────

def classify_risk(prob: float) -> dict:
    if prob >= 0.80:
        return dict(level="High",   label="High Risk",   color="#ef4444",
                    cell_cls="risk-high",   rb_cls="rb-high")
    if prob >= 0.40:
        return dict(level="Medium", label="Medium Risk", color="#eab308",
                    cell_cls="risk-medium", rb_cls="rb-medium")
    return     dict(level="Low",    label="Low Risk",    color="#22c55e",
                    cell_cls="risk-low",    rb_cls="rb-low")


def make_gauge(prob: float, risk: dict) -> go.Figure:
    # AUDIT FIX (Fix #1 — Gauge Threshold Consistency): the gauge's color
    # steps were 0-30/30-60/60-100, while classify_risk() actually branches
    # at 0.40/0.80 (Low 0-39.9% / Medium 40-79.9% / High 80-100% — verified
    # directly from classify_risk()'s source, not assumed). That mismatch
    # is real: a 45% prediction was "Medium Risk" by classify_risk() but
    # sat inside the gauge's 30-60 YELLOW zone only by coincidence, while a
    # 35% prediction (Low Risk) would incorrectly render in that same
    # yellow zone. Steps below now exactly match classify_risk()'s real
    # thresholds, so the gauge's color can never disagree with the badge.
    pct = prob * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 26, "color": risk["color"],
                                         "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "rgba(255,255,255,0.12)", "tickfont": {"size": 8}},
            "bar":  {"color": risk["color"], "thickness": 0.22},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40],  "color": "rgba(34,197,94,0.07)"},   # matches Low  (<40%)
                {"range": [40, 80],  "color": "rgba(234,179,8,0.07)"},  # matches Medium (40-79.9%)
                {"range": [80, 100], "color": "rgba(239,68,68,0.07)"},   # matches High (>=80%)
            ],
        },
    ))
    fig.update_layout(
        height=150, margin=dict(l=8, r=8, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "rgba(255,255,255,0.5)"},
    )
    return fig


FEATURE_LABELS: dict[str, str] = {
    "tenure":                    "Customer Tenure",
    "MonthlyCharges":            "Monthly Charges",
    "TotalCharges":               "Total Charges",
    "tenure_monthly_interaction": "Tenure × Monthly",
    "avg_monthly_charges":        "Avg Monthly Spend",
    "charges_per_service":        "Spend per Service",
    "total_services":             "Total Services",
    "Contract":                   "Contract Type",
    "InternetService":            "Internet Type",
    "PaymentMethod":               "Payment Method",
    "OnlineSecurity":              "Online Security",
    "TechSupport":                 "Tech Support",
    "OnlineBackup":                "Online Backup",
    "PaperlessBilling":            "Paperless Billing",
    "DeviceProtection":            "Device Protection",
    "StreamingTV":                 "Streaming TV",
    "StreamingMovies":             "Streaming Movies",
    "MultipleLines":               "Multiple Lines",
    "tenure_group":                "Tenure Group",
    "is_long_term":                "Long-term Contract",
    "has_any_security":            "Has Security/Support",
    "has_any_backup":              "Has Backup/Protection",
    "paperless_electronic":        "Paperless + E-Check",
    "SeniorCitizen":               "Senior Citizen",
    "Partner":                     "Has Partner",
    "Dependents":                  "Has Dependents",
    "PhoneService":                "Phone Service",
    "gender":                      "Gender",
}


def humanize_factor_label(raw_feature: str, raw: dict) -> str:
    """
    AUDIT FIX (Issue 2 — SHAP Feature Naming Problem):
    FEATURE_LABELS above is a STATIC name-only lookup. "is_long_term" always
    rendered as "Long-term Contract" regardless of whether this customer's
    actual value was 0 or 1. For a month-to-month customer (is_long_term=0)
    where that absence raises risk, the old UI showed "Long-term Contract
    ▲ raises risk" — which reads as "having a long-term contract raises
    risk", the OPPOSITE of what's true, directly contradicting the AI
    explanation's (correct) statement that the customer is month-to-month.

    This function replaces that static lookup wherever the underlying value
    is binary/state-dependent: it always names the customer's ACTUAL state,
    so the label can never read as the opposite of reality. Purely
    numeric/neutral features (MonthlyCharges, tenure, Contract, etc., where
    "X ▲ raises risk" is unambiguous because there's no implied opposite
    state) keep a simple value-annotated label.
    """
    yes = lambda k: raw.get(k) == "Yes"

    if raw_feature == "Contract":
        return f"Contract Type: {raw.get('Contract', 'Unknown')}"
    if raw_feature == "is_long_term":
        return ("Long-term Contract" if raw.get("Contract") in ("One year", "Two year")
                else "Month-to-month Contract")
    if raw_feature in ("tenure", "tenure_group"):
        months = raw.get("tenure", 0)
        unit = "month" if months == 1 else "months"
        return f"Tenure ({months} {unit})"
    if raw_feature == "InternetService":
        return f"Internet Service: {raw.get('InternetService', 'Unknown')}"
    if raw_feature == "OnlineSecurity":
        return "Has Online Security" if yes("OnlineSecurity") else "No Online Security"
    if raw_feature == "TechSupport":
        return "Has Tech Support" if yes("TechSupport") else "No Tech Support"
    if raw_feature == "OnlineBackup":
        return "Has Online Backup" if yes("OnlineBackup") else "No Online Backup"
    if raw_feature == "DeviceProtection":
        return "Has Device Protection" if yes("DeviceProtection") else "No Device Protection"
    if raw_feature == "has_any_security":
        return ("Has Security/Support Services" if (yes("OnlineSecurity") or yes("TechSupport"))
                else "No Security/Support Services")
    if raw_feature == "has_any_backup":
        return ("Has Backup/Protection Services" if (yes("OnlineBackup") or yes("DeviceProtection"))
                else "No Backup/Protection Services")
    if raw_feature == "PaymentMethod":
        return f"Payment Method: {raw.get('PaymentMethod', 'Unknown')}"
    if raw_feature == "PaperlessBilling":
        return "Paperless Billing" if yes("PaperlessBilling") else "Mailed/Paper Billing"
    if raw_feature == "paperless_electronic":
        is_pe = yes("PaperlessBilling") and raw.get("PaymentMethod") == "Electronic check"
        return ("Paperless + Electronic Check" if is_pe
                else "Different Billing Setup (not Paperless+E-Check)")
    if raw_feature == "MonthlyCharges":
        return f"Monthly Charges (${raw.get('MonthlyCharges', 0):.0f})"
    if raw_feature == "TotalCharges":
        return f"Total Charges (${raw.get('TotalCharges', 0):.0f})"
    if raw_feature == "SeniorCitizen":
        return "Senior Citizen" if raw.get("SeniorCitizen") == 1 else "Not a Senior Citizen"
    if raw_feature == "Partner":
        return "Has a Partner" if yes("Partner") else "No Partner"
    if raw_feature == "Dependents":
        return "Has Dependents" if yes("Dependents") else "No Dependents"
    if raw_feature == "PhoneService":
        return "Has Phone Service" if yes("PhoneService") else "No Phone Service"
    if raw_feature == "MultipleLines":
        return "Has Multiple Lines" if yes("MultipleLines") else "No Multiple Lines"
    if raw_feature == "StreamingTV":
        return "Streams TV" if yes("StreamingTV") else "No Streaming TV"
    if raw_feature == "StreamingMovies":
        return "Streams Movies" if yes("StreamingMovies") else "No Streaming Movies"
    if raw_feature == "gender":
        return f"Gender: {raw.get('gender', 'Unknown')}"

    # Numeric/engineered features with no binary "opposite state" to confuse —
    # the static label is unambiguous for these.
    return FEATURE_LABELS.get(raw_feature, raw_feature)


@st.cache_resource
def load_shap_explainer(_model) -> "shap.TreeExplainer":
    """
    Built once per deployed model (cached — TreeExplainer construction is
    cheap but no reason to redo it on every prediction). The leading
    underscore on `_model` tells Streamlit's cache_resource to skip trying
    to hash the estimator.

    TreeExplainer computes EXACT Shapley values for tree ensembles
    (Random Forest / XGBoost / Gradient Boosting — every model
    select_production_model() can deploy) via the Tree SHAP algorithm.
    This is what makes Key Factors customer-specific instead of global.
    """
    if not _SHAP_AVAILABLE:
        st.error(
            "The `shap` package is required for customer-specific Key "
            "Factors. Install it with `pip install shap` (already added "
            "to requirements.txt) and restart the app."
        )
        st.stop()
    return shap.TreeExplainer(_model)


# AUDIT FIX (Fix #2): maps every raw/engineered column to the business
# concept it represents. Columns sharing a group key are treated as "the
# same driver" for Key Factors purposes — only the highest-|SHAP| one in a
# group is ever shown. Columns not listed here are their own group (i.e.
# unique already). Derived directly from preprocessing.engineer_features()
# (which documents exactly which raw columns feed which engineered ones).
_SEMANTIC_GROUPS: dict[str, str] = {
    "Contract":                  "contract",
    "is_long_term":               "contract",
    "tenure":                     "tenure",
    "tenure_group":                "tenure",
    "tenure_monthly_interaction":  "tenure",
    "PaymentMethod":               "billing",
    "PaperlessBilling":            "billing",
    "paperless_electronic":        "billing",
    "OnlineSecurity":              "security",
    "TechSupport":                 "security",
    "has_any_security":            "security",
    "OnlineBackup":                "backup",
    "DeviceProtection":            "backup",
    "has_any_backup":              "backup",
    "MonthlyCharges":              "charges",
    "TotalCharges":                "charges",
    "avg_monthly_charges":         "charges",
    "charges_per_service":         "charges",
    "StreamingTV":                 "entertainment",
    "StreamingMovies":             "entertainment",
    "PhoneService":                "phone",
    "MultipleLines":               "phone",
}


def compute_shap_factors(
    explainer: "shap.TreeExplainer",
    X,
    feature_names: list,
    raw: dict,
    n: int = 3,
) -> list[dict]:
    """
    Customer-specific (local) explanation of THIS prediction, replacing the
    old global-importance heuristic.

    Mechanism: Tree SHAP decomposes this customer's predicted log-odds into
    a sum of per-feature contributions relative to the model's average
    prediction (additivity holds exactly: base_value + sum(shap) == this
    customer's raw margin). Each feature's contribution:
      • is specific to this customer's own feature values (not fixed like
        feature_importances_, which is identical for every customer)
      • carries a SIGN — positive pushes THIS customer's prediction toward
        churn, negative pushes it away — something the old heuristic threw
        away entirely via np.abs()
      • reflects the actual decision path the trees took for this row, not
        a dataset-wide average

    `raw` (the customer's raw profile dict) is required so the displayed
    "name" can use humanize_factor_label() — a value-aware label that
    states this customer's actual state, instead of a static name that can
    contradict the SHAP direction shown next to it (Issue 2 fix).

    AUDIT FIX (Fix #2 — SHAP Duplicate Business Meanings): several raw
    columns encode the SAME underlying business concept (Contract /
    is_long_term; tenure / tenure_group / tenure_monthly_interaction;
    PaymentMethod / PaperlessBilling / paperless_electronic; etc — see
    _SEMANTIC_GROUPS below). Because they're correlated, it's possible for
    two of them to land in the top-n by |SHAP value| for the same customer,
    which reads to a user as a duplicate ("Month-to-month Contract" AND
    "Long-term Contract" both shown, or the same tenure stated twice).
    Selection now walks the FULL feature list sorted by |SHAP value|
    (not just the top n) and keeps at most one feature per semantic group —
    the highest-impact one — continuing down the list until `n` distinct
    business concepts are collected. Output is still "top n", just
    guaranteed to be n *unique* concepts rather than n raw columns.

    Returns the same dict shape the UI/AI-engine already expect
    (name/imp_pct/bar_pct), plus "feature" (raw column name, used by
    ai_engine.py to scope what customer data it's allowed to mention) and
    "direction".
    """
    shap_vals = explainer.shap_values(X)
    # Some shap versions return a list [class0_array, class1_array] for
    # binary classifiers; normalise to the positive-class (churn) array.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    row = np.asarray(shap_vals)[0]

    total_abs = np.abs(row).sum() + 1e-9
    order = np.argsort(np.abs(row))[::-1]  # ALL features, descending |SHAP|

    results = []
    used_groups: set[str] = set()
    for i in order:
        raw_name = feature_names[i]
        group = _SEMANTIC_GROUPS.get(raw_name, raw_name)  # ungrouped features are their own group
        if group in used_groups:
            continue
        used_groups.add(group)
        results.append({
            "feature":   raw_name,
            "name":      humanize_factor_label(raw_name, raw),
            "shap":      round(float(row[i]), 4),
            "direction": "increases" if row[i] > 0 else "decreases",
            "imp_pct":   round(abs(row[i]) / total_abs * 100, 1),
        })
        if len(results) >= n:
            break

    # bar_pct is relative to the top SELECTED factor (results[0] is always
    # the global #1 by |SHAP value|, since used_groups starts empty — the
    # very first feature examined can never be skipped).
    top_abs = abs(results[0]["shap"]) + 1e-9 if results else 1e-9
    for r in results:
        r["bar_pct"] = round(min(abs(r["shap"]) / top_abs * 100, 100), 1)
    return results


_PRIORITY_ORDER = {"URGENT": 0, "HIGH": 1, "STANDARD": 2}
_PRIORITY_LINE_RE = re.compile(r"^\[(URGENT|HIGH|STANDARD)\]", re.IGNORECASE)


def sort_retention_actions(text: str) -> str:
    """
    Force the AI-generated retention strategy into URGENT -> HIGH ->
    STANDARD order.

    ai_engine.py's strategy prompt *asks* the LLM to emit lines in that
    order, but nothing previously enforced it — LLM output order is not
    guaranteed even when instructed, so app.py must guarantee it
    deterministically after generation completes. Lines that don't match
    the "[PRIORITY] ..." format (stray preamble, blank lines) are kept,
    appended after the sorted action lines, in their original order —
    nothing is silently dropped.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    tagged, other = [], []
    for i, ln in enumerate(lines):
        m = _PRIORITY_LINE_RE.match(ln.strip())
        if m:
            tagged.append((_PRIORITY_ORDER[m.group(1).upper()], i, ln))
        else:
            other.append(ln)
    tagged.sort(key=lambda t: (t[0], t[1]))  # stable: priority, then original order
    ordered = [ln for _, _, ln in tagged] + other
    return "\n".join(ordered)


# AUDIT NOTE: a rule-based build_explanation(raw, risk, factors) function
# previously lived here. It was dead code — grepped for call sites across
# the whole file and found none; _render_ai_section() only ever calls
# ai_engine.stream_explanation(). It has been removed because it hardcoded
# the exact failure pattern this audit fixes (e.g. it always checked
# `internet == "Fiber optic" and security == "No"` and `PaymentMethod ==
# "Electronic check"` regardless of whether those were actually this
# customer's top factors) — leaving it in place as unused, contradictory
# logic was a maintenance hazard.


def get_recommendations(raw: dict, risk_level: str, facs: list[dict] | None = None) -> list[dict]:
    """
    AUDIT FIX (Fix #6 — Recommendation Consistency): previously built only
    from raw profile attributes, with no link to the actual SHAP factors
    shown as this customer's Key Factors — so a recommendation could
    reference an attribute the AI explanation never mentions, reading as
    unrelated or contradictory next to it.

    `facs` (optional — defaults to None, preserving old behavior when not
    supplied) is used to PRIORITIZE recommendations whose underlying raw
    field is one of the actual top factors for THIS customer. Nothing is
    removed — a few recommendations (loyalty enrollment, proactive
    check-in) are intentionally generic and still useful regardless of the
    specific driver — they're just sorted after the factor-linked ones
    instead of being interleaved arbitrarily.
    """
    recs = []

    contract = raw.get("Contract", "")
    internet = raw.get("InternetService", "No")
    security = raw.get("OnlineSecurity", "No")
    tenure   = raw.get("tenure", 0)
    payment  = raw.get("PaymentMethod", "")
    monthly  = raw.get("MonthlyCharges", 0)

    # each candidate is (raw_field_or_None, rec_dict) — None marks a
    # generic recommendation not tied to one specific attribute.
    candidates: list[tuple[str | None, dict]] = []

    if risk_level == "High":
        if contract == "Month-to-month":
            candidates.append(("Contract", {"tag": "Contract",  "text": "Offer annual contract at 15% discount — priority outreach within 24h"}))
        if security == "No" and internet != "No":
            candidates.append(("OnlineSecurity", {"tag": "Upsell",    "text": "Present 60-day free Online Security trial"}))
        if tenure <= 12:
            candidates.append(("tenure", {"tag": "Onboard",   "text": "Assign dedicated success manager for first-year onboarding"}))
        if payment == "Electronic check":
            candidates.append(("PaymentMethod", {"tag": "Payment",   "text": "Incentivise switch to auto-pay: $5/month credit"}))
        if monthly > 80:
            candidates.append(("MonthlyCharges", {"tag": "Review",    "text": "Schedule plan review — possible downgrade or bundle optimisation"}))

    elif risk_level == "Medium":
        candidates.append((None, {"tag": "Engage",    "text": "Schedule proactive check-in within 7 days"}))
        if security == "No":
            candidates.append(("OnlineSecurity", {"tag": "Bundle",    "text": "Propose security + backup bundle at promotional rate"}))
        if contract == "Month-to-month":
            candidates.append(("Contract", {"tag": "Contract",  "text": "Offer one-year contract lock-in with loyalty pricing"}))
        candidates.append((None, {"tag": "Loyalty",   "text": "Enrol in loyalty programme — milestone reward at month 12"}))

    else:
        candidates.append((None, {"tag": "Upsell",    "text": "Ideal candidate for premium upgrade"}))
        candidates.append((None, {"tag": "Referral",  "text": "Invite to referral programme — $25 credit per referral"}))
        candidates.append((None, {"tag": "Renew",     "text": "Pre-emptive two-year renewal with free device upgrade"}))

    if not facs or not _AI_ENGINE_IMPORTABLE:
        return [rec for _, rec in candidates][:5]

    relevant_fields: set[str] = set()
    for f in facs:
        relevant_fields.add(f.get("feature", ""))
        relevant_fields.update(_ChurnAIEngine._FACTOR_PROFILE_FIELDS.get(f.get("feature", ""), []))

    def _is_factor_linked(field: str | None) -> bool:
        return field is not None and field in relevant_fields

    # stable sort: factor-linked candidates first, each bucket keeping its
    # original priority order — nothing dropped, just reprioritized.
    ordered = sorted(candidates, key=lambda c: 0 if _is_factor_linked(c[0]) else 1)
    return [rec for _, rec in ordered][:5]


# ─────────────────────────────────────────────────────────────────────────
# PREDICTION RESULT — the centerpiece. Same merged content (Gauge + Score
# + Top Factors) as before; visual treatment strengthened with a
# risk-colored top accent, glow shadow, and larger type.
# ─────────────────────────────────────────────────────────────────────────

def _render_prediction_result(prob: float, risk: dict, facs: list) -> None:
    pct = prob * 100
    st.markdown(f'<div class="result-card {risk["cell_cls"]}">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">⚡ Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="risk-badge {risk["rb_cls"]}">'
        f'<span class="rb-dot"></span>{risk["label"]}</span>',
        unsafe_allow_html=True,
    )

    col_gauge, col_factors = st.columns([2, 3], gap="large")
    with col_gauge:
        st.markdown(
            f'<div class="prob-number" style="color:{risk["color"]}">{pct:.1f}%</div>'
            f'<div class="prob-label">Predicted churn probability</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_gauge(prob, risk), width="stretch",
            config={"displayModeBar": False, "staticPlot": True},
            key="gauge_chart",
        )
    with col_factors:
        st.markdown('<div class="sec" style="margin-top:0.3rem">Key Factors</div>', unsafe_allow_html=True)
        for fac in facs:
            up = fac.get("direction") == "increases"
            dir_color = "#ef4444" if up else "#22c55e"
            dir_label = "▲ raises risk" if up else "▼ lowers risk"
            st.markdown(f"""
            <div class="factor-row">
              <div class="factor-hdr">
                <span class="factor-name">{fac['name']}</span>
                <span class="factor-pct">{fac['imp_pct']:.1f}%</span>
              </div>
              <div class="factor-track">
                <div class="factor-fill" style="width:{fac['bar_pct']}%; background:{dir_color}"></div>
              </div>
              <div class="factor-dir" style="color:{dir_color}; font-size:0.78rem; margin-top:0.15rem">{dir_label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close result-card


# ─────────────────────────────────────────────────────────────────────────
# AI SECTION — flagship cards. Explanation (2nd priority) + Retention
# Strategy (3rd priority). Always rendered in full, never behind an
# expander. Visually distinct indigo/violet treatment marks these as the
# product's signature AI features, set apart from ordinary content cards.
# ─────────────────────────────────────────────────────────────────────────

def _render_debug_diagnostic(label: str, meta: dict) -> None:
    """
    Fix #7 — Demo Reliability Logging. Shown only when debug_mode is on.
    Surfaces exactly what generate_explanation_safe() / generate_
    retention_strategy_safe() already recorded in `meta` — no separate
    tracking needed, this just exposes it in the UI for whoever is
    watching during the defense.
    """
    if meta.get("fallback_used"):
        st.warning(f"**{label}: Fallback Used**")
        if meta.get("issues"):
            st.caption("Reason(s) for fallback:")
            for reason in meta["issues"]:
                st.caption(f"• {reason}")
    else:
        note = " (regenerated once)" if meta.get("regenerated") else ""
        note = note or (" (repaired)" if meta.get("repaired") else "")
        st.success(f"**{label}: LLM Output Accepted**{note}")


def _render_ai_section(ai_engine, raw: dict, prob: float, risk: dict, facs: list, debug_mode: bool = False) -> None:
    if ai_engine is None:
        model = os.getenv("CHURN_LLM_MODEL", "llama3.2:3b")
        st.markdown(
            f'<div class="notice">AI insights unavailable — start Ollama '
            f'(<code>ollama serve</code>) and pull the model '
            f'(<code>ollama pull {model}</code>)</div>',
            unsafe_allow_html=True,
        )
        return

    # ── #2: AI Churn Explanation ──────────────────────────────────────────
    # AUDIT FIX (Issue 8 — Demo Safety Mode): token-by-token streaming
    # directly to the UI necessarily means the audience sees raw,
    # unvalidated LLM output while it's still arriving — there's no way to
    # validate text that doesn't exist yet. For the defense build this is
    # replaced with a blocking call to generate_explanation_safe(), which
    # internally runs the full pipeline (generate -> validate -> one
    # regeneration attempt -> deterministic fallback) and returns only a
    # final string that has already passed validate_explanation(). A
    # spinner covers the (sub-second to few-second) generation time instead
    # of a token-by-token reveal.
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ai-card-title">🧠 AI Churn Explanation '
        '<span class="ai-badge">AI Generated</span></div>',
        unsafe_allow_html=True,
    )
    exp_container = st.empty()
    try:
        with st.spinner("Generating explanation..."):
            exp_text, exp_meta = ai_engine.generate_explanation_safe(raw, prob, risk["level"], facs)
        exp_container.markdown(f'<div class="ai-body">{exp_text}</div>', unsafe_allow_html=True)
        if exp_meta.get("fallback_used"):
            st.caption("ℹ️ Generated using the deterministic safety template (the AI draft "
                       "didn't pass validation).")
        if debug_mode:
            _render_debug_diagnostic("Explanation", exp_meta)
    except Exception as exc:
        exp_text = ""
        exp_container.warning(f"AI explanation error: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── #3: AI Retention Strategy ─────────────────────────────────────────
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ai-card-title">🎯 AI Retention Strategy '
        '<span class="ai-badge">AI Generated</span></div>',
        unsafe_allow_html=True,
    )
    strat_container = st.empty()
    context = exp_text if exp_text else "Churn explanation not available."
    try:
        with st.spinner("Generating retention strategy..."):
            strat_text, strat_meta = ai_engine.generate_retention_strategy_safe(
                raw, prob, risk["level"], context, facs
            )
        # sort_retention_actions() is now belt-and-suspenders: repair_strategy()
        # inside generate_retention_strategy_safe() already sorts URGENT->HIGH->
        # STANDARD, but re-applying a pure, idempotent sort here costs nothing
        # and guarantees the UI never regresses if that internal ordering ever
        # changes.
        strat_text = sort_retention_actions(strat_text)
        strat_container.markdown(f'<div class="ai-body">{strat_text}</div>', unsafe_allow_html=True)
        if strat_meta.get("fallback_used"):
            st.caption("ℹ️ Generated using the deterministic safety template (the AI draft "
                       "didn't pass validation).")
        if debug_mode:
            _render_debug_diagnostic("Strategy", strat_meta)
    except Exception as exc:
        strat_container.warning(f"AI strategy error: {exc}")

    # Rule-based recommendations — quiet supplementary chips, not a 4th
    # priority block. Preserves the feature without competing visually
    # with the Result/Explanation/Strategy hierarchy.
    recs = get_recommendations(raw, risk["level"], facs)
    if recs:
        chips = "".join(
            f'<span class="chip"><strong>{r["tag"]}:</strong> {r["text"]}</span>'
            for r in recs
        )
        st.markdown(
            f'<div class="sec" style="margin-top:1rem">Also consider</div>'
            f'<div class="chip-row">{chips}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# KNOWLEDGE ASSISTANT TAB — chat-first, management tools in a popover
# ─────────────────────────────────────────────────────────────────────────

def _render_kb_management_popover(rag_engine) -> None:
    """All operational controls live here — upload, rebuild, index stats.
    Nothing in this function is new functionality; everything is moved
    from the old always-visible left column into a single popover."""
    stats = rag_engine.get_index_stats()
    status_label = "Ready" if stats["ready"] else "Empty"

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card"><span class="stat-val">{status_label}</span><span class="stat-lbl">Index Status</span></div>
      <div class="stat-card"><span class="stat-val">{stats['total_vectors']}</span><span class="stat-lbl">Vectors</span></div>
      <div class="stat-card"><span class="stat-val">{len(st.session_state.rag_history) // 2}</span><span class="stat-lbl">Queries</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec">📤 Upload Documents</div>', unsafe_allow_html=True)
    st.caption("CSV, PDF, TXT, MD — adds to the existing index")
    uploaded_files = st.file_uploader(
        "Drop files", type=["csv", "pdf", "txt", "md"],
        accept_multiple_files=True, label_visibility="collapsed", key="kb_uploader",
    )
    if uploaded_files:
        if st.button("Ingest documents", width="stretch", key="ingest_btn"):
            total_chunks, errors = 0, []
            prog = st.progress(0.0, text="Ingesting …")
            for i, uf in enumerate(uploaded_files):
                try:
                    total_chunks += rag_engine.add_documents_from_upload(uf.read(), uf.name)
                except Exception as exc:
                    errors.append(f"{uf.name}: {exc}")
                prog.progress((i + 1) / len(uploaded_files), text=f"Processing {uf.name} …")
            prog.empty()
            for err in errors:
                st.error(err)
            if total_chunks:
                st.success(f"{total_chunks} chunks indexed from {len(uploaded_files) - len(errors)} file(s)")
                st.rerun()

    st.markdown('<div class="sec" style="margin-top:1rem">🔄 Rebuild Index</div>', unsafe_allow_html=True)
    if st.button("Rebuild from data/knowledge_base/", width="stretch", key="rebuild_btn"):
        with st.spinner("Scanning and embedding documents …"):
            n = rag_engine.build_index_from_directory(str(ROOT / "data" / "knowledge_base"))
        if n:
            st.success(f"{n} chunks indexed")
            st.rerun()
        else:
            st.warning("No supported files found in data/knowledge_base/")

    if st.session_state.rag_history:
        st.markdown('<div class="sec" style="margin-top:1rem">💬 Conversation</div>', unsafe_allow_html=True)
        if st.button("Clear conversation", width="stretch", key="clear_chat_btn"):
            st.session_state.rag_history = []
            st.rerun()


def _render_knowledge_base_tab(rag_engine) -> None:
    if rag_engine is None:
        model = os.getenv("CHURN_LLM_MODEL",  "llama3.2:3b")
        embed = os.getenv("CHURN_EMBED_MODEL", "nomic-embed-text")
        st.markdown(f"""
        <div class="empty">
          <div class="empty-h">AI Engine Not Configured</div>
          <div class="empty-p">
            Start Ollama (<code>ollama serve</code>), then pull both models:<br>
            <code>ollama pull {model}</code> · <code>ollama pull {embed}</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    stats = rag_engine.get_index_stats()
    header_l, header_r = st.columns([5, 1])
    with header_l:
        if stats["ready"]:
            st.markdown(
                f'<div class="kb-status-line">📚 Knowledge base ready · '
                f'<strong>{stats["total_vectors"]}</strong> vectors indexed</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="kb-status-line">📚 Knowledge base is empty</div>', unsafe_allow_html=True)
    with header_r:
        with st.popover("⚙ Manage", width="stretch"):
            _render_kb_management_popover(rag_engine)

    QUICK_QUESTIONS = [
        "Explain customer 7590-VHVEG.",
        "What retention action is recommended for customer 7590-VHVEG?",
        "List the churn risk indicators for customer 7590-VHVEG.",
        "What internet service does customer 7590-VHVEG use?",
        "Explain customer 7590-VHVEG's payment method.",
        "Why is Online Security listed as a churn risk indicator?",
        "What payment method does customer 7590-VHVEG use?",
        "Explain the risk indicators for Fiber Optic customers.",
    ]
    q1, q2, q3, q4 = st.columns(4)
    quick_q_clicked = None
    for i, qq in enumerate(QUICK_QUESTIONS):
        col = [q1, q2, q3, q4][i % 4]
        with col:
            st.markdown('<div class="quick-q-btn">', unsafe_allow_html=True)
            if st.button(qq, key=f"qq_{i}", width="stretch"):
                quick_q_clicked = qq
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not rag_engine.is_ready():
        st.markdown("""
        <div class="empty">
          <div class="empty-h">No Knowledge Base Loaded</div>
          <div class="empty-p">
            Open <strong>⚙ Manage</strong> above to rebuild from
            <strong>data/knowledge_base/</strong> or upload your own files.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    chat_box = st.container(height=420)
    with chat_box:
        if not st.session_state.rag_history:
            st.markdown(
                '<div style="text-align:center;padding:3rem 1rem;color:var(--t3);font-size:0.78rem">'
                'Ask a question below or choose a quick topic above.</div>',
                unsafe_allow_html=True,
            )
        for msg in st.session_state.rag_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    chips = " ".join(
                        f'<span class="chat-source-chip">{Path(s).name}</span>'
                        for s in msg["sources"]
                    )
                    st.markdown(f'<div class="chat-source-row">{chips}</div>', unsafe_allow_html=True)

    user_question = st.chat_input(
        "Ask about a customer, a churn driver, or a retention recommendation...", key="rag_chat_input",
    )
    active_question = quick_q_clicked or user_question

    if active_question:
        st.session_state.rag_history.append({"role": "user", "content": active_question})
        with st.spinner("Searching knowledge base …"):
            result = rag_engine.query(active_question)
        st.session_state.rag_history.append({
            "role": "assistant", "content": result["answer"], "sources": result.get("sources", []),
        })
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# MODEL INSIGHTS TAB — deployed model, metrics, dataset stats, feature
# importance, full comparison table. Uses only data already in the model
# bundle — no retraining, no new computation.
# ─────────────────────────────────────────────────────────────────────────

def _render_model_insights_tab(
    deployed_name: str,
    deployed_auc: float,
    deployed_acc: float,
    results_d: dict,
    features: list,
    importances: np.ndarray,
) -> None:
    st.markdown('<div class="sec">🏆 Deployed Model</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, icon, val, lbl in [
        (m1, "🤖", deployed_name,        "Model"),
        (m2, "📈", f"{deployed_auc:.3f}", "ROC-AUC"),
        (m3, "🎯", f"{deployed_acc:.1%}", "Accuracy"),
        (m4, "🗂️", "7,043",               "Training Samples"),
    ]:
        with col:
            st.markdown(
                f'<div class="card metric-card"><div class="metric-icon">{icon}</div>'
                f'<div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sec" style="margin-top:1.7rem">📊 Dataset Statistics</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for col, icon, val, lbl in [
        (d1, "👥", "7,043",  "Total Customers"),
        (d2, "📉", "26.5%",  "Churn Rate"),
        (d3, "🧬", "28",     "Engineered Features"),
    ]:
        with col:
            st.markdown(
                f'<div class="card metric-card"><div class="metric-icon">{icon}</div>'
                f'<div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    col_imp, col_cmp = st.columns([1, 1], gap="large")

    with col_imp:
        st.markdown('<div class="sec" style="margin-top:1.7rem">📈 Feature Importance</div>', unsafe_allow_html=True)
        order = np.argsort(importances)[::-1][:10]
        names = [FEATURE_LABELS.get(features[i], features[i]) for i in order][::-1]
        vals  = [importances[i] for i in order][::-1]

        fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker=dict(color=vals, colorscale=[[0, "#fb923c"], [1, "#f97316"]]),
        ))
        fig.update_layout(
            height=340, margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "rgba(255,255,255,0.65)", "size": 11},
            xaxis={"gridcolor": "rgba(255,255,255,0.06)", "title": "Importance"},
            yaxis={"automargin": True},
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with col_cmp:
        st.markdown('<div class="sec" style="margin-top:1.7rem">🏅 Model Comparison</div>', unsafe_allow_html=True)
        rows = ""
        for name, m in sorted(results_d.items(), key=lambda kv: -kv[1].get("roc_auc", 0)):
            is_deployed = (name == deployed_name)
            tag = '<span class="deployed-tag">DEPLOYED</span>' if is_deployed else ""
            row_cls = "deployed" if is_deployed else ""
            rows += (
                f'<tr class="{row_cls}"><td>{name}{tag}</td>'
                f'<td class="mono">{m.get("roc_auc", 0):.3f}</td>'
                f'<td class="mono">{m.get("accuracy", 0):.1%}</td></tr>'
            )
        st.markdown(f"""
        <div class="card">
          <table class="cmp-table">
            <thead><tr><th>Model</th><th>ROC-AUC</th><th>Accuracy</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Resources ──────────────────────────────────────────────────────────
    bundle     = load_bundle()
    ai_engine  = load_ai_engine()
    rag_engine = load_rag_engine()

    encoders    = bundle["encoder"]
    scaler      = bundle["scaler"]
    features    = bundle["features"]
    model       = bundle["model"]
    importances = model.feature_importances_       # global — used ONLY by Model Insights tab
    shap_explainer = load_shap_explainer(model)     # local/per-customer — used by Key Factors
    results_d   = bundle.get("results", {})

    deployed_name    = bundle.get("best_model_name", "Unknown")
    deployed_metrics = results_d.get(deployed_name, {})
    deployed_auc     = deployed_metrics.get("roc_auc",  0.0)
    deployed_acc     = deployed_metrics.get("accuracy", 0.0)

    # ── Session state ──────────────────────────────────────────────────────
    if "preset" not in st.session_state:
        st.session_state.preset = None
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    # AUDIT FIX (Fix #7 — Demo Reliability Logging): off by default so the
    # graduation-defense run looks exactly like before. When switched on,
    # _render_ai_section() shows whether each AI card's text was the LLM's
    # own (validated) output or a deterministic fallback, and why — pulled
    # straight from the meta dict generate_explanation_safe() /
    # generate_retention_strategy_safe() already return.
    with st.sidebar:
        st.markdown("### 🔧 Diagnostics")
        debug_mode = st.checkbox("Debug mode", value=False,
                                 help="Show whether each AI card used the LLM's own output "
                                      "or a safety-layer fallback, and why.")

    def apply_preset(name: str) -> None:
        st.session_state.preset = name

    # ── Hero — premium first-open impression ───────────────────────────────
    ai_status  = ("on", "AI Engine")  if ai_engine else ("off", "AI Engine")
    rag_status = ("on", "Knowledge Base") if (rag_engine and rag_engine.is_ready()) else ("off", "Knowledge Base")

    st.markdown(f"""
    <div class="hero-wrap">
      <div class="hero-glow"></div>
      <div class="hero-top">
        <div>
          <div class="hero-brand-row">
            <div class="hero-icon">⚡</div>
            <div>
              <div class="hero-title">ChurnGuard AI</div>
              <div class="hero-subtitle">AI-Powered Customer Retention Platform</div>
            </div>
          </div>
          <div class="hero-statement">
            Predict churn risk, understand the "why" behind every score, and get
            an AI-generated retention plan - powered by a locally-hosted LLM and
            a RAG knowledge base built on real customer data.
          </div>
        </div>
        <div class="hero-status-col">
          <div class="status-pill"><span class="status-dot {ai_status[0]}"></span>{ai_status[1]}</div>
          <div class="status-pill"><span class="status-dot {rag_status[0]}"></span>{rag_status[1]}</div>
        </div>
      </div>
      <div class="hero-metrics">
        <div class="hero-metric"><div class="hero-metric-val">{deployed_auc:.3f}</div><div class="hero-metric-lbl">ROC-AUC</div></div>
        <div class="hero-metric"><div class="hero-metric-val">{deployed_acc:.1%}</div><div class="hero-metric-lbl">Accuracy</div></div>
        <div class="hero-metric"><div class="hero-metric-val">7,043</div><div class="hero-metric-lbl">Customers Analyzed</div></div>
        <div class="hero-metric"><div class="hero-metric-val">26.5%</div><div class="hero-metric-lbl">Churn Rate</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tab layout ─────────────────────────────────────────────────────────
    tab_pred, tab_kb, tab_insights = st.tabs(["⚡ Predict", "📚 Knowledge Assistant", "📊 Model Insights"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — PREDICT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_pred:
        left, right = st.columns([5, 4], gap="large")

        with left:
            st.markdown('<div class="input-panel">', unsafe_allow_html=True)
            st.caption("Try an example")
            s1, s2, s3 = st.columns(3)
            if s1.button("High Risk", width="stretch"):
                apply_preset("high"); st.rerun()
            if s2.button("Medium Risk", width="stretch"):
                apply_preset("mid");  st.rerun()
            if s3.button("Low Risk", width="stretch"):
                apply_preset("low");  st.rerun()

            preset = st.session_state.preset
            P = (HIGH_RISK_PRESET if preset == "high" else
                 MID_RISK_PRESET  if preset == "mid"  else
                 LOW_RISK_PRESET  if preset == "low"  else {})

            st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
            profile_tab, services_tab, billing_tab = st.tabs(["Profile", "Services", "Billing"])

            _yn   = ["No", "Yes"]
            _ynni = ["No", "Yes", "No internet service"]
            _ynnp = ["No", "Yes", "No phone service"]

            with profile_tab:
                c1, c2, c3 = st.columns(3)
                gender = c1.selectbox("Gender", ["Female", "Male"],
                                       index=["Female","Male"].index(P.get("gender","Female")))
                senior = c2.selectbox("Senior Citizen", _yn, index=_yn.index(P.get("senior","No")))
                tenure = c3.number_input("Tenure (months)", 0, 72, value=int(P.get("tenure", 12)))
                c4, c5 = st.columns(2)
                partner    = c4.selectbox("Partner",    _yn, index=_yn.index(P.get("partner","No")))
                dependents = c5.selectbox("Dependents", _yn, index=_yn.index(P.get("dependents","No")))

            with services_tab:
                c6, c7, c8 = st.columns(3)
                phone    = c6.selectbox("Phone Service",     _yn,   index=_yn.index(P.get("phone","Yes")))
                lines    = c6.selectbox("Multiple Lines",    _ynnp, index=_ynnp.index(P.get("lines","No")))
                internet = c7.selectbox("Internet Service",  ["DSL","Fiber optic","No"],
                                         index=["DSL","Fiber optic","No"].index(P.get("internet","DSL")))
                security = c7.selectbox("Online Security",   _ynni, index=_ynni.index(P.get("security","No")))
                backup   = c7.selectbox("Online Backup",     _ynni, index=_ynni.index(P.get("backup","No")))
                device   = c8.selectbox("Device Protection", _ynni, index=_ynni.index(P.get("device","No")))
                techsup  = c8.selectbox("Tech Support",      _ynni, index=_ynni.index(P.get("techsup","No")))
                tvstream = c8.selectbox("Streaming TV",      _ynni, index=_ynni.index(P.get("tvstream","No")))
                movies   = c8.selectbox("Streaming Movies",  _ynni, index=_ynni.index(P.get("movies","No")))

            with billing_tab:
                c9, c10 = st.columns(2)
                _contracts = ["Month-to-month","One year","Two year"]
                _payments  = ["Electronic check","Mailed check",
                              "Bank transfer (automatic)","Credit card (automatic)"]
                contract  = c9.selectbox("Contract Type", _contracts,
                                          index=_contracts.index(P.get("contract","Month-to-month")))
                paperless = c9.selectbox("Paperless Billing", _yn, index=_yn.index(P.get("paperless","Yes")))
                payment   = c10.selectbox("Payment Method", _payments,
                                           index=_payments.index(P.get("payment","Electronic check")))
                monthly   = c10.number_input("Monthly Charges ($)", 0.0, 200.0,
                                              value=float(P.get("monthly", 65.0)), step=1.0, format="%.2f")
                total = float(tenure) * float(monthly)
                st.markdown(f"""
                <div class="total-box">
                  <span class="total-lbl">Estimated Total Charges</span>
                  <span class="total-val">${total:,.2f}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("⚡ Run Prediction", width="stretch", type="primary")
            st.markdown("</div>", unsafe_allow_html=True)  # close input-panel

        with right:
            if predict_btn:
                if float(monthly) <= 0:
                    st.error("Monthly charges must be greater than $0.")
                    return

                raw = {
                    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
                    "Partner": partner, "Dependents": dependents, "tenure": int(tenure),
                    "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
                    "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": device,
                    "TechSupport": techsup, "StreamingTV": tvstream, "StreamingMovies": movies,
                    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
                    "MonthlyCharges": float(monthly), "TotalCharges": float(total),
                }

                X    = prepare_input(raw, encoders, scaler)
                prob = model.predict_proba(X)[0][1]
                risk = classify_risk(prob)
                facs = compute_shap_factors(shap_explainer, X, features, raw, n=3)

                # #1 — Prediction Result: the centerpiece
                _render_prediction_result(prob, risk, facs)
                st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
                # #2 + #3 — AI Explanation, AI Retention Strategy (flagship, always visible)
                _render_ai_section(ai_engine, raw, prob, risk, facs, debug_mode)

            else:
                st.markdown("""
                <div class="empty">
                  <div class="empty-h">Awaiting Customer Profile</div>
                  <div class="empty-p">
                    Fill in the form on the left - or choose an <strong>example</strong> -
                    then click <strong>Run Prediction</strong> to see the result,
                    AI explanation, and retention strategy.
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — KNOWLEDGE ASSISTANT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_kb:
        _render_knowledge_base_tab(rag_engine)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3 — MODEL INSIGHTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_insights:
        _render_model_insights_tab(
            deployed_name, deployed_auc, deployed_acc, results_d, features, importances,
        )


if __name__ == "__main__":
    main()
