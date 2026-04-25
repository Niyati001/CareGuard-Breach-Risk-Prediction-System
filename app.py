"""
CareGuard — ED Breach Risk Prediction Dashboard
================================================
Streamlit app that lets users input ED parameters and get
a real-time breach probability prediction from the trained XGBoost model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CareGuard — ED Breach Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>

/* Container */
.block-container {
    padding-top: 1.5rem !important;
    overflow: visible !important;  
}

.main-title {
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: var(--text-color) !important;
    margin: 0.5rem 0 0 0 !important;
    letter-spacing: -0.5px;
    line-height: 1.35;
}
            

/* Subtitle */
.subtitle {
    font-size: 0.95rem !important;
    color: #6c757d !important;
    margin-top: 0.3rem !important;
    margin-bottom: 1rem !important;
}

/* Risk box */
.risk-box {
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
            
.risk-low { background: rgba(40, 167, 69, 0.15); border: 2px solid #28a745;}

.risk-medium { background: rgba(255, 193, 7, 0.15); border: 2px solid #ffc107;}

.risk-high { background: rgba(220, 53, 69, 0.15); border: 2px solid #dc3545;}

.risk-number { font-size: 3.5rem; font-weight: 800; margin: 0; }
.risk-label  { font-size: 1.1rem; font-weight: 600; margin: 0; }

/* Insights */
.insight-card {
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
}
.insight-card span,
.insight-card b {
    color: var(--text-color) !important;
}

/* Metrics */
.metric-row {
    display: flex;
    gap: 1rem;
}

div[data-testid="stMetric"] {
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
}

.block-container {
    padding-top: 3.5rem !important;
    overflow: visible !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0.3rem !important;
}
            

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL + FEATURE CONFIG
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        model = joblib.load("careguard_model.pkl")
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def load_feature_config():
    with open("feature_config.json") as f:
        return json.load(f)

model = load_model()
config = load_feature_config()
FEATURES = config["features_realtime"]


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (mirrors eda.py exactly)
# ─────────────────────────────────────────────

def engineer_features(raw: dict) -> pd.DataFrame:
    """Apply the same feature engineering used during training."""
    d = raw.copy()
    d["doctor_to_arrival_ratio"] = d["num_doctors"] / d["arrival_rate"]
    d["bed_pressure"]            = d["arrival_rate"] / d["num_beds"]
    d["is_high_triage"]          = int(d["triage_level"] <= 2)
    d["arrived_to_congestion"]   = int(d["queue_length_on_arrival"] >= 3)
    d["complexity_score"]        = (6 - d["triage_level"]) + (d["needs_lab"] * 2)
    d["shift_pressure"]          = d["arrival_rate"] / (d["num_doctors"] * d["num_triage_nurses"])
    return pd.DataFrame([d])[FEATURES]


# ─────────────────────────────────────────────
# WHAT-IF SIMULATION (runs model across range)
# ─────────────────────────────────────────────

def whatif_doctors(base_raw: dict, model) -> tuple:
    """Vary number of doctors 1–8, return (x, breach_probs)."""
    docs = list(range(1, 9))
    probs = []
    for d in docs:
        row = base_raw.copy()
        row["num_doctors"] = d
        X = engineer_features(row)
        prob = model.predict_proba(X)[0][1]
        probs.append(round(prob * 100))
    return docs, probs


def whatif_arrival(base_raw: dict, model) -> tuple:
    """Vary arrival rate 5–30, return (x, breach_probs)."""
    rates = list(range(5, 31, 1))
    probs = []
    for r in rates:
        row = base_raw.copy()
        row["arrival_rate"] = r
        X = engineer_features(row)
        prob = model.predict_proba(X)[0][1]
        probs.append(round(prob * 100))
    return rates, probs


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom: 0.5rem;">
    <span style="font-size:2.8rem; font-weight:800; color:#1a1a2e; letter-spacing:-0.5px; line-height:1.1; display:block;">🏥 CareGuard</span>
    <span style="font-size:0.9rem; color:#6c757d; margin-top:0.3rem; display:block;">Emergency Department · Breach Risk Prediction</span>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("""
    **Model file not found.**
    Please train the model first by running the notebook `dataGen_Sim.ipynb`
    and saving it with:
    ```python
    import joblib
    joblib.dump(model, "careguard_model.pkl")
    ```
    Then restart this app.
    """)
    st.stop()

st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — INPUT PARAMETERS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔧 ED Configuration")
    st.caption("Set current shift parameters")

    st.markdown("**Staffing**")
    num_doctors       = st.slider("Doctors on shift",       min_value=1, max_value=10, value=4)
    num_triage_nurses = st.slider("Triage nurses",          min_value=1, max_value=6,  value=2)
    num_beds          = st.slider("ED beds available",      min_value=2, max_value=25, value=10)
    lab_capacity      = st.slider("Lab capacity (slots)",   min_value=1, max_value=8,  value=3)

    st.markdown("---")
    st.markdown("**Demand**")
    arrival_rate = st.slider("Arrival rate (patients/hr)", min_value=1.0, max_value=30.0,
                             value=12.0, step=0.5)

    st.markdown("---")
    st.markdown("### 🧑‍⚕️ Patient Profile")
    st.caption("Current patient being assessed")

    triage_level = st.select_slider(
        "Triage level",
        options=[1, 2, 3, 4, 5],
        value=3,
        format_func=lambda x: {
            1: "1 — Critical",
            2: "2 — Urgent",
            3: "3 — Standard",
            4: "4 — Minor",
            5: "5 — Non-urgent"
        }[x]
    )
    needs_lab         = st.checkbox("Lab test required?", value=False)
    arrival_hour      = st.slider("Arrival hour (0–24)", min_value=0.0, max_value=24.0,
                                  value=9.0, step=0.25)
    queue_on_arrival  = st.slider("Doctor queue on arrival", min_value=0, max_value=20, value=2)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Breach Risk", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

raw_input = {
    "num_doctors":             num_doctors,
    "num_triage_nurses":       num_triage_nurses,
    "num_beds":                num_beds,
    "arrival_rate":            arrival_rate,
    "lab_capacity":            lab_capacity,
    "triage_level":            triage_level,
    "needs_lab":               int(needs_lab),
    "arrival_hour":            arrival_hour,
    "queue_length_on_arrival": queue_on_arrival,
}

X = engineer_features(raw_input)
breach_prob   = model.predict_proba(X)[0][1]
breach_pct    = round(breach_prob * 100, 1)
breach_pct_display = round(breach_prob * 100)
no_breach_pct = round(100 - breach_pct, 1)

if breach_prob < 0.35:
    risk_level = "LOW"
    risk_class = "risk-low"
    risk_color = "#28a745"
    risk_emoji = "✅"
elif breach_prob < 0.65:
    risk_level = "MEDIUM"
    risk_class = "risk-medium"
    risk_color = "#ffc107"
    risk_emoji = "⚠️"
else:
    risk_level = "HIGH"
    risk_class = "risk-high"
    risk_color = "#dc3545"
    risk_emoji = "🚨"


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────

col_pred, col_insights = st.columns([1, 1.6], gap="large")

# ── LEFT: Prediction result ──────────────────
with col_pred:
    st.markdown("#### Breach Risk Prediction")

    st.markdown(f"""
    <div class="risk-box {risk_class}">
        <p class="risk-number" style="color:{risk_color}">{breach_pct_display}%</p>
        <p class="risk-label">{risk_emoji} {risk_level} RISK</p>
        <p style="font-size:0.85rem; color:#555; margin-top:0.5rem;">
            Probability this patient will wait > 4 hours
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Gauge bar
    fig_gauge, ax = plt.subplots(figsize=(5, 0.6))
    ax.barh(0, 100, color="#e9ecef", height=0.5)
    ax.barh(0, breach_pct, color=risk_color, height=0.5)
    ax.axvline(35, color="white", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(65, color="white", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 100)
    ax.axis("off")
    fig_gauge.patch.set_alpha(0)
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close()

    st.caption("Thresholds: Low < 35% · Medium 35–65% · High > 65%")

    st.markdown("---")

    # Derived feature summary
    st.markdown("#### Computed Indicators")
    dr = raw_input["num_doctors"] / raw_input["arrival_rate"]
    bp = raw_input["arrival_rate"] / raw_input["num_beds"]
    sp = raw_input["arrival_rate"] / (raw_input["num_doctors"] * raw_input["num_triage_nurses"])
    cs = (6 - raw_input["triage_level"]) + (raw_input["needs_lab"] * 2)

    m1, m2 = st.columns(2)
    m1.metric("Doctor/arrival ratio", f"{dr:.2f}",
              delta="adequate" if dr >= 0.3 else "understaffed",
              delta_color="normal" if dr >= 0.3 else "inverse")
    m2.metric("Bed pressure", f"{bp:.2f}",
              delta="manageable" if bp <= 1.5 else "high",
              delta_color="normal" if bp <= 1.5 else "inverse")
    m3, m4 = st.columns(2)
    m3.metric("Shift pressure", f"{sp:.2f}",
              delta="ok" if sp <= 4 else "stressed",
              delta_color="normal" if sp <= 4 else "inverse")
    m4.metric("Complexity score", f"{cs}",
              delta="simple" if cs <= 3 else "complex",
              delta_color="normal" if cs <= 3 else "inverse")


# ── RIGHT: Insights + What-If ─────────────────
with col_insights:
    st.markdown("#### Clinical Insights")

    insights = []

    if queue_on_arrival >= 3:
        insights.append(f"""🔴 <span style="font-weight:800; color:{'#fff' if st.get_option('theme.base')=='dark' else '#000'};"> High queue on arrival </span> — patient joining a congested queue significantly increases breach risk""")
    if triage_level == 2:
        insights.append(f"""🔴 <span style="font-weight:800; color:{'#fff' if st.get_option('theme.base')=='dark' else '#000'};"> Triage level 2 (Urgent) </span> — highest-risk triage group in this model; these patients often wait behind level-1 critical cases""")
    if needs_lab:
        insights.append(f"""🟡 <span style="font-weight:800; color:{'#fff' if st.get_option('theme.base')=='dark' else '#000'};"> Lab test required </span> — adds 20–90 min downstream; breach risk increases significantly""")
    if arrival_hour >= 18:
        insights.append(f"""🟡 <span style="font-weight:800; color:{'#fff' if st.get_option('theme.base')=='dark' else '#000'};"> Evening arrival </span> — ED congestion typically peaks in evening hours""")
    if raw_input["num_doctors"] / raw_input["arrival_rate"] < 0.25:
        insights.append(f"""🔴 <span style="font-weight:800; color:{'#fff' if st.get_option('theme.base')=='dark' else '#000'};"> Critically understaffed </span> — doctor-to-arrival ratio below 0.25; breach risk is elevated for all patients""")
    if triage_level >= 4 and queue_on_arrival == 0:
        insights.append(f"""🟢 <span style="font-weight:800; color:{'#fff' if st.get_option('theme.base')=='dark' else '#000'};"> Low-acuity patient with no queue </span> — favourable conditions for timely discharge""")
    if not insights:
        insights.append(f"""🟢 <span style="font-weight:800; color:{'#fff' if st.get_option('theme.base')=='dark' else '#000'};"> No major risk flags detected </span> — current configuration looks manageable""")

    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### What-If Analysis")
    tab1, tab2 = st.tabs(["📊 Vary doctor count", "📈 Vary arrival rate"])

    with tab1:
        docs, doc_probs = whatif_doctors(raw_input, model)
        fig1, ax1 = plt.subplots(figsize=(7.5, 4))

        theme = st.get_option("theme.base")
        if theme == "dark":
            bg_color = "#1e1e1e"
            text_color = "white"
        else:
            bg_color = "#ffffff"
            text_color = "#333333"

        fig1.patch.set_facecolor(bg_color)
        ax1.set_facecolor(bg_color)
        ax1.tick_params(colors=text_color)
        ax1.xaxis.label.set_color(text_color)
        ax1.yaxis.label.set_color(text_color)
        ax1.title.set_color(text_color)
        
        colors = ["#dc3545" if p >= 65 else "#ffc107" if p >= 35 else "#28a745"
                  for p in doc_probs]
        bars = ax1.bar(docs, doc_probs, color=colors, edgecolor="white", width=0.6, linewidth=1.2)
        ax1.axvline(num_doctors, color="#333333", linewidth=1.8, linestyle="--", alpha=0.85)
        ax1.axhline(35, color="#28a745", linewidth=1, linestyle=":", alpha=0.6)
        ax1.axhline(65, color="#dc3545", linewidth=1, linestyle=":", alpha=0.6)
        ax1.set_xlabel("Number of doctors on shift", fontsize=9, color="#444")
        ax1.set_ylabel("Breach probability (%)", fontsize=9, color="#444")
        ax1.set_title("How adding/removing doctors affects breach risk", fontsize=10, fontweight="bold", color="#1a1a2e")
        ax1.set_ylim(0, 110)
        ax1.set_xticks(docs)
        ax1.tick_params(axis='both', labelsize=8, colors='#555')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#dee2e6')
        ax1.spines['bottom'].set_color('#dee2e6')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
        for bar, prob in zip(bars, doc_probs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{round(prob)}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")

            ax1.text(num_doctors, 105, f"Current: {num_doctors} doctors", ha="center", fontsize=9, fontweight="bold", color="#333", bbox=dict(facecolor="white", edgecolor="#dee2e6", boxstyle="round,pad=0.3"))

        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close()

        # Recommendation
        safe_docs = [d for d, p in zip(docs, doc_probs) if p < 35]
        if safe_docs:
            st.success(f"➡ Adding doctors until **{min(safe_docs)}** on shift brings breach risk below 35%")
        else:
            st.warning("Doctor count alone may not bring risk below 35% — consider demand management too")

    with tab2:
        rates, rate_probs = whatif_arrival(raw_input, model)
        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
        fig2.patch.set_facecolor('#ffffff')
        ax2.set_facecolor('#f8f9fa')
        ax2.plot(rates, rate_probs, color="#007bff", linewidth=2.5, marker="o",
                 markersize=4, markerfacecolor="white", markeredgewidth=1.5, markeredgecolor="#007bff")
        ax2.axvline(arrival_rate, color="#333333", linewidth=1.8, linestyle="--",
                    label=f"Current: {arrival_rate}/hr", alpha=0.85)
        ax2.axhline(35, color="#28a745", linewidth=1, linestyle=":", alpha=0.6, label="Low threshold (35%)")
        ax2.axhline(65, color="#dc3545", linewidth=1, linestyle=":", alpha=0.6, label="High threshold (65%)")
        ax2.fill_between(rates, rate_probs, 35,
                         where=[p > 35 for p in rate_probs],
                         alpha=0.08, color="#dc3545")
        ax2.set_xlabel("Patient arrival rate (per hour)", fontsize=9, color="#444")
        ax2.set_ylabel("Breach probability (%)", fontsize=9, color="#444")
        ax2.set_title("How arrival rate affects breach risk", fontsize=10, fontweight="bold", color="#1a1a2e")
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis='both', labelsize=8, colors='#555')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#dee2e6')
        ax2.spines['bottom'].set_color('#dee2e6')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
        ax2.legend(fontsize=8, framealpha=0.9)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        safe_rates = [r for r, p in zip(rates, rate_probs) if p < 35]
        if safe_rates:
            st.success(f"➡ Breach risk stays below 35% if arrivals remain at or below **{max(safe_rates)}/hr**")
        else:
            st.warning("Breach risk stays high across all tested arrival rates — staffing increase needed")


# ─────────────────────────────────────────────
# BOTTOM — INPUT SUMMARY
# ─────────────────────────────────────────────

st.divider()
with st.expander("📋 Full input summary", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**ED Configuration**")
        st.dataframe(pd.DataFrame({
            "Parameter": ["Doctors", "Triage nurses", "Beds", "Arrival rate", "Lab capacity"],
            "Value": [num_doctors, num_triage_nurses, num_beds, f"{arrival_rate}/hr", lab_capacity]
        }), hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("**Patient Profile**")
        st.dataframe(pd.DataFrame({
            "Parameter": ["Triage level", "Needs lab", "Arrival hour", "Queue on arrival"],
            "Value": [triage_level, "Yes" if needs_lab else "No",
                      f"{arrival_hour:.2f}h", queue_on_arrival]
        }), hide_index=True, use_container_width=True)

st.caption("CareGuard · Emergency Department Decision Support · Niyati")
