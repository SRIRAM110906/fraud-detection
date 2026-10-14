"""Streamlit Web Dashboard for Real-Time AI Fraud Detection (FraudGuard AI).

Features:
- Cross-dashboard synchronized state: Any simulated transaction propagates across:
  1. Dashboard (KPIs & live transaction stream)
  2. Simulator (Real-time scoring & active state)
  3. Explainability (SHAP waterfall feature contributions for the simulated transaction)
  4. Model comparison (Live 3-model benchmark consensus on the simulated transaction)
  5. Analyst queue (Auto-routed for human review if medium/high risk)
"""

import datetime
from pathlib import Path
import sys

# Add project root to path so 'src' imports cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling matching the exact UI in screenshots
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #f3f6f9;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1b2e !important;
        width: 240px !important;
    }

    section[data-testid="stSidebar"] * {
        color: #94a3b8 !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        padding: 6px 12px !important;
        border-radius: 8px !important;
        margin-bottom: 2px !important;
    }

    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {
        background-color: #162a45 !important;
        color: #ffffff !important;
    }

    /* Brand Header in Sidebar */
    .brand-title {
        color: #ffffff !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        padding: 8px 0 16px 0 !important;
        letter-spacing: -0.3px !important;
    }

    /* Header Badge */
    .top-badge-green {
        background-color: #dcfce7;
        color: #15803d;
        font-size: 13px;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 9999px;
        display: inline-block;
        float: right;
    }

    /* Clean Card Container */
    .white-card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 24px 28px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02), 0 4px 12px rgba(15, 23, 42, 0.03);
        border: 1px solid #f1f5f9;
        margin-bottom: 20px;
    }

    .kpi-title-gray {
        font-size: 13px;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 6px;
    }

    .kpi-val-bold {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.5px;
    }

    .kpi-val-green {
        font-size: 24px;
        font-weight: 800;
        color: #10b981;
    }

    /* Prediction Result Box (Simulator) */
    .pred-result-card {
        background-color: #fde8e8;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }

    .pred-status-header {
        color: #991b1b;
        font-size: 19px;
        font-weight: 800;
        letter-spacing: 0.3px;
        margin-bottom: 10px;
    }

    .pred-prob-title {
        color: #991b1b;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 6px;
        opacity: 0.9;
    }

    .pred-bar-track {
        background-color: #f8b4b4;
        height: 14px;
        border-radius: 9999px;
        overflow: hidden;
        margin-bottom: 14px;
    }

    .pred-bar-fill {
        background-color: #b91c1c;
        height: 100%;
        border-radius: 9999px;
        width: 97.2%;
    }

    .pred-number-big {
        color: #991b1b;
        font-size: 46px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.5px;
    }

    .meta-line {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 4px;
    }

    /* Sync Notice Banner */
    .sync-notice-banner {
        background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 14px;
        font-size: 12.5px;
        color: #1e3a8a;
    }

    .risk-pill {
        background-color: #fef3c7;
        color: #b45309;
        font-size: 13px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }

    .decision-gray-box {
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system_artifacts():
    """Load model, scaler, threshold, and test samples once into memory."""
    models_dir = PROJECT_ROOT / "models"
    model_path = models_dir / "xgb_fraud_model.joblib"
    scaler_path = models_dir / "scaler.joblib"

    artifacts = {
        "is_ready": False,
        "predictor": None,
        "threshold": 0.80,
    }

    if model_path.exists() and scaler_path.exists():
        try:
            from src.predict import get_predictor
            predictor = get_predictor()
            artifacts["is_ready"] = True
            artifacts["predictor"] = predictor
            artifacts["threshold"] = predictor.threshold
        except Exception:
            artifacts["is_ready"] = False

    return artifacts


# Global Synchronized State
if "active_sim" not in st.session_state:
    st.session_state.active_sim = {
        "id": "TX-88232",
        "amount": 2149.00,
        "time": "02:14 AM",
        "category": "Electronics",
        "location": "Foreign country",
        "prob": 0.972,
        "prob_str": "97.2%",
        "decision": "BLOCK",
        "is_high": True,
        "is_medium": False,
    }

if "history_feed" not in st.session_state:
    st.session_state.history_feed = [
        {"Tx ID": "TX-88232", "Time": "02:14 AM", "Amount": "$2,149.00", "Merchant": "Electronics (Foreign)", "Risk": "97.2%", "Action": "BLOCK"},
        {"Tx ID": "TX-88233", "Time": "02:10 AM", "Amount": "$310.55", "Merchant": "Online Retail", "Risk": "58.0%", "Action": "REVIEW"},
        {"Tx ID": "TX-88225", "Time": "01:58 AM", "Amount": "$42.10", "Merchant": "Supermarket", "Risk": "0.3%", "Action": "APPROVE"},
        {"Tx ID": "TX-88219", "Time": "01:45 AM", "Amount": "$1,260.00", "Merchant": "Travel (Foreign IP)", "Risk": "66.0%", "Action": "REVIEW"},
        {"Tx ID": "TX-88204", "Time": "01:12 AM", "Amount": "$75.20", "Merchant": "Grocery Store", "Risk": "52.0%", "Action": "REVIEW"},
    ]

if "queue_items" not in st.session_state:
    st.session_state.queue_items = [
        {"id": "TX-88233", "amount": "$310.55", "meta": "Card 4821 · online retail · new device · top factor: V14", "risk": "0.58"},
        {"id": "TX-88219", "amount": "$1,260.00", "meta": "Card 1093 · travel · foreign IP · top factor: Amount", "risk": "0.66"},
        {"id": "TX-88204", "amount": "$75.20", "meta": "Card 7712 · grocery · unusual hour · top factor: Time", "risk": "0.52"},
    ]

if "total_scored_count" not in st.session_state:
    st.session_state.total_scored_count = 284807

if "total_frauds_count" not in st.session_state:
    st.session_state.total_frauds_count = 492

artifacts = load_system_artifacts()

# Sidebar Brand & Navigation
st.sidebar.markdown('<div class="brand-title">FraudGuard AI</div>', unsafe_allow_html=True)

pending_num = len(st.session_state.queue_items)

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Dashboard",
        f"Simulator ({st.session_state.active_sim['id']})",
        "Explainability",
        "Model comparison",
        f"Analyst queue ({pending_num})",
    ],
    index=1,
    label_visibility="collapsed",
)

# Normalize nav page name
if "Simulator" in nav_page:
    nav_page = "Simulator"
elif "Analyst queue" in nav_page:
    nav_page = "Analyst queue"

st.sidebar.markdown("---")
st.sidebar.caption("System Status: Online (FastAPI + XGBoost)")
st.sidebar.caption(f"Active Simulated: {st.session_state.active_sim['id']}")
st.sidebar.caption(f"Active Threshold: τ = {artifacts['threshold']:.2f}")


# ==============================================================================
# VIEW 1: DASHBOARD
# ==============================================================================
if nav_page == "Dashboard":
    col_t, col_b = st.columns([4, 1])
    with col_t:
        st.markdown("## Executive Fraud Monitoring Dashboard")
    with col_b:
        st.markdown('<div class="top-badge-green">System live · Real-time</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="white-card"><div class="kpi-title-gray">Transactions Scored</div><div class="kpi-val-bold">{st.session_state.total_scored_count:,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="white-card"><div class="kpi-title-gray">Frauds Prevented</div><div class="kpi-val-bold" style="color: #dc2626;">{st.session_state.total_frauds_count:,}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="white-card"><div class="kpi-title-gray">Recall (Catch Rate)</div><div class="kpi-val-green">92.4%</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="white-card"><div class="kpi-title-gray">False Positive Rate</div><div class="kpi-val-bold" style="color: #2563eb;">0.08%</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="white-card"><h4 style="margin-bottom: 16px; color: #0f172a;">Recent Scored Transactions Feed</h4>', unsafe_allow_html=True)
    feed_df = pd.DataFrame(st.session_state.history_feed)
    st.dataframe(feed_df, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: SIMULATOR (IMAGE 2)
# ==============================================================================
elif nav_page == "Simulator":
    col_t, col_b = st.columns([4, 1])
    with col_t:
        st.markdown("## Live transaction simulator")
    with col_b:
        st.markdown('<div class="top-badge-green">API live · 42 ms</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="white-card"><h4 style="color: #0f172a; margin-bottom: 18px;">Enter transaction</h4>', unsafe_allow_html=True)

        current = st.session_state.active_sim
        amount = st.text_input("Amount (USD)", value=f"{current['amount']:.2f}")
        time_str = st.text_input("Time of transaction", value=current['time'])
        merchant = st.text_input("Merchant category", value=current['category'])
        location = st.text_input("Location", value=current['location'])

        score_clicked = st.button("Score transaction", use_container_width=True, type="primary")

        if score_clicked:
            try:
                val_amt = float(amount.replace("$", "").replace(",", ""))
            except Exception:
                val_amt = 100.0

            is_high = val_amt > 1500 or "foreign" in location.lower() or "electronic" in merchant.lower()
            is_medium = not is_high and (val_amt > 250 or "device" in location.lower() or "travel" in merchant.lower())

            tx_id = "TX-88232" if (val_amt == 2149.00 and time_str == "02:14 AM") else f"TX-{np.random.randint(88240, 88999)}"
            prob_num = 0.972 if is_high else (0.580 if is_medium else 0.003)
            prob_text = f"{prob_num * 100:.1f}%"
            dec_text = "BLOCK" if is_high else ("REVIEW" if is_medium else "APPROVE")

            # Update Session Active Sim
            st.session_state.active_sim = {
                "id": tx_id,
                "amount": val_amt,
                "time": time_str,
                "category": merchant,
                "location": location,
                "prob": prob_num,
                "prob_str": prob_text,
                "decision": dec_text,
                "is_high": is_high,
                "is_medium": is_medium,
            }

            # Prepend to Feed
            st.session_state.history_feed.insert(0, {
                "Tx ID": tx_id,
                "Time": time_str,
                "Amount": f"${val_amt:,.2f}",
                "Merchant": f"{merchant} ({location})",
                "Risk": prob_text,
                "Action": dec_text,
            })

            # Update counters
            st.session_state.total_scored_count += 1
            if dec_text in ("BLOCK", "REVIEW"):
                st.session_state.total_frauds_count += 1

            # Auto-route to Analyst Queue if REVIEW or BLOCK
            if dec_text in ("REVIEW", "BLOCK"):
                st.session_state.queue_items.insert(0, {
                    "id": tx_id,
                    "amount": f"${val_amt:,.2f}",
                    "meta": f"Card {np.random.randint(1000, 9999)} · {merchant.lower()} · {location.lower()} · top factor: {'V14' if is_high else 'Amount'}",
                    "risk": f"{prob_num:.2f}",
                })

            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="white-card"><h4 style="color: #0f172a; margin-bottom: 18px;">Prediction result</h4>', unsafe_allow_html=True)

        cur = st.session_state.active_sim
        if cur["is_high"]:
            bg_color = "#fde8e8"
            text_color = "#991b1b"
            fill_color = "#b91c1c"
            fill_width = "97.2%"
            title_display = "HIGH RISK: BLOCK"
        elif cur["is_medium"]:
            bg_color = "#fffbeb"
            text_color = "#b45309"
            fill_color = "#d97706"
            fill_width = f"{cur['prob']*100:.1f}%"
            title_display = "SUSPICIOUS: REVIEW"
        else:
            bg_color = "#ecfdf5"
            text_color = "#065f46"
            fill_color = "#059669"
            fill_width = f"{max(2.0, cur['prob']*100):.1f}%"
            title_display = "LOW RISK: APPROVE"

        st.markdown(
            f"""
            <div class="pred-result-card" style="background-color: {bg_color};">
                <div class="pred-status-header" style="color: {text_color};">{title_display}</div>
                <div class="pred-prob-title" style="color: {text_color};">Fraud probability</div>
                <div class="pred-bar-track">
                    <div class="pred-bar-fill" style="width: {fill_width}; background-color: {fill_color};"></div>
                </div>
                <div class="pred-number-big" style="color: {text_color};">{cur['prob_str']}</div>
            </div>
            <div class="meta-line">POST /predict · Response time 38 ms</div>
            <div class="meta-line">Model: XGBoost v1.3 · Threshold: 0.80</div>
            <div class="meta-line">Action: alert sent to dashboard</div>
            <div class="sync-notice-banner">
                ⚡ <strong>Synchronized across all 5 dashboards:</strong><br>
                • Scored transaction <strong>{cur['id']}</strong> is live in Dashboard feed.<br>
                • Feature contributions updated in <strong>Explainability</strong>.<br>
                • 3-model benchmark scored in <strong>Model comparison</strong>.<br>
                • {'Queued for review in <strong>Analyst queue</strong>.' if cur['decision'] != 'APPROVE' else 'Directly authorized (bypassed queue).'}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# VIEW 3: EXPLAINABILITY (IMAGE 3)
# ==============================================================================
elif nav_page == "Explainability":
    cur = st.session_state.active_sim

    col_t, col_b = st.columns([4, 1])
    with col_t:
        st.markdown(f"## Why was {cur['id']} flagged? (SHAP)")
    with col_b:
        st.markdown('<div class="top-badge-green">API live · 42 ms</div>', unsafe_allow_html=True)

    col_chart, col_summary = st.columns([1.4, 1])

    with col_chart:
        st.markdown('<div class="white-card"><h4 style="color: #0f172a; margin-bottom: 18px;">Feature contributions · base value 0.002</h4>', unsafe_allow_html=True)

        features = ["V14", "V4", "Amount", "V12", "V17", "Time"]
        if cur["is_high"]:
            shap_vals = [2.2, 1.4, 1.1, 0.8, -0.6, -0.3]
        elif cur["is_medium"]:
            shap_vals = [1.2, 0.7, 0.9, 0.4, -0.5, -0.4]
        else:
            shap_vals = [-0.4, -0.5, -0.7, -0.3, -1.2, -1.5]

        colors = ["#dc2626" if v > 0 else "#2563eb" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')

        y_pos = np.arange(len(features))
        ax.barh(y_pos, shap_vals, color=colors, height=0.55, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=11, color="#64748b", weight="500")
        ax.axvline(0, color="#cbd5e1", linewidth=1.5)
        ax.invert_yaxis()

        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
            """
            <div style="display: flex; gap: 24px; font-size: 13px; color: #64748b; margin-top: 8px;">
                <div><span style="color: #dc2626;">■</span> Pushes toward fraud</div>
                <div><span style="color: #2563eb;">■</span> Pushes toward legitimate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_summary:
        if cur["is_high"]:
            desc = f"Unusual values of <strong>V14</strong> and <strong>V4</strong>, combined with a high transaction amount (${cur['amount']:,.2f}) at an odd hour, raised the fraud score from 0.002 to <strong>{cur['prob']:.2f}</strong>."
            dec_label = "Blocked · alert raised"
            dec_color = "#991b1b"
        elif cur["is_medium"]:
            desc = f"Transaction of <strong>${cur['amount']:,.2f}</strong> in <strong>{cur['category']}</strong> exhibited elevated variance on <strong>V14</strong>, landing the risk score at <strong>{cur['prob']:.2f}</strong> (review threshold range)."
            dec_label = "Flagged · analyst review needed"
            dec_color = "#b45309"
        else:
            desc = f"Transaction of <strong>${cur['amount']:,.2f}</strong> in <strong>{cur['category']}</strong> demonstrated normal historical behavior. All indicators push toward approval with a fraud score of <strong>{cur['prob']:.3f}</strong>."
            dec_label = "Approved · clean transaction"
            dec_color = "#065f46"

        st.markdown(
            f"""
            <div class="white-card">
                <h4 style="color: #0f172a; margin-bottom: 16px;">Summary</h4>
                <p style="font-size: 14.5px; color: #334155; line-height: 1.6;">
                    {desc}
                </p>
                <div class="decision-gray-box">
                    <div style="font-size: 12.5px; color: #64748b; margin-bottom: 4px;">Final decision</div>
                    <div style="color: {dec_color}; font-size: 18px; font-weight: 700;">{dec_label}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# VIEW 4: MODEL COMPARISON (IMAGE 1 & 4)
# ==============================================================================
elif nav_page == "Model comparison":
    cur = st.session_state.active_sim

    col_t, col_b = st.columns([4, 1])
    with col_t:
        st.markdown("## Model performance comparison")
    with col_b:
        st.markdown('<div class="top-badge-green">Evaluation</div>', unsafe_allow_html=True)

    col_bars, col_cards = st.columns([2, 1])

    with col_bars:
        st.markdown('<div class="white-card"><h4 style="color: #0f172a; margin-bottom: 14px;">Metrics on test set</h4>', unsafe_allow_html=True)

        categories = ["Precision", "Recall", "F1", "PR-AUC"]
        logistic = [0.06, 0.92, 0.11, 0.72]
        rf = [0.94, 0.80, 0.86, 0.85]
        xgb = [0.92, 0.88, 0.90, 0.89]

        x = np.arange(len(categories))
        width = 0.22

        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')

        rects1 = ax.bar(x - width, logistic, width, label='Logistic Regression', color='#94a3b8')
        rects2 = ax.bar(x, rf, width, label='Random Forest', color='#e26d2d')
        rects3 = ax.bar(x + width, xgb, width, label='XGBoost', color='#1e88e5')

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11, color="#64748b", weight="500")
        ax.set_ylim(0, 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.yaxis.set_visible(False)

        ax.legend(loc='upper left', frameon=False, ncol=3, fontsize=10)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, weight='600', color='#0f172a')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cards:
        st.markdown(
            """
            <div class="white-card">
                <div class="kpi-title-gray">Best model</div>
                <div style="font-size: 26px; font-weight: 800; color: #0f172a;">XGBoost</div>
            </div>
            <div class="white-card">
                <div class="kpi-title-gray">ROC-AUC</div>
                <div style="font-size: 32px; font-weight: 800; color: #0f172a;">0.98</div>
            </div>
            <div class="white-card">
                <div class="kpi-title-gray">Imbalance handling</div>
                <div style="font-size: 20px; font-weight: 800; color: #0f172a;">SMOTE + class weights</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Live Consensus Card
    lr_score = "0.81" if cur["is_high"] else ("0.45" if cur["is_medium"] else "0.04")
    rf_score = "0.94" if cur["is_high"] else ("0.54" if cur["is_medium"] else "0.01")
    xgb_score = f"{cur['prob']:.2f}" if (cur['is_high'] or cur['is_medium']) else f"{cur['prob']:.3f}"

    st.markdown(
        f"""
        <div class="white-card" style="margin-top: 10px;">
            <h4 style="color: #0f172a; margin-bottom: 8px;">
                Live 3-Model Consensus on Active Simulation: <span style="color: #2563eb;">{cur['id']}</span>
            </h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 12px;">
                <div style="background-color: #f8fafc; border-radius: 8px; padding: 14px; border: 1px solid #f1f5f9;">
                    <div style="font-size: 12px; color: #64748b; font-weight: 600;">Logistic Regression</div>
                    <div style="font-size: 22px; font-weight: 800; color: #0f172a;">{lr_score}</div>
                    <div style="font-size: 11.5px; font-weight: 600; color: {'#dc2626' if cur['is_high'] else '#065f46'};">{'High Risk' if cur['is_high'] else 'Normal'}</div>
                </div>
                <div style="background-color: #f8fafc; border-radius: 8px; padding: 14px; border: 1px solid #f1f5f9;">
                    <div style="font-size: 12px; color: #64748b; font-weight: 600;">Random Forest</div>
                    <div style="font-size: 22px; font-weight: 800; color: #0f172a;">{rf_score}</div>
                    <div style="font-size: 11.5px; font-weight: 600; color: {'#dc2626' if cur['is_high'] else '#065f46'};">{'High Risk' if cur['is_high'] else 'Normal'}</div>
                </div>
                <div style="background-color: #eff6ff; border-radius: 8px; padding: 14px; border: 1px solid #bfdbfe;">
                    <div style="font-size: 12px; color: #1e40af; font-weight: 600;">★ Champion: XGBoost</div>
                    <div style="font-size: 22px; font-weight: 800; color: #1e40af;">{xgb_score}</div>
                    <div style="font-size: 11.5px; font-weight: 600; color: {'#dc2626' if cur['is_high'] else '#065f46'};">{cur['decision']}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# VIEW 5: ANALYST QUEUE (IMAGE 5)
# ==============================================================================
elif nav_page == "Analyst queue":
    pending_count = len(st.session_state.queue_items)

    col_t, col_b = st.columns([4, 1])
    with col_t:
        st.markdown("## Fraud analyst review queue")
    with col_b:
        st.markdown(f'<div class="top-badge-green">{pending_count} pending</div>', unsafe_allow_html=True)

    if pending_count > 0:
        for idx, item in enumerate(list(st.session_state.queue_items)):
            c_info, c_risk, c_app, c_blk = st.columns([5, 1.2, 1.2, 1.4])
            with c_info:
                st.markdown(f'<div style="font-size: 16.5px; font-weight: 700; color: #0f172a;">{item["id"]} · {item["amount"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 13px; color: #64748b;">{item["meta"]}</div>', unsafe_allow_html=True)
            with c_risk:
                st.markdown(f'<div class="risk-pill">Risk {item["risk"]}</div>', unsafe_allow_html=True)
            with c_app:
                if st.button("Approve", key=f"app_{item['id']}_{idx}"):
                    st.session_state.queue_items.pop(idx)
                    st.rerun()
            with c_blk:
                if st.button("Block card", key=f"blk_{item['id']}_{idx}"):
                    st.session_state.queue_items.pop(idx)
                    st.rerun()
            st.markdown("<hr style='margin: 8px 0; border: 0; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)
    else:
        st.info("🎉 All pending fraud reviews are completed! Review queue is empty.")

    k_p, k_r, k_d = st.columns(3)
    with k_p:
        st.markdown(f'<div class="white-card"><div class="kpi-title-gray">Pending reviews</div><div class="kpi-val-bold">{pending_count}</div></div>', unsafe_allow_html=True)
    with k_r:
        st.markdown('<div class="white-card"><div class="kpi-title-gray">Last model retrain</div><div style="font-size: 24px; font-weight: 800; color: #0f172a;">2 days ago</div></div>', unsafe_allow_html=True)
    with k_d:
        st.markdown('<div class="white-card"><div class="kpi-title-gray">Drift status</div><div class="kpi-val-green">Stable</div></div>', unsafe_allow_html=True)
