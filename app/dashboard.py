"""Streamlit Web Dashboard for Real-Time AI Fraud Detection.

Pages:
1. Dashboard: High-level KPIs, operational monitoring, and recent scored feed.
2. Live Simulator: Interactive transaction tester with instant scoring and SHAP explainability.
3. Model Comparison: Benchmark table across 9 configurations and high-resolution diagnostic plots.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path so 'src' imports cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AI Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 4px;
    }

    /* Decision Badges */
    .badge-approve {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-review {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-block {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system_artifacts():
    """Load model, scaler, threshold, and test samples once into memory."""
    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"

    model_path = models_dir / "xgb_fraud_model.joblib"
    scaler_path = models_dir / "scaler.joblib"
    threshold_path = models_dir / "threshold.json"
    samples_path = models_dir / "test_sample.joblib"
    comparison_csv_path = reports_dir / "model_comparison.csv"

    artifacts = {
        "is_ready": False,
        "model": None,
        "scaler": None,
        "threshold": 0.50,
        "threshold_metadata": {},
        "test_samples": None,
        "comparison_df": None,
    }

    if model_path.exists() and scaler_path.exists():
        try:
            from src.predict import get_predictor
            predictor = get_predictor()
            artifacts["is_ready"] = True
            artifacts["predictor"] = predictor
            artifacts["model"] = predictor.model
            artifacts["scaler"] = predictor.scaler
            artifacts["threshold"] = predictor.threshold
            artifacts["threshold_metadata"] = predictor.threshold_metadata
        except Exception as e:
            st.error(f"Error loading model predictor: {e}")

    if samples_path.exists():
        try:
            artifacts["test_samples"] = joblib.load(samples_path)
        except Exception:
            artifacts["test_samples"] = None

    if comparison_csv_path.exists():
        try:
            artifacts["comparison_df"] = pd.read_csv(comparison_csv_path)
        except Exception:
            artifacts["comparison_df"] = None

    return artifacts


# Initialize session state for real-time history
if "history" not in st.session_state:
    st.session_state.history = []

artifacts = load_system_artifacts()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("Fraud Guard AI")
st.sidebar.caption("Real-Time Intelligent Risk Scoring")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "⚡ Live Simulator", "🔬 Model Comparison"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Operational Controls")
if artifacts["is_ready"]:
    st.sidebar.success("● ML Engine Online (XGBoost)")
    st.sidebar.info(f"Active Threshold: **{artifacts['threshold']:.3f}**")
else:
    st.sidebar.error("● ML Engine Offline")
    st.sidebar.caption("Please run `python -m src.train` in terminal.")


# ==============================================================================
# Page 1: Dashboard
# ==============================================================================
if page == "📊 Dashboard":
    st.title("🛡️ Executive Fraud Monitoring Dashboard")
    st.markdown("Real-time transaction stream, fraud mitigation KPIs, and model health.")

    if not artifacts["is_ready"]:
        st.warning(
            "⚠️ The model pipeline has not been executed yet. "
            "Run `python -m src.train` in your terminal to train and evaluate all models."
        )

    # Compute live session KPIs
    total_scored = len(st.session_state.history)
    frauds_flagged = sum(1 for h in st.session_state.history if h["decision"] in ("BLOCK", "REVIEW"))

    # Benchmark metrics from threshold JSON or default
    thresh_meta = artifacts["threshold_metadata"].get("metrics_at_tuned_threshold", {})
    recall_rate = thresh_meta.get("recall", 0.88)
    fpr_rate = thresh_meta.get("false_positive_rate", 0.0008)
    pr_auc_score = thresh_meta.get("pr_auc", 0.89)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Transactions Scored</div>
                <div class="metric-value">{total_scored:,}</div>
                <div class="metric-sub">Session volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Frauds Flagged</div>
                <div class="metric-value" style="color: #f87171;">{frauds_flagged:,}</div>
                <div class="metric-sub">Block + Review actions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Fraud Recall (Catch Rate)</div>
                <div class="metric-value" style="color: #34d399;">{recall_rate * 100:.1f}%</div>
                <div class="metric-sub">Target constraint: >= 85%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">False Positive Rate (FPR)</div>
                <div class="metric-value" style="color: #60a5fa;">{fpr_rate * 100:.3f}%</div>
                <div class="metric-sub">Legitimate flagged</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Real-Time Recent Scored Stream
    st.subheader("📋 Recent Transaction Stream")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            history_df[["timestamp", "transaction_id", "amount", "fraud_probability", "decision", "latency_ms"]],
            column_config={
                "timestamp": "Timestamp",
                "transaction_id": "Tx ID",
                "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                "fraud_probability": st.column_config.ProgressColumn(
                    "Fraud Risk",
                    help="Predicted probability of fraud",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "decision": "Action",
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.1f ms"),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No live transactions scored yet. Head to the **Live Simulator** tab to score sample transactions!")

    # System Architecture Highlights
    st.markdown("---")
    st.subheader("⚙️ System Architecture & Tiers")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("🟢 **APPROVE (< 0.40)**\n\nInstant authorization. Seamless user experience with zero friction.")
    with c2:
        st.warning(f"🟡 **REVIEW (0.40 - {artifacts['threshold']:.2f})**\n\nStep-up authentication (SMS OTP, biometric check, or fraud analyst review).")
    with c3:
        st.error(f"🔴 **BLOCK (>= {artifacts['threshold']:.2f})**\n\nImmediate transaction decline and account security lockdown.")


# ==============================================================================
# Page 2: Live Simulator
# ==============================================================================
elif page == "⚡ Live Simulator":
    st.title("⚡ Live Transaction Simulator & SHAP Explainer")
    st.markdown("Test single transactions against the production XGBoost model with instant SHAP attribution.")

    if not artifacts["is_ready"]:
        st.error("ML engine artifacts not found. Run `python -m src.train` first.")
        st.stop()

    predictor = artifacts["predictor"]
    test_samples = artifacts.get("test_samples")

    # Sample selection buttons
    st.write("##### Quick Load Samples from Test Set:")
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    loaded_data = None
    if btn_col1.button("🎲 Load Normal Transaction (Legit)"):
        if test_samples and test_samples.get("legit"):
            loaded_data = np.random.choice(test_samples["legit"])
            st.session_state["simulator_input"] = loaded_data
            st.toast("Loaded legitimate test sample!", icon="✅")

    if btn_col2.button("🚨 Load High-Risk Transaction (Fraud)"):
        if test_samples and test_samples.get("fraud"):
            loaded_data = np.random.choice(test_samples["fraud"])
            st.session_state["simulator_input"] = loaded_data
            st.toast("Loaded fraudulent test sample!", icon="🚨")

    if btn_col3.button("🔄 Clear / Reset Form"):
        st.session_state["simulator_input"] = None
        st.rerun()

    # Form defaults
    current_input = st.session_state.get("simulator_input") or {}

    with st.form("transaction_scoring_form"):
        st.write("##### Transaction Parameters")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            amount_val = float(current_input.get("Amount", 125.50))
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=50000.0, value=amount_val, step=5.0)
        with fcol2:
            time_val = float(current_input.get("Time", 45000.0))
            time_sec = st.number_input("Time Elapsed (seconds)", min_value=0.0, max_value=200000.0, value=time_val, step=100.0)

        st.caption("Key PCA Latent Features (Identified as most predictive by Gini Importance & SHAP):")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)

        v14 = pcol1.number_input("V14 (Strong Fraud Driver)", value=float(current_input.get("V14", -4.2)), format="%.4f")
        v12 = pcol2.number_input("V12 (PCA Component)", value=float(current_input.get("V12", -2.8)), format="%.4f")
        v10 = pcol3.number_input("V10 (PCA Component)", value=float(current_input.get("V10", -2.5)), format="%.4f")
        v4  = pcol4.number_input("V4 (PCA Component)", value=float(current_input.get("V4", 3.8)), format="%.4f")

        with st.expander("Show all remaining PCA features (V1-V28)"):
            all_v_cols = st.columns(4)
            full_tx_data = {"Time": time_sec, "Amount": amount, "V14": v14, "V12": v12, "V10": v10, "V4": v4}
            for i in range(1, 29):
                v_name = f"V{i}"
                if v_name in ["V14", "V12", "V10", "V4"]:
                    continue
                col_idx = (i - 1) % 4
                default_v = float(current_input.get(v_name, np.random.normal(0, 1)))
                val = all_v_cols[col_idx].number_input(v_name, value=default_v, format="%.4f", key=f"inp_{v_name}")
                full_tx_data[v_name] = val

        submit = st.form_submit_button("⚡ Run Fraud Risk Assessment", use_container_width=True)

    if submit:
        # Assemble dictionary
        tx_dict = {"Time": time_sec, "Amount": amount}
        for i in range(1, 29):
            v_name = f"V{i}"
            if v_name in ["V14", "V12", "V10", "V4"]:
                tx_dict[v_name] = {"V14": v14, "V12": v12, "V10": v10, "V4": v4}[v_name]
            else:
                tx_dict[v_name] = full_tx_data.get(v_name, 0.0)

        # Run inference
        result = predictor.predict(tx_dict)

        # Record in history
        import datetime
        st.session_state.history.insert(0, {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "transaction_id": f"TX-{np.random.randint(100000, 999999)}",
            "amount": amount,
            "fraud_probability": result["fraud_probability"],
            "decision": result["decision"],
            "latency_ms": result["response_time_ms"],
        })

        st.markdown("---")
        st.subheader("🎯 Scoring Results")

        rcol1, rcol2, rcol3 = st.columns(3)

        decision = result["decision"]
        prob = result["fraud_probability"]

        with rcol1:
            st.metric("Fraud Probability", f"{prob * 100:.2f}%")
            st.progress(min(1.0, max(0.0, prob)))

        with rcol2:
            st.write("**Operational Decision:**")
            if decision == "APPROVE":
                st.markdown('<span class="badge-approve">🟢 APPROVE</span>', unsafe_allow_html=True)
                st.caption("Low risk (< 0.40). Auto-authorized.")
            elif decision == "REVIEW":
                st.markdown('<span class="badge-review">🟡 REVIEW</span>', unsafe_allow_html=True)
                st.caption(f"Borderline risk (0.40 - {result['tuned_threshold']:.2f}). Trigger step-up authentication.")
            else:
                st.markdown('<span class="badge-block">🔴 BLOCK</span>', unsafe_allow_html=True)
                st.caption(f"High risk (>= {result['tuned_threshold']:.2f}). Transaction declined.")

        with rcol3:
            st.metric("Inference Latency", f"{result['response_time_ms']:.1f} ms")
            st.caption(f"Tuned Threshold: {result['tuned_threshold']:.3f}")

        # Local SHAP Explanation Chart
        st.markdown("---")
        st.subheader("🔍 Local Explainability (SHAP Top Feature Drivers)")
        st.caption("Identifies the exact variables pushing this transaction toward fraud or toward legitimate approval.")

        shap_features = result.get("top_5_shap_features", [])
        if shap_features:
            features = [item["feature"] for item in shap_features]
            shap_vals = [item["shap_value"] for item in shap_features]
            colors = ["#ef4444" if val > 0 else "#10b981" for val in shap_vals]

            fig, ax = plt.subplots(figsize=(8, 3.8))
            y_pos = np.arange(len(features))
            ax.barh(y_pos, shap_vals, color=colors, align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10, weight="bold")
            ax.axvline(0, color="#94a3b8", linestyle="--", alpha=0.7)
            ax.set_xlabel("SHAP Value (Contribution to Fraud Log-Odds)", fontsize=10)
            ax.set_title("Top 5 Decision Drivers for this Transaction", fontsize=11, weight="bold")
            plt.tight_layout()
            st.pyplot(fig)

            # Table of details
            shap_table = pd.DataFrame(shap_features)
            st.dataframe(shap_table, hide_index=True, use_container_width=True)


# ==============================================================================
# Page 3: Model Comparison
# ==============================================================================
elif page == "🔬 Model Comparison":
    st.title("🔬 Model & Strategy Benchmarking")
    st.markdown("Comprehensive performance matrix comparing **3 ML Algorithms** across **3 Imbalance Handling Strategies**.")

    reports_dir = PROJECT_ROOT / "reports"
    comparison_df = artifacts.get("comparison_df")

    if comparison_df is not None:
        st.subheader("📊 3x3 Benchmark Matrix (Evaluated on Untouched Test Set)")
        st.dataframe(
            comparison_df.style.highlight_max(subset=["PR-AUC", "F1", "Recall"], color="#1e3a8a"),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Note: The champion model is automatically selected based on PR-AUC "
            "subject to a minimum recall constraint (>= 0.70)."
        )

    st.markdown("---")
    st.subheader("📈 Diagnostic Visualizations")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model Comparison",
        "PR Curve & ROC",
        "Confusion Matrix",
        "Feature Importance",
        "SHAP Global Summary",
    ])

    with tab1:
        img_path = reports_dir / "model_comparison.png"
        if img_path.exists():
            st.image(str(img_path), caption="Benchmark comparison across all 9 configurations", use_column_width=True)
        else:
            st.info("Run `python -m src.train` to generate benchmark plot.")

    with tab2:
        c1, c2 = st.columns(2)
        pr_path = reports_dir / "precision_recall_curve.png"
        roc_path = reports_dir / "roc_curve.png"
        with c1:
            if pr_path.exists():
                st.image(str(pr_path), caption="Precision-Recall Curve with Tuned Threshold", use_column_width=True)
            else:
                st.info("PR curve not generated yet.")
        with c2:
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC Curve", use_column_width=True)
            else:
                st.info("ROC curve not generated yet.")

    with tab3:
        cm_path = reports_dir / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix at Tuned Threshold", use_column_width=True)
        else:
            st.info("Confusion matrix plot not generated yet.")

    with tab4:
        fi_path = reports_dir / "feature_importance.png"
        if fi_path.exists():
            st.image(str(fi_path), caption="Top 15 Most Predictive Features (XGBoost)", use_column_width=True)
        else:
            st.info("Feature importance plot not generated yet.")

    with tab5:
        shap_path = reports_dir / "shap_summary.png"
        if shap_path.exists():
            st.image(str(shap_path), caption="Global SHAP Beeswarm Summary Plot", use_column_width=True)
        else:
            st.info("SHAP summary plot not generated yet.")
