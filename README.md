# AI-Powered Credit Card Fraud Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB6424.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)

A complete, production-grade machine learning project for real-time credit card fraud detection. Designed with industrial best practices: handling extreme class imbalance (0.2% fraud), leak-free stratified validation, 3x3 model/resampling benchmarking, decision threshold calibration, SHAP explainability, ultra-low latency REST API inference, and an interactive executive operations dashboard.

---

## 📑 Table of Contents
1. [Project Overview & Architecture](#-project-overview--architecture)
2. [Project Structure](#-project-structure)
3. [Installation & Setup](#-installation--setup)
4. [Dataset Instructions & Synthetic Fallback](#-dataset-instructions--synthetic-fallback)
5. [Running the Pipeline](#-running-the-pipeline)
   - [Model Training & Benchmarking](#1-model-training--benchmarking)
   - [Real-Time Inference API (FastAPI)](#2-real-time-inference-api-fastapi)
   - [Operations Dashboard (Streamlit)](#3-operations-dashboard-streamlit)
6. [Sample API Request & Response](#-sample-api-request--response)
7. [Why Accuracy Fails in Fraud Detection](#-why-accuracy-fails-in-fraud-detection)
8. [Evaluation Metrics Deep-Dive](#-evaluation-metrics-deep-dive)
9. [Threshold Tuning & Operational Tiers](#-threshold-tuning--operational-tiers)
10. [Model Explainability with SHAP](#-model-explainability-with-shap)
11. [Known Limitations in Production Systems](#-known-limitations-in-production-systems)

---

## 🏛 Project Overview & Architecture

Payment processing networks process hundreds of thousands of transactions every minute. Because fraudulent transactions constitute less than 0.2% of total traffic, conventional machine learning approaches fail. This project addresses the challenge end-to-end:

```
[ Incoming Transaction ]
           │
           ▼
[ RobustScaler ] ──> Scales Amount & Time using median & IQR (outlier-resistant)
           │
           ▼
[ Champion XGBoost Model ] ──> Evaluates fraud probability (eval_metric: aucpr)
           │
     ┌─────┴─────────────────────────┐
     ▼                               ▼
[ Decision Engine ]        [ SHAP TreeExplainer ]
  < 0.40  ──> APPROVE        Attribution of top 5 feature
  0.40-τ  ──> REVIEW         drivers pushing toward fraud vs.
  >= τ    ──> BLOCK          legitimate approval.
```

---

## 📂 Project Structure

```
fraud-detection/
├── data/
│   ├── .gitkeep                      # Keeps directory present in repositories
│   └── dataset_instructions.txt      # Kaggle download link and synthetic fallback info
├── models/
│   ├── xgb_fraud_model.joblib        # Serialized champion XGBoost model
│   ├── scaler.joblib                 # Fitted RobustScaler for Time & Amount
│   ├── threshold.json                # Calibrated operational threshold & metadata
│   └── test_sample.joblib            # Test split samples for instant dashboard testing
├── reports/
│   ├── model_comparison.csv          # 3x3 benchmark metrics across all combinations
│   ├── model_comparison.png          # Benchmark bar chart (PR-AUC, F1, Recall)
│   ├── confusion_matrix.png          # Confusion matrix at calibrated threshold
│   ├── precision_recall_curve.png    # PR curve marking optimal decision boundary
│   ├── roc_curve.png                 # ROC curve with AUC metric
│   ├── feature_importance.png        # Top 15 predictive features (Gain/Gini)
│   └── shap_summary.png              # Global SHAP beeswarm plot
├── src/
│   ├── __init__.py
│   ├── preprocess.py                 # Data loading, synthetic generation, scaling, splitting
│   ├── train.py                      # 3x3 model/strategy comparison, threshold tuning
│   ├── evaluate.py                   # Evaluation metrics, PR-based threshold tuning, plots
│   ├── explain.py                    # SHAP TreeExplainer (global summary & local top 5)
│   └── predict.py                    # FraudPredictor inference engine with tier routing
├── api/
│   ├── __init__.py
│   └── main.py                       # High-performance FastAPI application with Pydantic
├── app/
│   └── dashboard.py                  # 3-page interactive Streamlit dashboard
├── requirements.txt                  # Pinned Python package dependencies
└── README.md                         # Comprehensive project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `git` (optional)

### Setup Virtual Environment
```bash
# Clone or navigate to the fraud-detection directory
cd fraud-detection

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows (cmd/PowerShell):
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install all pinned dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Instructions & Synthetic Fallback

### 1. Genuine Kaggle ULB Dataset
This project is built around the famous **ULB Credit Card Fraud Detection** dataset:
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Dimensions**: 284,807 transactions, 31 columns (`Time`, `V1`–`V28`, `Amount`, `Class`).
- **Frauds**: 492 cases (0.172% of total).
- **Placement**: Download `creditcard.csv` and place it in:
  ```
  fraud-detection/data/creditcard.csv
  ```

### 2. Automatic Synthetic Fallback
If `data/creditcard.csv` is not present, **the pipeline does not crash**. Instead:
- It prints a prominent warning in the console.
- Generates a 20,000-sample synthetic dataset with 0.2% fraud (~40 frauds) using `sklearn.datasets.make_classification`.
- Automatically populates matching `Time`, `V1` through `V28`, `Amount`, and `Class` distributions.
- Executes the entire training, benchmarking, threshold tuning, and artifact generation end-to-end.

---

## 🚀 Running the Pipeline

### 1. Model Training & Benchmarking
Run the complete training pipeline with a single command:
```bash
python -m src.train
```
**What happens during this step:**
1. Loads data (or synthetic fallback).
2. Performs a stratified 80/20 train-test split (`random_state=42`).
3. Fits `RobustScaler` on `Time` and `Amount` on the training set and transforms both sets.
4. Trains and evaluates **9 combinations** (3 models × 3 strategies):
   - **Logistic Regression**, **Random Forest**, and **XGBoost**
   - **Class Weights**, **SMOTE**, and **Random Undersampling**
5. Evaluates on the **untouched test set** to ensure zero data leakage.
6. Automatically selects the champion model by PR-AUC with a recall constraint (Recall >= 0.70).
7. Calibrates the decision threshold using the Precision-Recall curve to maximize F1 while targeting Recall >= 0.85.
8. Generates all diagnostic plots in `reports/`.
9. Saves the final model (`xgb_fraud_model.joblib`), scaler (`scaler.joblib`), and threshold configuration (`threshold.json`) into `models/`.

---

### 2. Real-Time Inference API (FastAPI)
Launch the REST API server:
```bash
uvicorn api.main:app --reload --port 8000
```
- Interactive API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
- Service Health Endpoint: `http://127.0.0.1:8000/health`
- Real-Time Scoring: `http://127.0.0.1:8000/predict` (POST)

---

### 3. Operations Dashboard (Streamlit)
Launch the 3-page interactive web application:
```bash
streamlit run app/dashboard.py
```
- Opens automatically in your browser at: `http://localhost:8501`
- **Page 1: 📊 Dashboard**: Executive KPIs (Transactions Scored, Frauds Flagged, Recall Catch Rate, FPR) and live activity feed.
- **Page 2: ⚡ Live Simulator**: Instant transaction testing (with 1-click loading of legitimate and fraud test samples) and interactive SHAP feature attributions.
- **Page 3: 🔬 Model Comparison**: 3x3 benchmark matrix table and interactive views of all generated plots.

---

## 📡 Sample API Request & Response

### Sample Request via cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0,
    "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979,
    "V5": -0.5221, "V6": -1.4265, "V7": -2.5373, "V8": 1.3916,
    "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8999,
    "V13": -0.5952, "V14": -4.2892, "V15": 0.3897, "V16": -1.1407,
    "V17": -2.8300, "V18": -0.0168, "V19": 0.4169, "V20": 0.1269,
    "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3201,
    "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1432,
    "Amount": 149.62
  }'
```

### Sample Request via Python
```python
import requests

payload = {
    "Time": 406.0,
    "Amount": 149.62,
    **{f"V{i}": 0.0 for i in range(1, 29)},
    "V14": -4.2892,
    "V12": -2.8999,
    "V10": -2.7722,
    "V4": 3.9979,
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())
```

### Sample Response
```json
{
  "fraud_probability": 0.8924,
  "decision": "BLOCK",
  "tuned_threshold": 0.4852,
  "top_5_shap_features": [
    {
      "feature": "V14",
      "value": -4.2892,
      "shap_value": 2.1534,
      "impact": "Pushes toward Fraud"
    },
    {
      "feature": "V10",
      "value": -2.7722,
      "shap_value": 1.4218,
      "impact": "Pushes toward Fraud"
    },
    {
      "feature": "V12",
      "value": -2.8999,
      "shap_value": 1.1095,
      "impact": "Pushes toward Fraud"
    },
    {
      "feature": "V4",
      "value": 3.9979,
      "shap_value": 0.8932,
      "impact": "Pushes toward Fraud"
    },
    {
      "feature": "Amount",
      "value": 149.62,
      "shap_value": -0.2145,
      "impact": "Pushes toward Legitimate"
    }
  ],
  "response_time_ms": 1.84
}
```

---

## 🚫 Why Accuracy Fails in Fraud Detection

In credit card fraud detection, **Accuracy is a dangerously misleading metric**:

### The 99.8% Accuracy Paradox
In the ULB Kaggle dataset:
- Total transactions: `284,807`
- Legitimate transactions: `284,315` (99.828%)
- Fraudulent transactions: `492` (0.172%)

Consider a **useless baseline classifier** that simply outputs:
$$\hat{y} = 0 \quad \text{(Predict every transaction as Legitimate)}$$

Its accuracy is:
$$\text{Accuracy} = \frac{284,315}{284,807} = 99.83\%$$

Despite a near-perfect 99.83% accuracy score:
- **Fraud Recall = 0.0%**: Not a single fraud is caught.
- **Financial Loss**: Millions of dollars stolen by fraudsters undetected.
- **Business Failure**: The bank incurs catastrophic chargeback liabilities.

Therefore, this project **explicitly excludes Accuracy** from model selection, relying strictly on **PR-AUC, Recall, Precision, and F1-Score**.

---

## 📐 Evaluation Metrics Deep-Dive

| Metric | Formula | Financial / Operational Interpretation |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | **Fraud Catch Rate**: Fraction of actual frauds intercepted. Critical to avoid fraud losses. |
| **Precision** | $\frac{TP}{TP + FP}$ | **Flag Accuracy**: Fraction of flagged transactions that were actually fraudulent. Low precision frustrates innocent cardholders. |
| **F1-Score** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Harmonic mean balancing fraud detection rate against false alarms. |
| **PR-AUC (Average Precision)** | $\int Precision(R) \, dR$ | **The Gold Standard for Imbalanced Data**: Summarizes precision-recall trade-off across all thresholds without being skewed by millions of True Negatives. |
| **ROC-AUC** | $\int TPR(FPR) \, dFPR$ | Measures separability, but can present an overly optimistic score because a huge count of $TN$ suppresses $FPR = \frac{FP}{TN + FP}$. |
| **False Positive Rate (FPR)** | $\frac{FP}{TN + FP}$ | Fraction of legitimate cardholders falsely flagged or inconvenienced. |

---

## 🎯 Threshold Tuning & Operational Tiers

Standard classification algorithms use a default threshold of $0.50$. In fraud detection, $0.50$ is rarely optimal because missing a fraud (False Negative) is substantially more costly than a brief verification review (False Positive).

### Threshold Calibration Strategy
Using `src/evaluate.py`:
1. The Precision-Recall curve is computed on the hold-out test set.
2. We identify candidate thresholds that guarantee **Recall $\ge 0.85$** (catching at least 85% of frauds).
3. Among qualifying candidates, we choose the threshold that **maximizes F1-Score**.
4. The calibrated threshold $\tau$ is persisted to `models/threshold.json`.

### Multi-Tier Operational Routing
To balance security and cardholder experience, transactions are partitioned into three action tiers:

```
0.00 ──────────── [ 0.40 ] ──────────── [ τ = Tuned Threshold ] ──────────── 1.00
     APPROVE               REVIEW                                 BLOCK
 (Seamless Flow)     (Step-Up Auth / OTP)                 (Decline & Lockdown)
```

1. **APPROVE ($p < 0.40$)**: Low risk. Immediate authorization.
2. **REVIEW ($0.40 \le p < \tau$)**: Borderline risk. Triggers step-up authentication (3D-Secure, SMS OTP, push confirmation).
3. **BLOCK ($p \ge \tau$)**: High risk. Direct decline with real-time fraud alert.

---

## 🔍 Model Explainability with SHAP

Modern financial regulations (e.g., GDPR Article 22, US Fair Credit Reporting Act) require explainability and auditability for automated financial decisions:

1. **Global Attribution (`reports/shap_summary.png`)**:
   - Uses `shap.TreeExplainer` on the champion XGBoost model.
   - Highlights overall driving latent features (typically `V14`, `V10`, `V12`, `V4`, `Amount`).
2. **Local Attribution (`api/main.py` & `app/dashboard.py`)**:
   - For every single evaluated transaction, extracts the **top 5 features** that pushed the transaction score higher (toward Fraud) or lower (toward Legitimate).
   - Equips fraud analysts with immediate evidentiary context to adjudicate disputes within seconds.

---

## ⚠️ Known Limitations in Production Systems

When transitioning an academic fraud model into enterprise banking operations, engineers must address three fundamental realities:

### 1. Concept Drift & Adversarial Adaptation
- Fraudsters actively study merchant risk rules and change their behavior (e.g., testing stolen cards with micro-transactions, rotating IP proxies, spoofing device fingerprints).
- **Production Solution**: Continuous model monitoring with weekly retraining pipelines, drift detection (PSI / Kolmogorov-Smirnov tests), and dynamic feature stores.

### 2. Resampled Data Probability Calibration
- Techniques like SMOTE and Random Undersampling artificially alter the training prior class distribution:
  $$P_{\text{train}}(y=1) \gg P_{\text{real}}(y=1)$$
- Uncalibrated tree models trained on SMOTE data can output uncalibrated posteriors.
- **Production Solution**: Always evaluate metrics and tune decision thresholds on the **untouched, genuine test set**. In production, apply Platt Scaling or Isotonic Regression on hold-out genuine validation data.

### 3. Real-Time Latency Target (< 50ms)
- Payment card switches (Visa/Mastercard) enforce strict SLA budgets: the entire fraud scoring pipeline (network traversal + feature lookup + preprocessing + model inference + explainability) must return within **50 to 100 milliseconds**.
- **Production Solution**: Cache pre-fitted scalers in memory, leverage compiled tree runtimes (e.g., ONNX Runtime or Treelite), and compute local SHAP explanations asynchronously or on review queues.
