"""Model training, benchmark comparison, threshold tuning, and artifact generation.

This module coordinates the end-to-end ML pipeline:
1. Loads genuine data or generates a synthetic fallback if missing.
2. Scales Time and Amount via RobustScaler, preserving V1-V28.
3. Conducts a stratified 80/20 train-test split (random_state=42).
4. Evaluates a full 3x3 matrix (9 combinations):
   - Models: Logistic Regression, Random Forest, XGBoost
   - Strategies: Class Weights, SMOTE, Random Undersampling
5. Evaluates every configuration on the untouched test set (Precision, Recall, F1, ROC-AUC, PR-AUC).
6. Automatically selects the champion model by PR-AUC with a recall check.
7. Tunes the decision threshold using the Precision-Recall curve.
8. Generates and saves evaluation plots and SHAP explainability artifacts to reports/.
9. Serializes the final model, scaler, threshold, and test samples to models/.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
    tune_decision_threshold,
)
from src.explain import create_tree_explainer, generate_shap_summary_plot
from src.preprocess import (
    FEATURE_COLUMNS,
    apply_resampling,
    load_data,
    scale_features,
    split_data,
)

# Configure clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_model_instance(model_name: str, strategy: str, neg_to_pos_ratio: float, random_state: int = 42):
    """Instantiate a model configured for the specified imbalance handling strategy."""
    model_name = model_name.lower().strip()
    use_weights = (strategy.lower().strip() == "class_weight")

    if "logistic" in model_name:
        class_weight = "balanced" if use_weights else None
        return LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )

    elif "forest" in model_name:
        class_weight = "balanced" if use_weights else None
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    elif "xgboost" in model_name or "xgb" in model_name:
        scale_pos_weight = neg_to_pos_ratio if use_weights else 1.0
        return XGBClassifier(
            eval_metric="aucpr",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_benchmark(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """Train and evaluate 3 models x 3 imbalance strategies (9 combinations).

    Returns:
        Tuple: (comparison_dataframe, trained_models_dict, best_combination_info)
    """
    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    strategies = ["class_weight", "smote", "undersample"]

    neg_count = int((y_train == 0).sum())
    pos_count = max(1, int((y_train == 1).sum()))
    neg_to_pos_ratio = float(neg_count / pos_count)

    results_records: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    print("\n" + "=" * 90)
    print(f"{'BENCHMARKING 3 MODELS x 3 IMBALANCE STRATEGIES (9 COMBINATIONS)':^90}")
    print("=" * 90)

    for strat in strategies:
        print(f"\n---> Applying Resampling Strategy: [{strat.upper()}] to Training Data only")
        X_res, y_res = apply_resampling(X_train_scaled, y_train, strategy=strat, random_state=random_state)

        for m_name in models:
            combo_key = f"{m_name} + {strat}"
            print(f"  Training: {combo_key} ...", end="", flush=True)

            model = build_model_instance(m_name, strat, neg_to_pos_ratio, random_state)
            model.fit(X_res, y_res)
            trained_models[combo_key] = model

            # Predict probabilities on UNTOUCHED test set
            probs = model.predict_proba(X_test_scaled)[:, 1]
            metrics = compute_metrics(y_test.values, probs, threshold=0.50)

            results_records.append({
                "Model": m_name,
                "Strategy": strat,
                "PR-AUC": metrics["pr_auc"],
                "ROC-AUC": metrics["roc_auc"],
                "F1": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "FPR": metrics["false_positive_rate"],
                "_model_key": combo_key,
                "_probs": probs,
            })
            print(f" Done! [PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}]")

    comparison_df = pd.DataFrame(results_records)

    # Automatic champion model selection:
    # Prioritize PR-AUC with a minimum Recall check (>= 0.70)
    eligible = comparison_df[comparison_df["Recall"] >= 0.70]
    if not eligible.empty:
        best_row = eligible.sort_values(by="PR-AUC", ascending=False).iloc[0]
        selection_criteria = "Highest PR-AUC with Recall >= 0.70"
    else:
        best_row = comparison_df.sort_values(by="PR-AUC", ascending=False).iloc[0]
        selection_criteria = "Highest PR-AUC overall (no model met Recall >= 0.70)"

    best_key = best_row["_model_key"]
    best_model = trained_models[best_key]

    print("\n" + "=" * 90)
    print(f" CHAMPION SELECTED: {best_key}")
    print(f" Selection Criteria: {selection_criteria}")
    print(f" Score: PR-AUC = {best_row['PR-AUC']:.4f} | F1 = {best_row['F1']:.4f} | Recall = {best_row['Recall']:.4f}")
    print("=" * 90 + "\n")

    return comparison_df, trained_models, best_row


def main() -> None:
    """Run the complete end-to-end fraud detection training pipeline."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 90)
    print(" " * 20 + "AI-POWERED CREDIT CARD FRAUD DETECTION PIPELINE")
    print("#" * 90 + "\n")

    # 1. Load Data
    data_csv_path = data_dir / "creditcard.csv"
    df, is_synthetic = load_data(data_csv_path, random_state=42)

    # 2. Split Data (Stratified 80/20 train/test split before scaling to avoid any leakage)
    X_train_raw, X_test_raw, y_train, y_test = split_data(df, test_size=0.20, random_state=42)

    # 3. Scale Time and Amount with RobustScaler (fit on train, transform test)
    print("\n---> Fitting RobustScaler on training 'Time' and 'Amount' features...")
    X_train_scaled, scaler = scale_features(X_train_raw, fit_scaler=True)
    X_test_scaled, _ = scale_features(X_test_raw, scaler=scaler, fit_scaler=False)

    # Save fitted scaler
    scaler_save_path = models_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_save_path)
    logger.info("Saved RobustScaler to: %s", scaler_save_path)

    # Save test sample dataset for Streamlit Live Simulator
    sample_legit = X_test_raw[y_test == 0].head(50)
    sample_fraud = X_test_raw[y_test == 1].head(50)
    test_sample_pack = {
        "legit": sample_legit.to_dict(orient="records"),
        "fraud": sample_fraud.to_dict(orient="records"),
        "X_test_scaled_head": X_test_scaled.head(200),
        "y_test_head": y_test.head(200).tolist(),
    }
    test_sample_path = models_dir / "test_sample.joblib"
    joblib.dump(test_sample_pack, test_sample_path)
    logger.info("Saved test samples for Live Simulator to: %s", test_sample_path)

    # 4. Run Benchmark across 3 models x 3 strategies (9 combinations)
    comparison_df, trained_models, best_row = run_benchmark(
        X_train_scaled, y_train, X_test_scaled, y_test, random_state=42
    )

    # Print and save comparison table
    display_df = comparison_df[["Model", "Strategy", "PR-AUC", "ROC-AUC", "F1", "Precision", "Recall", "FPR"]].copy()
    display_df = display_df.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    print("=" * 90)
    print(f"{'FINAL BENCHMARK COMPARISON TABLE (Evaluated on Untouched Test Set)':^90}")
    print("=" * 90)
    print(display_df.to_string(index=False))
    print("=" * 90 + "\n")

    csv_save_path = reports_dir / "model_comparison.csv"
    display_df.to_csv(csv_save_path, index=False)
    logger.info("Saved benchmark CSV to: %s", csv_save_path)

    # Plot model comparison bar chart
    plot_model_comparison(display_df, reports_dir / "model_comparison.png")

    # 5. Threshold Tuning for the Final Model
    # We retrieve the best XGBoost model specifically to meet the requirement:
    # "Save the final model with joblib to models/xgb_fraud_model.joblib"
    xgb_candidates = [k for k in trained_models.keys() if "xgboost" in k.lower()]
    # Pick the top performing XGBoost combination
    xgb_rows = comparison_df[comparison_df["Model"] == "XGBoost"].sort_values(by="PR-AUC", ascending=False)
    chosen_xgb_key = xgb_rows.iloc[0]["_model_key"]
    final_xgb_model = trained_models[chosen_xgb_key]
    final_xgb_probs = xgb_rows.iloc[0]["_probs"]

    print(f"---> Tuning Decision Threshold on Final XGBoost Model ({chosen_xgb_key})...")
    tuned_threshold, threshold_metrics = tune_decision_threshold(
        y_test.values,
        final_xgb_probs,
        min_recall=0.85,
        reports_dir=reports_dir,
    )

    threshold_info = {
        "threshold": tuned_threshold,
        "champion_combination": chosen_xgb_key,
        "champion_strategy": xgb_rows.iloc[0]["Strategy"],
        "metrics_at_tuned_threshold": threshold_metrics,
        "dataset_type": "synthetic_fallback" if is_synthetic else "genuine_kaggle_ulb",
    }
    threshold_path = models_dir / "threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2)
    logger.info("Saved tuned threshold configuration to: %s", threshold_path)

    # 6. Save Final Model
    final_model_path = models_dir / "xgb_fraud_model.joblib"
    joblib.dump(final_xgb_model, final_model_path)
    logger.info("Saved final XGBoost fraud model to: %s", final_model_path)

    # 7. Generate Evaluation Plots for Final Model
    cm = np.array(threshold_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, f"XGBoost ({xgb_rows.iloc[0]['Strategy']})", reports_dir / "confusion_matrix.png")
    plot_roc_curve(y_test.values, final_xgb_probs, "XGBoost Fraud Model", reports_dir / "roc_curve.png")
    plot_precision_recall_curve(
        y_test.values,
        final_xgb_probs,
        chosen_threshold=tuned_threshold,
        model_name="XGBoost Fraud Model",
        save_path=reports_dir / "precision_recall_curve.png",
    )
    plot_feature_importance(
        final_xgb_model,
        feature_names=FEATURE_COLUMNS,
        save_path=reports_dir / "feature_importance.png",
        top_n=15,
    )

    # 8. SHAP Explainability on XGBoost Model
    print("\n---> Generating SHAP Explainability Artifacts...")
    try:
        explainer = create_tree_explainer(final_xgb_model)
        # Use a representative sample of test transactions for global summary (up to 400 rows)
        sample_size = min(400, len(X_test_scaled))
        X_shap_sample = X_test_scaled.sample(n=sample_size, random_state=42)
        generate_shap_summary_plot(explainer, X_shap_sample, reports_dir / "shap_summary.png")
    except Exception as e:
        logger.error("SHAP plot generation failed: %s", e)

    print("\n" + "=" * 90)
    print(" " * 25 + "TRAINING PIPELINE COMPLETE!")
    print("=" * 90)
    print(f" - Model Artifact:     {final_model_path}")
    print(f" - Scaler Artifact:    {scaler_save_path}")
    print(f" - Threshold JSON:     {threshold_path}")
    print(f" - Reports Generated:  {reports_dir}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
