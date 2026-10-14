"""Evaluation metrics, threshold tuning, and reporting plots.

This module provides:
1. Computation of Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
2. Threshold tuning using the Precision-Recall curve to maximize F1 or meet target recall.
3. Visualization functions saved to the reports/ directory.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate key fraud detection metrics for a given classification threshold.

    Why accuracy is NOT included in decision metrics:
    In a dataset where 99.8% of transactions are legitimate, a naive model that predicts
    ALL transactions as legitimate achieves 99.8% accuracy while detecting ZERO frauds.
    Instead, we rely on PR-AUC, Recall, Precision, and F1.

    Args:
        y_true: Binary ground truth labels (0 = Legitimate, 1 = Fraud).
        y_probs: Predicted probabilities for the positive fraud class.
        threshold: Classification decision boundary (default: 0.50).

    Returns:
        Dict[str, Any]: Dictionary containing precision, recall, f1, roc_auc, pr_auc,
                        confusion_matrix, and false_positive_rate.
    """
    y_pred = (y_probs >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "false_positive_rate": round(fpr, 6),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    min_recall: float = 0.85,
    reports_dir: Optional[Path] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Tune classification threshold using Precision-Recall curve.

    Finds the optimal threshold that maximizes F1 score while prioritizing
    a recall threshold of at least min_recall (e.g. 0.85) to catch critical frauds.

    Args:
        y_true: True binary targets.
        y_probs: Predicted probabilities of fraud.
        min_recall: Minimum required recall target (default: 0.85).
        reports_dir: Optional directory to save threshold tuning summary.

    Returns:
        Tuple[float, Dict[str, Any]]: (Chosen threshold, evaluation metrics dict)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    # Avoid division by zero
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)

    # Note: precision_recall_curve returns thresholds with length len(precisions) - 1
    # Match lengths
    p_thresh = precisions[:-1]
    r_thresh = recalls[:-1]
    f1_thresh = f1_scores[:-1]

    # Find candidates meeting minimum recall requirement
    qualifying_indices = np.where(r_thresh >= min_recall)[0]

    if len(qualifying_indices) > 0:
        # Among qualifying thresholds, pick the one maximizing F1 (best balance)
        best_sub_idx = np.argmax(f1_thresh[qualifying_indices])
        best_idx = qualifying_indices[best_sub_idx]
        chosen_threshold = float(thresholds[best_idx])
        selection_reason = f"Maximized F1 among candidates with Recall >= {min_recall}"
    else:
        # Fall back to overall maximum F1 score
        best_idx = int(np.argmax(f1_thresh))
        chosen_threshold = float(thresholds[best_idx])
        selection_reason = f"Global maximum F1 score (no threshold reached target recall {min_recall})"

    # Ensure threshold is within reasonable bounds [0.05, 0.95]
    chosen_threshold = float(np.clip(chosen_threshold, 0.05, 0.95))
    metrics = compute_metrics(y_true, y_probs, threshold=chosen_threshold)
    metrics["selection_reason"] = selection_reason
    metrics["target_recall_constraint"] = min_recall

    logger.info(
        "Tuned Decision Threshold: %.4f | Recall: %.4f | Precision: %.4f | F1: %.4f | Reason: %s",
        chosen_threshold,
        metrics["recall"],
        metrics["precision"],
        metrics["f1"],
        selection_reason,
    )

    return chosen_threshold, metrics


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: Path,
) -> None:
    """Plot an annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax)

    labels = [
        ["True Negatives\n(Legit Approved)", "False Positives\n(Legit Flagged)"],
        ["False Negatives\n(Fraud Missed)", "True Positives\n(Fraud Caught)"],
    ]

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            text = f"{labels[i][j]}\n{count:,}"
            color = "white" if count > (cm.sum() * 0.4) else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, weight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Legit (0)", "Pred Fraud (1)"], fontsize=10)
    ax.set_yticklabels(["Actual Legit (0)", "Actual Fraud (1)"], fontsize=10)
    ax.set_title(f"Confusion Matrix - {model_name}", pad=20, fontsize=12, weight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix plot: %s", save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    model_name: str,
    save_path: Path,
) -> None:
    """Plot Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"{model_name} (ROC-AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Classifier (0.50)")

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / TPR)", fontsize=11)
    plt.title(f"ROC Curve - {model_name}", fontsize=13, weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved ROC curve plot: %s", save_path)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    chosen_threshold: float,
    model_name: str,
    save_path: Path,
) -> None:
    """Plot Precision-Recall curve with tuned threshold marker.

    Crucial for highly imbalanced fraud datasets because it focuses entirely
    on the rare positive class (frauds), unlike ROC-AUC which can be artificially
    inflated by large numbers of True Negatives.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)

    # Find closest point to chosen threshold
    closest_idx = np.argmin(np.abs(thresholds - chosen_threshold)) if len(thresholds) > 0 else 0
    t_prec = precisions[closest_idx] if len(precisions) > closest_idx else 0.0
    t_rec = recalls[closest_idx] if len(recalls) > closest_idx else 0.0

    plt.figure(figsize=(7, 5.5))
    plt.plot(recalls, precisions, color="#059669", lw=2.5, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    plt.scatter(
        [t_rec],
        [t_prec],
        color="#DC2626",
        s=120,
        zorder=5,
        label=f"Tuned Threshold ({chosen_threshold:.2f})\n[Prec: {t_prec:.2f}, Rec: {t_rec:.2f}]",
    )

    baseline = float(y_true.sum() / len(y_true))
    plt.axhline(baseline, color="#94A3B8", linestyle="--", label=f"Random Guess Baseline ({baseline:.4f})")

    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Fraud Catch Rate)", fontsize=11)
    plt.ylabel("Precision (Flagged Transactions Accuracy)", fontsize=11)
    plt.title(f"Precision-Recall Curve - {model_name}", fontsize=13, weight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved PR curve plot: %s", save_path)


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    save_path: Path,
    top_n: int = 15,
) -> None:
    """Plot feature importance ranking for tree models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.warning("Model does not expose feature_importances_. Skipping plot.")
        return

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importances[::-1], color="#4F46E5", align="center", alpha=0.85)
    plt.yticks(y_pos, top_features[::-1], fontsize=10)
    plt.xlabel("Relative Feature Importance (Gini / Gain)", fontsize=11)
    plt.title(f"Top {top_n} Most Predictive Fraud Features", fontsize=13, weight="bold", pad=12)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved feature importance plot: %s", save_path)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Plot comparative benchmark bar chart across all model and strategy combinations."""
    plt.figure(figsize=(11, 6))
    models_labels = comparison_df["Model"] + " (" + comparison_df["Strategy"] + ")"
    x = np.arange(len(models_labels))
    width = 0.25

    pr_auc_scores = comparison_df["PR-AUC"]
    f1_scores = comparison_df["F1"]
    recall_scores = comparison_df["Recall"]

    plt.bar(x - width, pr_auc_scores, width, label="PR-AUC", color="#2563EB", alpha=0.9)
    plt.bar(x, f1_scores, width, label="F1-Score", color="#10B981", alpha=0.9)
    plt.bar(x + width, recall_scores, width, label="Recall", color="#F59E0B", alpha=0.9)

    plt.xticks(x, models_labels, rotation=35, ha="right", fontsize=9)
    plt.ylim([0.0, 1.08])
    plt.ylabel("Score", fontsize=11)
    plt.title("Model & Imbalance Strategy Benchmark Comparison", fontsize=13, weight="bold", pad=15)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved model comparison plot: %s", save_path)
