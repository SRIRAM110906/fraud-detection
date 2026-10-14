"""SHAP (SHapley Additive exPlanations) model explainability.

This module provides:
1. TreeExplainer initialization for XGBoost.
2. Global SHAP summary plot generation saved to reports/shap_summary.png.
3. Local transaction explainability extracting the top 5 driving features
   (pushing toward fraud vs. pushing toward legitimate).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def create_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Initialize a SHAP TreeExplainer for tree-based models like XGBoost."""
    try:
        explainer = shap.TreeExplainer(model)
        logger.info("Initialized shap.TreeExplainer successfully.")
        return explainer
    except Exception as e:
        logger.error("Failed to initialize TreeExplainer: %s", e)
        raise


def generate_shap_summary_plot(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
    save_path: Union[str, Path],
    max_display: int = 15,
) -> None:
    """Generate and save a global SHAP beeswarm / summary plot.

    Args:
        explainer: Fitted TreeExplainer instance.
        X_sample: Representative sample of features (e.g. 300-500 rows).
        save_path: Path where the PNG image will be written.
        max_display: Maximum number of top features to display.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Computing SHAP values for global summary plot on %d samples...", len(X_sample))
    shap_values = explainer(X_sample)

    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.title("Global Feature Attribution (SHAP Summary)", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Global SHAP summary plot saved to: %s", save_path)


def explain_single_transaction(
    explainer: shap.TreeExplainer,
    transaction_df: pd.DataFrame,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Explain a single transaction using SHAP local feature attribution.

    Determines the top N features that exerted the greatest influence on the model's
    fraud score for this specific transaction.

    Args:
        explainer: Fitted TreeExplainer.
        transaction_df: Single row DataFrame (1, 30) with all feature columns.
        top_n: Number of most impactful features to return (default: 5).

    Returns:
        List[Dict[str, Any]]: Top features ordered by absolute SHAP impact:
            - feature: Name of the feature (e.g. 'V14', 'Amount')
            - value: Actual value of the feature for this transaction
            - shap_value: SHAP contribution (positive pushes toward Fraud, negative toward Legit)
            - impact: Human-readable interpretation string
    """
    shap_explanation = explainer(transaction_df)

    # Handle shape differences between raw numpy values and Explanation object
    if hasattr(shap_explanation, "values"):
        vals = shap_explanation.values
        if vals.ndim == 2:
            single_shap = vals[0]
        elif vals.ndim == 3:
            # Multi-class output: binary positive class is at index 1
            single_shap = vals[0, :, 1]
        else:
            single_shap = vals.flatten()
    else:
        single_shap = np.array(shap_explanation).flatten()

    feature_names = list(transaction_df.columns)
    raw_values = transaction_df.iloc[0].values

    # Pair features with their SHAP values and raw values
    attributions = []
    for f_name, r_val, s_val in zip(feature_names, raw_values, single_shap):
        s_float = float(s_val)
        r_float = float(r_val)
        impact = "Pushes toward Fraud" if s_float > 0 else "Pushes toward Legitimate"
        attributions.append({
            "feature": str(f_name),
            "value": round(r_float, 4),
            "shap_value": round(s_float, 4),
            "abs_impact": abs(s_float),
            "impact": impact,
        })

    # Sort descending by absolute impact
    attributions.sort(key=lambda x: x["abs_impact"], reverse=True)
    top_features = attributions[:top_n]

    # Clean up internal sorting key before returning
    for item in top_features:
        item.pop("abs_impact", None)

    return top_features
