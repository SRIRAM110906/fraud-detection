"""Real-time transaction scoring and risk decision engine.

This module provides the FraudPredictor class, which:
1. Loads the trained model, RobustScaler, and tuned decision threshold.
2. Scales raw transaction inputs (Time and Amount) while keeping V1-V28 intact.
3. Computes the fraud probability.
4. Categorizes transactions into actionable operational decisions:
   - APPROVE: fraud_probability < 0.40
   - REVIEW:  0.40 <= fraud_probability < tuned_threshold
   - BLOCK:   fraud_probability >= tuned_threshold
5. Generates local SHAP feature attributions for regulatory auditability and compliance.
6. Measures end-to-end response latency in milliseconds.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.explain import create_tree_explainer, explain_single_transaction
from src.preprocess import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Production scoring engine for credit card fraud detection."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        threshold_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize predictor by loading serialized artifacts from models/ directory."""
        base_dir = Path(__file__).resolve().parent.parent
        self.model_path = Path(model_path) if model_path else base_dir / "models" / "xgb_fraud_model.joblib"
        self.scaler_path = Path(scaler_path) if scaler_path else base_dir / "models" / "scaler.joblib"
        self.threshold_path = Path(threshold_path) if threshold_path else base_dir / "models" / "threshold.json"

        self.model = None
        self.scaler = None
        self.threshold: float = 0.50
        self.threshold_metadata: Dict[str, Any] = {}
        self.explainer = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model, scaler, and threshold metadata from disk into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}. "
                "Please run training first: 'python -m src.train'"
            )

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found at: {self.scaler_path}. "
                "Please run training first: 'python -m src.train'"
            )

        logger.info("Loading model from: %s", self.model_path)
        self.model = joblib.load(self.model_path)

        logger.info("Loading RobustScaler from: %s", self.scaler_path)
        self.scaler = joblib.load(self.scaler_path)

        if self.threshold_path.exists():
            try:
                with open(self.threshold_path, "r", encoding="utf-8") as f:
                    self.threshold_metadata = json.load(f)
                    self.threshold = float(self.threshold_metadata.get("threshold", 0.50))
                logger.info("Loaded tuned decision threshold: %.4f", self.threshold)
            except Exception as e:
                logger.warning("Could not read threshold JSON: %s. Using default 0.50.", e)
                self.threshold = 0.50
        else:
            logger.warning("Threshold JSON not found. Defaulting to 0.50.")
            self.threshold = 0.50

        # Initialize SHAP TreeExplainer for local feature attribution
        try:
            self.explainer = create_tree_explainer(self.model)
        except Exception as e:
            logger.warning("SHAP explainer initialization deferred or failed: %s", e)
            self.explainer = None

    def make_decision(self, probability: float) -> str:
        """Classify transaction risk into business tiers:

        - APPROVE: probability < min(0.40, tuned_threshold)
        - REVIEW:  min(0.40, tuned_threshold) <= probability < max(0.40, tuned_threshold)
        - BLOCK:   probability >= max(0.40, tuned_threshold)

        Ensures clean tier boundaries even if tuned threshold varies.
        """
        lower_bound = min(0.40, self.threshold)
        upper_bound = max(0.40, self.threshold)

        if probability < lower_bound:
            return "APPROVE"
        elif probability < upper_bound:
            return "REVIEW"
        else:
            return "BLOCK"

    def predict(self, transaction: Dict[str, Union[float, int]]) -> Dict[str, Any]:
        """Score a single transaction and return probability, decision, and SHAP drivers.

        Args:
            transaction: Dictionary containing 'Time', 'V1' through 'V28', and 'Amount'.

        Returns:
            Dict[str, Any]: Scoring outcome with latency in milliseconds.
        """
        start_time = time.perf_counter()

        # Validate input features
        missing_features = [f for f in FEATURE_COLUMNS if f not in transaction]
        if missing_features:
            raise ValueError(f"Transaction missing required features: {missing_features}")

        # Assemble DataFrame with strict feature order
        df_raw = pd.DataFrame([{col: float(transaction[col]) for col in FEATURE_COLUMNS}])

        # Robust-scale Time and Amount using the training scaler
        df_scaled = df_raw.copy()
        df_scaled[["Time", "Amount"]] = self.scaler.transform(df_scaled[["Time", "Amount"]])

        # Model inference
        prob_array = self.model.predict_proba(df_scaled)
        fraud_prob = float(prob_array[0, 1])

        # Business decision tier
        decision = self.make_decision(fraud_prob)

        # Local SHAP attribution
        top_5_features = []
        if self.explainer is not None:
            try:
                top_5_features = explain_single_transaction(self.explainer, df_scaled, top_n=5)
            except Exception as e:
                logger.warning("Failed to calculate SHAP explanation: %s", e)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "fraud_probability": round(fraud_prob, 4),
            "decision": decision,
            "tuned_threshold": round(self.threshold, 4),
            "top_5_shap_features": top_5_features,
            "response_time_ms": latency_ms,
        }


# Module-level singleton predictor for fast reuse in API / Dashboard
_cached_predictor: Optional[FraudPredictor] = None


def get_predictor() -> FraudPredictor:
    """Retrieve or initialize the singleton FraudPredictor instance."""
    global _cached_predictor
    if _cached_predictor is None:
        _cached_predictor = FraudPredictor()
    return _cached_predictor


def score_transaction(transaction: Dict[str, Union[float, int]]) -> Dict[str, Any]:
    """Convenience functional interface to score a transaction dictionary."""
    predictor = get_predictor()
    return predictor.predict(transaction)
