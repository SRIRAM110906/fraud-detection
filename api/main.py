"""FastAPI Real-Time Fraud Detection Inference Service.

Endpoints:
- POST /predict: Evaluates transaction risk, assigns decision tier, and explains via SHAP.
- GET  /health:  Verifies model and service availability.
- GET  /:        API metadata and interactive documentation reference.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import FraudPredictor, get_predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Predictor instance holder
predictor_instance: Optional[FraudPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload machine learning artifacts on server startup for ultra-low latency."""
    global predictor_instance
    try:
        predictor_instance = get_predictor()
        logger.info("FastAPI successfully initialized FraudPredictor in memory.")
    except Exception as e:
        logger.warning(
            "Predictor could not be preloaded at startup (pipeline may need to be trained first): %s",
            e,
        )
        predictor_instance = None
    yield
    # Cleanup on shutdown if needed
    predictor_instance = None


app = FastAPI(
    title="AI-Powered Credit Card Fraud Detection API",
    description=(
        "Ultra-low latency real-time scoring engine powered by XGBoost, "
        "RobustScaler, and SHAP TreeExplainer. Evaluates transaction risk "
        "and categorizes transactions into APPROVE, REVIEW, and BLOCK actions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend dashboard and cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class TransactionInput(BaseModel):
    """Credit card transaction schema with PCA components, elapsed time, and amount."""
    Time: float = Field(
        ...,
        description="Elapsed seconds between this transaction and the first transaction in dataset",
        examples=[406.0],
    )
    V1: float = Field(..., description="PCA Component 1", examples=[-2.3122])
    V2: float = Field(..., description="PCA Component 2", examples=[1.9519])
    V3: float = Field(..., description="PCA Component 3", examples=[-1.6098])
    V4: float = Field(..., description="PCA Component 4", examples=[3.9979])
    V5: float = Field(..., description="PCA Component 5", examples=[-0.5221])
    V6: float = Field(..., description="PCA Component 6", examples=[-1.4265])
    V7: float = Field(..., description="PCA Component 7", examples=[-2.5373])
    V8: float = Field(..., description="PCA Component 8", examples=[1.3916])
    V9: float = Field(..., description="PCA Component 9", examples=[-2.7700])
    V10: float = Field(..., description="PCA Component 10", examples=[-2.7722])
    V11: float = Field(..., description="PCA Component 11", examples=[3.2020])
    V12: float = Field(..., description="PCA Component 12", examples=[-2.8999])
    V13: float = Field(..., description="PCA Component 13", examples=[-0.5952])
    V14: float = Field(..., description="PCA Component 14", examples=[-4.2892])
    V15: float = Field(..., description="PCA Component 15", examples=[0.3897])
    V16: float = Field(..., description="PCA Component 16", examples=[-1.1407])
    V17: float = Field(..., description="PCA Component 17", examples=[-2.8300])
    V18: float = Field(..., description="PCA Component 18", examples=[-0.0168])
    V19: float = Field(..., description="PCA Component 19", examples=[0.4169])
    V20: float = Field(..., description="PCA Component 20", examples=[0.1269])
    V21: float = Field(..., description="PCA Component 21", examples=[0.5172])
    V22: float = Field(..., description="PCA Component 22", examples=[-0.0350])
    V23: float = Field(..., description="PCA Component 23", examples=[-0.4652])
    V24: float = Field(..., description="PCA Component 24", examples=[0.3201])
    V25: float = Field(..., description="PCA Component 25", examples=[0.0445])
    V26: float = Field(..., description="PCA Component 26", examples=[0.1778])
    V27: float = Field(..., description="PCA Component 27", examples=[0.2611])
    V28: float = Field(..., description="PCA Component 28", examples=[-0.1432])
    Amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount in currency units",
        examples=[149.62],
    )


class ShapFeature(BaseModel):
    """Local SHAP feature attribution element."""
    feature: str = Field(..., description="Feature identifier (e.g. V14, Amount)")
    value: float = Field(..., description="Feature value in the transaction")
    shap_value: float = Field(..., description="SHAP score contribution")
    impact: str = Field(..., description="'Pushes toward Fraud' or 'Pushes toward Legitimate'")


class PredictionResponse(BaseModel):
    """Standard response returned by /predict endpoint."""
    fraud_probability: float = Field(..., description="Estimated probability of fraud (0.0 to 1.0)")
    decision: str = Field(..., description="Actionable decision: 'APPROVE', 'REVIEW', or 'BLOCK'")
    tuned_threshold: float = Field(..., description="Calibrated operational decision threshold")
    top_5_shap_features: List[ShapFeature] = Field(..., description="Top 5 feature drivers from SHAP")
    response_time_ms: float = Field(..., description="End-to-end inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Health check payload."""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    threshold: Optional[float]
    model_type: str
    details: Dict[str, Any]


# ==============================================================================
# Route Handlers
# ==============================================================================

@app.get("/", tags=["Info"])
def root() -> Dict[str, Any]:
    """Root info endpoint."""
    return {
        "service": "AI-Powered Credit Card Fraud Detection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Verify service health and model availability."""
    global predictor_instance
    if predictor_instance is None:
        try:
            predictor_instance = get_predictor()
        except Exception:
            predictor_instance = None

    is_ready = predictor_instance is not None and predictor_instance.model is not None

    return HealthResponse(
        status="healthy" if is_ready else "degraded (models not loaded - run training first)",
        model_loaded=is_ready,
        scaler_loaded=is_ready and (predictor_instance.scaler is not None),
        threshold=predictor_instance.threshold if is_ready else None,
        model_type="XGBoost Classifier" if is_ready else "None",
        details={
            "threshold_metadata": predictor_instance.threshold_metadata if is_ready else {},
        },
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict_fraud(transaction: TransactionInput) -> PredictionResponse:
    """Score a transaction in real-time, determine operational tier, and explain via SHAP."""
    global predictor_instance

    if predictor_instance is None:
        try:
            predictor_instance = get_predictor()
        except FileNotFoundError as fnf:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(fnf),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference initialization failed: {e}",
            )

    try:
        transaction_dict = transaction.model_dump()
        result = predictor_instance.predict(transaction_dict)
        return PredictionResponse(**result)

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Input validation error: {ve}",
        )
    except Exception as e:
        logger.error("Scoring error encountered: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring pipeline execution error: {e}",
        )
