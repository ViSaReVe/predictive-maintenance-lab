"""
FastAPI inference service for Predictive Maintenance Lab.

Mirrors the pattern from reference MLOps repo:
  GET  /            → health check
  POST /train       → trigger training pipeline
  POST /predict/rul → XGBoost RUL prediction from sensor windows
  POST /predict/anomaly → Isolation Forest anomaly score from feature vector

Models are loaded from the MLflow Model Registry at startup.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient

from api.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    RULRequest,
    RULResponse,
    TrainRequest,
    TrainResponse,
)

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
CMAPSS_MODEL_NAME = "pdm-cmapss-rul-xgboost"
CWRU_MODEL_NAME = "pdm-cwru-anomaly-isoforest"

# ── Model cache (loaded once at startup) ──────────────────────────────────────
_models: dict = {}


def _load_model(name: str, as_sklearn: bool = False) -> Optional[object]:
    """Load model from MLflow registry — tries @champion alias, then latest version."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    loader = mlflow.sklearn.load_model if as_sklearn else mlflow.pyfunc.load_model
    for uri in [f"models:/{name}@champion", f"models:/{name}/latest"]:
        try:
            model = loader(uri)
            logger.info(f"Loaded model '{name}' from {uri}")
            return model
        except Exception as e:
            logger.debug(f"Could not load from {uri}: {e}")
    logger.warning(f"Model '{name}' not found in registry — predictions will fail")
    return None


def _get_model_version(name: str) -> Optional[str]:
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"name='{name}'")
        return str(versions[0].version) if versions else None
    except Exception:
        return None


def _get_cwru_threshold() -> float:
    """Retrieve the deployed anomaly threshold logged as a param in the champion run."""
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"name='{CWRU_MODEL_NAME}'")
        if not versions:
            return 0.0
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        threshold = run.data.params.get("deployed_threshold")
        if threshold is not None:
            return float(threshold)
    except Exception as e:
        logger.warning(f"Could not load CWRU threshold: {e}")
    return 0.0


# ── Lifespan: load models on startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models from MLflow registry...")
    _models["cmapss"] = _load_model(CMAPSS_MODEL_NAME)
    _models["cwru"] = _load_model(CWRU_MODEL_NAME, as_sklearn=True)
    # Cache model versions + CWRU anomaly threshold for response metadata
    _models["cmapss_version"] = _get_model_version(CMAPSS_MODEL_NAME)
    _models["cwru_version"] = _get_model_version(CWRU_MODEL_NAME)
    _models["cwru_threshold"] = _get_cwru_threshold()
    logger.info(f"CWRU anomaly threshold: {_models['cwru_threshold']:.6f}")
    logger.info("Models loaded. API ready.")
    yield
    _models.clear()


# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "End-to-end PdM inference service: RUL prediction (CMAPSS) + "
        "bearing anomaly detection (CWRU). Models served from MLflow registry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def health():
    """Health check — shows which models are loaded."""
    return {
        "status": "ok",
        "endpoints": ["/predict/rul", "/predict/anomaly", "/train"],
        "models": {
            "cmapss_rul": "loaded" if _models.get("cmapss") else "not_loaded",
            "cwru_anomaly": "loaded" if _models.get("cwru") else "not_loaded",
        },
    }


@app.post("/predict/rul", response_model=RULResponse, tags=["inference"])
def predict_rul(req: RULRequest):
    """
    Predict Remaining Useful Life for one or more sensor windows.

    Each window should be a flat list of (window_size × n_sensors) values.
    The model was trained with window_size=30 and 17 sensors → 510 values per window.
    """
    model = _models.get("cmapss")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="RUL model not loaded. Run POST /train first, then restart the API.",
        )

    try:
        X = np.array(req.windows, dtype=np.float32)
        predictions = model.predict(X).tolist()
        # Clip to valid RUL range
        predictions = [max(0.0, float(p)) for p in predictions]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

    return RULResponse(
        engine_id=req.engine_id,
        rul_predictions=predictions,
        model_name=CMAPSS_MODEL_NAME,
        model_version=_models.get("cmapss_version"),
    )


@app.post("/predict/anomaly", response_model=AnomalyResponse, tags=["inference"])
def predict_anomaly(req: AnomalyRequest):
    """
    Score a bearing vibration window for anomaly.

    Input: feature vector [rms, kurtosis, peak_to_peak, spectral_entropy]
    Output: anomaly score (higher = more anomalous) + binary is_anomaly flag.

    The threshold is the 95th percentile of training scores, stored as a
    logged param in MLflow at training time.
    """
    model = _models.get("cwru")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Anomaly model not loaded. Run POST /train first, then restart the API.",
        )

    try:
        X = np.array([req.features], dtype=np.float32)
        # Negate decision_function: higher score = more anomalous
        raw_score = float(-model.decision_function(X)[0])
        # Retrieve threshold from cached model metadata (fallback: 0.0)
        threshold = _models.get("cwru_threshold", 0.0)
        is_anomaly = raw_score > threshold
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Scoring failed: {e}")

    return AnomalyResponse(
        anomaly_score=round(raw_score, 6),
        is_anomaly=is_anomaly,
        threshold=threshold,
        model_name=CWRU_MODEL_NAME,
        model_version=_models.get("cwru_version"),
    )


@app.post("/train", response_model=TrainResponse, tags=["training"])
def train(req: TrainRequest):
    """
    Trigger the full training pipeline (CMAPSS + CWRU).
    Logs experiments to MLflow, registers champion models.

    Note: training is synchronous and may take 1-3 minutes.
    Refresh the API (restart uvicorn) after training to reload the new champion models.
    """
    from pipelines.train import run as run_pipeline

    try:
        summary = run_pipeline(
            cmapss_config=req.cmapss_config,
            cwru_config=req.cwru_config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training pipeline failed: {e}")

    return TrainResponse(
        success=True,
        run_timestamp=summary["run_timestamp"],
        cmapss_rul=summary["cmapss_rul"],
        cwru_anomaly=summary["cwru_anomaly"],
        message="Training complete. View experiments at http://localhost:5000",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
