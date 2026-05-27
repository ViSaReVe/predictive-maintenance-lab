"""
FastAPI inference service for Predictive Maintenance Lab.

Mirrors the pattern from reference MLOps repo:
  GET  /            → health check
  POST /train       → trigger training pipeline
  POST /predict/rul → XGBoost RUL prediction from sensor windows
  POST /predict/anomaly → Isolation Forest anomaly score from feature vector

Models are loaded from the MLflow Model Registry at startup.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
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
# MODEL_DIR: set this env var in cloud to load baked-in joblib files.
# Falls back to MLflow registry for local development.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
CMAPSS_MODEL_NAME = "pdm-cmapss-rul-xgboost"
CWRU_MODEL_NAME = "pdm-cwru-anomaly-isoforest"

# ── Model cache (loaded once at startup) ──────────────────────────────────────
_models: dict = {}


def _load_from_files() -> bool:
    """Try loading models from joblib files in MODEL_DIR. Returns True on success."""
    cmapss_path = Path(MODEL_DIR) / "cmapss_rul_xgboost.joblib"
    cwru_path = Path(MODEL_DIR) / "cwru_anomaly_isoforest.joblib"
    threshold_path = Path(MODEL_DIR) / "cwru_threshold.json"
    if not (cmapss_path.exists() and cwru_path.exists()):
        return False
    _models["cmapss"] = joblib.load(cmapss_path)
    _models["cwru"] = joblib.load(cwru_path)
    _models["cmapss_version"] = "file"
    _models["cwru_version"] = "file"
    _models["cwru_threshold"] = (
        json.loads(threshold_path.read_text())["threshold"]
        if threshold_path.exists()
        else 0.0
    )
    logger.info(f"Loaded models from {MODEL_DIR}/")
    return True


def _load_from_mlflow() -> None:
    """Load models from MLflow registry (local dev path)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def _load(name: str, as_sklearn: bool = False) -> Optional[object]:
        loader = mlflow.sklearn.load_model if as_sklearn else mlflow.pyfunc.load_model
        for uri in [f"models:/{name}@champion", f"models:/{name}/latest"]:
            try:
                m = loader(uri)
                logger.info(f"Loaded '{name}' from {uri}")
                return m
            except Exception as e:
                logger.debug(f"Could not load {uri}: {e}")
        logger.warning(f"Model '{name}' not found — predictions will fail")
        return None

    _models["cmapss"] = _load(CMAPSS_MODEL_NAME)
    _models["cwru"] = _load(CWRU_MODEL_NAME, as_sklearn=True)

    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        cv = client.search_model_versions(f"name='{CMAPSS_MODEL_NAME}'")
        _models["cmapss_version"] = str(cv[0].version) if cv else None
        wv = client.search_model_versions(f"name='{CWRU_MODEL_NAME}'")
        _models["cwru_version"] = str(wv[0].version) if wv else None
        if wv:
            run = client.get_run(wv[0].run_id)
            t = run.data.params.get("deployed_threshold")
            _models["cwru_threshold"] = float(t) if t else 0.0
        else:
            _models["cwru_threshold"] = 0.0
    except Exception as e:
        logger.warning(f"Could not load metadata from MLflow: {e}")
        _models.setdefault("cwru_threshold", 0.0)

    logger.info(f"CWRU threshold: {_models.get('cwru_threshold', 0.0):.6f}")


# ── Lifespan: load models on startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _load_from_files():
        logger.info("No model files found — loading from MLflow registry...")
        _load_from_mlflow()
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
