from typing import Optional

from pydantic import BaseModel, Field

# ── Request schemas ────────────────────────────────────────────────────────────

class RULRequest(BaseModel):
    """
    Input for RUL prediction.
    windows: list of sensor windows, each window is a flat list of (window_size * n_sensors) values.
    engine_id: optional identifier, returned in the response for traceability.
    """
    windows: list[list[float]] = Field(
        ...,
        description="One or more sensor windows. Each flat: window_size × n_sensors values.",
        example=[[0.1, 0.2, 0.3] * 30],  # 30 timesteps × 3 sensors (simplified)
    )
    engine_id: Optional[str] = Field(None, description="Optional engine identifier")


class AnomalyRequest(BaseModel):
    """
    Input for bearing anomaly detection.
    features: pre-extracted feature vector [rms, kurtosis, peak_to_peak, spectral_entropy]
              OR raw vibration window (will be feature-extracted inside the endpoint).
    """
    features: list[float] = Field(
        ...,
        description="Feature vector [rms, kurtosis, peak_to_peak, spectral_entropy].",
        example=[0.12, 3.1, 0.45, 0.89],
    )


class TrainRequest(BaseModel):
    cmapss_config: str = Field("configs/cmapss.yaml", description="Path to CMAPSS config file")
    cwru_config: str = Field("configs/cwru.yaml", description="Path to CWRU config file")


# ── Response schemas ───────────────────────────────────────────────────────────

class RULResponse(BaseModel):
    engine_id: Optional[str]
    rul_predictions: list[float] = Field(
        ..., description="Predicted RUL (cycles) for each input window"
    )
    model_name: str
    model_version: Optional[str]


class AnomalyResponse(BaseModel):
    anomaly_score: float = Field(..., description="Anomaly score — higher = more anomalous")
    is_anomaly: bool = Field(..., description="True if score exceeds the deployment threshold")
    threshold: float = Field(..., description="Threshold used for binary decision")
    model_name: str
    model_version: Optional[str]


class TrainResponse(BaseModel):
    success: bool
    run_timestamp: str
    cmapss_rul: dict
    cwru_anomaly: dict
    message: str
