# Predictive Maintenance Lab

End-to-end ML system for industrial sensor data: RUL prediction from turbofan degradation trajectories (NASA CMAPSS) and bearing fault detection from vibration signals (CWRU). Built as a production system — not just notebooks.

[![CI](https://github.com/ViSaReVe/predictive-maintenance-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ViSaReVe/predictive-maintenance-lab/actions/workflows/ci.yml)

**Live API:** `https://pdm-api.politebeach-8c05f79b.canadacentral.azurecontainerapps.io`  
**Swagger UI:** `https://pdm-api.politebeach-8c05f79b.canadacentral.azurecontainerapps.io/docs`

---

## Try It Now

**RUL Prediction** — how many cycles remain before engine failure:
```bash
curl -X POST https://pdm-api.politebeach-8c05f79b.canadacentral.azurecontainerapps.io/predict/rul \
  -H "Content-Type: application/json" \
  -d '{"engine_id": "engine_001", "windows": [[0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08, 0.05, 0.1, -0.02, 0.08, 0.15, 0.03, -0.05, 0.12, 0.07, -0.03, 0.09, 0.04, 0.11, -0.06, 0.08, 0.02, 0.14, 0.05, -0.08, 0.07, 0.03, 0.10, -0.04, 0.06, 0.09, 0.01, 0.13, -0.07, 0.05, 0.08]]}'
```

**Bearing Fault Detection** — is this vibration signal anomalous:
```bash
curl -X POST https://pdm-api.politebeach-8c05f79b.canadacentral.azurecontainerapps.io/predict/anomaly \
  -H "Content-Type: application/json" \
  -d '{"features": [0.07, 3.1, 0.26, 0.35]}'
```

---

## What This Is

Two complementary industrial ML problems, productionized end-to-end:

| Problem | Dataset | Model | Metric |
|---------|---------|-------|--------|
| RUL Prediction | NASA CMAPSS FD001 | XGBoost | RMSE 16.80 (lit: 16.14) |
| Bearing Fault Detection | CWRU Vibration | Isolation Forest | PR-AUC 0.9998 |

**RUL Prediction:** Given 30 cycles of multivariate sensor readings from a turbofan engine, predict how many cycles remain before failure. Used for maintenance scheduling.

**Fault Detection:** Given a vibration window from a rotating bearing, score it for anomaly. Trained on normal-only data — detects fault modes never seen during training.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│  python main.py                                             │
│                                                             │
│  configs/ ──► pipelines/train.py ──► MLflow Registry       │
│  (yaml)        (XGBoost + IsoForest)   (sqlite:///mlflow.db)│
└─────────────────────────────────────────────────────────────┘
                              │
                              │ models/ (joblib export)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Inference Service                         │
│  uvicorn api.app:app                                        │
│                                                             │
│  POST /predict/rul      ──► XGBoost RUL prediction          │
│  POST /predict/anomaly  ──► Isolation Forest scoring        │
│  POST /train            ──► disabled in prod (403)          │
│  GET  /docs             ──► Swagger UI                      │
└─────────────────────────────────────────────────────────────┘
                              │
                     Docker + Azure Container Apps
                     (live at canadacentral)
                              │
                              │ production features (CSV)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Drift Monitoring                          │
│  python monitoring/drift_report.py                          │
│  python monitoring/score_monitor.py                         │
│                                                             │
│  Evidently drift report ──► reports/drift_report.html       │
│  Score distribution    ──► reports/score_summary.json       │
│  GitHub Actions        ──► weekly scheduled run             │
└─────────────────────────────────────────────────────────────┘
```

---

## Results

### CMAPSS FD001 — RUL Prediction

| Model | RMSE (cycles) | MAE (cycles) | NASA Score |
|-------|--------------|-------------|------------|
| **XGBoost (300 trees, window=30)** | **16.80** | **12.77** | **519** |
| LSTM (128 hidden, 2 layers, LR scheduler) | 22.43 | 17.33 | 1170 |
| Literature best (LSTM) | ~16.14 | — | — |

XGBoost matches published deep model performance on FD001 because the dataset has a single operating condition — the degradation trajectory is smooth and consistent, no temporal memory needed. The NASA asymmetric score matters more than RMSE operationally: late predictions (underestimating damage) are penalized exponentially harder. XGBoost scores 519 vs LSTM 1170 — safer for maintenance scheduling.

### CWRU Bearing Fault Detection

Position-based split: normal 60% train / 40% test, fault signals entirely in test.

| Model | PR-AUC |
|-------|--------|
| **Isolation Forest (200 trees, normal-only)** | **0.9998** |
| Conv1d Autoencoder | 1.0000 |
| One-Class SVM | 1.0000 |
| XGBoost + oversample (10 fault windows) | 0.9493 |

Near-perfect PR-AUC reflects CWRU's controlled lab conditions — consistent fault signatures, clean SNR. In production, kurtosis alone separates fault from normal. These numbers demonstrate correct methodology (normal-only training, position-based split, PR-AUC over accuracy), not production-ready claims.

---

## Key Design Decisions

**RUL cap at 125 cycles** — Early-life data (RUL > 125) adds label noise without degradation signal. Capping isolates the zone that matters for scheduling.

**Per-engine normalisation** — Global normalisation conflates inter-engine manufacturing variation with within-engine degradation. Per-engine baseline (first 30 cycles) isolates the true signal.

**Normal-only training** — Production fault data is rare. Isolation Forest learns the normal envelope; anything outside is anomalous, including undocumented failure modes.

**PR-AUC not accuracy** — A classifier that always predicts "normal" gets 95%+ accuracy on imbalanced bearing data but catches zero faults. PR-AUC is the correct metric.

**Position-based split** — Adjacent vibration windows are temporally correlated. Random splitting inflates metrics with near-duplicate train/test samples.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Modeling | XGBoost, Isolation Forest, PyTorch LSTM, Conv1d Autoencoder |
| Experiment tracking | MLflow (sqlite backend, model registry, @champion alias) |
| Inference API | FastAPI + Pydantic + uvicorn + slowapi rate limiting |
| Containerization | Docker (multi-stage), Docker Compose |
| Cloud deployment | Azure Container Apps (canadacentral, scale-to-zero) |
| CI/CD | GitHub Actions (lint → test → docker build + model load assertion) |
| Drift monitoring | Evidently (feature drift report + score distribution tracker) |
| Testing | pytest, 46 tests, synthetic fixtures (no data files in CI) |
| Linting | ruff |

---

## Run Locally

**Prerequisites:** Python 3.11+, data files in `data/` (see below)

```bash
git clone https://github.com/ViSaReVe/predictive-maintenance-lab
cd predictive-maintenance-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Download data:**
- CMAPSS FD001: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) → place in `data/cmapss/`
- CWRU: [Case Western Reserve University](https://engineering.case.edu/bearingdatacenter) → place in `data/cwru/`

**Train both models:**
```bash
python main.py
# Models registered in MLflow registry
# View experiments: mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Start inference API:**
```bash
uvicorn api.app:app --reload
# Swagger UI: http://localhost:8000/docs
```

**Run tests:**
```bash
pytest tests/ -v
```

**Run drift monitoring:**
```bash
# Synthetic demo (no data files needed):
python monitoring/drift_report.py

# With real CWRU data (compares train half vs test half of normal signal):
python monitoring/drift_report.py --reference data/cwru/ --current data/cwru/
# → reports/drift_report.html  (open in browser)

# Anomaly score distribution over a feature CSV:
python monitoring/score_monitor.py --features path/to/features.csv
# → reports/score_summary.json
```

---

## Run with Docker

```bash
# Export models first (required to bake into image)
python -c "
import mlflow, mlflow.sklearn, joblib, json
mlflow.set_tracking_uri('sqlite:///mlflow.db')
joblib.dump(mlflow.sklearn.load_model('models:/pdm-cmapss-rul-xgboost@champion'), 'models/cmapss_rul_xgboost.joblib')
joblib.dump(mlflow.sklearn.load_model('models:/pdm-cwru-anomaly-isoforest@champion'), 'models/cwru_anomaly_isoforest.joblib')
"

# Start API + MLflow UI
docker compose up

# API: http://localhost:8000/docs
# MLflow UI: http://localhost:5000
```

---

## Production Deployment Thinking

```
Sensor (accelerometer, 12 kHz)
    └─► Edge node (Raspberry Pi / industrial PC)
            ├─ Ring buffer → 1024-sample windows every 43 ms
            ├─ Feature extraction: RMS, kurtosis, peak-to-peak, spectral entropy
            ├─ Isolation Forest inference (< 1 ms per window)
            ├─ Anomaly score → MQTT / OPC-UA → plant historian
            └─ Threshold breach → CMMS work order (Maximo / SAP PM)
                    └─► Technician feedback → labelled outcomes
                                └─► Periodic model retraining
```

**Threshold selection:** The model outputs a score; the threshold is a business decision. At a cost ratio of ~100:1 (motor failure vs. false work order), set the threshold at ≤2% missed-failure rate and accept the false alarm rate. Show the operations team the false-alarm vs. missed-failure curve (logged in MLflow) and let them choose the operating point.

---

## What I'd Change at Scale

- **Streaming inference:** Kafka/Kinesis consumer updating a ring buffer per asset instead of batch CSV
- **Managed retraining:** Azure ML pipelines triggered when technician-confirmed label accumulation exceeds threshold
- **Per-asset models:** Hierarchical model (global prior + per-asset fine-tuning) for a real fleet with individual wear histories
- **Explainability:** SHAP values on Isolation Forest features so technicians see "kurtosis spike drove this alarm"
- **Drift alerting:** Wire Evidently drift report to PagerDuty/Teams webhook; current implementation generates reports but does not push alerts
