# predictive-maintenance-lab

> A predictive-maintenance learning lab built on two standard sensor benchmarks. CMAPSS teaches supervised RUL prediction from multivariate degradation trajectories. CWRU teaches normal-only anomaly detection and bearing fault diagnosis from vibration signals. Goal: understand the complete PdM workflow — cleaning, normalisation, windowing, feature extraction, modelling, evaluation, and operational threshold selection.

---

## What Problem This Solves

Unplanned equipment failure is one of the largest cost drivers in manufacturing. Predictive maintenance (PdM) aims to move from reactive repair ("fix it when it breaks") and scheduled maintenance ("replace it every 6 months") to condition-based maintenance ("replace it when sensor data says it's degrading").

This lab covers two complementary PdM problems:

1. **RUL Prediction (CMAPSS):** Given a multivariate sensor history, predict how many cycles remain before engine failure. Used for long-horizon maintenance scheduling.
2. **Fault Detection (CWRU):** Given raw vibration from a rotating machine, flag deviation from healthy behaviour. Used for real-time alerting.

---

## Datasets

### NASA CMAPSS — Turbofan Degradation (FD001)
- 100 training engines, each run to failure; 100 test engines truncated before failure
- 21 sensors + 3 operational settings per cycle; FD001 uses one operating condition and one fault mode (HPC degradation)
- Ground-truth RUL provided for test engines
- **Why CMAPSS:** Clean, well-documented benchmark with well-established RMSE baselines (~12–18 for deep models). Ideal for learning the supervised RUL workflow without distractions.

### CWRU Bearing Dataset — Vibration Fault Diagnosis
- Drive-end accelerometer, 12 kHz sampling, four conditions: normal, inner-race fault, ball fault, outer-race fault (all 0.007" defect diameter, 0 HP load)
- **Why CWRU:** The standard fault-detection benchmark. Controlled lab environment makes it possible to understand model behaviour before dealing with production noise.
- **Caveat:** CWRU is a controlled lab benchmark — not a production dataset. Models trained on CWRU do not directly transfer to production motors without domain adaptation. The PR-AUC reported here is honest under a position-based split (see below), but should not be claimed as production-ready performance.

---

## Key Design Decisions

### 1. Piecewise-Linear RUL Cap at 125 (CMAPSS)
Early-life data points (RUL > 125) add label noise without useful degradation signal. Capping tells the model: "beyond 125 cycles of life, report healthy." The degradation zone is what matters for scheduling.

### 2. Per-Engine Normalisation
Global min-max normalisation conflates inter-engine variation (manufacturing tolerance, baseline differences) with within-engine degradation signal. Per-engine normalisation using each engine's first 30 cycles as baseline isolates true degradation.

### 3. Normal-Only Training for Fault Detection
Production fault data is rare. Training a binary classifier on 3 documented failures out of 10,000 operating hours is pathological. The unsupervised approach (Isolation Forest, Autoencoder) learns the normal envelope; anything outside it is anomalous — including failure modes that were never documented.

### 4. PR-AUC Not Accuracy (Fault Detection)
Bearing faults are rare — a classifier that always predicts "normal" gets 95%+ accuracy but catches zero faults. PR-AUC measures precision and recall at all thresholds and is the correct metric for imbalanced anomaly detection.

### 5. Position-Based Train/Test Split (CWRU)
Adjacent vibration windows from the same recording are temporally correlated. Random splitting puts near-duplicate windows in both train and test, inflating metrics. The split here: normal signal split at 60% by position; fault signals are separate files with no overlap.

---

## Results

### CMAPSS FD001 — RUL Prediction

| Model | RMSE (cycles) | MAE (cycles) | NASA Score |
|-------|--------------|-------------|------------|
| XGBoost (300 trees, window=30) | **16.80** ✅ | 12.77 | **519** ✅ |
| LSTM (128 hidden, 2 layers, LR scheduler) | 22.43 | 17.33 | 1170 |

**Why XGBoost beats LSTM on FD001:** FD001 is a single operating condition, single fault mode dataset — the degradation trajectory is smooth and consistent across engines. XGBoost captures this well by learning from the full 30-cycle flattened feature vector. LSTMs typically outperform tree models on FD002/FD003/FD004, which have multiple operating conditions and more complex temporal patterns where sequence memory matters.

**NASA score matters more than RMSE operationally:** The asymmetric NASA score penalizes late predictions (optimistic about remaining life) much harder than early ones. LSTM 1170 vs XGBoost 519 means the LSTM skews toward predicting longer remaining life than actual — the dangerous direction for maintenance scheduling. XGBoost is both more accurate and safer on this dataset.

### CWRU Bearing Fault Detection

Split: normal signal 60% train / 40% test by position (no leakage). Fault signals entirely in test.  
Conditions: normal vs inner race, ball, outer race fault (0.007", 1 HP load, 12 kHz).

| Model | PR-AUC | Notes |
|-------|--------|-------|
| Isolation Forest | **0.9998** | 200 trees, normal-only, 4 statistical features |
| Conv1d Autoencoder | **1.0000** | 30 epochs, normal-only, raw 1024-sample windows |
| Z-Score Baseline | **1.0000** | Max z-score across 4 features |
| One-Class SVM | **1.0000** | RBF kernel, normal-only training |
| Conv1d Autoencoder | **1.0000** | 30 epochs, normal-only training |
| Isolation Forest | **0.9998** | 200 trees, normal-only training |
| XGBoost + oversample | 0.9493 | 10 labelled fault windows, repeated oversampling |

**Why near-perfect PR-AUC?** CWRU is a controlled lab benchmark — large, consistent fault signatures in a clean environment. Kurtosis alone separates fault conditions from normal almost perfectly. These numbers should not be claimed as production-ready; in a real plant, SNR is lower and fault signatures are less distinct. The value here is demonstrating the correct methodology (normal-only training, position-based split, PR-AUC over accuracy).

---

## Project Structure

```
predictive-maintenance-lab/
├── data/
│   ├── cmapss/          # train_FD001.txt, test_FD001.txt, RUL_FD001.txt
│   └── cwru/            # Normal_0.mat, IR007_0.mat, B007_0.mat, OR007_0.mat
├── notebooks/
│   ├── 01_cmapss_rul_prediction.ipynb
│   ├── 02_cwru_bearing_fault_detection.ipynb
│   └── 03_model_comparison_and_operational_metrics.ipynb
├── src/
│   ├── data_loader.py    # load_cmapss(), load_cwru()
│   ├── preprocessing.py  # add_rul_labels(), normalize_per_engine(), sliding_windows_*()
│   ├── features.py       # rms, kurtosis, peak_to_peak, spectral_entropy, extract_features()
│   ├── models.py         # LSTMRegressor, ConvAutoencoder, fit_isolation_forest(), fit_one_class_svm()
│   └── evaluate.py       # rmse, mae, pr_auc, threshold_sweep, plotting helpers
├── requirements.txt
└── BUILD.md
```

---

## Setup

```bash
cd ~/Documents/predictive-maintenance-lab
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download data and place in the correct directories (see BUILD.md for direct links), then:

```bash
cd notebooks
jupyter notebook
```

---

## Production Deployment Thinking

A minimal viable deployment of the fault detection system:

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

**Threshold selection:** The model outputs a score; the threshold is a business decision. At a cost ratio of ~100:1 (motor failure vs. false work order), set the threshold at ≤2% missed-failure rate and accept the resulting false alarm rate. Show the operations team the false-alarm vs. missed-failure curve and let them choose the operating point.

---

## What I'd Change at Scale

- **Streaming inference:** Replace batch CSV processing with a Kafka/Kinesis consumer updating a ring buffer per asset
- **Managed training pipeline:** Azure ML or SageMaker for scheduled retraining when technician-confirmed label accumulation triggers a retrain
- **Drift monitoring:** Track KL divergence of input feature distributions week-over-week; alert when distribution shift exceeds a threshold
- **Per-asset models:** In a real fleet, each motor has its own wear history. Hierarchical model (global prior + per-asset fine-tuning) outperforms a single fleet model
- **Explainability layer:** SHAP values on the Isolation Forest features so technicians can see "kurtosis spike drove this alarm" rather than a black-box score
