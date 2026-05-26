"""
Training pipeline for Predictive Maintenance Lab.

Mirrors the pattern from the reference time-series MLOps repo:
  - Single run() entry point
  - MLflow tracks params, metrics, artifacts (plots + model)
  - Models registered in MLflow Model Registry with @champion alias
  - Returns summary dict so FastAPI /train endpoint can surface results

Two tasks share one MLflow experiment:
  1. CMAPSS RUL Prediction   → XGBoost regressor
  2. CWRU Anomaly Detection  → Isolation Forest (normal-only training)
"""

import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import xgboost as xgb  # must be imported before torch to avoid macOS ARM segfault
import yaml
from mlflow.tracking import MlflowClient

# Modeling library (src/)
# NOTE: src.models contains torch-based classes — import it AFTER xgboost is loaded
from src.data_loader import CMAPSS_DROP, load_cmapss, load_cwru
from src.evaluate import mae, pr_auc, rmse, score_fn, threshold_sweep
from src.features import extract_features
from src.preprocessing import (
    add_rul_labels,
    normalize_per_engine,
    prepare_test_windows,
    sliding_windows_cmapss,
    sliding_windows_cwru,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "predictive-maintenance"
CMAPSS_MODEL_NAME = "pdm-cmapss-rul-xgboost"
CWRU_MODEL_NAME = "pdm-cwru-anomaly-isoforest"
# SQLite backend — single file, no file-store deprecation warnings, works with mlflow ui default
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Feature / data helpers ─────────────────────────────────────────────────────

def _sensor_cols(df) -> list:
    exclude = {"engine_id", "cycle", "rul", "op1", "op2", "op3"}
    drop_set = set(CMAPSS_DROP)
    return [c for c in df.columns if c not in exclude and c not in drop_set]


def _save_plot(fig, name: str, run_ts: str) -> str:
    """Save figure to reports/ and return path for mlflow.log_artifact."""
    reports_dir = Path("reports") / run_ts
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = str(reports_dir / f"{name}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def _register_champion(model_name: str, run_id: str, artifact_path: str) -> None:
    """Register a model version and promote it to @champion alias."""
    client = MlflowClient()
    # Register
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri, model_name)
    # Set @champion alias (MLflow ≥ 2.9 supports aliases)
    try:
        client.set_registered_model_alias(model_name, "champion", mv.version)
        logger.info(f"Registered {model_name} v{mv.version} as @champion")
    except Exception:
        logger.warning("MLflow aliases not supported — skipping champion alias")


# ── Task 1: CMAPSS RUL Prediction ─────────────────────────────────────────────

def _run_cmapss(cfg: dict, run_ts: str) -> dict:
    logger.info("=== CMAPSS RUL Prediction ===")

    # Load + preprocess
    subset = cfg["data"]["subset"]
    data_dir = cfg["data"]["data_dir"]
    train_df, test_df, rul_gt = load_cmapss(data_dir=data_dir, subset=subset)

    sensor_cols = _sensor_cols(train_df)
    w = cfg["preprocessing"]["window_size"]
    stride = cfg["preprocessing"]["stride"]

    train_df = add_rul_labels(train_df)
    train_df = normalize_per_engine(train_df, sensor_cols)
    test_df = normalize_per_engine(test_df, sensor_cols)

    X_train, y_train = sliding_windows_cmapss(train_df, sensor_cols, window=w, stride=stride)
    X_test, y_test = prepare_test_windows(test_df, rul_gt, sensor_cols, window=w)

    # Flatten windows for XGBoost: (n, window * n_sensors)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # Train
    xgb_params = cfg["xgboost"]
    model = xgb.XGBRegressor(
        n_estimators=xgb_params["n_estimators"],
        max_depth=xgb_params["max_depth"],
        learning_rate=xgb_params["learning_rate"],
        subsample=xgb_params["subsample"],
        colsample_bytree=xgb_params["colsample_bytree"],
        random_state=xgb_params["random_state"],
    )
    model.fit(X_train_flat, y_train)

    # Evaluate
    y_pred = model.predict(X_test_flat)
    test_rmse = rmse(y_test, y_pred)
    test_mae = mae(y_test, y_pred)
    nasa = score_fn(y_test, y_pred)
    logger.info(f"  RMSE={test_rmse:.2f}  MAE={test_mae:.2f}  NASA={nasa:.0f}")

    # MLflow run
    with mlflow.start_run(run_name="cmapss-xgboost", nested=True) as run:
        # Log params
        mlflow.log_params({
            "subset": subset,
            "window_size": w,
            "rul_cap": cfg["preprocessing"]["rul_cap"],
            **{f"xgb_{k}": v for k, v in xgb_params.items()},
        })

        # Log metrics
        mlflow.log_metrics({
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "nasa_score": nasa,
        })

        # Log feature importance plot
        fig, ax = plt.subplots(figsize=(10, 4))
        top_n = 20
        fi = model.feature_importances_
        top_idx = np.argsort(fi)[-top_n:]
        ax.barh(range(top_n), fi[top_idx])
        ax.set_title("XGBoost Feature Importances (Top 20)")
        ax.set_xlabel("Importance")
        plot_path = _save_plot(fig, "cmapss_feature_importance", run_ts)
        mlflow.log_artifact(plot_path)

        # Log RUL predictions scatter
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.4, s=10)
        ax.plot([0, 125], [0, 125], "r--", linewidth=1)
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title(f"RUL Predictions — RMSE={test_rmse:.2f}")
        plot_path = _save_plot(fig, "cmapss_rul_scatter", run_ts)
        mlflow.log_artifact(plot_path)

        # Log model + register
        mlflow.sklearn.log_model(model, artifact_path="model")
        _register_champion(CMAPSS_MODEL_NAME, run.info.run_id, "model")

    return {
        "test_rmse": round(test_rmse, 4),
        "test_mae": round(test_mae, 4),
        "nasa_score": round(nasa, 2),
    }


# ── Task 2: CWRU Anomaly Detection ────────────────────────────────────────────

def _run_cwru(cfg: dict, run_ts: str) -> dict:
    from src.classical_models import fit_isolation_forest

    logger.info("=== CWRU Anomaly Detection ===")

    data_dir = cfg["data"]["data_dir"]
    cwru_signals = load_cwru(data_dir=data_dir)

    w = cfg["preprocessing"]["window_size"]
    stride = cfg["preprocessing"]["stride"]
    train_frac = cfg["preprocessing"]["train_fraction"]

    # Position-based split of normal signal to avoid temporal leakage
    normal = cwru_signals["normal"]
    split = int(len(normal) * train_frac)
    train_windows = sliding_windows_cwru(normal[:split], window=w, stride=stride)

    # Test set: unseen normal + all fault windows
    normal_test = sliding_windows_cwru(normal[split:], window=w, stride=stride)
    fault_windows = np.vstack([
        sliding_windows_cwru(sig, window=w, stride=stride)
        for label, sig in cwru_signals.items()
        if label != "normal"
    ])

    X_test = np.vstack([normal_test, fault_windows])
    y_test = np.concatenate([np.zeros(len(normal_test)), np.ones(len(fault_windows))])

    # Extract features
    X_train_feats = extract_features(train_windows)
    X_test_feats = extract_features(X_test)

    # Train Isolation Forest (normal-only)
    if_params = cfg["isolation_forest"]
    model = fit_isolation_forest(
        X_train_feats,
        contamination=if_params["contamination"],
    )

    # Anomaly scores: negate decision_function so higher = more anomalous
    scores_train = -model.decision_function(X_train_feats)
    scores_test = -model.decision_function(X_test_feats)

    # Threshold: 95th percentile of training scores
    threshold = float(np.percentile(scores_train, 95))
    test_pr_auc = pr_auc(y_test, scores_test)
    thresh_arr, far, mfr = threshold_sweep(y_test, scores_test)

    logger.info(f"  PR-AUC={test_pr_auc:.4f}  threshold={threshold:.4f}")

    # MLflow run
    with mlflow.start_run(run_name="cwru-isolation-forest", nested=True) as run:
        mlflow.log_params({
            "window_size": w,
            "train_fraction": train_frac,
            "if_n_estimators": if_params["n_estimators"],
            "if_contamination": if_params["contamination"],
            "threshold_percentile": 95,
        })

        mlflow.log_metrics({
            "test_pr_auc": test_pr_auc,
            "anomaly_threshold": threshold,
            "n_train_windows": len(train_windows),
            "n_test_normal": len(normal_test),
            "n_test_fault": len(fault_windows),
        })

        # False alarm vs missed failure plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(thresh_arr, far, label="False Alarm Rate", color="orange")
        ax.plot(thresh_arr, mfr, label="Missed Failure Rate", color="red")
        ax.axvline(threshold, color="blue", linestyle="--", label=f"Threshold={threshold:.3f}")
        ax.set_xlabel("Anomaly Score Threshold")
        ax.set_ylabel("Rate")
        ax.set_title(f"Threshold Tradeoff — PR-AUC={test_pr_auc:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = _save_plot(fig, "cwru_threshold_tradeoff", run_ts)
        mlflow.log_artifact(plot_path)

        # Score distribution by class
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(
            [scores_test[y_test == 0], scores_test[y_test == 1]],
            labels=["Normal", "Fault"],
            patch_artist=True,
        )
        ax.set_ylabel("Anomaly Score")
        ax.set_title("Anomaly Score Distribution")
        ax.grid(True, alpha=0.3, axis="y")
        plot_path = _save_plot(fig, "cwru_score_distribution", run_ts)
        mlflow.log_artifact(plot_path)

        # Log model + threshold as param artifact
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_param("deployed_threshold", threshold)
        _register_champion(CWRU_MODEL_NAME, run.info.run_id, "model")

    return {
        "test_pr_auc": round(test_pr_auc, 4),
        "anomaly_threshold": round(threshold, 6),
    }


# ── Main pipeline entry point ──────────────────────────────────────────────────

def run(
    cmapss_config: str = "configs/cmapss.yaml",
    cwru_config: str = "configs/cwru.yaml",
) -> dict:
    """
    Full training pipeline: train both models, log everything to MLflow,
    register champion models. Returns summary dict for callers (e.g. FastAPI /train).

    Usage:
        python main.py
        # or from FastAPI: POST /train
    """
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    cmapss_cfg = _load_config(cmapss_config)
    cwru_cfg = _load_config(cwru_config)

    logger.info(f"Starting training pipeline — run_ts={run_ts}")

    cmapss_results = {}
    cwru_results = {}

    with mlflow.start_run(run_name=f"full-pipeline-{run_ts}"):
        mlflow.log_param("run_timestamp", run_ts)

        try:
            cmapss_results = _run_cmapss(cmapss_cfg, run_ts)
            mlflow.log_metrics({f"cmapss_{k}": v for k, v in cmapss_results.items()})
        except Exception as e:
            logger.error(f"CMAPSS training failed: {e}", exc_info=True)
            cmapss_results = {"error": str(e)}

        try:
            cwru_results = _run_cwru(cwru_cfg, run_ts)
            mlflow.log_metrics({f"cwru_{k}": v for k, v in cwru_results.items()})
        except Exception as e:
            logger.error(f"CWRU training failed: {e}", exc_info=True)
            cwru_results = {"error": str(e)}

    summary = {
        "run_timestamp": run_ts,
        "cmapss_rul": cmapss_results,
        "cwru_anomaly": cwru_results,
    }

    logger.info("Training pipeline complete")
    logger.info(f"  CMAPSS → {cmapss_results}")
    logger.info(f"  CWRU   → {cwru_results}")

    return summary


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    summary = run()
    print("\n" + "=" * 50)
    print("Pipeline Summary:")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v}")
