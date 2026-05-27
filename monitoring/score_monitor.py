"""
Anomaly score distribution monitor.

Tracks the distribution of Isolation Forest anomaly scores over a batch of
feature vectors. In production, feed this script a CSV of features logged
by the inference API. In demo/CI mode, uses synthetic features.

Usage
-----
# Synthetic demo:
    python monitoring/score_monitor.py

# Real feature CSV (columns: rms, kurtosis, peak_to_peak, spectral_entropy):
    python monitoring/score_monitor.py --features path/to/features.csv

Output: reports/score_summary.json  +  console summary
"""

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_PATH = "models/cwru_anomaly_isoforest.joblib"
THRESHOLD_PATH = "models/cwru_threshold.json"
FEATURE_COLS = ["rms", "kurtosis", "peak_to_peak", "spectral_entropy"]

# Synthetic normal-ish features for demo/CI (matches training distribution)
_DEMO_STATS = {
    "rms":              {"mean": 0.175, "std": 0.025},
    "kurtosis":         {"mean": 3.05,  "std": 0.42},
    "peak_to_peak":     {"mean": 0.80,  "std": 0.12},
    "spectral_entropy": {"mean": 9.20,  "std": 0.60},
}


def _synthetic_features(n: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.normal(s["mean"], s["std"], n) for s in _DEMO_STATS.values()
    ]).astype(np.float32)


def compute_score_summary(features: np.ndarray, model, threshold: float) -> dict:
    scores = -model.decision_function(features)
    flagged = (scores > threshold).sum()
    return {
        "n_samples":       int(len(scores)),
        "mean_score":      round(float(np.mean(scores)), 6),
        "std_score":       round(float(np.std(scores)), 6),
        "p50_score":       round(float(np.percentile(scores, 50)), 6),
        "p95_score":       round(float(np.percentile(scores, 95)), 6),
        "p99_score":       round(float(np.percentile(scores, 99)), 6),
        "threshold":       round(threshold, 6),
        "n_flagged":       int(flagged),
        "pct_flagged":     round(float(flagged / len(scores)) * 100, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Anomaly score distribution monitor")
    parser.add_argument("--features", default=None, help="CSV of feature vectors (optional)")
    parser.add_argument("--output",   default="reports/score_summary.json")
    args = parser.parse_args()

    import joblib
    model_path = Path(MODEL_PATH)
    threshold_path = Path(THRESHOLD_PATH)

    if not model_path.exists():
        print(f"Model not found at {MODEL_PATH} — run 'python main.py' first to train and export")
        return

    model = joblib.load(model_path)
    threshold = (
        json.loads(threshold_path.read_text())["threshold"]
        if threshold_path.exists() else 0.0
    )

    demo_mode = False
    if args.features:
        import pandas as pd
        df = pd.read_csv(args.features)
        features = df[FEATURE_COLS].values.astype(np.float32)
        print(f"Loaded {len(features)} feature vectors from {args.features}")
    else:
        features = _synthetic_features()
        demo_mode = True
        print(f"Using synthetic demo features ({len(features)} samples)")
        print("NOTE: Synthetic features are for pipeline validation only.")
        print("      For accurate score distribution, provide real feature CSV.")

    summary = compute_score_summary(features, model, threshold)

    print("\n── Score Monitor Summary ──────────────────────────")
    print(f"  Samples:       {summary['n_samples']}")
    print(f"  Mean score:    {summary['mean_score']:.4f}")
    print(f"  95th pct:      {summary['p95_score']:.4f}")
    print(f"  Threshold:     {summary['threshold']:.4f}")
    flagged_str = f"{summary['n_flagged']} / {summary['n_samples']} ({summary['pct_flagged']:.1f}%)"
    print(f"  Flagged:       {flagged_str}")
    if not demo_mode and summary["pct_flagged"] > 10:
        print("  WARNING: >10% of samples flagged — possible sensor drift or model degradation")
    print("───────────────────────────────────────────────────\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"Score summary → {args.output}")


if __name__ == "__main__":
    main()
