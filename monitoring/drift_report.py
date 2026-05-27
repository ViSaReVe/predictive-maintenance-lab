"""
Bearing vibration feature drift report.

Compares the distribution of CWRU features (rms, kurtosis, peak_to_peak,
spectral_entropy) between a reference window (training data) and a current
window (new production data). Uses Evidently to generate an HTML report.

Usage
-----
# Synthetic demo (no real data needed — works in CI):
    python monitoring/drift_report.py

# Real data (load .mat files, extract features, compare):
    python monitoring/drift_report.py --reference data/cwru/ --current data/cwru/

Output: reports/drift_report.html
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = ["rms", "kurtosis", "peak_to_peak", "spectral_entropy"]

# Reference feature stats from training run (normal CWRU bearings, 0 HP load).
# These are the distribution parameters the Isolation Forest was trained on.
_REF_STATS = {
    "rms":              {"mean": 0.175, "std": 0.025},
    "kurtosis":         {"mean": 3.05,  "std": 0.42},
    "peak_to_peak":     {"mean": 0.80,  "std": 0.12},
    "spectral_entropy": {"mean": 9.20,  "std": 0.60},
}

# Simulated current-window stats — slight degradation (rising kurtosis, falling entropy).
# Used for CI and demo. In production, replace with real extracted features.
_CUR_STATS = {
    "rms":              {"mean": 0.188, "std": 0.032},
    "kurtosis":         {"mean": 3.65,  "std": 0.78},
    "peak_to_peak":     {"mean": 0.92,  "std": 0.16},
    "spectral_entropy": {"mean": 8.75,  "std": 0.72},
}


def _synthetic_df(stats: dict, n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {col: rng.normal(s["mean"], s["std"], n) for col, s in stats.items()}
    )


def _load_real_features(data_dir: str) -> pd.DataFrame:
    """Extract features from .mat files in data_dir. Requires real CWRU data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loader import load_cwru
    from src.features import extract_features
    from src.preprocessing import sliding_windows_cwru

    signals = load_cwru(data_dir=data_dir)
    normal = signals["normal"]
    windows = sliding_windows_cwru(normal, window=1024, stride=512)
    feats = extract_features(windows)
    return pd.DataFrame(feats, columns=FEATURE_COLS)


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "reports/drift_report.html",
) -> dict:
    """Run Evidently drift report and save HTML. Returns per-feature drift flags."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference_df, current_data=current_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(output_path)
    print(f"Drift report saved → {output_path}")

    # Parse per-column drift from snapshot metrics
    drift_flags = {}
    try:
        for m in snapshot.dict().get("metrics", []):
            name = m.get("metric_name", "")
            value = m.get("value")
            # DriftedColumnsCount gives dataset-level summary
            if name.startswith("DriftedColumnsCount"):
                if isinstance(value, dict):
                    share = value.get("share", 0)
                    print(f"  Drifted feature share: {share:.0%}")
            # ValueDrift(column=X, method=..., threshold=T) → value is the p-value
            elif name.startswith("ValueDrift(column="):
                col = name.split("column=")[1].split(",")[0]
                threshold_str = name.split("threshold=")[1].rstrip(")")
                threshold = float(threshold_str)
                drifted = bool(isinstance(value, (float, int)) and value < threshold)
                drift_flags[col] = drifted
    except (KeyError, TypeError, AttributeError, IndexError, ValueError):
        pass

    return drift_flags


def main() -> None:
    parser = argparse.ArgumentParser(description="CWRU feature drift report")
    parser.add_argument("--reference", default=None, help="Path to reference data dir (optional)")
    parser.add_argument("--current",   default=None, help="Path to current data dir (optional)")
    parser.add_argument("--output",    default="reports/drift_report.html")
    parser.add_argument("--summary",   default="reports/drift_summary.json")
    args = parser.parse_args()

    if args.reference:
        print(f"Loading reference features from {args.reference}")
        ref_df = _load_real_features(args.reference)
    else:
        print("No reference data provided — using synthetic training distribution")
        ref_df = _synthetic_df(_REF_STATS, n=500, seed=0)

    if args.current:
        print(f"Loading current features from {args.current}")
        cur_df = _load_real_features(args.current)
    else:
        print("No current data provided — using synthetic drifted distribution")
        cur_df = _synthetic_df(_CUR_STATS, n=500, seed=1)

    print(f"\nReference: {len(ref_df)} samples  |  Current: {len(cur_df)} samples")
    print(f"Features:  {FEATURE_COLS}\n")

    drift_flags = run_drift_report(ref_df, cur_df, args.output)

    summary = {
        "output": args.output,
        "n_reference": len(ref_df),
        "n_current": len(cur_df),
        "feature_drift": drift_flags,
        "any_drift": any(drift_flags.values()) if drift_flags else None,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"\nDrift summary → {args.summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
