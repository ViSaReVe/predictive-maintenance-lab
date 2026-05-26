"""
Entry point for the Predictive Maintenance training pipeline.

Usage:
    python main.py
    python main.py --cmapss-config configs/cmapss.yaml --cwru-config configs/cwru.yaml

After training:
    - View MLflow experiments: mlflow ui (or docker compose up mlflow)
    - Start inference API:     uvicorn api.app:app --reload
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from pipelines.train import run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="PdM training pipeline")
    parser.add_argument("--cmapss-config", default="configs/cmapss.yaml")
    parser.add_argument("--cwru-config", default="configs/cwru.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run(cmapss_config=args.cmapss_config, cwru_config=args.cwru_config)

    print("\n" + "=" * 55)
    print("  Training Complete")
    print("=" * 55)
    print(f"  Run timestamp : {summary['run_timestamp']}")
    print(f"  CMAPSS RUL    : {summary['cmapss_rul']}")
    print(f"  CWRU Anomaly  : {summary['cwru_anomaly']}")
    print("=" * 55)
    print("  Next steps:")
    print("    MLflow UI  → mlflow ui   (port 5000)")
    print("    Start API  → uvicorn api.app:app --reload  (port 8000)")
    print("    Docker     → docker compose up")
    print("=" * 55)
