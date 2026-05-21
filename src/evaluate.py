import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
)


# ── Regression metrics (CMAPSS RUL) ──────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA CMAPSS asymmetric scoring function (penalises late predictions more).
    S = sum( exp( d/13 ) - 1  if d < 0   (early prediction)
             exp( d/10 ) - 1  if d >= 0  (late prediction) )
    where d = y_pred - y_true.
    """
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.sum())


# ── Anomaly detection metrics (CWRU fault detection) ─────────────────────────

def pr_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    """Area under the Precision-Recall curve (average precision)."""
    return float(average_precision_score(y_true_binary, scores))


def threshold_sweep(y_true_binary: np.ndarray, scores: np.ndarray):
    """
    Sweep decision thresholds and compute false-alarm rate + missed-failure rate.

    Returns
    -------
    thresh : ndarray — threshold values
    far    : ndarray — false alarm rate ≈ 1 - precision  (FP / predicted positives)
    mfr    : ndarray — missed failure rate = 1 - recall   (FN / actual positives)
    """
    prec, rec, thresh = precision_recall_curve(y_true_binary, scores)
    far = 1 - prec[:-1]   # false alarm rate at each threshold
    mfr = 1 - rec[:-1]    # missed failure rate at each threshold
    return thresh, far, mfr


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_threshold_tradeoff(thresh, far, mfr, title: str = 'Threshold Tradeoff'):
    """Plot false-alarm rate vs. missed-failure rate across threshold values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresh, far, label='False Alarm Rate', color='orange', linewidth=2)
    ax.plot(thresh, mfr, label='Missed Failure Rate', color='red', linewidth=2)
    ax.set_xlabel('Anomaly Score Threshold')
    ax.set_ylabel('Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_rul_predictions(y_true_list: list, y_pred_list: list, title: str = 'RUL Predictions vs True'):
    """
    Plot predicted vs. true RUL for a list of engines.

    Parameters
    ----------
    y_true_list : list of 1D arrays, one per engine
    y_pred_list : list of 1D arrays, one per engine
    """
    n = len(y_true_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(y_true_list[i], label='True RUL', linewidth=2)
        ax.plot(y_pred_list[i], label='Predicted RUL', linewidth=2, linestyle='--')
        ax.set_title(f'Engine {i + 1}')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('RUL')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_score_distribution(scores_dict: dict, title: str = 'Anomaly Score Distributions'):
    """
    Box/violin plot of anomaly scores per condition.

    Parameters
    ----------
    scores_dict : {label: 1D array of scores}
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = list(scores_dict.keys())
    data   = [scores_dict[k] for k in labels]
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_ylabel('Anomaly Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig
