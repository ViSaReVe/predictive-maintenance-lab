"""
Classical (non-deep) anomaly detection models — no torch dependency.
Imported by pipelines/train.py to avoid the macOS ARM torch+xgboost segfault.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def fit_isolation_forest(X_normal: np.ndarray, contamination: float = 0.01) -> IsolationForest:
    """
    Train an Isolation Forest on normal-only feature vectors.
    contamination=0.01 → assume 1% of 'normal' training data is noisy.
    Score convention: lower (more negative) = more anomalous.
    """
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    clf.fit(X_normal)
    return clf


def fit_one_class_svm(X_normal: np.ndarray, nu: float = 0.05) -> OneClassSVM:
    """
    Train a One-Class SVM on normal-only feature vectors.
    nu ≈ upper bound on fraction of training outliers.
    Score convention: lower (more negative) = more anomalous.
    """
    clf = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    clf.fit(X_normal)
    return clf
