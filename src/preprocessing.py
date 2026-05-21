import numpy as np
import pandas as pd

RUL_CAP = 125  # piecewise-linear cap — industry standard for CMAPSS


def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RUL column to CMAPSS training data (capped at RUL_CAP).

    The last cycle of each engine is RUL=0 (failure); earlier cycles count
    backward. Values above RUL_CAP are clipped — we don't need precise
    predictions far from failure, only that the engine is 'healthy'.
    """
    max_cycle = df.groupby('engine_id')['cycle'].max()
    df = df.join(max_cycle.rename('max_cycle'), on='engine_id')
    df['rul'] = (df['max_cycle'] - df['cycle']).clip(upper=RUL_CAP)
    df.drop(columns=['max_cycle'], inplace=True)
    return df


def normalize_per_engine(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Per-engine min-max normalization using each engine's first 30 cycles as
    its baseline (healthy state). Avoids global normalization which leaks
    fleet-wide knowledge and conflates inter-engine variation with degradation.
    """
    df = df.copy()
    df[sensor_cols] = df[sensor_cols].astype(float)   # ensure float columns before normalisation
    for eid, group in df.groupby('engine_id'):
        baseline = group[group['cycle'] <= 30][sensor_cols]
        lo = baseline.min()
        hi = baseline.max()
        denom = (hi - lo).replace(0, 1)
        normalised = (df.loc[df['engine_id'] == eid, sensor_cols] - lo) / denom
        df.loc[df['engine_id'] == eid, sensor_cols] = normalised.astype(float)
    return df


def sliding_windows_cmapss(
    df: pd.DataFrame,
    sensor_cols: list,
    window: int = 30,
    stride: int = 1,
):
    """
    Extract sliding windows from CMAPSS data.

    Returns
    -------
    X : ndarray, shape (n_windows, window, n_sensors)
    y : ndarray, shape (n_windows,) — RUL at the last timestep of each window
    """
    X, y = [], []
    for _, group in df.groupby('engine_id'):
        vals = group[sensor_cols].values
        rul  = group['rul'].values
        for i in range(0, len(vals) - window + 1, stride):
            X.append(vals[i:i + window])
            y.append(rul[i + window - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_test_windows(
    test_df: pd.DataFrame,
    rul_gt: pd.DataFrame,
    sensor_cols: list,
    window: int = 30,
):
    """
    Build one window per test engine (last `window` cycles) and align with
    ground-truth RUL values from RUL_FD001.txt.

    Returns
    -------
    X_test : ndarray, shape (n_engines, window, n_sensors)
    y_test : ndarray, shape (n_engines,)
    """
    X_test, y_test = [], []
    rul_values = rul_gt['true_rul'].values
    for idx, (eid, group) in enumerate(test_df.groupby('engine_id')):
        vals = group[sensor_cols].values
        # Take last `window` cycles (or pad with zeros if engine is shorter)
        if len(vals) >= window:
            X_test.append(vals[-window:])
        else:
            pad = np.zeros((window - len(vals), len(sensor_cols)), dtype=np.float32)
            X_test.append(np.vstack([pad, vals]))
        y_test.append(rul_values[idx])
    return np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32)


def sliding_windows_cwru(signal: np.ndarray, window: int = 1024, stride: int = 512) -> np.ndarray:
    """Segment a 1D vibration signal into overlapping windows."""
    windows = []
    for i in range(0, len(signal) - window + 1, stride):
        windows.append(signal[i:i + window])
    return np.array(windows, dtype=np.float32)
