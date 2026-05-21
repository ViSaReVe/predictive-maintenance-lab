import numpy as np
from scipy.fft import fft


def rms(x: np.ndarray) -> float:
    """Root mean square — overall vibration energy."""
    return float(np.sqrt(np.mean(x ** 2)))


def kurtosis(x: np.ndarray) -> float:
    """
    Kurtosis — sensitivity to impulsive peaks (bearing defect indicator).
    Healthy bearings ≈ 3; fault conditions often > 6.
    """
    mu = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 4))


def peak_to_peak(x: np.ndarray) -> float:
    """Peak-to-peak amplitude."""
    return float(np.max(x) - np.min(x))


def spectral_entropy(x: np.ndarray) -> float:
    """
    Spectral entropy — how spread-out the frequency content is.
    Healthy = broad spectrum (high entropy).
    Fault = energy concentrates at fault frequencies (lower entropy).
    """
    freqs = np.abs(fft(x))[:len(x) // 2]
    psd = freqs ** 2
    psd_norm = psd / (psd.sum() + 1e-10)
    return float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))


def extract_features(windows: np.ndarray) -> np.ndarray:
    """
    Extract 4 statistical/spectral features from each window.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, window_len)

    Returns
    -------
    features : ndarray, shape (n_windows, 4)
        Columns: [rms, kurtosis, peak_to_peak, spectral_entropy]
    """
    feats = []
    for w in windows:
        feats.append([rms(w), kurtosis(w), peak_to_peak(w), spectral_entropy(w)])
    return np.array(feats, dtype=np.float32)
