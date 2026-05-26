"""Tests for src/features.py — shapes, edge cases, and physical sanity checks."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import extract_features, kurtosis, peak_to_peak, rms, spectral_entropy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pure_sine():
    """Pure 50 Hz sine wave — known kurtosis ≈ 1.5, low spectral entropy."""
    t = np.linspace(0, 1, 4096)
    return np.sin(2 * np.pi * 50 * t)


@pytest.fixture
def white_noise():
    """Gaussian white noise — high spectral entropy, kurtosis ≈ 3."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 4096)


@pytest.fixture
def impulsive_signal():
    """Signal with a single spike — high kurtosis (fault-like)."""
    x = np.zeros(4096)
    x[2048] = 100.0
    return x


@pytest.fixture
def batch_windows():
    """Batch of 20 windows, each 2048 samples."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, (20, 2048))


# ---------------------------------------------------------------------------
# rms
# ---------------------------------------------------------------------------

class TestRms:
    def test_returns_scalar(self, pure_sine):
        assert np.isscalar(rms(pure_sine))

    def test_sine_rms_near_point7(self, pure_sine):
        # RMS of sin(x) = 1/sqrt(2) ≈ 0.707
        assert abs(rms(pure_sine) - 1 / np.sqrt(2)) < 0.01

    def test_zero_signal(self):
        assert rms(np.zeros(1024)) == pytest.approx(0.0)

    def test_nonnegative(self, white_noise):
        assert rms(white_noise) >= 0


# ---------------------------------------------------------------------------
# kurtosis
# ---------------------------------------------------------------------------

class TestKurtosis:
    def test_returns_scalar(self, white_noise):
        assert np.isscalar(kurtosis(white_noise))

    def test_gaussian_kurtosis_near_3(self, white_noise):
        # Gaussian kurtosis ≈ 3 (excess kurtosis ≈ 0)
        k = kurtosis(white_noise)
        assert abs(k - 3.0) < 0.5, f"Expected ~3 for Gaussian noise, got {k}"

    def test_impulsive_signal_high_kurtosis(self, impulsive_signal):
        # Impulsive (fault-like) signal should have kurtosis >> 3
        k = kurtosis(impulsive_signal)
        assert k > 10, f"Expected high kurtosis for impulsive signal, got {k}"


# ---------------------------------------------------------------------------
# peak_to_peak
# ---------------------------------------------------------------------------

class TestPeakToPeak:
    def test_returns_scalar(self, pure_sine):
        assert np.isscalar(peak_to_peak(pure_sine))

    def test_sine_peak_to_peak_near_2(self, pure_sine):
        assert abs(peak_to_peak(pure_sine) - 2.0) < 0.01

    def test_constant_signal_is_zero(self):
        assert peak_to_peak(np.ones(1024)) == pytest.approx(0.0)

    def test_nonnegative(self, white_noise):
        assert peak_to_peak(white_noise) >= 0


# ---------------------------------------------------------------------------
# spectral_entropy
# ---------------------------------------------------------------------------

class TestSpectralEntropy:
    def test_returns_scalar(self, pure_sine):
        assert np.isscalar(spectral_entropy(pure_sine))

    def test_nonnegative(self, pure_sine):
        assert spectral_entropy(pure_sine) >= 0

    def test_white_noise_higher_entropy_than_sine(self, white_noise, pure_sine):
        # White noise spreads energy across all frequencies → higher entropy
        assert spectral_entropy(white_noise) > spectral_entropy(pure_sine)


# ---------------------------------------------------------------------------
# extract_features (batch)
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_output_shape(self, batch_windows):
        feats = extract_features(batch_windows)
        # 4 features: rms, kurtosis, peak_to_peak, spectral_entropy
        assert feats.shape == (20, 4)

    def test_no_nans(self, batch_windows):
        feats = extract_features(batch_windows)
        assert not np.any(np.isnan(feats))

    def test_all_finite(self, batch_windows):
        feats = extract_features(batch_windows)
        assert np.all(np.isfinite(feats))

    def test_single_window(self):
        window = np.random.rand(1, 2048)
        feats = extract_features(window)
        assert feats.shape == (1, 4)
