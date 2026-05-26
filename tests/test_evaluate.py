"""Tests for src/evaluate.py — metric correctness and asymmetry verification."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import mae, pr_auc, rmse, score_fn, threshold_sweep

# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------

class TestRmse:
    def test_perfect_predictions(self):
        y = np.array([10.0, 20.0, 30.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # sqrt((9+16)/2) = sqrt(12.5)
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, 50)
        y_pred = rng.uniform(0, 100, 50)
        assert rmse(y_true, y_pred) >= 0


# ---------------------------------------------------------------------------
# mae
# ---------------------------------------------------------------------------

class TestMae:
    def test_perfect_predictions(self):
        y = np.array([5.0, 10.0, 15.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mae(y_true, y_pred) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# score_fn — NASA asymmetric penalty
# ---------------------------------------------------------------------------

class TestScoreFn:
    def test_perfect_predictions_score_zero(self):
        y = np.array([50.0, 100.0, 10.0])
        assert score_fn(y, y) == pytest.approx(0.0)

    def test_asymmetry_late_prediction_penalized_more(self):
        """Late prediction (predicted > true = optimistic about remaining life) costs more."""
        y_true = np.array([50.0])
        y_late = np.array([60.0])   # predicts 10 cycles MORE than actual → late/optimistic
        y_early = np.array([40.0])  # predicts 10 cycles LESS → early/conservative

        score_late = score_fn(y_true, y_late)
        score_early = score_fn(y_true, y_early)

        assert score_late > score_early, (
            f"Late prediction (score={score_late:.2f}) should cost more than "
            f"early prediction (score={score_early:.2f})"
        )

    def test_nonnegative_output(self):
        rng = np.random.default_rng(1)
        y_true = rng.uniform(0, 125, 100)
        y_pred = rng.uniform(0, 125, 100)
        assert score_fn(y_true, y_pred) >= 0

    def test_single_late_prediction(self):
        # d = y_pred - y_true = 60 - 50 = +10 (late/optimistic)
        # score = exp(d/10) - 1 = exp(1) - 1 ≈ 1.718
        result = score_fn(np.array([50.0]), np.array([60.0]))
        expected = np.exp(10.0 / 10.0) - 1
        assert result == pytest.approx(expected, rel=1e-3)

    def test_single_early_prediction(self):
        # d = y_pred - y_true = 40 - 50 = -10 (early/conservative)
        # score = exp(-d/13) - 1 = exp(10/13) - 1 ≈ 1.158
        result = score_fn(np.array([50.0]), np.array([40.0]))
        expected = np.exp(10.0 / 13.0) - 1
        assert result == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# pr_auc
# ---------------------------------------------------------------------------

class TestPrAuc:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        assert pr_auc(y_true, scores) == pytest.approx(1.0)

    def test_range_zero_to_one(self):
        rng = np.random.default_rng(42)
        y_true = (rng.random(100) > 0.8).astype(int)
        scores = rng.random(100)
        result = pr_auc(y_true, scores)
        assert 0.0 <= result <= 1.0

    def test_all_normal_no_faults(self):
        # If no positive labels, pr_auc should handle gracefully (not crash)
        y_true = np.zeros(10, dtype=int)
        scores = np.random.rand(10)
        # sklearn raises UndefinedMetricWarning but returns 0.0 — just don't crash
        try:
            pr_auc(y_true, scores)
        except Exception as e:
            pytest.fail(f"pr_auc raised unexpectedly: {e}")


# ---------------------------------------------------------------------------
# threshold_sweep
# ---------------------------------------------------------------------------

class TestThresholdSweep:
    def test_returns_arrays(self):
        rng = np.random.default_rng(7)
        y_true = (rng.random(100) > 0.85).astype(int)
        scores = rng.random(100)
        thresholds, far, mfr = threshold_sweep(y_true, scores)
        assert len(thresholds) == len(far) == len(mfr)

    def test_far_and_mfr_in_range(self):
        rng = np.random.default_rng(8)
        y_true = (rng.random(200) > 0.8).astype(int)
        scores = rng.random(200)
        _, far, mfr = threshold_sweep(y_true, scores)
        assert np.all((far >= 0) & (far <= 1))
        assert np.all((mfr >= 0) & (mfr <= 1))
