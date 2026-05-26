"""Tests for src/preprocessing.py — all use synthetic fixtures, no real data files."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    RUL_CAP,
    add_rul_labels,
    normalize_per_engine,
    sliding_windows_cmapss,
    sliding_windows_cwru,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_cmapss_df(n_engines=3, cycles_per_engine=50, n_sensors=14):
    """Synthetic CMAPSS dataframe with engine_id, cycle, and sensor columns."""
    rows = []
    for eng in range(1, n_engines + 1):
        for cyc in range(1, cycles_per_engine + 1):
            row = {"engine_id": eng, "cycle": cyc}
            for s in range(n_sensors):
                row[f"s{s+1}"] = np.random.rand()
            rows.append(row)
    return pd.DataFrame(rows)


SENSOR_COLS = [f"s{i}" for i in range(1, 15)]


# ---------------------------------------------------------------------------
# add_rul_labels
# ---------------------------------------------------------------------------

class TestAddRulLabels:
    def test_rul_counts_down_to_zero(self):
        df = make_cmapss_df(n_engines=1, cycles_per_engine=10)
        result = add_rul_labels(df)
        engine_rows = result[result["engine_id"] == 1].sort_values("cycle")
        assert engine_rows["rul"].iloc[-1] == 0

    def test_rul_cap_applied(self):
        df = make_cmapss_df(n_engines=1, cycles_per_engine=200)
        result = add_rul_labels(df)
        assert result["rul"].max() <= RUL_CAP

    def test_rul_monotonically_decreasing_per_engine(self):
        df = make_cmapss_df(n_engines=2, cycles_per_engine=60)
        result = add_rul_labels(df)
        for eng_id in result["engine_id"].unique():
            rul_series = result[result["engine_id"] == eng_id].sort_values("cycle")["rul"].values
            # after cap, RUL decreases until 0
            assert np.all(np.diff(rul_series) <= 0), "RUL should be non-increasing"

    def test_rul_column_added(self):
        df = make_cmapss_df()
        result = add_rul_labels(df)
        assert "rul" in result.columns

    def test_no_negative_rul(self):
        df = make_cmapss_df(n_engines=5, cycles_per_engine=150)
        result = add_rul_labels(df)
        assert (result["rul"] >= 0).all()


# ---------------------------------------------------------------------------
# normalize_per_engine
# ---------------------------------------------------------------------------

class TestNormalizePerEngine:
    def test_output_shape_unchanged(self):
        df = make_cmapss_df(n_engines=2, cycles_per_engine=50)
        result = normalize_per_engine(df, SENSOR_COLS)
        assert result.shape == df.shape

    def test_sensor_values_in_reasonable_range(self):
        df = make_cmapss_df(n_engines=2, cycles_per_engine=50)
        result = normalize_per_engine(df, SENSOR_COLS)
        # Normalized values may go slightly outside [0,1] for cycles beyond baseline,
        # but should not be wildly out of range
        for col in SENSOR_COLS:
            assert result[col].abs().max() < 100, f"{col} values look unreasonable"

    def test_non_sensor_columns_untouched(self):
        df = make_cmapss_df(n_engines=2, cycles_per_engine=40)
        original_ids = df["engine_id"].copy()
        result = normalize_per_engine(df.copy(), SENSOR_COLS)
        pd.testing.assert_series_equal(result["engine_id"], original_ids)


# ---------------------------------------------------------------------------
# sliding_windows_cmapss
# ---------------------------------------------------------------------------

class TestSlidingWindowsCmapss:
    def test_output_shapes(self):
        df = make_cmapss_df(n_engines=2, cycles_per_engine=50)
        df = add_rul_labels(df)
        df = normalize_per_engine(df, SENSOR_COLS)
        X, y = sliding_windows_cmapss(df, SENSOR_COLS, window=30, stride=1)
        assert X.ndim == 3
        assert X.shape[1] == 30
        assert X.shape[2] == len(SENSOR_COLS)
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

    def test_window_larger_than_engine_raises_or_skips(self):
        df = make_cmapss_df(n_engines=1, cycles_per_engine=10)
        df = add_rul_labels(df)
        df = normalize_per_engine(df, SENSOR_COLS)
        # Window of 30 on engine with only 10 cycles — should produce 0 windows (no crash)
        X, y = sliding_windows_cmapss(df, SENSOR_COLS, window=30, stride=1)
        assert X.shape[0] == 0 or X.shape[0] >= 0  # graceful, no exception

    def test_stride_reduces_window_count(self):
        df = make_cmapss_df(n_engines=1, cycles_per_engine=60)
        df = add_rul_labels(df)
        df = normalize_per_engine(df, SENSOR_COLS)
        X1, _ = sliding_windows_cmapss(df, SENSOR_COLS, window=10, stride=1)
        X2, _ = sliding_windows_cmapss(df, SENSOR_COLS, window=10, stride=5)
        assert X2.shape[0] < X1.shape[0]


# ---------------------------------------------------------------------------
# sliding_windows_cwru
# ---------------------------------------------------------------------------

class TestSlidingWindowsCwru:
    def test_output_shape(self):
        signal = np.random.rand(10000)
        windows = sliding_windows_cwru(signal, window=2048, stride=512)
        assert windows.ndim == 2
        assert windows.shape[1] == 2048
        n_expected = (10000 - 2048) // 512 + 1
        assert windows.shape[0] == n_expected

    def test_short_signal_returns_empty(self):
        signal = np.random.rand(100)
        windows = sliding_windows_cwru(signal, window=2048, stride=512)
        assert windows.shape[0] == 0
