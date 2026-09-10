"""
tests/test_forecast.py — Unit tests for forecasting utilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from agents.ml_agent import (
    _naive_forecast, _seasonal_naive, _metrics, _safe_smape, _safe_wape,
)


def _make_ts(n: int = 36) -> pd.DataFrame:
    """Creates a simple monthly time-series."""
    dates = pd.date_range("2016-01-01", periods=n, freq="ME")
    values = np.linspace(100, 200, n) + np.random.default_rng(0).normal(0, 5, n)
    return pd.DataFrame({"ds": dates, "y": values})


class TestNaiveForecast:
    def test_carries_last_value(self):
        ts    = _make_ts(24)
        pred  = _naive_forecast(ts, 3)
        assert len(pred) == 3
        assert all(p == ts["y"].iloc[-1] for p in pred)


class TestSeasonalNaive:
    def test_repeats_last_season(self):
        ts   = _make_ts(24)
        pred = _seasonal_naive(ts, 3, season_length=12)
        assert len(pred) == 3

    def test_falls_back_if_short(self):
        ts   = _make_ts(5)
        pred = _seasonal_naive(ts, 3, season_length=12)
        assert len(pred) == 3  # should fall back to naive


class TestMetrics:
    def test_perfect_forecast(self):
        a = np.array([100.0, 200.0, 300.0])
        m = _metrics(a, a)
        assert m["MAE"]    == 0.0
        assert m["RMSE"]   == 0.0
        assert m["WAPE_%"] == 0.0

    def test_nonzero_metrics(self):
        a = np.array([100.0, 200.0, 300.0])
        p = np.array([110.0, 190.0, 290.0])
        m = _metrics(a, p)
        assert m["MAE"]  > 0
        assert m["RMSE"] > 0

    def test_smape_safe_on_zeros(self):
        a = np.array([0.0, 100.0])
        p = np.array([0.0, 110.0])
        val = _safe_smape(a, p)
        assert not np.isnan(val)
        assert val >= 0

    def test_wape_safe_on_zeros(self):
        a = np.array([0.0, 0.0])
        p = np.array([1.0, 1.0])
        val = _safe_wape(a, p)
        assert val == 0.0  # total is 0, returns 0
