"""
tests/test_stats.py — Unit tests for statistical analysis functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from agents.eda_stats_agent import (
    _descriptive, _outliers, _run_ttest, _run_mannwhitney,
    _run_anova, _run_chisquare, _run_correlation,
)


class TestDescriptive:
    def test_returns_stats_for_numeric(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        result = _descriptive(df)
        assert "x" in result
        assert "mean" in result["x"]

    def test_empty_numeric_returns_empty(self):
        df = pd.DataFrame({"name": ["a", "b"]})
        assert _descriptive(df) == {}


class TestOutliers:
    def test_detects_outlier(self):
        data = [1.0] * 100 + [1000.0]
        df = pd.DataFrame({"val": data})
        result = _outliers(df)
        assert "val" in result
        assert result["val"]["n_outliers"] >= 1

    def test_no_outlier_clean_data(self):
        df = pd.DataFrame({"val": range(100)})
        result = _outliers(df)
        assert "val" not in result  # No IQR outliers in a uniform range


class TestTTest:
    def _df_two_groups(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        g1  = rng.normal(10, 1, 100)
        g2  = rng.normal(20, 1, 100)  # clearly different
        return pd.DataFrame({
            "group": ["A"] * 100 + ["B"] * 100,
            "value": list(g1) + list(g2),
        })

    def test_significant_difference_detected(self):
        result = _run_ttest(self._df_two_groups(), "group", "value")
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert "effect_size_cohens_d" in result
        assert result["test"] == "Welch t-test"

    def test_small_sample_returns_error(self):
        df = pd.DataFrame({"g": ["A", "B"], "v": [1, 2]})
        result = _run_ttest(df, "g", "v")
        assert "error" in result

    def test_three_groups_returns_error(self):
        df = pd.DataFrame({"g": ["A", "B", "C"] * 30, "v": list(range(90))})
        result = _run_ttest(df, "g", "v")
        assert "error" in result


class TestAnova:
    def test_significant_anova(self):
        rng = np.random.default_rng(0)
        df  = pd.DataFrame({
            "group": (["A"] * 50 + ["B"] * 50 + ["C"] * 50),
            "value": list(rng.normal(10, 1, 50)) +
                     list(rng.normal(20, 1, 50)) +
                     list(rng.normal(30, 1, 50)),
        })
        result = _run_anova(df, "group", "value")
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_insufficient_groups(self):
        df = pd.DataFrame({"g": ["A"] * 20 + ["B"] * 20, "v": range(40)})
        result = _run_anova(df, "g", "v")
        assert "error" in result  # needs 3+ groups


class TestChiSquare:
    def test_detects_association(self):
        # Two clearly associated categorical columns
        df = pd.DataFrame({
            "cat1": ["A"] * 100 + ["B"] * 100,
            "cat2": ["X"] * 100 + ["Y"] * 100,
        })
        result = _run_chisquare(df, "cat1", "cat2")
        assert result["significant"] is True
        assert "effect_size_cramers_v" in result

    def test_no_association(self):
        rng = np.random.default_rng(1)
        df  = pd.DataFrame({
            "cat1": rng.choice(["A", "B"], 300),
            "cat2": rng.choice(["X", "Y"], 300),
        })
        result = _run_chisquare(df, "cat1", "cat2")
        assert "p_value" in result  # might or might not be significant


class TestCorrelation:
    def test_positive_correlation(self):
        x = list(range(100))
        df = pd.DataFrame({"x": x, "y": [v + np.random.randn() for v in x]})
        result = _run_correlation(df, "x", "y")
        assert result["rho"] > 0.9
        assert result["significant"] is True
        assert result["test"] == "Spearman correlation"

    def test_insufficient_data(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = _run_correlation(df, "x", "y")
        assert "error" in result
