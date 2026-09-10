"""
tests/test_sql_eval.py — Unit tests for the SQL gold-comparison evaluation functions.

Tests that:
  - correct SQL passes all checks
  - wrong but executable SQL fails
  - tiny floating-point differences pass within tolerance
  - materially different numeric results fail
  - unordered results pass after canonical sorting
  - missing / extra columns fail correctly

Run with: python -m pytest tests/test_sql_eval.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate import (
    check_schema,
    check_row_count,
    check_values,
    check_ordering,
    _normalise_colnames,
    _coerce_types,
    _canonical_sort,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _gold() -> pd.DataFrame:
    """Reference 'gold' result for most tests."""
    return pd.DataFrame({
        "city":    ["sao paulo", "rio de janeiro", "belo horizonte"],
        "n_customers": [40000.0, 12000.0, 6000.0],
    })


def _correct_gen() -> pd.DataFrame:
    """Generated result that exactly matches gold (different column order)."""
    return pd.DataFrame({
        "N_Customers": [40000.0, 12000.0, 6000.0],   # different case
        "City":        ["sao paulo", "rio de janeiro", "belo horizonte"],
    })


def _wrong_gen() -> pd.DataFrame:
    """Generated result that is wrong — different numeric values."""
    return pd.DataFrame({
        "city":        ["sao paulo", "rio de janeiro", "belo horizonte"],
        "n_customers": [1.0, 2.0, 3.0],   # completely wrong
    })


def _tiny_diff_gen() -> pd.DataFrame:
    """Generated result with tiny floating-point differences (< 5% rtol)."""
    return pd.DataFrame({
        "city":        ["sao paulo", "rio de janeiro", "belo horizonte"],
        "n_customers": [40001.0, 11999.5, 6000.0001],  # within tolerance
    })


def _extra_col_gen() -> pd.DataFrame:
    """Generated result with an extra column not in gold."""
    return pd.DataFrame({
        "city":        ["sao paulo", "rio de janeiro", "belo horizonte"],
        "n_customers": [40000.0, 12000.0, 6000.0],
        "extra_col":   ["a", "b", "c"],   # not in gold — should not fail schema
    })


def _missing_col_gen() -> pd.DataFrame:
    """Generated result missing a column that exists in gold."""
    return pd.DataFrame({
        "city": ["sao paulo", "rio de janeiro", "belo horizonte"],
        # n_customers missing
    })


def _unordered_gen() -> pd.DataFrame:
    """Generated result with rows in different order than gold."""
    return pd.DataFrame({
        "city":        ["belo horizonte", "sao paulo", "rio de janeiro"],  # scrambled
        "n_customers": [6000.0, 40000.0, 12000.0],
    })


def _desc_sorted() -> pd.DataFrame:
    """Result sorted descending by n_customers."""
    return pd.DataFrame({
        "city":        ["sao paulo", "rio de janeiro", "belo horizonte"],
        "n_customers": [40000.0, 12000.0, 6000.0],
    })


def _asc_sorted() -> pd.DataFrame:
    """Result sorted ascending by n_customers."""
    return pd.DataFrame({
        "city":        ["belo horizonte", "rio de janeiro", "sao paulo"],
        "n_customers": [6000.0, 12000.0, 40000.0],
    })


# ── check_schema ─────────────────────────────────────────────────────────────

class TestCheckSchema:
    def test_exact_match_passes(self):
        ok, msg = check_schema(_correct_gen(), _gold(), ["city", "n_customers"])
        assert ok is True, msg

    def test_case_insensitive_passes(self):
        ok, msg = check_schema(_correct_gen(), _gold(), ["City", "N_Customers"])
        assert ok is True, msg

    def test_missing_column_fails(self):
        ok, msg = check_schema(_missing_col_gen(), _gold(), ["city", "n_customers"])
        assert ok is False
        assert "n_customers" in msg.lower()

    def test_extra_column_in_gen_passes(self):
        """Extra columns in generated output are not penalised."""
        ok, msg = check_schema(_extra_col_gen(), _gold(), ["city", "n_customers"])
        assert ok is True, msg

    def test_empty_expected_columns_uses_gold(self):
        """With no explicit expected_columns, schema is derived from gold."""
        ok, msg = check_schema(_correct_gen(), _gold(), [])
        assert ok is True, msg

    def test_no_gold_uses_expected_columns(self):
        ok, msg = check_schema(_correct_gen(), None, ["city", "n_customers"])
        assert ok is True, msg

    def test_expected_columns_normalised_through_alias_dict(self):
        """
        expected_columns containing an alias such as 'n_orders'
        must match the canonical generated column 'order_count'.
        """
        gen = pd.DataFrame({
            "order_status": ["delivered"],
            "order_count": [96000],
        })

        gold = pd.DataFrame({
            "order_status": ["delivered"],
            "order_count": [96000],
        })

        ok, msg = check_schema(
            gen,
            gold,
            ["order_status", "n_orders"],
        )

        assert ok is True, (
            f"Schema check should accept alias-equivalent columns, "
            f"but got: {msg}"
        )

# ── check_row_count ───────────────────────────────────────────────────────────

class TestCheckRowCount:
    def test_matching_row_count_passes(self):
        ok, msg = check_row_count(_correct_gen(), _gold(), 1, None)
        assert ok is True, msg

    def test_different_row_count_fails(self):
        gen_extra = pd.DataFrame({
            "city": ["a", "b", "c", "d"],
            "n_customers": [1.0, 2.0, 3.0, 4.0],
        })
        ok, msg = check_row_count(gen_extra, _gold(), 1, None)
        assert ok is False
        assert "mismatch" in msg.lower()

    def test_fallback_min_rows_no_gold(self):
        ok, msg = check_row_count(_correct_gen(), None, 2, 5)
        assert ok is True, msg

    def test_fallback_too_few_rows(self):
        ok, msg = check_row_count(_correct_gen(), None, 10, None)
        assert ok is False
        assert "too few" in msg.lower()

    def test_fallback_too_many_rows(self):
        ok, msg = check_row_count(_correct_gen(), None, 1, 2)
        assert ok is False
        assert "too many" in msg.lower()


# ── check_values ─────────────────────────────────────────────────────────────

class TestCheckValues:
    RTOL = 0.05
    ATOL = 1.0

    def test_correct_result_passes(self):
        ok, msg = check_values(_correct_gen(), _gold(), ["city"], self.RTOL, self.ATOL)
        assert ok is True, msg

    def test_wrong_numeric_values_fail(self):
        ok, msg = check_values(_wrong_gen(), _gold(), ["city"], self.RTOL, self.ATOL)
        assert ok is False
        assert "mismatch" in msg.lower()

    def test_tiny_float_diff_passes(self):
        """Differences within rtol=5% and atol=1 should pass."""
        ok, msg = check_values(_tiny_diff_gen(), _gold(), ["city"], self.RTOL, self.ATOL)
        assert ok is True, msg

    def test_material_diff_fails(self):
        """10x different values must fail."""
        gen = pd.DataFrame({
            "city":        ["sao paulo", "rio de janeiro", "belo horizonte"],
            "n_customers": [400000.0, 120000.0, 60000.0],   # 10× too high
        })
        ok, msg = check_values(gen, _gold(), ["city"], self.RTOL, self.ATOL)
        assert ok is False

    def test_unordered_rows_pass_after_canonical_sort(self):
        """Scrambled row order must pass when canonical_key is provided."""
        ok, msg = check_values(
            _unordered_gen(), _gold(), canonical_key=["city"],
            numeric_rtol=self.RTOL, numeric_atol=self.ATOL
        )
        assert ok is True, msg

    def test_unordered_without_key_may_fail(self):
        """Without a canonical key, unordered rows are sorted by all cols —
        numeric sort may produce a different order and expose a mismatch."""
        # The scrambled df sorts differently on n_customers → row alignment differs
        gen = _unordered_gen()
        gold = _gold()
        # This test documents the behaviour: with no canonical key, the sort
        # falls back to all columns. Because city strings sort identically
        # to n_customers DESC here, this may still pass — the test just
        # checks that the function runs without error.
        ok, msg = check_values(gen, gold, canonical_key=[], numeric_rtol=self.RTOL,
                               numeric_atol=self.ATOL)
        # We don't assert pass/fail here — just that it returns a bool
        assert isinstance(ok, bool)

    def test_extra_column_in_gen_ignored(self):
        """Extra columns in gen that don't appear in gold are not compared."""
        ok, msg = check_values(_extra_col_gen(), _gold(), ["city"], self.RTOL, self.ATOL)
        assert ok is True, msg

    def test_no_gold_returns_true(self):
        ok, msg = check_values(_correct_gen(), None, [], self.RTOL, self.ATOL)
        assert ok is True

    def test_string_mismatch_fails(self):
        gen = pd.DataFrame({
            "city":        ["wrong city", "rio de janeiro", "belo horizonte"],
            "n_customers": [40000.0, 12000.0, 6000.0],
        })
        ok, msg = check_values(gen, _gold(), ["city"], self.RTOL, self.ATOL)
        assert ok is False
        assert "mismatch" in msg.lower()

    def test_case_insensitive_string_match(self):
        gen = pd.DataFrame({
            "city":        ["SAO PAULO", "RIO DE JANEIRO", "BELO HORIZONTE"],
            "n_customers": [40000.0, 12000.0, 6000.0],
        })
        ok, msg = check_values(gen, _gold(), ["city"], self.RTOL, self.ATOL)
        assert ok is True, msg


# ── check_ordering ────────────────────────────────────────────────────────────

class TestCheckOrdering:
    def test_correct_desc_order_passes(self):
        ok, msg = check_ordering(_desc_sorted(), _gold(), "n_customers", True)
        assert ok is True, msg

    def test_correct_asc_order_passes(self):
        ok, msg = check_ordering(_asc_sorted(), _gold(), "n_customers", False)
        assert ok is True, msg

    def test_wrong_order_fails(self):
        ok, msg = check_ordering(_asc_sorted(), _gold(), "n_customers", True)
        assert ok is False
        assert "ordering" in msg.lower() or "order" in msg.lower()

    def test_no_sort_column_skips(self):
        ok, msg = check_ordering(_wrong_gen(), _gold(), None, False)
        assert ok is True

    def test_missing_sort_column_fails(self):
        ok, msg = check_ordering(_correct_gen(), _gold(), "nonexistent_col", False)
        assert ok is False

    def test_single_row_passes(self):
        df = pd.DataFrame({"x": [42.0]})
        ok, msg = check_ordering(df, None, "x", True)
        assert ok is True

    def test_alias_normalisation_for_ordering_column(self):
        """
        Ordering checks should treat n_orders and order_count
        as the same canonical metric.
        """
        df = pd.DataFrame({
            "order_status": ["delivered", "shipped", "canceled"],
            "order_count": [100, 50, 10],
        })

        ok, msg = check_ordering(df, None, "n_orders", True)

        assert ok is True, msg


# ── _normalise_colnames, _coerce_types, _canonical_sort ──────────────────────

class TestHelpers:
    def test_normalise_lower_cases(self):
        df = pd.DataFrame({"MyCol": [1], "UPPER": [2]})
        out = _normalise_colnames(df)
        assert list(out.columns) == ["mycol", "upper"]

    def test_coerce_numeric_strings(self):
        # Values that clearly cannot be parsed as dates — large integers
        df  = pd.DataFrame({"val": ["150000", "230000", "310000"]})
        out = _coerce_types(df)
        # After coercion they should be numeric (either int or float)
        assert pd.api.types.is_numeric_dtype(out["val"]), (
            f"Expected numeric dtype, got {out['val'].dtype}"
        )

    def test_canonical_sort_stable(self):
        df = pd.DataFrame({"a": ["c", "a", "b"], "b": [3, 1, 2]})
        out = _canonical_sort(df, ["a"])
        assert list(out["a"]) == ["a", "b", "c"]

    def test_canonical_sort_fallback_all_cols(self):
        df = pd.DataFrame({"x": [3, 1, 2]})
        out = _canonical_sort(df, ["nonexistent"])
        assert list(out["x"]) == [1, 2, 3]
