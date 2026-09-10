"""
tests/test_critic.py — Unit tests for the Critic Agent.

Tests:
  - Valid inputs correctly pass
  - Invalid inputs correctly fail
  - Structured retry info is populated on failure
  - failed_node produces a non-empty failure report
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from agents.critic import critic_node, failed_node


def _base() -> dict:
    return {
        "query":            "test query",
        "query_complexity": "SIMPLE_SQL",
        "plan":             [{"agent": "sql", "task": "Run SQL query"}],
        "current_step":     0,
        "pipeline_status":  "RUNNING",
        "sql_query":        "SELECT 1 AS x",
        "sql_result":       pd.DataFrame({"x": [1, 2, 3]}),
        "sql_meta":         None, "sql_error": None,
        "eda_result":       None, "stat_result": None,
        "segment_result":   None, "forecast_result": None,
        # Evidence registry: must match IDs cited in _good_report()
        "evidence":         [
            {"id": "SQL-01", "agent": "sql", "type": "dataframe",
             "summary": "3 rows × 2 cols: category, revenue", "artifact": "sql_result"},
        ],
        "figures":          [],
        "report":           _good_report(),
        "critic_result":    None,
        "iteration":        0, "agent_timings": {},
        "llm_calls":        0, "total_retries": 0,
    }


def _good_report() -> str:
    return (
        "## Executive Summary\nGood report.\n\n"
        "## Key Data Findings\n- Finding 1 [SQL-01]\n- Finding 2 [SQL-01]\n\n"
        "## Statistical Evidence\nNo testing.\n\n"
        "## Business Implications\nSome implications.\n\n"
        "## Recommended Actions\n1. Do this.\n2. Do that.\n\n"
        "## Limitations & Confidence\nLimited data.\n\n"
        "## Evidence Trace\n[SQL-01] sql — 3 rows"
    )


class TestCriticValidInputs:
    """Valid inputs should PASS."""

    def test_basic_valid_state_passes(self):
        out = critic_node(_base())
        cr  = out["critic_result"]
        assert cr["passed"] is True

    def test_valid_with_stats_passes(self):
        state = {**_base(), "stat_result": {
            "test": "Welch t-test",
            "p_value": 0.001,
            "significant": True,
            "interpretation": "Significant difference.",
        }}
        cr = critic_node(state)["critic_result"]
        assert cr["passed"] is True

    def test_valid_with_segmentation_passes(self):
        state = {**_base(), "segment_result": {
            "silhouette_score": 0.45,
            "best_k": 3,
            "n_customers": 500,
            "cluster_profiles": [
                {"cluster": 0, "n_customers": 200, "mean_recency": 30.0,
                 "mean_frequency": 5.0, "mean_monetary": 800.0,
                 "suggested_interpretation": "High-value active customers"},
                {"cluster": 1, "n_customers": 200, "mean_recency": 200.0,
                 "mean_frequency": 1.0, "mean_monetary": 100.0,
                 "suggested_interpretation": "At-risk or lapsed customers"},
                {"cluster": 2, "n_customers": 100, "mean_recency": 10.0,
                 "mean_frequency": 1.0, "mean_monetary": 120.0,
                 "suggested_interpretation": "New or occasional customers"},
            ],
        }}
        cr = critic_node(state)["critic_result"]
        assert cr["passed"] is True

    def test_valid_forecast_passes(self):
        state = {**_base(), "forecast_result": {
            "method": "Prophet + Baselines",
            "best_model": "Prophet",
            "model_metrics": {
                "Naive":   {"MAE": 500.0, "RMSE": 600.0},
                "Prophet": {"MAE": 200.0, "RMSE": 300.0},
            },
        }}
        cr = critic_node(state)["critic_result"]
        assert cr["passed"] is True


class TestCriticInvalidInputs:
    """Invalid inputs should FAIL with structured feedback."""

    def test_sql_error_fails(self):
        state = {**_base(), "sql_error": "Table does not exist", "sql_result": None}
        out = critic_node(state)
        cr  = out["critic_result"]
        assert cr["passed"] is False
        assert cr["failed_component"] == "sql"
        assert cr["retry_agent"] == "sql"
        assert cr["retry_instruction"] is not None

    def test_empty_sql_result_fails(self):
        state = {**_base(), "sql_result": pd.DataFrame()}
        cr    = critic_node(state)["critic_result"]
        assert cr["passed"] is False
        assert cr["failed_component"] == "sql"

    def test_none_sql_result_fails(self):
        state = {**_base(), "sql_result": None, "sql_error": None}
        cr    = critic_node(state)["critic_result"]
        assert cr["passed"] is False
        assert cr["failed_component"] == "sql"

    def test_low_silhouette_fails(self):
        state = {**_base(), "segment_result": {
            "silhouette_score": 0.05,
            "best_k": 2, "n_customers": 100,
            "cluster_profiles": [
                {"cluster": 0, "n_customers": 50, "mean_recency": 30.0,
                 "mean_frequency": 2.0, "mean_monetary": 200.0,
                 "suggested_interpretation": "Moderate customers"},
                {"cluster": 1, "n_customers": 50, "mean_recency": 100.0,
                 "mean_frequency": 1.0, "mean_monetary": 80.0,
                 "suggested_interpretation": "Moderate customers"},
            ],
        }}
        cr = critic_node(state)["critic_result"]
        assert cr["passed"] is False
        assert cr["failed_component"] == "ml"
        assert cr["retry_agent"] == "ml"

    def test_forecast_error_fails(self):
        state = {**_base(), "forecast_result": {"error": "No date column found"}}
        cr    = critic_node(state)["critic_result"]
        assert cr["passed"] is False
        assert cr["retry_agent"] == "ml"

    def test_short_report_fails(self):
        state = {**_base(), "report": "Too short."}
        cr    = critic_node(state)["critic_result"]
        assert cr["passed"] is False
        assert cr["failed_component"] == "business_analyst"

    def test_missing_sections_fails(self):
        # Long enough to pass length check but missing required sections
        report = (
            "## Executive Summary\n"
            "This is a long enough report to pass the length check. " * 5 + "\n\n"
            "## Statistical Evidence\nSome stats here.\n\n"
            "## Business Implications\nSome implications.\n"
        )
        state = {**_base(), "report": report}
        cr    = critic_node(state)["critic_result"]
        assert cr["passed"] is False
        # Should fail because Key Data Findings and Recommended Actions are missing
        assert cr["failed_component"] == "business_analyst"

    def test_stats_error_fails(self):
        state = {**_base(), "stat_result": {
            "test": "Welch t-test",
            "error": "Insufficient sample size",
        }}
        cr = critic_node(state)["critic_result"]
        assert cr["passed"] is False
        assert cr["retry_agent"] == "eda_stats"

    def test_iteration_incremented_on_failure(self):
        state = {**_base(), "sql_result": pd.DataFrame(), "iteration": 0}
        out   = critic_node(state)
        assert out["iteration"] == 1


class TestEvidenceValidation:
    """Critic should catch phantom evidence IDs in the report."""

    def test_phantom_evidence_id_fails(self):
        # Report cites [SQL-99] but evidence registry only has SQL-01
        state = {
            **_base(),
            "evidence": [{"id": "SQL-01", "agent": "sql", "type": "dataframe",
                           "summary": "3 rows", "artifact": "sql_result"}],
            "report": (
                "## Executive Summary\nRevenue is high. [SQL-99]\n\n"
                "## Key Data Findings\n- Revenue R$2M [SQL-01]\n\n"
                "## Statistical Evidence\nNone.\n\n"
                "## Business Implications\nGood results.\n\n"
                "## Recommended Actions\n1. Expand. 2. Invest. 3. Track.\n\n"
                "## Limitations & Confidence\nLimited period.\n\n"
                "## Evidence Trace\n[SQL-01] sql — revenue data"
            ),
        }
        cr = critic_node(state)["critic_result"]
        assert cr["passed"] is False
        assert cr["failed_component"] == "business_analyst"
        assert "SQL-99" in cr["reason"]

    def test_valid_evidence_ids_pass(self):
        # Report cites only IDs that exist in evidence
        state = {
            **_base(),
            "evidence": [
                {"id": "SQL-01", "agent": "sql", "type": "dataframe",
                 "summary": "3 rows", "artifact": "sql_result"},
                {"id": "STAT-01", "agent": "eda_stats", "type": "statistical_test",
                 "summary": "p=0.001", "artifact": "stat_result"},
            ],
            "report": _good_report().replace("[SQL-01]", "[SQL-01]").replace(
                "## Evidence Trace\n[SQL-01]",
                "## Evidence Trace\n[SQL-01]\n[STAT-01]"
            ),
        }
        cr = critic_node(state)["critic_result"]
        assert cr["passed"] is True

    def test_no_evidence_citations_still_passes(self):
        # Report with NO [ID] citations — no phantom IDs to catch, should pass
        report_no_citations = (
            "## Executive Summary\nRevenue analysis. High earnings noted.\n\n"
            "## Key Data Findings\n"
            "- Electronics leads revenue.\n"
            "- Fashion is second.\n"
            "- Average order value is reasonable.\n\n"
            "## Statistical Evidence\nNo statistical testing performed.\n\n"
            "## Business Implications\nElectronics and fashion dominate.\n\n"
            "## Recommended Actions\n"
            "1. Expand electronics inventory.\n"
            "2. Invest in fashion campaigns.\n"
            "3. Review underperforming categories.\n\n"
            "## Limitations & Confidence\n"
            "Data covers 2016-2018 only.\n\n"
            "## Evidence Trace\nNo evidence IDs cited."
        )
        state = {**_base(), "evidence": [], "report": report_no_citations}
        cr = critic_node(state)["critic_result"]
        assert cr["passed"] is True


class TestFailedNode:
    """failed_node should write an honest failure report."""

    def test_failed_node_writes_report(self):
        state = {
            **_base(),
            "critic_result": {
                "passed":           False,
                "failed_component": "sql",
                "reason":           "SQL returned 0 rows.",
                "retry_agent":      "sql",
                "retry_instruction": "Fix the query.",
            },
            "iteration": 3,
        }
        out = failed_node(state)
        assert out["pipeline_status"] == "FAILED"
        assert out["report"] is not None
        assert len(out["report"]) > 100
        assert "sql" in out["report"].lower() or "failed" in out["report"].lower()

    def test_failed_node_does_not_claim_success(self):
        state = {**_base(), "critic_result": {
            "passed": False, "failed_component": "ml",
            "reason": "Silhouette too low.", "retry_agent": "ml",
            "retry_instruction": "Retry with fewer clusters.",
        }, "iteration": 2}
        out = failed_node(state)
        report = out["report"].lower()
        assert "could not" in report or "failed" in report or "unable" in report
