"""
agents/critic.py — Critic Agent

Runs deterministic validation checks on every upstream agent output.

On failure, returns a STRUCTURED critic_result with:
  - passed: False
  - failed_component: which agent produced the bad output
  - reason: specific description of what failed
  - retry_agent: which agent to retry
  - retry_instruction: targeted instruction for the retry

FAIL-CLOSED BEHAVIOUR
─────────────────────
If critic_passed=False and iteration >= MAX_RETRIES, the pipeline routes
to the "failed" terminal node — it does NOT continue to business_analyst.
The "failed" node writes an honest failure message into state["report"].

This prevents the system from generating plausible-looking reports backed
by failed or unchecked analysis.
"""

import time
from config import MIN_SILHOUETTE, MIN_CLUSTER_SIZE, SIGNIFICANCE_LEVEL, MAX_RETRIES
from state import AgentState


REQUIRED_REPORT_SECTIONS = [
    "## Executive Summary",
    "## Key Data Findings",
    "## Recommended Actions",
]

MIN_REPORT_LENGTH = 200  # characters


def _check_sql(state: AgentState) -> dict | None:
    """Returns failure dict or None if SQL is valid or not required."""

    # SQL is only required when the Supervisor included the SQL agent
    # in the current execution plan.
    plan = state.get("plan", []) or []
    sql_required = any(
        step.get("agent") == "sql"
        for step in plan
    )

    if not sql_required:
        return None
    
    if state.get("sql_error"):
        return {
            "passed":            False,
            "failed_component":  "sql",
            "reason":            f"SQL execution failed: {state['sql_error']}",
            "retry_agent":       "sql",
            "retry_instruction": (
                f"The previous SQL failed with: {state['sql_error']}. "
                f"Original task: {(state.get('plan') or [{}])[0].get('task', state['query'])}. "
                "Rewrite the SQL query to fix this error."
            ),
        }

    df = state.get("sql_result")
    if df is None:
        return {
            "passed":            False,
            "failed_component":  "sql",
            "reason":            "SQL agent returned no result (None).",
            "retry_agent":       "sql",
            "retry_instruction": "No data was returned. Rewrite the SQL to retrieve data.",
        }

    if hasattr(df, "empty") and df.empty:
        return {
            "passed":            False,
            "failed_component":  "sql",
            "reason":            "SQL query returned 0 rows.",
            "retry_agent":       "sql",
            "retry_instruction": (
                "The query returned 0 rows. Check filters, JOINs, and WHERE clauses. "
                "The Olist dataset has ~100k orders from 2016–2018."
            ),
        }

    # Sanity check: result should have at least 1 column and 1 row
    if hasattr(df, "shape") and (df.shape[0] < 1 or df.shape[1] < 1):
        return {
            "passed":            False,
            "failed_component":  "sql",
            "reason":            f"SQL result has invalid shape: {df.shape}",
            "retry_agent":       "sql",
            "retry_instruction": "The result DataFrame is malformed. Rewrite the SQL.",
        }

    return None


def _check_stats(state: AgentState) -> dict | None:
    """Returns failure dict or None. Stats are optional — only checked if present."""
    stat = state.get("stat_result")
    if not stat or stat.get("test") in (None, "none"):
        return None  # not run — that's fine

    if "error" in stat:
        return {
            "passed":            False,
            "failed_component":  "eda_stats",
            "reason":            f"Statistical test error: {stat['error']}",
            "retry_agent":       "eda_stats",
            "retry_instruction": (
                f"The statistical test failed: {stat['error']}. "
                "Check column types and sample sizes, then retry."
            ),
        }

    # Must have a p_value or chi2 to be considered run
    has_result = "p_value" in stat or "chi2" in stat or "rho" in stat
    if not has_result:
        return {
            "passed":            False,
            "failed_component":  "eda_stats",
            "reason":            "Statistical test ran but produced no result value.",
            "retry_agent":       "eda_stats",
            "retry_instruction": "Re-run statistical testing and ensure a valid result is returned.",
        }

    return None


def _check_segmentation(state: AgentState) -> dict | None:
    """Returns failure dict or None. Segmentation is optional."""
    seg = state.get("segment_result")
    if not seg:
        return None

    if "error" in seg:
        return {
            "passed":            False,
            "failed_component":  "ml",
            "reason":            f"Segmentation failed: {seg['error']}",
            "retry_agent":       "ml",
            "retry_instruction": (
                f"Segmentation failed: {seg['error']}. "
                "Retry RFM segmentation from the database."
            ),
        }

    score = seg.get("silhouette_score", 1.0)
    if isinstance(score, (int, float)) and score < MIN_SILHOUETTE:
        return {
            "passed":            False,
            "failed_component":  "ml",
            "reason":            (
                f"Silhouette score {score:.3f} is below minimum threshold "
                f"{MIN_SILHOUETTE}. Clusters are not well-separated."
            ),
            "retry_agent":       "ml",
            "retry_instruction": (
                f"Silhouette score was {score:.3f}. "
                "Retry segmentation — reduce k or use a different k range."
            ),
        }

    # Check no empty clusters
    profiles = seg.get("cluster_profiles", [])
    if profiles:
        for p in profiles:
            if p.get("n_customers", MIN_CLUSTER_SIZE + 1) < MIN_CLUSTER_SIZE:
                return {
                    "passed":            False,
                    "failed_component":  "ml",
                    "reason":            (
                        f"Cluster '{p.get('cluster_label','?')}' has fewer than "
                        f"{MIN_CLUSTER_SIZE} customers."
                    ),
                    "retry_agent":       "ml",
                    "retry_instruction": (
                        f"A cluster has fewer than {MIN_CLUSTER_SIZE} customers. "
                        "Reduce k or increase MIN_CLUSTER_SIZE."
                    ),
                }

    return None


def _check_forecast(state: AgentState) -> dict | None:
    """Returns failure dict or None. Forecasting is optional."""
    fc = state.get("forecast_result")
    if not fc:
        return None

    if "error" in fc:
        return {
            "passed":            False,
            "failed_component":  "ml",
            "reason":            f"Forecast failed: {fc['error']}",
            "retry_agent":       "ml",
            "retry_instruction": (
                f"Forecasting failed: {fc['error']}. "
                "Ensure the SQL result contains a date column and a numeric target column."
            ),
        }

    metrics = fc.get("model_metrics", {})
    if not metrics:
        return {
            "passed":            False,
            "failed_component":  "ml",
            "reason":            "Forecast completed but produced no evaluation metrics.",
            "retry_agent":       "ml",
            "retry_instruction": "Retry forecasting and ensure metrics are computed.",
        }

    return None


def _extract_evidence_ids(report: str) -> list[str]:
    """
    Extracts all evidence citation IDs from a report.
    Matches patterns like [SQL-01], [STAT-02], [RFM-01], [FORECAST-01].
    Returns the full ID strings e.g. ["SQL-01", "RFM-01"].
    """
    import re
    # Match the full [TYPE-NN] pattern; capture just the ID portion
    matches = re.findall(r"\[((?:SQL|STAT|RFM|FORECAST)-\d+)\]", report)
    return matches


def _check_report(state: AgentState) -> dict | None:
    """
    Validates report structure and evidence traceability.

    Checks:
      1. Report is long enough
      2. All required sections are present
      3. Any evidence IDs cited in the report ([SQL-01], [RFM-01], etc.)
         correspond to actual entries in state["evidence"]

    The third check is lightweight but deterministic — it prevents the LLM
    from citing evidence IDs that don't exist in the analytical record.
    """
    report = state.get("report", "") or ""

    # 1. Length check
    if len(report.strip()) < MIN_REPORT_LENGTH:
        return {
            "passed":            False,
            "failed_component":  "business_analyst",
            "reason":            "Report is too short or empty.",
            "retry_agent":       "business_analyst",
            "retry_instruction": "Generate a complete business intelligence report.",
        }

    # 2. Required sections check
    missing = [s for s in REQUIRED_REPORT_SECTIONS if s not in report]
    if missing:
        return {
            "passed":            False,
            "failed_component":  "business_analyst",
            "reason":            f"Report is missing required sections: {missing}",
            "retry_agent":       "business_analyst",
            "retry_instruction": (
                f"The report is missing these sections: {missing}. "
                "Rewrite the complete report including all required sections."
            ),
        }

    # 3. Evidence ID traceability check
    cited_ids  = _extract_evidence_ids(report)
    evidence   = state.get("evidence", []) or []
    valid_ids  = {e["id"] for e in evidence}

    # Find IDs cited in the report that do not exist in the evidence registry
    phantom_ids = [cid for cid in cited_ids if cid not in valid_ids]
    if phantom_ids:
        return {
            "passed":            False,
            "failed_component":  "business_analyst",
            "reason":            (
                f"Report cites evidence IDs that do not exist: {phantom_ids}. "
                f"Valid evidence IDs are: {sorted(valid_ids)}."
            ),
            "retry_agent":       "business_analyst",
            "retry_instruction": (
                f"The report cited non-existent evidence IDs: {phantom_ids}. "
                f"Only cite IDs from this list: {sorted(valid_ids)}. "
                "Rewrite the report using only evidence that actually exists."
            ),
        }

    return None


def critic_node(state: AgentState) -> AgentState:
    """
    LangGraph node: runs all validation checks.
    Sets critic_result with structured failure info for targeted retries.
    Increments iteration counter.
    """
    t0 = time.time()

    # Run all checks — stop at first failure (targeted retry needs one clear fix)
    checks = [
        _check_sql(state),
        _check_stats(state),
        _check_segmentation(state),
        _check_forecast(state),
        _check_report(state),
    ]

    first_failure = next((c for c in checks if c is not None), None)

    elapsed = round(time.time() - t0, 2)
    timings = dict(state.get("agent_timings", {}))
    timings["critic"] = elapsed

    if first_failure:
        first_failure["passed"] = False
        return {
            **state,
            "critic_result": first_failure,
            "iteration":     state.get("iteration", 0) + 1,
            "agent_timings": timings,
        }

    return {
        **state,
        "critic_result": {"passed": True, "failed_component": None,
                          "reason": None, "retry_agent": None,
                          "retry_instruction": None},
        "pipeline_status": "SUCCESS",
        "agent_timings":   timings,
    }


def failed_node(state: AgentState) -> AgentState:
    """
    Terminal failure node — reached when MAX_RETRIES is exhausted.
    Writes an honest failure report instead of a hallucinated one.
    FAIL-CLOSED: never generates a report from unvalidated analysis.
    """
    cr = state.get("critic_result", {}) or {}
    failed_component = cr.get("failed_component", "unknown")
    reason           = cr.get("reason", "Unknown validation failure.")
    iterations       = state.get("iteration", 0)

    failure_report = (
        f"## Analysis Could Not Be Completed\n\n"
        f"The system attempted to answer your question but could not produce "
        f"a validated result after {iterations} attempt(s).\n\n"
        f"**Failed component:** `{failed_component}`\n\n"
        f"**Reason:** {reason}\n\n"
        f"**What you can try:**\n"
        f"1. Rephrase your question with more specific column or table references.\n"
        f"2. Simplify the question — break it into smaller sub-questions.\n"
        f"3. Check that the Olist dataset is fully loaded (run `python setup.py`).\n\n"
        f"*This message is shown instead of a potentially unreliable report.*"
    )

    timings = dict(state.get("agent_timings", {}))
    timings["failed_node"] = 0.0

    return {
        **state,
        "report":          failure_report,
        "pipeline_status": "FAILED",
        "agent_timings":   timings,
    }
