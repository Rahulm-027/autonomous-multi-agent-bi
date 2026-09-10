"""
state.py — Shared AgentState TypedDict for the LangGraph pipeline.

Every agent reads from and writes to this object.
Fields are grouped by producing agent. Every agent result includes
a standard envelope (status, agent_name, execution_time, errors, warnings)
so downstream agents know exactly what they received and from where.

EVIDENCE REGISTRY
─────────────────
state["evidence"] is a list of evidence records. Every agent that produces
an analytical artifact appends a record:

  {
    "id":       "SQL-01",
    "agent":    "sql",
    "type":     "dataframe",
    "summary":  "99k rows, columns: [customer_state, avg_days]",
    "artifact": "sql_result",   # key in state that holds the artifact
  }

The Business Analyst uses these records to cite sources in the report.
The Critic uses them to verify that report claims have traceable support.
"""

from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str                      # Original NL business question
    query_complexity: str           # SIMPLE_SQL | SQL_PLUS_STATS | SQL_PLUS_ML | MULTI_AGENT

    # ── Supervisor ────────────────────────────────────────────────────────────
    plan: list                      # [{step, agent, task, ...}, ...]
    current_step: int
    pipeline_status: str            # RUNNING | FAILED | SUCCESS

    # ── SQL Agent ─────────────────────────────────────────────────────────────
    sql_query: Optional[str]
    sql_result: Optional[Any]       # pandas DataFrame
    sql_meta: Optional[dict]        # {rows, columns, execution_time_s, retries}
    sql_error: Optional[str]        # last error message

    # ── EDA / Stats Agent ─────────────────────────────────────────────────────
    eda_result: Optional[dict]      # descriptive stats, outliers, shape
    stat_result: Optional[dict]     # test, statistic, p_value, effect_size, interpretation

    # ── ML Agent ──────────────────────────────────────────────────────────────
    segment_result: Optional[dict]  # RFM values, clusters, silhouette, profiles
    forecast_result: Optional[dict] # model, metrics (MAE/RMSE/WAPE/sMAPE), baseline

    # ── Evidence registry ─────────────────────────────────────────────────────
    evidence: list                  # [{id, agent, type, summary, artifact}, ...]

    # ── Visualization ─────────────────────────────────────────────────────────
    figures: list                   # paths to saved Plotly HTML files

    # ── Business Analyst ──────────────────────────────────────────────────────
    report: Optional[str]           # final Markdown BI report

    # ── Critic ────────────────────────────────────────────────────────────────
    critic_result: Optional[dict]   # {passed, failed_component, reason, retry_agent, retry_instruction}
    iteration: int                  # retry counter

    # ── Execution tracking ────────────────────────────────────────────────────
    agent_timings: dict             # {agent_name: elapsed_seconds}
    llm_calls: int                  # total LLM calls this run
    total_retries: int              # total retries across all agents
