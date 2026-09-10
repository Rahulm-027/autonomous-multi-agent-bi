"""
agents/supervisor.py — Supervisor Agent

Responsibilities:
  1. Classify query complexity (SIMPLE_SQL / SQL_PLUS_STATS / SQL_PLUS_ML / MULTI_AGENT)
  2. Generate a validated structured analytical plan (Pydantic model)
  3. Route to the correct first worker agent

The plan specifies WHAT each agent should do, including explicit parameters
for statistical tests, ML tasks, and forecasting. Agents do not guess —
they receive explicit instructions from the plan.

Routing (conditional edge):
  After supervisor: go to first planned agent
  After each worker: go to next planned agent OR critic
  After critic PASS: END
  After critic FAIL: targeted retry agent (not full pipeline)
"""

import json
import time
from typing import Optional
from pydantic import BaseModel, field_validator

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from config import get_llm, MAX_RETRIES
from tools.db_loader import load_schema, schema_to_prompt_str


# ── Plan step schema ──────────────────────────────────────────────────────────
class PlanStep(BaseModel):
    step: int
    agent: str                          # sql | eda_stats | ml | business_analyst
    task: str                           # human-readable task description

    # SQL agent hints
    tables: list[str] = []
    expected_columns: list[str] = []

    # Stats agent hints
    stat_question: Optional[str] = None # what question to test
    null_hypothesis: Optional[str] = None
    stat_test: Optional[str] = None     # ttest | mannwhitney | chisquare | anova | correlation
    group_column: Optional[str] = None  # column that defines groups
    value_column: Optional[str] = None  # numeric column to compare

    # ML agent hints
    ml_task: Optional[str] = None       # segmentation | forecasting
    forecast_target: Optional[str] = None
    forecast_freq: Optional[str] = None # M | W | D | Q
    forecast_horizon: int = 3

    @field_validator("agent")
    @classmethod
    def agent_must_be_valid(cls, v: str) -> str:
        valid = {"sql", "eda_stats", "ml", "business_analyst"}
        if v not in valid:
            raise ValueError(f"agent must be one of {valid}, got '{v}'")
        return v


# ── Complexity classifier ─────────────────────────────────────────────────────
COMPLEXITY_PROMPT = """Classify this business question into exactly ONE category.

Categories:
  SIMPLE_SQL       - Only needs data retrieval (aggregations, rankings, filters).
  SQL_PLUS_STATS   - Needs SQL data AND statistical hypothesis testing.
  SQL_PLUS_ML      - Needs SQL data AND segmentation or forecasting.
  ML_ONLY          - Needs only ML-based segmentation or forecasting where the ML
                     agent can build the required input data directly.
  MULTI_AGENT      - Needs SQL + statistics + ML together.

Important:
- RFM customer segmentation is ML_ONLY because the ML agent builds the canonical
  RFM table directly from the database.
- Do NOT classify RFM segmentation as SQL_PLUS_ML.
- Do NOT add SQL merely to retrieve RFM data for the ML agent.

Reply with ONLY the category name, nothing else.

Question: {query}"""

# ── Plan generation prompt ────────────────────────────────────────────────────
PLAN_PROMPT = """You are the Supervisor of a multi-agent business intelligence system.

Produce a JSON analytical plan for this question.

DATABASE SCHEMA:
{schema}

QUESTION: {query}
COMPLEXITY: {complexity}

AGENT RULES:
- "sql": Retrieves data from DuckDB when the requested analysis requires SQL-derived data.
- "eda_stats": Only for statistical tests (t-test, chi-square, correlation, ANOVA).
- "ml": Only for segmentation (RFM + K-Means) OR time-series forecasting.
- "business_analyst": Always last. Writes the final report.
- DO NOT include "business_analyst" if it is not the final step.
- The critic runs automatically — do NOT add it.

ROUTING RULES:
- For SIMPLE_SQL: [sql, business_analyst]
- For SQL_PLUS_STATS: [sql, eda_stats, business_analyst]
- For SQL_PLUS_ML forecasting: [sql, ml, business_analyst]
- For SQL_PLUS_ML segmentation: [ml, business_analyst]
- For ML_ONLY segmentation: [ml, business_analyst]
- For ML_ONLY forecasting: [ml, business_analyst]
- For MULTI_AGENT: include only the agents actually required by the question.
- IMPORTANT: RFM segmentation is ML_ONLY and does NOT require a preceding SQL step.
- Do not add an SQL step merely to provide raw RFM data to the ML agent.

IMPORTANT for ml steps:
- Set "ml_task" to "segmentation" or "forecasting"
- For forecasting: set "forecast_target" (column name), "forecast_freq" (M/W/D), "forecast_horizon" (integer)
- For segmentation: task must say "RFM segmentation" — the agent builds RFM from the DB directly

IMPORTANT for eda_stats steps:
- Set "stat_question", "null_hypothesis", "stat_test"
- Set "group_column" and "value_column" based on the data the SQL step will return

Output ONLY a valid JSON array. No markdown. No explanation.
Example:
[
  {{"step": 1, "agent": "sql", "task": "Retrieve monthly order counts", "tables": ["orders"], "expected_columns": ["month", "order_count"]}},
  {{"step": 2, "agent": "business_analyst", "task": "Summarise findings"}}
]
"""


def _classify_complexity(query: str, llm) -> str:
    """
    Classify the analytical complexity of the business query.

    RFM segmentation is handled directly by the ML agent, which builds
    the canonical RFM dataset from DuckDB. Therefore, RFM queries do
    not require a preceding SQL agent.
    """

    query_lower = query.lower()

    # RFM segmentation is a direct ML workflow.
    rfm_terms = (
        "rfm",
        "recency frequency monetary",
        "customer segmentation",
        "customer segments",
        "segment customers",
        "segment customer",
    )

    if any(term in query_lower for term in rfm_terms):
        return "ML_ONLY"

    prompt = COMPLEXITY_PROMPT.format(query=query)

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])

        classification = resp.content.strip().upper()

        valid = {
            "SIMPLE_SQL",
            "SQL_PLUS_STATS",
            "SQL_PLUS_ML",
            "MULTI_AGENT",
        }

        return classification if classification in valid else "SIMPLE_SQL"

    except Exception:
        return "SIMPLE_SQL"


def _parse_plan(raw: str, query: str) -> list[dict]:
    """Parse and validate plan JSON. Falls back to minimal safe plan."""
    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
        steps = []
        for item in data:
            try:
                step = PlanStep(**item)
                steps.append(step.model_dump())
            except Exception:
                # Skip invalid steps
                continue
        if steps:
            return steps
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback: minimal safe plan
    return [
        {"step": 1, "agent": "sql",
         "task": f"Retrieve data relevant to: {query}",
         "tables": [], "expected_columns": []},
        {"step": 2, "agent": "business_analyst",
         "task": "Summarise findings and provide recommendations"},
    ]


def supervisor_node(state: AgentState) -> AgentState:
    """
    LangGraph node: classifies query, generates plan, initialises state.
    Only runs once at the start — skipped on critic-triggered retries.
    """
    # Do not re-plan on retries
    if state.get("plan"):
        return state

    t0 = time.time()
    llm = get_llm()
    schema = load_schema()
    schema_str = schema_to_prompt_str(schema)

    complexity = _classify_complexity(state["query"], llm)

    plan_prompt = PLAN_PROMPT.format(
        schema=schema_str,
        query=state["query"],
        complexity=complexity,
    )

    try:
        resp = llm.invoke([
            SystemMessage(content="You produce analytical plans as valid JSON arrays."),
            HumanMessage(content=plan_prompt),
        ])
        plan = _parse_plan(resp.content, state["query"])
        llm_calls = state.get("llm_calls", 0) + 2  # classify + plan
    except Exception as e:
        plan = [
            {"step": 1, "agent": "sql",
             "task": f"Retrieve data for: {state['query']}",
             "tables": [], "expected_columns": []},
            {"step": 2, "agent": "business_analyst",
             "task": "Summarise findings"},
        ]
        llm_calls = state.get("llm_calls", 0) + 1

    elapsed = round(time.time() - t0, 2)
    timings = dict(state.get("agent_timings", {}))
    timings["supervisor"] = elapsed

    return {
        **state,
        "plan":             plan,
        "query_complexity": complexity,
        "current_step":     0,
        "iteration":        0,
        "total_retries":    0,
        "pipeline_status":  "RUNNING",
        "figures":          [],
        "evidence":         [],
        "agent_timings":    timings,
        "llm_calls":        llm_calls,
        "critic_result":    None,
    }


# ── Routing ───────────────────────────────────────────────────────────────────
def route_next(state: AgentState) -> str:
    """
    LangGraph conditional edge: decides the next node.

    Priority order:
      1. If critic failed → targeted retry agent (from critic_result)
      2. If max retries exceeded → "failed" terminal node
      3. If more plan steps → next planned agent
      4. All steps done → "critic"
    """
    critic_result = state.get("critic_result")
    iteration     = state.get("iteration", 0)
    plan          = state.get("plan", [])
    current_step  = state.get("current_step", 0)

    # Targeted critic retry
    if critic_result and not critic_result.get("passed", True):
        if iteration <= MAX_RETRIES:
            retry_agent = critic_result.get("retry_agent", "business_analyst")
            if retry_agent in ("sql", "eda_stats", "ml", "business_analyst"):
                return retry_agent
        # Max retries exhausted — fail closed
        return "failed"

    # Next planned step
    if current_step < len(plan):
        agent = plan[current_step].get("agent", "business_analyst")
        return agent

    # All steps done — validate
    return "critic"
