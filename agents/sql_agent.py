"""
agents/sql_agent.py — SQL Agent

Converts a natural-language task into DuckDB SQL, validates it for safety
and schema correctness, executes it, and returns a DataFrame.

Self-correction loop:
  1. Generate SQL
  2. Validate (safety + schema)
  3. Execute
  4. On failure → send original task + SQL + error back to LLM → retry
  5. Repeat up to SQL_MAX_RETRIES times

On final failure: sets sql_error in state (Critic will catch this).
"""

import re
import time
import duckdb
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from config import get_llm, DB_PATH, SQL_MAX_RETRIES, MAX_QUERY_ROWS
from tools.db_loader import load_schema, schema_to_prompt_str, get_table_names
from tools.sql_validator import validate_sql, SQLValidationError


SYSTEM_PROMPT = """You are an expert DuckDB SQL analyst working with the Olist \
Brazilian e-commerce database.

Given a task and the database schema, write ONE valid DuckDB SQL SELECT query.

STRICT RULES:
1. Output ONLY the raw SQL — no markdown fences, no explanation, no comments.
2. Only use SELECT statements. No DROP, DELETE, UPDATE, INSERT, CREATE, ALTER.
3. Use exact table and column names from the schema.
4. Always include LIMIT {limit} unless the task explicitly requires all rows or an aggregation.
5. DuckDB date functions: DATE_TRUNC('month', col), DATEDIFF('day', start, end), CAST(col AS DATE).
6. For window functions: ROW_NUMBER() OVER (PARTITION BY x ORDER BY y DESC).
7. Use double quotes for column aliases with spaces: SELECT col AS "My Column".
8. If the previous attempt failed, study the error carefully and fix it.
9. Round monetary values to 2 decimal places.
10. Ranking / Top-N:
    - If the task asks for "top N", "highest N", "best N", or similar,
      use ORDER BY on the requested metric followed by LIMIT N.
    - For "top N" or "highest N", sort the ranking metric DESC.
    - For "bottom N" or "lowest N", sort the ranking metric ASC.
    - Apply LIMIT N after ORDER BY.
    - Never omit the LIMIT when the task explicitly requests a specific N.
    - Do not invent a different N from the task.
11. Grouped categorical distributions:
    - When counting or aggregating across categories, statuses, types, or
      states without an explicit Top-N limit, order the result by the
      aggregated metric DESC so the most relevant rows appear first.
    - Examples include:
        * "count by payment type"
        * "orders per status"
        * "sellers per state"
        * "revenue by category"
    - Use the aggregate alias in ORDER BY when possible.
    - Do NOT add a LIMIT unless the user explicitly requests one
      (e.g. "top 10" or "the 5 most common").
    - Do NOT exclude NULL values unless the question explicitly asks
      to filter them.
12. Date-based business comparisons:
    - When a question refers to a calendar "date" (e.g., "after the estimated
      delivery date", "before the due date", "on the same date"), compare
      calendar dates rather than full timestamps when the underlying columns
      contain timestamps.
    - Use CAST(timestamp_column AS DATE) when time-of-day should not affect
      the business definition.
    - For example, "delivered after the estimated delivery date" means:
      CAST(order_delivered_customer_date AS DATE)
      > CAST(order_estimated_delivery_date AS DATE).
    - When comparing dates, handle NULL timestamp values explicitly when
      necessary to avoid counting incomplete records.

SCHEMA:
{schema}
"""

CORRECTION_TEMPLATE = """Your previous SQL failed.

TASK: {task}

PREVIOUS SQL:
{sql}

ERROR:
{error}

Fix the error and write a corrected SQL query.
Output ONLY the corrected SQL — no explanation, no markdown.
"""


def _extract_sql(raw: str) -> str:
    """Strips markdown code fences from LLM output."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lower().startswith("sql"):
            text = text[3:]
        text = text.strip()
    # Remove inline comments that break some DuckDB versions
    text = re.sub(r"--[^\n]*", "", text).strip()
    return text


def sql_agent_node(state: AgentState) -> AgentState:
    """LangGraph node: generates, validates, and executes SQL."""
    t0 = time.time()

    plan         = state.get("plan", [])
    current      = state.get("current_step", 0)
    critic_result = state.get("critic_result")

    # Determine task: from plan step or critic retry instruction
    if critic_result and not critic_result.get("passed", True) \
            and critic_result.get("retry_agent") == "sql":
        task = critic_result.get("retry_instruction", state["query"])
    elif current < len(plan):
        task = plan[current].get("task", state["query"])
    else:
        task = state["query"]

    llm        = get_llm()
    schema     = load_schema()
    schema_str = schema_to_prompt_str(schema)
    tables     = get_table_names()

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT.format(schema=schema_str, limit=MAX_QUERY_ROWS)
    )

    last_sql   = None
    last_error = None
    result_df  = None
    retries    = 0
    llm_calls  = state.get("llm_calls", 0)

    user_content = f"TASK: {task}"

    for attempt in range(SQL_MAX_RETRIES):
        # LLM call
        resp = llm.invoke([system_msg, HumanMessage(content=user_content)])
        sql  = _extract_sql(resp.content)
        last_sql  = sql
        llm_calls += 1

        # Safety + schema validation
        try:
            validate_sql(sql, tables)
        except SQLValidationError as ve:
            last_error = f"Validation error: {ve}"
            user_content = CORRECTION_TEMPLATE.format(
                task=task, sql=sql, error=last_error
            )
            retries += 1
            continue

        # Execution
        try:
            conn      = duckdb.connect(DB_PATH, read_only=True)
            result_df = conn.execute(sql).df()
            conn.close()
            last_error = None
            break
        except Exception as e:
            last_error = str(e)
            user_content = CORRECTION_TEMPLATE.format(
                task=task, sql=sql, error=last_error
            )
            retries += 1

    elapsed = round(time.time() - t0, 2)
    timings = dict(state.get("agent_timings", {}))
    timings["sql"] = elapsed

    evidence = list(state.get("evidence", []))
    sql_meta = {
        "rows":            int(len(result_df)) if result_df is not None else 0,
        "columns":         list(result_df.columns) if result_df is not None else [],
        "execution_time_s": elapsed,
        "retries":         retries,
    }

    if result_df is not None and not result_df.empty:
        evidence.append({
            "id":       f"SQL-{len(evidence)+1:02d}",
            "agent":    "sql",
            "type":     "dataframe",
            "summary":  f"{sql_meta['rows']} rows × {len(sql_meta['columns'])} cols: "
                        f"{sql_meta['columns']}",
            "artifact": "sql_result",
        })

    return {
        **state,
        "sql_query":    last_sql,
        "sql_result":   result_df,
        "sql_meta":     sql_meta,
        "sql_error":    last_error,
        "evidence":     evidence,
        "current_step": current + 1 if not (critic_result and not critic_result.get("passed", True)) else current,
        "agent_timings": timings,
        "llm_calls":    llm_calls,
        "critic_result": None,   # clear after handling
        "total_retries": state.get("total_retries", 0) + retries,
    }
