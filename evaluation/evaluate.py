"""
evaluation/evaluate.py — Evaluation Suite

Three evaluation modules:

1. SQL Evaluation
   - 15 test cases with expected schemas and result properties
   - Metrics: execution success rate, schema correctness
   - Does NOT count non-empty = correct

2. Critic Evaluation (Confusion Matrix)
   - 5 valid inputs (should PASS) → TN if passed, FP if failed
   - 5 invalid inputs (should FAIL) → TP if caught, FN if missed
   - Reports: TP, TN, FP, FN, Precision, Recall, F1

3. End-to-End LLM-as-Judge
   - 4 full pipeline runs
   - Scored 0–10 on accuracy, completeness, actionability
   - Labelled as "LLM-based quality estimate" NOT ground-truth accuracy

Results written to evaluation/results.json (loaded by Streamlit Evaluation tab).

Usage:
    python evaluation/evaluate.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import duckdb
from langchain_core.messages import HumanMessage

from config import get_llm, DB_PATH
from agents.critic import critic_node
from state import AgentState


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SQL EVALUATION — Gold-query correctness
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each test case has:
#   question           — the NL question sent to the SQL Agent
#   gold_sql           — reference SQL executed against DuckDB to get the correct result
#   expected_columns   — columns that MUST appear in the generated result
#   expected_min_rows  — minimum acceptable row count
#   expected_max_rows  — maximum acceptable row count (None = no upper bound)
#   sort_column        — if set, the result should be sorted by this column
#   sort_descending    — sort direction
#   value_check        — optional dict: {column: expected_scalar} to verify a
#                        specific aggregate value within numeric tolerance
#   tolerance          — relative tolerance for value comparisons (default 0.05 = 5%)
#
# Evaluation dimensions:
#   Execution Success  — did the generated SQL execute without error?
#   Schema Accuracy    — are all expected_columns present in the result?
#   Row Count Accuracy — does the row count fall within [min, max]?
#   Value Accuracy     — does the top-row value match gold within tolerance?
#   Sort Accuracy      — is the sort order correct (when checkable)?

SQL_TEST_CASES = [
    {
        "name": "Review score distribution",
        "question": "What is the count of orders for each review score from 1 to 5?",
        "gold_sql": """
            SELECT review_score, COUNT(*) AS order_count
            FROM reviews
            WHERE review_score BETWEEN 1 AND 5
            GROUP BY review_score
            ORDER BY review_score
        """,
        "expected_columns":  ["review_score", "order_count"],
        "expected_min_rows": 5,
        "expected_max_rows": 5,
        "sort_column":       "review_score",
        "sort_descending":   False,
        "canonical_key":     ["review_score"],
        "value_check":       None,
        "tolerance":         0.05,
        "note": "Exactly 5 rows (one per score 1–5); order_count must be positive.",
    },
    {
        "name": "Payment type breakdown",
        "question": "What is the count of orders by payment type?",
        "gold_sql": """
            SELECT payment_type, COUNT(DISTINCT order_id) AS order_count
            FROM payments
            GROUP BY payment_type
            ORDER BY order_count DESC
        """,
        "expected_columns":  ["payment_type", "order_count"],
        "expected_min_rows": 3,
        "expected_max_rows": 10,
        "sort_column":       "order_count",
        "sort_descending":   True,
        "canonical_key":     ["payment_type"],
        "value_check":       None,
        "tolerance":         0.05,
        "note": "credit_card should be the most common payment type.",
    },
    {
        "name": "Monthly order volume",
        "question": "Show the total number of orders per month.",
        "gold_sql": """
            SELECT DATE_TRUNC('month', CAST(order_purchase_timestamp AS DATE)) AS month,
                   COUNT(*) AS order_count
            FROM orders
            GROUP BY 1
            ORDER BY 1
        """,
        "expected_columns":  ["month", "order_count"],
        "expected_min_rows": 24,
        "expected_max_rows": 30,
        "sort_column":       "month",
        "sort_descending":   False,
        "canonical_key":     ["month"],
        "value_check":       None,
        "tolerance":         0.05,
        "note": "Dataset spans ~25 months; rows should be in chronological order.",
    },
    {
        "name": "Total orders by status",
        "question": "How many orders exist for each order status?",
        "gold_sql": """
            SELECT order_status, COUNT(*) AS n_orders
            FROM orders
            GROUP BY order_status
            ORDER BY n_orders DESC
        """,
        "expected_columns":  ["order_status", "n_orders"],
        "expected_min_rows": 5,
        "expected_max_rows": 10,
        "sort_column":       "n_orders",
        "sort_descending":   True,
        "canonical_key":     ["order_status"],
        "value_check":       None,
        "tolerance":         0.05,
        "note": "'delivered' must be the largest status by count.",
    },
    {
        "name": "Top 10 customer cities",
        "question": "What are the top 10 cities by number of customers?",
        "gold_sql": """
            SELECT customer_city, COUNT(*) AS n_customers
            FROM customers
            GROUP BY customer_city
            ORDER BY n_customers DESC
            LIMIT 10
        """,
        "expected_columns":  ["customer_city", "n_customers"],
        "expected_min_rows": 10,
        "expected_max_rows": 10,
        "sort_column":       "n_customers",
        "sort_descending":   True,
        "canonical_key":     ["customer_city"],
        "value_check":       None,
        "tolerance":         0.05,
        "note": "Exactly 10 rows; sao paulo should appear near the top.",
    },
    {
        "name": "Sellers by state",
        "question": "How many sellers are there in each state?",
        "gold_sql": """
            SELECT seller_state, COUNT(*) AS n_sellers
            FROM sellers
            GROUP BY seller_state
            ORDER BY n_sellers DESC
        """,
        "expected_columns":  ["seller_state", "n_sellers"],
        "expected_min_rows": 10,
        "expected_max_rows": 30,
        "sort_column":       "n_sellers",
        "sort_descending":   True,
        "canonical_key":     ["seller_state"],
        "value_check":       None,
        "tolerance":         0.05,
        "note": "SP should have the most sellers.",
    },
    {
        "name": "Late deliveries count",
        "question": "How many orders were delivered after the estimated delivery date?",
        "gold_sql": """
            SELECT COUNT(*) AS late_orders
            FROM orders
            WHERE order_delivered_customer_date IS NOT NULL
              AND order_estimated_delivery_date IS NOT NULL
              AND CAST(order_delivered_customer_date AS DATE)
                  > CAST(order_estimated_delivery_date AS DATE)
        """,
        "expected_columns":  ["late_orders"],
        "expected_min_rows": 1,
        "expected_max_rows": 1,
        "sort_column":       None,
        "sort_descending":   False,
        "canonical_key":     [],
        "value_check":       {"late_orders": 6600},
        "tolerance":         0.15,
        "note": "Single aggregate row; value should be in range 5k–8k.",
    },
    {
        "name": "Total customers",
        "question": "How many total customers are in the dataset?",
        "gold_sql": "SELECT COUNT(*) AS total_customers FROM customers",
        "expected_columns":  ["total_customers"],
        "expected_min_rows": 1,
        "expected_max_rows": 1,
        "sort_column":       None,
        "sort_descending":   False,
        "canonical_key":     [],
        "value_check":       {"total_customers": 99441},
        "tolerance":         0.01,
        "note": "Should return exactly 99,441 (or very close).",
    },
    {
        "name": "Distinct order statuses",
        "question": "What are the distinct order statuses in the dataset?",
        "gold_sql": """
            SELECT DISTINCT order_status FROM orders ORDER BY order_status
        """,
        "expected_columns":  ["order_status"],
        "expected_min_rows": 5,
        "expected_max_rows": 10,
        "sort_column":       None,
        "sort_descending":   False,
        "canonical_key":     ["order_status"],
        "value_check":       None,
        "tolerance":         0.05,
        "note": "Should include delivered, shipped, canceled, etc.",
    },
    {
        "name": "Average review score overall",
        "question": "What is the average review score across all orders?",
        "gold_sql": "SELECT ROUND(AVG(review_score), 4) AS avg_review_score FROM reviews",
        "expected_columns":  ["avg_review_score"],
        "expected_min_rows": 1,
        "expected_max_rows": 1,
        "sort_column":       None,
        "sort_descending":   False,
        "canonical_key":     [],
        "value_check":       {"avg_review_score": 4.09},
        "tolerance":         0.05,
        "note": "Should be approximately 4.09.",
    },
    {
        "name": "Monthly revenue trend",
        "question": "Show total payment revenue per month, ordered chronologically.",
        "gold_sql": """
            SELECT DATE_TRUNC('month', CAST(o.order_purchase_timestamp AS DATE)) AS month,
                   ROUND(SUM(p.payment_value), 2) AS total_revenue
            FROM orders o
            JOIN payments p ON o.order_id = p.order_id
            GROUP BY 1
            ORDER BY 1
        """,
        "expected_columns":  ["month", "total_revenue"],
        "expected_min_rows": 20,
        "expected_max_rows": 35,
        "sort_column":       "month",
        "sort_descending":   False,
        "canonical_key":     ["month"],
        "value_check":       None,
        "tolerance":         0.05,
        "note": "Revenue should increase over time. Sort must be chronological.",
    },
    {
        "name": "Top 20 product categories",
        "question": "What are the top 20 product categories by number of products?",
        "gold_sql": """
            SELECT product_category_name, COUNT(*) AS n_products
            FROM products
            GROUP BY product_category_name
            ORDER BY n_products DESC
            LIMIT 20
        """,
        "expected_columns": ["product_category_name", "n_products"],
        "expected_rows_min": 20,
        "expected_rows_max": 20,
        "sort_column": "n_products",
        "sort_descending": True,
        "value_check": None,
        "tolerance": 0.05,
        "note": "Expected top 20 product categories by product count."
    },
    {
        "name": "Total sellers",
        "question": "How many sellers are in the dataset?",
        "gold_sql": "SELECT COUNT(*) AS total_sellers FROM sellers",
        "expected_columns":  ["n_sellers"],
        "expected_min_rows": 1,
        "expected_max_rows": 1,
        "sort_column":       None,
        "sort_descending":   False,
        "canonical_key":     [],
        "value_check":       {"n_sellers": 3095},
        "tolerance":         0.02,
        "note": "Should return approximately 3,095.",
    },
    {
        "name": "Orders with 5-star reviews",
        "question": "How many orders received a 5-star review?",
        "gold_sql": """
            SELECT COUNT(*) AS five_star_orders
            FROM reviews
            WHERE review_score = 5
        """,
        "expected_columns":  ["five_star_orders"],
        "expected_min_rows": 1,
        "expected_max_rows": 1,
        "sort_column":       None,
        "sort_descending":   False,
        "canonical_key":     [],
        "value_check":       {"five_star_orders": 57328},
        "tolerance":         0.05,
        "note": "Should be approximately 57,000–58,000.",
    },
    {
        "name": "Customer states",
        "question": "How many customers are there per state?",
        "gold_sql": """
            SELECT customer_state, COUNT(*) AS n_customers
            FROM customers
            GROUP BY customer_state
            ORDER BY n_customers DESC
        """,
        "expected_columns":  ["customer_state", "n_customers"],
        "expected_min_rows": 20,
        "expected_max_rows": 30,
        "sort_column":       "n_customers",
        "sort_descending":   True,
        "canonical_key":     ["customer_state"],
        "value_check":       None,
        "tolerance":         0.05,
        "note": "SP should have the most customers by a large margin.",
    },
]


def _run_gold_sql(gold_sql: str) -> pd.DataFrame | None:
    """Executes the gold SQL against DuckDB. Returns None on failure."""
    try:
        conn   = duckdb.connect(DB_PATH, read_only=True)
        result = conn.execute(gold_sql).df()
        conn.close()
        return result
    except Exception as e:
        print(f"[WARNING] Gold SQL failed: {e}")
        return None


# ── Normalisation helpers ─────────────────────────────────────────────────────

_COLUMN_ALIASES = {
    # Order counts
    "total_orders": "n_orders",
    "order count": "n_orders",
    "order_count": "n_orders",
    "n_orders": "n_orders",

    # Customer counts
    "customer count": "n_customers",
    "customer_count": "n_customers",
    "n_customers": "n_customers",

    # Seller counts
    "seller count": "n_sellers",
    "seller_count": "n_sellers",
    "n_sellers": "n_sellers",
    "number of sellers": "n_sellers",
    "total_sellers": "n_sellers",

    # Product counts
    "product count": "n_products",
    "product_count": "n_products",
    "n_products": "n_products",

    # Category
    "product category name": "category",
    "product_category_name": "category",
    "category": "category",

    # Revenue
    "total revenue": "total_revenue",
    "total_revenue": "total_revenue",

    # Review score
    "average review score": "avg_review_score",
    "avg_review_score": "avg_review_score",

    # Late deliveries
    "late deliveries": "late_orders",
    "late_orders": "late_orders",
    "delayed_orders": "late_orders",

    # Five-star orders
    "5-star orders": "five_star_orders",
    "5_star_orders": "five_star_orders",
}


def _normalise_col_str(col: str) -> str:
    key = str(col).lower().strip()
    key = " ".join(key.split())
    return _COLUMN_ALIASES.get(key, key)


def _normalise_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalise_col_str(c) for c in df.columns]
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerces columns to comparable types:
    - Object columns that parse successfully as dates → datetime (UTC-naive)
    - Object columns where ALL non-null values are numeric → float
    Everything else left as-is.
    """
    df = df.copy()
    for col in df.columns:
        # Skip columns that are already numeric or datetime
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        # Process string-like columns (object dtype or StringDtype in Pandas 2.x)
        if df[col].dtype != object and not isinstance(df[col].dtype, pd.StringDtype):
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        # Try datetime first (use a sample to avoid false positives on plain numbers)
        sample = series.iloc[0]
        looks_like_date = (
            isinstance(sample, str)
            and any(c in sample for c in ("-", "/", "T", " "))
            and not sample.replace(".", "").replace("-", "").isdigit()
        )
        if looks_like_date:
            try:
                converted = pd.to_datetime(df[col], utc=False, errors="raise")
                df[col] = converted.dt.tz_localize(None)
                continue
            except Exception:
                pass

        # Try numeric: coerce and check how many succeeded
        numeric_attempt = pd.to_numeric(df[col], errors="coerce")
        n_non_null      = df[col].notna().sum()
        n_converted     = numeric_attempt.notna().sum()
        if n_non_null > 0 and n_converted == n_non_null:
            df[col] = numeric_attempt

    return df


def _canonical_sort(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """
    Sorts a DataFrame by the declared canonical key columns (lower-cased).
    Falls back to sorting by all columns if key is absent.
    """
    available = [c for c in key_cols if c in df.columns]
    if not available:
        available = sorted(df.columns.tolist())
    return df.sort_values(available).reset_index(drop=True)


# ── Per-dimension correctness checks ─────────────────────────────────────────

def check_schema(
    gen: pd.DataFrame,
    gold: pd.DataFrame,
    expected_columns: list[str],
) -> tuple[bool, str]:

    # Normalise generated and gold column names
    gen_norm = _normalise_colnames(gen)
    gold_norm = _normalise_colnames(gold) if gold is not None else None

    # Normalise expected column names as well
    expected_norm = _normalise_colnames(
        pd.DataFrame(columns=expected_columns)
    ).columns.tolist()

    gen_cols = {c.lower() for c in gen_norm.columns}
    gold_cols = (
        {c.lower() for c in gold_norm.columns}
        if gold_norm is not None
        else set()
    )

    required = {c.lower() for c in expected_norm} | gold_cols

    missing = required - gen_cols

    if missing:
        return (
            False,
            f"Missing columns: {sorted(missing)}. "
            f"Generated has: {sorted(gen_cols)}"
        )

    return True, ""


def check_row_count(
    gen: pd.DataFrame,
    gold: pd.DataFrame | None,
    min_rows: int,
    max_rows: int | None,
) -> tuple[bool, str]:
    """
    Primary check: generated row count must equal gold row count (if gold exists).
    Fallback: must be within [min_rows, max_rows].
    Returns (passed, reason).
    """
    n_gen = len(gen)

    if gold is not None:
        n_gold = len(gold)
        if n_gen != n_gold:
            return False, (
                f"Row count mismatch: generated={n_gen}, gold={n_gold}"
            )
        return True, ""

    # Fallback to range
    if n_gen < min_rows:
        return False, f"Too few rows: {n_gen} < {min_rows}"
    if max_rows is not None and n_gen > max_rows:
        return False, f"Too many rows: {n_gen} > {max_rows}"
    return True, ""


def check_values(
    gen: pd.DataFrame,
    gold: pd.DataFrame,
    canonical_key: list[str],
    numeric_rtol: float,
    numeric_atol: float,
) -> tuple[bool, str]:
    """
    Compares generated result to gold result cell-by-cell.

    Steps:
      1. Normalise column names and coerce compatible types.
      2. Sort both by canonical_key so row order is comparable.
      3. For each column that appears in both:
         - numeric: pass if |gen - gold| <= atol + rtol * |gold|  (numpy allclose semantics)
         - string/date: require exact equality after str normalisation
      4. Return (True, "") if all columns pass, else (False, first failure reason).

    Skips columns present in gen but absent from gold (extra columns are not penalised).
    Missing gold columns have already been caught by check_schema.
    """
    if gold is None or gold.empty:
        return True, "No gold result — value check skipped"

    gen_n  = _normalise_colnames(_coerce_types(gen))
    gold_n = _normalise_colnames(_coerce_types(gold))

    # Only compare columns that appear in gold
    shared = [c for c in gold_n.columns if c in gen_n.columns]
    if not shared:
        return False, "No shared columns between generated and gold after normalisation"

    # Align row order via canonical sort
    key_lower = [c.lower() for c in canonical_key]
    gen_s  = _canonical_sort(gen_n[shared],  key_lower)
    gold_s = _canonical_sort(gold_n[shared], key_lower)

    # Truncate to the shorter length if row counts differ (row_count check handles the mismatch)
    n = min(len(gen_s), len(gold_s))
    gen_s  = gen_s.iloc[:n].reset_index(drop=True)
    gold_s = gold_s.iloc[:n].reset_index(drop=True)

    for col in shared:
        g_col = gold_s[col]
        p_col = gen_s[col]

        # Numeric comparison
        if pd.api.types.is_numeric_dtype(g_col) and pd.api.types.is_numeric_dtype(p_col):
            g_vals = g_col.astype(float).values
            p_vals = p_col.astype(float).values
            abs_diff = abs(p_vals - g_vals)
            allowed  = numeric_atol + numeric_rtol * abs(g_vals)
            bad_idx  = (abs_diff > allowed).nonzero()[0]
            if len(bad_idx) > 0:
                i = bad_idx[0]
                return False, (
                    f"Numeric mismatch in '{col}' at row {i}: "
                    f"generated={p_vals[i]:.6g}, gold={g_vals[i]:.6g}, "
                    f"abs_diff={abs_diff[i]:.6g}, allowed={allowed[i]:.6g}"
                )

        # Datetime comparison
        elif pd.api.types.is_datetime64_any_dtype(g_col):
            g_str = g_col.dt.strftime("%Y-%m").values
            p_str = (p_col.dt.strftime("%Y-%m").values
                     if pd.api.types.is_datetime64_any_dtype(p_col)
                     else p_col.astype(str).values)
            bad = g_str != p_str
            if bad.any():
                i = bad.nonzero()[0][0]
                return False, (
                    f"Date mismatch in '{col}' at row {i}: "
                    f"generated={p_str[i]}, gold={g_str[i]}"
                )

        # String / categorical comparison (case-insensitive strip)
        else:
            g_str = g_col.astype(str).str.lower().str.strip().values
            p_str = p_col.astype(str).str.lower().str.strip().values
            bad   = g_str != p_str
            if bad.any():
                i = bad.nonzero()[0][0]
                return False, (
                    f"String mismatch in '{col}' at row {i}: "
                    f"generated='{p_str[i]}', gold='{g_str[i]}'"
                )

    return True, ""


def check_ordering(
    gen: pd.DataFrame,
    gold: pd.DataFrame | None,
    sort_column: str | None,
    sort_descending: bool,
) -> tuple[bool, str]:
    """
    Checks that gen is sorted by sort_column in the declared direction.
    Uses gold to determine whether the query is inherently ordered:
    if gold itself is not sorted by sort_column, this check is skipped.

    Requires ≥ 80% of consecutive pairs to be in order (allows minor ties).
    Returns (passed, reason). Returns (True, "") if sort_column is None.
    """
    if not sort_column:
        return True, ""

    sort_column_normalised = _normalise_col_str(sort_column)
    matched = next(
        (c for c in gen.columns if _normalise_col_str(c) == sort_column_normalised),
        None,
    )
    if not matched:
        return False, f"Sort column '{sort_column}' not found in generated result"

    vals = gen[matched].reset_index(drop=True)
    total_pairs = len(vals) - 1
    if total_pairs <= 0:
        return True, ""

    pairs_ok = sum(
        1
        for i in range(total_pairs)
        if (vals.iloc[i] >= vals.iloc[i + 1]) == sort_descending
        or vals.iloc[i] == vals.iloc[i + 1]  # ties are fine
    )
    fraction = pairs_ok / total_pairs
    if fraction < 0.80:
        direction = "DESC" if sort_descending else "ASC"
        return False, (
            f"Ordering check failed for '{sort_column}' {direction}: "
            f"only {fraction:.0%} of consecutive pairs in correct order"
        )
    return True, ""


def _sanity_check(
    gen: pd.DataFrame,
    value_check: dict | None,
    tolerance: float,
) -> tuple[bool, str]:
    """
    Optional hardcoded scalar sanity checks (kept as a secondary guard, not
    the primary correctness reference). Only runs when value_check is set.
    Returns (passed, reason).
    """
    if not value_check:
        return True, ""
    for col, expected_val in value_check.items():
        target_col = _normalise_col_str(col)

        matched = next(
            (c for c in gen.columns if _normalise_col_str(c) == target_col),
            None,
        )
        if matched is None:
            return False, f"Sanity-check column '{col}' not in generated result"
        try:
            actual   = float(gen[matched].iloc[0])
            expected = float(expected_val)
            rel_err  = abs(actual - expected) / max(abs(expected), 1)
            if rel_err > tolerance:
                return False, (
                    f"Sanity check failed for '{col}': "
                    f"got {actual:.2f}, expected ≈{expected:.2f} "
                    f"(rel_err {rel_err:.1%} > {tolerance:.0%})"
                )
        except (TypeError, ValueError):
            pass
    return True, ""


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_sql_evaluation() -> dict:
    """
    Evaluates SQL Agent correctness using generated-vs-gold DataFrame comparison.

    For each test case:
      1. Execute the generated SQL (via SQL Agent with self-correction).
      2. Execute the gold SQL directly against DuckDB.
      3. Compare on five dimensions:
           execution_success — SQL ran without error
           schema_correct    — all expected/gold columns present
           row_count_correct — generated row count == gold row count
           value_correct     — cell-by-cell comparison within tolerance
           ordering_correct  — sort order matches declaration
      4. overall_correct requires all five to pass.

    Numeric comparison uses relative + absolute tolerance.
    String/date columns require exact equality (case-insensitive, stripped).
    Both DataFrames are sorted by a declared canonical key before cell comparison
    so that logically unordered queries can still be compared correctly.

    Hardcoded scalar sanity checks (value_check) remain as a secondary guard
    but the gold DataFrame is the primary correctness reference.
    """
    print("\n── SQL Evaluation (Generated vs Gold) ──────────────")
    from agents.sql_agent import sql_agent_node

    # Configurable tolerances
    NUMERIC_RTOL = 0.05   # 5 % relative tolerance for numeric columns
    NUMERIC_ATOL = 1.0    # absolute tolerance (handles near-zero values)

    total            = len(SQL_TEST_CASES)
    exec_successes   = 0
    schema_passes    = 0
    row_count_passes = 0
    value_passes     = 0
    order_passes     = 0
    overall_passes   = 0
    total_retries    = 0
    results          = []

    for i, tc in enumerate(SQL_TEST_CASES):
        name = tc["name"]
        print(f"  [{i+1:>2}/{total}] {name:<42}", end=" ", flush=True)

        # ── Run gold SQL first (synchronous, no LLM) ──────────────────────────
        gold_df = _run_gold_sql(tc["gold_sql"])

        # ── Run generated SQL through the SQL Agent ───────────────────────────
        state: AgentState = {
            "query":            tc["question"],
            "query_complexity": "SIMPLE_SQL",
            "plan":             [{"step": 1, "agent": "sql", "task": tc["question"],
                                  "tables": [], "expected_columns": []}],
            "current_step":     0, "pipeline_status": "RUNNING",
            "sql_query":        None, "sql_result":   None,
            "sql_meta":         None, "sql_error":    None,
            "eda_result":       None, "stat_result":  None,
            "segment_result":   None, "forecast_result": None,
            "evidence":         [], "figures": [],
            "report":           None, "critic_result": None,
            "iteration":        0,   "agent_timings": {},
            "llm_calls":        0,   "total_retries": 0,
        }

        record = {
            "name":               name,
            "execution_success":  False,
            "schema_correct":     None,
            "row_count_correct":  None,
            "value_correct":      None,
            "ordering_correct":   None,
            "overall_correct":    False,
            "rows_generated":     0,
            "rows_gold":          len(gold_df) if gold_df is not None else None,
            "retries":            0,
            "failure_reasons":    [],
        }

        try:
            out     = sql_agent_node(state)
            gen_df  = out.get("sql_result")
            err     = out.get("sql_error")
            meta    = out.get("sql_meta", {}) or {}
            print(f"\n       SQL: {out.get('sql_query')}")
            retries = meta.get("retries", 0)
            total_retries     += retries
            record["retries"]  = retries

            if err or gen_df is None or (hasattr(gen_df, "empty") and gen_df.empty):
                reason = err or "empty result"
                print(f"EXEC_FAIL  {reason[:60]}")
                record["failure_reasons"].append(reason)
                results.append(record)
                time.sleep(0.8)
                continue

            exec_successes              += 1
            record["execution_success"]  = True
            record["rows_generated"]     = len(gen_df)

            # Normalise generated column names once so schema, value, and ordering
            # checks all use the same canonical names.
            gen_df = _normalise_colnames(gen_df)

            # ── Schema ────────────────────────────────────────────────────────
            schema_ok, schema_msg = check_schema(
                gen_df, gold_df, tc.get("expected_columns", [])
            )
            record["schema_correct"] = schema_ok
            if schema_ok:
                schema_passes += 1
            else:
                record["failure_reasons"].append(schema_msg)

            # ── Row count ─────────────────────────────────────────────────────
            rc_ok, rc_msg = check_row_count(
                gen_df, gold_df,
                tc.get("expected_min_rows", 1),
                tc.get("expected_max_rows"),
            )
            record["row_count_correct"] = rc_ok
            if rc_ok:
                row_count_passes += 1
            else:
                record["failure_reasons"].append(rc_msg)

            # ── Value comparison (gold DataFrame is primary reference) ─────────
            if gold_df is not None and not gold_df.empty:
                val_ok, val_msg = check_values(
                    gen_df, gold_df,
                    canonical_key=tc.get("canonical_key", []),
                    numeric_rtol=NUMERIC_RTOL,
                    numeric_atol=NUMERIC_ATOL,
                )
            else:
                # No gold → fall back to hardcoded sanity check only
                val_ok, val_msg = _sanity_check(
                    gen_df, tc.get("value_check"), tc.get("tolerance", 0.05)
                )
            record["value_correct"] = val_ok
            if val_ok:
                value_passes += 1
            else:
                record["failure_reasons"].append(val_msg)

            # ── Ordering ──────────────────────────────────────────────────────
            ord_ok, ord_msg = check_ordering(
                gen_df, gold_df,
                tc.get("sort_column"),
                tc.get("sort_descending", False),
            )
            record["ordering_correct"] = ord_ok
            # Only score ordering when the test explicitly requires an order.
            if tc.get("sort_column"):
                if ord_ok:
                    order_passes += 1
                else:
                    record["failure_reasons"].append(ord_msg)

            # ── Optional scalar sanity check (secondary guard) ─────────────────
            if tc.get("value_check"):
                san_ok, san_msg = _sanity_check(
                    gen_df, tc["value_check"], tc.get("tolerance", 0.05)
                )
                if not san_ok:
                    record["failure_reasons"].append(f"[sanity] {san_msg}")
                    # Sanity failure degrades overall even if gold comparison passed
                    val_ok = val_ok and san_ok

            overall = schema_ok and rc_ok and val_ok and ord_ok
            record["overall_correct"] = overall
            if overall:
                overall_passes += 1

            status = "PASS" if overall else "PARTIAL"
            suffix = (f"  issues={record['failure_reasons']}" if not overall else "")
            print(f"{status}  gen={len(gen_df)} gold={record['rows_gold']}{suffix}")

        except Exception as e:
            print(f"ERROR  {e}")
            record["failure_reasons"].append(str(e))

        results.append(record)
        time.sleep(0.8)

    n_with_sort = sum(1 for tc in SQL_TEST_CASES if tc.get("sort_column"))

    metrics = {
        "total":                      total,
        "execution_success":          exec_successes,
        "schema_correct":             schema_passes,
        "row_count_correct":          row_count_passes,
        "value_correct":              value_passes,
        "ordering_correct":           order_passes,
        "overall_correct":            overall_passes,
        "avg_retries":                round(total_retries / total, 2),
        "execution_success_rate_pct": round(exec_successes   / total             * 100, 1),
        "schema_accuracy_pct":        round(schema_passes     / total             * 100, 1),
        "row_count_accuracy_pct":     round(row_count_passes  / total             * 100, 1),
        "value_accuracy_pct":         round(value_passes      / total             * 100, 1),
        "ordering_accuracy_pct":      round(order_passes      / max(n_with_sort, 1) * 100, 1),
        "overall_accuracy_pct":       round(overall_passes    / total             * 100, 1),
        "details":                    results,
    }

    print(f"\n  Execution success:  {exec_successes}/{total}  = {metrics['execution_success_rate_pct']}%")
    print(f"  Schema accuracy:    {schema_passes}/{total}  = {metrics['schema_accuracy_pct']}%")
    print(f"  Row count accuracy: {row_count_passes}/{total}  = {metrics['row_count_accuracy_pct']}%")
    print(f"  Value accuracy:     {value_passes}/{total}  = {metrics['value_accuracy_pct']}%")
    if n_with_sort:
        print(f"  Ordering accuracy:  {order_passes}/{n_with_sort}  = {metrics['ordering_accuracy_pct']}%")
    print(f"  OVERALL accuracy:   {overall_passes}/{total}  = {metrics['overall_accuracy_pct']}%")
    print(f"  Avg retries:        {metrics['avg_retries']}\n")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CRITIC EVALUATION — Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════════

def _base_state() -> AgentState:
    return {
        "query":            "test", "query_complexity": "SIMPLE_SQL",
        "plan":             [{"agent": "sql", "task": "Run SQL query"}],
        "current_step":     0,
        "pipeline_status":  "RUNNING",
        "sql_query":        "SELECT 1 AS x", "sql_result": None,
        "sql_meta":         None, "sql_error": None,
        "eda_result":       None, "stat_result": None,
        "segment_result":   None, "forecast_result": None,
        "evidence":         [
            {
                "id": "SQL-01",
                "agent": "sql",
                "type": "dataframe",
                "summary": "3 rows × 2 cols: category, revenue",
                "artifact": "sql_result",
            }
        ],
        "figures":          [],
        "report":           None, "critic_result": None,
        "iteration":        0, "agent_timings": {},
        "llm_calls":        0, "total_retries": 0,
    }


def _good_report() -> str:
    return (
        "## Executive Summary\nRevenue analysis across 9 product categories. "
        "Electronics leads with R$2.1M. [SQL-01]\n\n"
        "## Key Data Findings\n"
        "- Electronics: R$2.1M [SQL-01]\n"
        "- Fashion: R$1.8M [SQL-01]\n"
        "- Average order value: R$160 [SQL-01]\n\n"
        "## Statistical Evidence\nNo statistical testing performed for this query.\n\n"
        "## Business Implications\n"
        "Electronics and fashion dominate revenue. Sellers should invest here.\n\n"
        "## Recommended Actions\n"
        "1. Expand electronics inventory.\n"
        "2. Launch targeted campaigns for fashion segment.\n"
        "3. Investigate underperforming categories.\n\n"
        "## Limitations & Confidence\n"
        "Analysis covers 2016–2018 only. Recent trends may differ.\n\n"
        "## Evidence Trace\n[SQL-01] sql — 10 rows × 2 cols: category, revenue"
    )


def _good_df() -> pd.DataFrame:
    return pd.DataFrame({
        "category": ["electronics", "fashion", "home"],
        "revenue":  [2100000.0, 1800000.0, 950000.0],
    })


VALID_CASES = [
    # Should PASS (TN when they pass, FP when they fail)
    {
        "name": "Valid: good SQL + good report",
        "state_overrides": {
            "sql_result": _good_df(),
            "report":     _good_report(),
        },
    },
    {
        "name": "Valid: SQL only, minimal report",
        "state_overrides": {
            "sql_result": _good_df(),
            "report":     _good_report(),
        },
    },
    {
        "name": "Valid: with stats result",
        "state_overrides": {
            "sql_result": _good_df(),
            "stat_result": {
                "test": "Welch t-test", "p_value": 0.001,
                "significant": True, "effect_size_cohens_d": 0.85,
                "interpretation": "Significant difference found.",
            },
            "report": _good_report(),
        },
    },
    {
        "name": "Valid: segmentation passes silhouette",
        "state_overrides": {
            "sql_result": _good_df(),
            "segment_result": {
                "silhouette_score": 0.45, "best_k": 3, "n_customers": 500,
                "cluster_profiles": [
                    {"cluster": 0, "n_customers": 200, "mean_recency": 30.0,
                     "mean_frequency": 5.0, "mean_monetary": 800.0,
                     "recency_level": "High (recent)", "frequency_level": "High",
                     "monetary_level": "High",
                     "suggested_interpretation": "High-value active customers"},
                    {"cluster": 1, "n_customers": 200, "mean_recency": 180.0,
                     "mean_frequency": 1.5, "mean_monetary": 150.0,
                     "recency_level": "Low (lapsed)", "frequency_level": "Low",
                     "monetary_level": "Low",
                     "suggested_interpretation": "At-risk or lapsed customers"},
                    {"cluster": 2, "n_customers": 100, "mean_recency": 10.0,
                     "mean_frequency": 1.0, "mean_monetary": 120.0,
                     "recency_level": "High (recent)", "frequency_level": "Low",
                     "monetary_level": "Low",
                     "suggested_interpretation": "New or occasional customers"},
                ],
            },
            "report": _good_report(),
        },
    },
    {
        "name": "Valid: forecast with metrics",
        "state_overrides": {
            "sql_result": _good_df(),
            "forecast_result": {
                "method": "Prophet + Baselines", "best_model": "Prophet",
                "model_metrics": {
                    "Naive":           {"MAE": 500.0, "RMSE": 600.0, "WAPE_%": 20.0, "sMAPE_%": 18.0},
                    "Seasonal Naive":  {"MAE": 450.0, "RMSE": 550.0, "WAPE_%": 18.0, "sMAPE_%": 16.0},
                    "Prophet":         {"MAE": 300.0, "RMSE": 380.0, "WAPE_%": 12.0, "sMAPE_%": 11.0},
                },
            },
            "report": _good_report(),
        },
    },
]

INVALID_CASES = [
    # Should FAIL (TP when caught, FN when missed)
    {
        "name": "Invalid: SQL execution error",
        "state_overrides": {
            "sql_error": "Table 'nonexistent_table' does not exist",
            "sql_result": None,
            "report":     _good_report(),
        },
    },
    {
        "name": "Invalid: empty SQL result",
        "state_overrides": {
            "sql_result": pd.DataFrame(),
            "report":     _good_report(),
        },
    },
    {
        "name": "Invalid: segmentation silhouette too low",
        "state_overrides": {
            "sql_result": _good_df(),
            "segment_result": {
                "silhouette_score": 0.05, "best_k": 3, "n_customers": 300,
                "cluster_profiles": [
                    {"cluster": 0, "n_customers": 100, "mean_recency": 30.0,
                     "mean_frequency": 3.0, "mean_monetary": 500.0,
                     "suggested_interpretation": "Moderate customers"},
                ],
            },
            "report": _good_report(),
        },
    },
    {
        "name": "Invalid: forecast error",
        "state_overrides": {
            "sql_result": _good_df(),
            "forecast_result": {"error": "No date column found"},
            "report": _good_report(),
        },
    },
    {
        "name": "Invalid: report too short / missing sections",
        "state_overrides": {
            "sql_result": _good_df(),
            "report":     "Short incomplete report.",
        },
    },
]

def run_critic_evaluation() -> dict:
    print("── Critic Evaluation (Confusion Matrix) ────────────")

    TP = TN = FP = FN = 0

    print("  VALID cases (should PASS):")
    for case in VALID_CASES:
        state = {**_base_state(), **case["state_overrides"]}
        out   = critic_node(state)
        cr    = out.get("critic_result", {}) or {}
        passed = cr.get("passed", False)
        if passed:
            TN += 1
            verdict = "TN ✓ (correctly passed)"
        else:
            FP += 1
            verdict = f"FP ✗ (incorrectly failed: {cr.get('reason','')})"
        print(f"    {case['name'][:55]:<55} {verdict}")

    print("  INVALID cases (should FAIL):")
    for case in INVALID_CASES:
        state = {**_base_state(), **case["state_overrides"]}
        out   = critic_node(state)
        cr    = out.get("critic_result", {}) or {}
        passed = cr.get("passed", False)
        if not passed:
            TP += 1
            verdict = f"TP ✓ (correctly caught: {cr.get('failed_component','')})"
        else:
            FN += 1
            verdict = "FN ✗ (missed — should have failed)"
        print(f"    {case['name'][:55]:<55} {verdict}")

    precision = round(TP / (TP + FP) * 100, 1) if (TP + FP) > 0 else 0.0
    recall    = round(TP / (TP + FN) * 100, 1) if (TP + FN) > 0 else 0.0
    f1_denom  = precision + recall
    f1        = round(2 * precision * recall / f1_denom, 1) if f1_denom > 0 else 0.0

    print(f"\n  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"  Precision={precision}%  Recall={recall}%  F1={f1}%\n")

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision_pct": precision,
        "recall_pct":    recall,
        "f1_pct":        f1,
        "note": (
            "Precision = TP/(TP+FP): fraction of flagged outputs that were truly bad. "
            "Recall = TP/(TP+FN): fraction of bad outputs that were caught."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. END-TO-END LLM-AS-JUDGE
# ═══════════════════════════════════════════════════════════════════════════════

E2E_QUERIES = [
    {
        "query":           "Which 10 product categories generated the highest revenue?",
        "expected_agents": ["sql", "business_analyst"],
        "type":            "SIMPLE_SQL",
    },
    {
        "query":           "Is there a significant relationship between delivery delay and review score?",
        "expected_agents": ["sql", "eda_stats", "business_analyst"],
        "type":            "SQL_PLUS_STATS",
    },
    {
        "query":           "What are the top 10 cities by number of customers and average order value?",
        "expected_agents": ["sql", "business_analyst"],
        "type":            "SIMPLE_SQL",
    },
    {
        "query":           "Which sellers have the highest cancellation rate?",
        "expected_agents": ["sql", "business_analyst"],
        "type":            "SIMPLE_SQL",
    },
]

JUDGE_PROMPT = """You are evaluating a business intelligence report.

Score from 0 to 10 based on:
- Accuracy: are the facts consistent with the data described?
- Completeness: does it cover the main sections?
- Actionability: are the recommendations specific and useful?

Return ONLY this JSON (no markdown, no explanation):
{{"score": <0-10>, "accuracy": <0-10>, "completeness": <0-10>, "actionability": <0-10>, "reason": "<one sentence>"}}

REPORT:
{report}
"""


def run_e2e_evaluation() -> list:
    print("── End-to-End LLM-as-Judge ─────────────────────────")
    print("  (Scores reflect LLM-perceived quality, not ground-truth accuracy)\n")

    from pipeline import run_query
    llm = get_llm()
    results = []

    for i, tc in enumerate(E2E_QUERIES):
        q = tc["query"]
        print(f"  [{i+1}/{len(E2E_QUERIES)}] {q[:60]}...")

        try:
            state  = run_query(q)
            report = state.get("report", "")

            if not report or len(report.strip()) < 50:
                results.append({"query": q, "score": 0, "reason": "No report generated",
                                 "pipeline_status": state.get("pipeline_status")})
                continue

            prompt = JUDGE_PROMPT.format(report=report[:4000])
            resp   = llm.invoke([HumanMessage(content=prompt)])
            raw    = resp.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            score  = parsed.get("score", 0)
            print(f"       Score: {score}/10 — {parsed.get('reason','')}")

            results.append({
                "query":           q,
                "type":            tc["type"],
                "score":           score,
                "accuracy":        parsed.get("accuracy", 0),
                "completeness":    parsed.get("completeness", 0),
                "actionability":   parsed.get("actionability", 0),
                "reason":          parsed.get("reason", ""),
                "pipeline_status": state.get("pipeline_status"),
                "critic_passed":   (state.get("critic_result") or {}).get("passed"),
                "llm_calls":       state.get("llm_calls", 0),
            })
        except Exception as e:
            print(f"       ERROR: {e}")
            results.append({"query": q, "score": 0, "reason": str(e)})

        time.sleep(3)  # Groq rate limit buffer

    if results:
        scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
        if scores:
            print(f"\n  Mean LLM-judge score: {sum(scores)/len(scores):.1f}/10")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "=" * 60)
    print("  RETAIL BI AGENT — EVALUATION SUITE")
    print("=" * 60)

    all_results: dict = {}

    all_results["sql"]       = run_sql_evaluation()
    all_results["critic"]    = run_critic_evaluation()

    print("Running end-to-end evaluation (4 full pipeline runs, ~3–6 min)…")
    all_results["end_to_end"] = run_e2e_evaluation()

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"  Results saved → {out_path}")
    print("  Open Streamlit → Evaluation tab to view.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
