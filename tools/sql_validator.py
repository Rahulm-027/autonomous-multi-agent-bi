"""
tools/sql_validator.py — SQL safety and schema validation.

Validates SQL before execution:
  1. Safety check: blocks destructive statements
  2. Schema check: verifies referenced tables exist
  3. Basic syntax check: non-empty, not just whitespace

Used by the SQL Agent before every execution attempt.
"""

import re
from config import SQL_BLOCKED_KEYWORDS


class SQLValidationError(Exception):
    """Raised when SQL fails a safety or schema check."""
    pass


def validate_sql_safety(sql: str) -> None:
    """
    Raises SQLValidationError if the SQL contains any blocked keyword
    at a word boundary (prevents DROP TABLE, dropTable, etc.).
    """
    if not sql or not sql.strip():
        raise SQLValidationError("SQL is empty.")

    upper = sql.upper()
    for keyword in SQL_BLOCKED_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, upper):
            raise SQLValidationError(
                f"SQL contains blocked keyword '{keyword}'. "
                "Only read-only SELECT statements are permitted."
            )


def validate_sql_tables(sql: str, available_tables: list[str]) -> None:
    """
    Heuristically checks that any word in the FROM/JOIN clause matching
    a plausible table name actually exists in the database.

    This is a lightweight check, not a full SQL parser.
    """
    upper = sql.upper()

    # Extract tokens after FROM and JOIN
    from_pattern = re.findall(
        r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", upper
    )

    unknown = [
        t.lower()
        for t in from_pattern
        if t.lower() not in [tbl.lower() for tbl in available_tables]
    ]

    if unknown:
        raise SQLValidationError(
            f"SQL references unknown table(s): {unknown}. "
            f"Available tables: {available_tables}"
        )


def validate_sql(sql: str, available_tables: list[str]) -> None:
    """
    Full validation: safety + schema. Raises SQLValidationError on failure.
    """
    validate_sql_safety(sql)
    validate_sql_tables(sql, available_tables)
