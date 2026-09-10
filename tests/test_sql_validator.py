"""
tests/test_sql_validator.py — Unit tests for SQL safety and schema validation.

Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tools.sql_validator import validate_sql_safety, validate_sql_tables, SQLValidationError


AVAILABLE_TABLES = ["orders", "customers", "order_items", "payments", "reviews",
                    "products", "sellers", "geolocation", "categories"]

# ── Safety tests ──────────────────────────────────────────────────────────────
class TestSQLSafety:
    def test_valid_select(self):
        """A plain SELECT should pass."""
        validate_sql_safety("SELECT * FROM orders LIMIT 10")

    def test_drop_blocked(self):
        with pytest.raises(SQLValidationError, match="DROP"):
            validate_sql_safety("DROP TABLE orders")

    def test_delete_blocked(self):
        with pytest.raises(SQLValidationError, match="DELETE"):
            validate_sql_safety("DELETE FROM orders WHERE 1=1")

    def test_update_blocked(self):
        with pytest.raises(SQLValidationError, match="UPDATE"):
            validate_sql_safety("UPDATE orders SET status='x'")

    def test_insert_blocked(self):
        with pytest.raises(SQLValidationError, match="INSERT"):
            validate_sql_safety("INSERT INTO orders VALUES (1)")

    def test_alter_blocked(self):
        with pytest.raises(SQLValidationError, match="ALTER"):
            validate_sql_safety("ALTER TABLE orders ADD COLUMN x INT")

    def test_create_blocked(self):
        with pytest.raises(SQLValidationError, match="CREATE"):
            validate_sql_safety("CREATE TABLE hack AS SELECT 1")

    def test_truncate_blocked(self):
        with pytest.raises(SQLValidationError, match="TRUNCATE"):
            validate_sql_safety("TRUNCATE orders")

    def test_empty_sql_blocked(self):
        with pytest.raises(SQLValidationError):
            validate_sql_safety("")

    def test_whitespace_only_blocked(self):
        with pytest.raises(SQLValidationError):
            validate_sql_safety("   ")

    def test_case_insensitive_drop(self):
        """Blocked keywords should be caught regardless of case."""
        with pytest.raises(SQLValidationError):
            validate_sql_safety("drop table orders")

    def test_word_boundary_respected(self):
        """'DROPDOWN' should NOT be blocked — word boundary check."""
        validate_sql_safety(
            "SELECT * FROM orders WHERE description LIKE '%dropdown%'"
        )


# ── Schema validation tests ───────────────────────────────────────────────────
class TestSQLTableValidation:
    def test_known_table_passes(self):
        validate_sql_tables("SELECT * FROM orders LIMIT 10", AVAILABLE_TABLES)

    def test_unknown_table_fails(self):
        with pytest.raises(SQLValidationError, match="unknown table"):
            validate_sql_tables("SELECT * FROM nonexistent_table", AVAILABLE_TABLES)

    def test_join_unknown_table_fails(self):
        with pytest.raises(SQLValidationError):
            validate_sql_tables(
                "SELECT * FROM orders JOIN fake_table ON orders.id = fake_table.id",
                AVAILABLE_TABLES
            )

    def test_multiple_known_tables(self):
        validate_sql_tables(
            "SELECT o.*, c.* FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
            AVAILABLE_TABLES
        )

    def test_subquery_known_table(self):
        validate_sql_tables(
            "SELECT * FROM (SELECT * FROM orders LIMIT 100) sub",
            AVAILABLE_TABLES
        )
