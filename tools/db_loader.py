"""
tools/db_loader.py — DuckDB loading and schema utilities.

Run once via setup.py. The database persists at db/olist.duckdb.
All reads use read_only=True to enforce analytical-only access.
"""

import os
import json
import duckdb
from config import DATA_DIR, DB_PATH, SCHEMA_PATH


TABLE_MAP: dict[str, str] = {
    "customers":   "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments":    "olist_order_payments_dataset.csv",
    "reviews":     "olist_order_reviews_dataset.csv",
    "orders":      "olist_orders_dataset.csv",
    "products":    "olist_products_dataset.csv",
    "sellers":     "olist_sellers_dataset.csv",
    "categories":  "product_category_name_translation.csv",
}


def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Returns a DuckDB connection. Always read-only for agent queries."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}.\n"
            "Run: python setup.py"
        )
    return duckdb.connect(DB_PATH, read_only=read_only)


def load_data(force_reload: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Loads all CSV files into DuckDB.
    Skips if the database already has the expected tables, unless force_reload=True.
    """
    conn = duckdb.connect(DB_PATH)

    existing = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }

    expected = set(TABLE_MAP.keys())
    if existing >= expected and not force_reload:
        print(f"Database already has all {len(expected)} tables. Skipping reload.")
        print("Use --reload flag to force a reload.\n")
        return conn

    print("Loading CSVs into DuckDB...\n")
    missing_files = []

    for table_name, csv_file in TABLE_MAP.items():
        csv_path = os.path.join(DATA_DIR, csv_file)
        if not os.path.exists(csv_path):
            missing_files.append(csv_path)
            continue

        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(
            f"CREATE TABLE {table_name} AS "
            f"SELECT * FROM read_csv_auto('{csv_path}', header=True)"
        )
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  ✓  {table_name:<22} {count:>10,} rows")

    if missing_files:
        print("\n  ✗  Missing CSV files:")
        for f in missing_files:
            print(f"       {f}")
        print(
            "\n  Download from: "
            "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce\n"
        )
        if len(missing_files) == len(TABLE_MAP):
            conn.close()
            raise FileNotFoundError(
                "No CSV files found in data/. "
                "Download the Olist dataset and place the 9 CSV files in data/."
            )

    print("\nLoad complete.\n")
    return conn


def generate_schema(conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Builds a schema dict describing every table: columns (name+type) and
    3 sample rows. Used to inject DB context into agent prompts.
    """
    schema: dict = {}

    tables = [
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
    ]

    for table in tables:
        cols = conn.execute(
            "SELECT column_name, data_type "
            "FROM information_schema.columns "
            f"WHERE table_name = '{table}' ORDER BY ordinal_position"
        ).fetchall()

        sample = conn.execute(f"SELECT * FROM {table} LIMIT 3").df()

        schema[table] = {
            "columns": {c: t for c, t in cols},
            "sample_rows": sample.to_dict(orient="records"),
        }

    return schema


def save_schema(schema: dict) -> None:
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, default=str)
    print(f"Schema saved → {SCHEMA_PATH}\n")


def load_schema() -> dict:
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(
            f"schema.json not found at {SCHEMA_PATH}.\n"
            "Run: python setup.py"
        )
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def schema_to_prompt_str(schema: dict) -> str:
    """
    Compact schema string injected into agent prompts.
    Format per table: TABLE_NAME(col:TYPE, ...)
    """
    lines = []
    for table, info in schema.items():
        cols = ", ".join(f"{c}:{t}" for c, t in info["columns"].items())
        lines.append(f"{table}({cols})")
    return "\n".join(lines)


def get_table_names() -> list[str]:
    """Returns list of table names in the database (for SQL validation)."""
    conn = get_connection()
    names = [
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    ]
    conn.close()
    return names


def validate_tables(conn: duckdb.DuckDBPyConnection) -> None:
    print("=" * 55)
    print(f"  {'TABLE':<22} {'ROWS':>12} {'COLS':>8}")
    print("=" * 55)

    tables = [
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
    ]

    for t in tables:
        rows = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cols = conn.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            f"WHERE table_name = '{t}'"
        ).fetchone()[0]
        print(f"  {t:<22} {rows:>12,} {cols:>8}")

    print("=" * 55 + "\n")
