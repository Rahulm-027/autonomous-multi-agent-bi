"""
setup.py — One-time setup script.

Run this once before using the application:
  python setup.py            # skip reload if DB already populated
  python setup.py --reload   # force reload from CSVs

What it does:
  1. Loads all 9 Olist CSVs into DuckDB
  2. Generates schema.json
  3. Validates all tables
  4. Tests Groq API connection
"""

import sys
import json
import argparse

from tools.db_loader import load_data, generate_schema, save_schema, validate_tables
from config import get_llm, SCHEMA_PATH


def test_llm() -> bool:
    print("Testing Groq API connection...")
    try:
        llm  = get_llm()
        resp = llm.invoke("Reply with exactly: GROQ_OK")
        print(f"  ✓  Groq connected — response: {resp.content.strip()}\n")
        return True
    except Exception as e:
        print(f"  ✗  Groq connection failed: {e}\n")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail BI Agent setup")
    parser.add_argument("--reload", action="store_true",
                        help="Force reload all CSVs even if DB already exists")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RETAIL BI AGENT — SETUP")
    print("=" * 60 + "\n")

    # 1. Load data
    try:
        conn = load_data(force_reload=args.reload)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # 2. Validate
    validate_tables(conn)

    # 3. Generate schema
    print("Generating schema.json ...")
    schema = generate_schema(conn)
    save_schema(schema)
    conn.close()

    # 4. Preview
    first_table = list(schema.keys())[0]
    print("Schema preview (first table):")
    print(json.dumps({first_table: schema[first_table]}, indent=2, default=str))
    print("...\n")

    # 5. Test LLM
    ok = test_llm()

    print("=" * 60)
    if ok:
        print("  SETUP COMPLETE\n")
        print("  Launch UI:    streamlit run app/streamlit_app.py")
        print("  CLI query:    python run_query.py")
        print("  Evaluation:   python evaluation/evaluate.py")
    else:
        print("  INCOMPLETE — fix GROQ_API_KEY in .env and re-run setup.py")
    print("=" * 60 + "\n")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
