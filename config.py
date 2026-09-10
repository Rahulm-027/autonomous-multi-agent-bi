"""
config.py — Central configuration for the Retail BI Agent system.

All tunable parameters live here. Agents import from this file only —
nothing is hard-coded inside agent modules.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL        = "openai/gpt-oss-120b"   # Groq free-tier model
LLM_TEMPERATURE  = 0.0                          # Deterministic for analytics

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
DB_DIR      = os.path.join(BASE_DIR, "db")
DB_PATH     = os.path.join(DB_DIR,   "olist.duckdb")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.json")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_RETRIES          = 2      # Maximum Critic-triggered retries per failed component
SQL_MAX_RETRIES      = 3      # SQL self-correction attempts before giving up
MAX_QUERY_ROWS       = 500    # Default LIMIT for SQL queries

# ── Statistical analysis ──────────────────────────────────────────────────────
SIGNIFICANCE_LEVEL   = 0.05
MIN_SAMPLE_SIZE      = 30     # Minimum rows per group for parametric tests

# ── Segmentation ──────────────────────────────────────────────────────────────
MIN_SILHOUETTE       = 0.20   # Minimum acceptable silhouette score
MIN_CLUSTER_SIZE     = 5      # Minimum customers per cluster
RANDOM_STATE         = 42
N_CLUSTERS_MIN       = 2
N_CLUSTERS_MAX       = 6

# ── Forecasting ───────────────────────────────────────────────────────────────
FORECAST_HORIZON     = 3      # Periods ahead to forecast (months by default)
MIN_TS_OBSERVATIONS  = 12     # Minimum time-series points required

# ── SQL safety: blocked keywords (case-insensitive, word-boundary matched) ────
SQL_BLOCKED_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "CREATE", "TRUNCATE", "REPLACE", "MERGE", "GRANT",
    "REVOKE", "EXEC", "EXECUTE",
]

# ── Directories ───────────────────────────────────────────────────────────────
for _d in [DATA_DIR, DB_DIR, FIGURES_DIR]:
    os.makedirs(_d, exist_ok=True)


def get_llm(temperature: float = LLM_TEMPERATURE):
    """Returns a ChatGroq LLM instance. Raises clearly if API key is missing."""
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "1. Copy .env.example to .env\n"
            "2. Add your key from https://console.groq.com\n"
            "3. Re-run the application."
        )
    return ChatGroq(
        model=LLM_MODEL,
        temperature=temperature,
        api_key=api_key,
        max_tokens=4096,
    )
