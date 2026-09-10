# Autonomous Multi-Agent Business Intelligence System
### Retail Operations Analytics · Olist Brazilian E-Commerce Dataset

---

## Overview

A multi-agent system that receives a natural-language business question,
autonomously classifies its complexity, generates a structured analytical plan,
routes work to specialised agents (SQL, EDA/Statistics, ML, Business Analyst),
validates every output through a deterministic Critic Agent, and produces a
structured business intelligence report — or fails honestly if validation cannot
be satisfied.

**This is not a chatbot or a RAG system.** It is an analytical pipeline in which
agents plan, execute, self-correct, and synthesise results from real data.

---

## Architecture

```
User NL Question
        │
        ▼
┌──────────────────────┐
│   SUPERVISOR         │  Classifies query complexity, generates structured plan
└──────────┬───────────┘
           │  Routes per plan
    ┌──────┼──────────┬──────────────┐
    ▼      ▼          ▼              ▼
  SQL   EDA/Stats    ML         Business
  Agent  Agent       Agent       Analyst
    │      │          │              │
    └──────┴──────────┘              │
           │ (all feed into)         │
           └─────────────────────────┘
                       │
                       ▼
             ┌─────────────────┐
             │  CRITIC AGENT   │  Deterministic checks — NOT an LLM critic
             └────────┬────────┘
                      │
              ┌───────┴────────┐
           PASS             FAIL
              │               │
             END     Targeted retry OR
                     fail-closed terminal
                              │
                             END
```

**Fail-closed:** If validation fails after MAX_RETRIES, the system routes to a
`failed_node` that writes an honest failure message. It does NOT continue to the
Business Analyst with unvalidated data.

---

## Agent Responsibilities

| Agent | What it does |
|---|---|
| **Supervisor** | Classifies query (SIMPLE_SQL / SQL_PLUS_STATS / SQL_PLUS_ML / MULTI_AGENT), generates a Pydantic-validated plan with explicit parameters for each agent |
| **SQL Agent** | NL → DuckDB SQL with safety validation (blocks DROP/DELETE/etc.), schema validation, and up to 3 self-correction retries on failure |
| **EDA/Stats Agent** | Descriptive stats, outlier detection, correlation; selects appropriate hypothesis test (t-test / Mann-Whitney / ANOVA / chi-squared / Spearman) from Supervisor-specified parameters; reports effect sizes |
| **ML Agent** | RFM segmentation built directly from transactional tables (not from arbitrary SQL result columns); K-Means with silhouette-selected k; Prophet forecasting with Naive + Seasonal Naive baselines and chronological holdout |
| **Business Analyst** | Writes the BI report from structured evidence only — the LLM is explicitly instructed not to invent numbers |
| **Critic** | Deterministic programmatic checks (SQL success, non-empty result, silhouette threshold, forecast metrics, report sections); returns structured retry instructions with `failed_component` and `retry_agent` |

---

## Dataset

**Olist Brazilian E-Commerce** — https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

- ~100,000 orders from 2016–2018
- 9 relational tables loaded into DuckDB
- RFM analysis uses: `orders` (delivered status, timestamps) + `order_payments` (payment values)

---

## Technology Stack (100% Free)

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph (StateGraph, conditional edges) |
| LLM | Groq API — Llama 3.3 70B (free tier) |
| Database | DuckDB (in-process, no server) |
| Data wrangling | Pandas, NumPy |
| Statistics | SciPy (t-test, Mann-Whitney, ANOVA, chi-squared, Spearman) |
| ML / Clustering | Scikit-learn (KMeans, StandardScaler, silhouette_score) |
| Forecasting | Prophet (+ Naive and Seasonal Naive baselines) |
| Visualisation | Plotly |
| UI | Streamlit |
| Validation | Pydantic (plan schemas) |
| Tests | pytest (86 tests, 100% passing) |

---

## Setup

### Prerequisites
- Python 3.10 or 3.11 (3.12 also works)
- Free Groq API key: https://console.groq.com (no credit card needed)
- Olist dataset CSVs: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

### 1. Place CSV files
Extract the Kaggle download and place all 9 CSV files in the `data/` folder:
```
data/
  olist_customers_dataset.csv
  olist_geolocation_dataset.csv
  olist_order_items_dataset.csv
  olist_order_payments_dataset.csv
  olist_order_reviews_dataset.csv
  olist_orders_dataset.csv
  olist_products_dataset.csv
  olist_sellers_dataset.csv
  product_category_name_translation.csv
```

### 2. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> Prophet takes 3–5 minutes to install. Let it finish.

### 4. Configure API key
```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```
Open `.env` and replace `your_groq_api_key_here` with your actual key.

### 5. Run setup (once)
```bash
python setup.py
```
Expected output: 9 tables confirmed + `GROQ_OK`

### 6. Run unit tests
```bash
python -m pytest tests/ -v
```
Expected: 86 passed

---

## Running the Application

### Streamlit UI (recommended)
```bash
streamlit run app/streamlit_app.py
```
Open http://localhost:8501

### Command-line interface
```bash
python run_query.py
python run_query.py --query "Which product categories generated the most revenue?"
```

### Evaluation suite
```bash
python evaluation/evaluate.py
```
Writes `evaluation/results.json` — displayed in the Streamlit Evaluation tab.

---

## Example Queries

| Type | Example |
|---|---|
| **Simple SQL** | Which 10 product categories generated the highest revenue? |
| **Simple SQL** | Which sellers have the highest cancellation rate? |
| **SQL + Stats** | Is there a significant relationship between delivery delay and review score? |
| **SQL + Stats** | Do customers in São Paulo spend significantly more than those in Rio? |
| **SQL + ML** | Segment customers based on recency, frequency and monetary value. |
| **SQL + ML** | Forecast monthly order volume for the next 3 months. |
| **Multi-agent** | Which customer segments are most valuable and how have their orders changed over time? |

---

## Key Design Decisions

### RFM is built from the database, not from SQL result columns

The ML Agent uses a hardcoded query against the Olist transactional tables
with explicit business definitions:
- **Recency** = days since last DELIVERED order (reference date = max order date)
- **Frequency** = count of distinct delivered orders
- **Monetary** = sum of payment_value for delivered orders

This is not inferred from whatever columns the SQL Agent happened to return.

### Critic uses deterministic checks, not an LLM

The Critic is a Python function — not an LLM call. It checks:
- SQL execution success and non-empty result
- Silhouette score ≥ MIN_SILHOUETTE (configurable)
- Forecast metrics exist and model_metrics is populated
- Report contains all required sections and meets minimum length

This is documented explicitly to avoid misrepresenting the architecture.

### Fail-closed on repeated failures

If validation fails more than MAX_RETRIES times, the pipeline routes to a
`failed_node` that writes an honest failure message. It never generates a
business report from unvalidated analysis.

### Evidence traceability

Every agent result appends a record to `state["evidence"]`. The Business
Analyst receives this registry and is instructed to cite evidence IDs
(e.g. [SQL-01], [STAT-01]) in the report.

---

## Evaluation Methodology

### SQL evaluation
- 15 test cases
- Metrics: execution success rate, schema correctness rate
- Does NOT count "non-empty result" as correct — schema is checked separately

### Critic evaluation (confusion matrix)
- 5 valid inputs (should PASS) → TN if passed, FP if failed
- 5 invalid inputs (should FAIL) → TP if caught, FN if missed
- Reports: TP, TN, FP, FN, Precision, Recall, F1

### End-to-end LLM-as-judge
- 4 full pipeline runs
- Scored 0–10 by a separate LLM call on accuracy, completeness, actionability
- **Labelled as LLM-based quality estimate, NOT ground-truth accuracy**

### Evaluation results
Run `python evaluation/evaluate.py` to get your actual measured results.
Results will appear in `evaluation/results.json` and in the Streamlit Evaluation tab.

**No fabricated metrics are included in this README.**

---

## Limitations

- The Groq free tier has rate limits (~14k requests/day for Llama 3.3 70B). Heavy
  testing may hit these limits. Switch to Ollama for unlimited local inference.
- Prophet forecasting requires ≥ 12 monthly observations (configurable).
- RFM segmentation uses only delivered orders. Customers with no delivered orders
  are excluded.
- The Business Analyst is an LLM — it can occasionally produce imprecise language
  even when constrained to the evidence. Always verify numerical claims against
  the Evidence Registry.
- Evaluation scores (LLM-as-judge) reflect perceived quality, not analytical accuracy.
- The dataset covers 2016–2018. Insights may not reflect current market conditions.

---

## Project Structure

```
retail_agent/
├── agents/
│   ├── supervisor.py         Complexity classification + plan generation
│   ├── sql_agent.py          NL→SQL with safety validation + self-correction
│   ├── eda_stats_agent.py    EDA + hypothesis testing with effect sizes
│   ├── ml_agent.py           RFM segmentation (from DB) + Prophet forecasting
│   ├── business_analyst.py   Evidence-grounded BI report generation
│   └── critic.py             Deterministic validation + fail-closed terminal
├── tools/
│   ├── db_loader.py          DuckDB loading + schema utilities
│   ├── plot_utils.py         Plotly chart helpers
│   └── sql_validator.py      SQL safety + schema validation
├── app/
│   └── streamlit_app.py      3-tab Streamlit UI
├── evaluation/
│   └── evaluate.py           SQL accuracy + Critic confusion matrix + LLM-judge
├── tests/
│   ├── test_sql_validator.py 17 SQL safety/schema tests
│   ├── test_critic.py        15 Critic tests (valid + invalid + fail-closed)
│   ├── test_stats.py         14 statistical function tests
│   └── test_forecast.py      7 forecasting utility tests
├── data/                     Place 9 Olist CSVs here
├── config.py                 All tunable parameters
├── state.py                  AgentState TypedDict
├── pipeline.py               LangGraph StateGraph
├── setup.py                  One-time setup
├── run_query.py              CLI interface
└── requirements.txt
```

---

## CV Bullet Points

Fill in your actual measured numbers from `python evaluation/evaluate.py`:

```
• Designed a LangGraph supervisor-worker multi-agent system (6 agents: Supervisor,
  SQL, EDA/Stats, ML, Business Analyst, Critic) with Pydantic-validated plan schemas,
  shared TypedDict state, conditional routing, and a fail-closed Critic Agent; achieved
  __% SQL execution success rate across 15 benchmark queries on the Olist dataset.

• Built an ML Agent performing RFM segmentation (recency/frequency/monetary computed
  directly from 100k transactional records, not proxy columns), K-Means with
  silhouette-selected k (best score: __), and Prophet demand forecasting benchmarked
  against Naive and Seasonal Naive baselines on a chronological holdout (MAE: __, RMSE: __).

• Implemented a deterministic Critic Agent with a confusion matrix evaluation:
  TP=__, TN=__, Precision=__%, Recall=__% across 10 valid/invalid test cases;
  integrated a fail-closed terminal node that halts the pipeline rather than generating
  reports from unvalidated analysis.

• Evaluated end-to-end report quality using LLM-as-judge scoring (mean __/10 on
  accuracy, completeness, actionability); all code verified by 86 unit tests (pytest)
  covering SQL safety, statistical functions, Critic logic, and forecasting utilities.
```

---

## Technical Skills Line

```
Python · LangGraph · LangChain · Pydantic · DuckDB · Pandas · Scikit-learn ·
SciPy · Prophet · Plotly · Streamlit · Groq API · pytest
```
