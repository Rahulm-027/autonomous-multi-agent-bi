"""
run_query.py — CLI for running a single pipeline query without the UI.

Usage:
  python run_query.py
  python run_query.py --query "Which states have the longest delivery times?"
"""

import argparse
import json
from pipeline import run_query

EXAMPLES = [
    "Which 10 product categories generated the highest revenue?",
    "Is there a significant relationship between delivery delay and review score?",
    "Segment customers based on recency, frequency and monetary value.",
    "Forecast monthly order volume for the next 3 months.",
    "What are the top 10 cities by number of customers and average order value?",
    "Which sellers have the highest cancellation rate?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BI pipeline query.")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        print("\nExample queries:")
        for i, q in enumerate(EXAMPLES, 1):
            print(f"  {i}. {q}")
        print()
        choice = input("Enter a number (1–6) or type your own query: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(EXAMPLES):
            query = EXAMPLES[int(choice) - 1]
        else:
            query = choice

    print(f"\n{'='*65}")
    print(f"  QUERY: {query}")
    print(f"{'='*65}\n")
    print("Running pipeline…\n")

    result = run_query(query)

    # Plan
    print(f"{'='*65}")
    print("  AGENT PLAN")
    print(f"{'='*65}")
    for step in result.get("plan", []):
        print(f"  Step {step['step']:>2}  [{step['agent']:<18}]  {step['task']}")
    print(f"  Complexity: {result.get('query_complexity','?')}\n")

    # SQL
    if result.get("sql_query"):
        print(f"{'='*65}\n  SQL\n{'='*65}")
        print(result["sql_query"])
        meta = result.get("sql_meta", {})
        if meta:
            print(f"\n  → {meta.get('rows',0):,} rows | "
                  f"{meta.get('retries',0)} retries | "
                  f"{meta.get('execution_time_s',0):.2f}s")

    # Stats
    stat = result.get("stat_result")
    if stat and stat.get("test") not in (None, "none"):
        print(f"\n{'='*65}\n  STATISTICAL TEST\n{'='*65}")
        for k, v in stat.items():
            if k not in ("test",):
                print(f"  {k}: {v}")

    # Segmentation
    seg = result.get("segment_result")
    if seg and "error" not in (seg or {}):
        print(f"\n{'='*65}\n  SEGMENTATION\n{'='*65}")
        print(f"  Method:     {seg.get('method','?')}")
        print(f"  Customers:  {seg.get('n_customers',0):,}")
        print(f"  Best k:     {seg.get('best_k','?')}")
        print(f"  Silhouette: {seg.get('silhouette_score','?')}")
        for p in seg.get("cluster_profiles", []):
            print(f"    [{p.get('cluster_label','?')}] "
                  f"n={p.get('n_customers',0):,}  "
                  f"R={p.get('mean_recency',0):.0f}d  "
                  f"F={p.get('mean_frequency',0):.1f}  "
                  f"M=R${p.get('mean_monetary',0):.0f}")

    # Forecast
    fc = result.get("forecast_result")
    if fc and "error" not in (fc or {}):
        print(f"\n{'='*65}\n  FORECAST\n{'='*65}")
        print(f"  Best model: {fc.get('best_model','?')}")
        for model, m in fc.get("model_metrics", {}).items():
            if isinstance(m, dict):
                print(f"    {model:<18} MAE={m.get('MAE',0):.2f}  "
                      f"RMSE={m.get('RMSE',0):.2f}  "
                      f"WAPE={m.get('WAPE_%',0):.1f}%  "
                      f"sMAPE={m.get('sMAPE_%',0):.1f}%")

    # Critic
    cr = result.get("critic_result") or {}
    print(f"\n{'='*65}\n  CRITIC\n{'='*65}")
    print(f"  Passed:  {cr.get('passed', False)}")
    if not cr.get("passed"):
        print(f"  Reason:  {cr.get('reason','?')}")
    print(f"  Status:  {result.get('pipeline_status','?')}")
    print(f"  Retries: {result.get('iteration',0)}")

    # Execution tracking
    timings = result.get("agent_timings", {})
    if timings:
        print(f"\n{'='*65}\n  EXECUTION\n{'='*65}")
        total = sum(timings.values())
        for agent, t in timings.items():
            print(f"  {agent:<22} {t:.2f}s")
        print(f"  {'TOTAL':<22} {total:.2f}s")
        print(f"  LLM calls: {result.get('llm_calls',0)}")

    # Report
    print(f"\n{'='*65}\n  FINAL REPORT\n{'='*65}")
    print(result.get("report", "No report generated."))

    # Charts
    figs = result.get("figures", [])
    if figs:
        print(f"\n  {len(figs)} chart(s) saved to figures/")


if __name__ == "__main__":
    main()
