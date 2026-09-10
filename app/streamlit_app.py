"""
app/streamlit_app.py — Streamlit UI

Three tabs:
  1. Query     — NL input, live agent trace, final report
  2. Analytics — browse all generated charts
  3. Evaluation — view evaluation/results.json metrics
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail BI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Retail BI Agent")
    st.caption("Autonomous Multi-Agent Business Intelligence")
    st.markdown(
        "**Dataset:** Olist Brazilian E-Commerce (100k orders)\n\n"
        "**Stack:** LangGraph · Groq (Llama 3.3 70B) · DuckDB · Prophet"
    )
    st.divider()

    st.markdown("**Example queries by type:**")

    simple_sql = [
        "Which 10 product categories generated the highest revenue?",
        "What are the top 10 cities by number of customers?",
        "Which sellers have the most fulfilled orders?",
    ]
    stat_queries = [
        "Is there a significant relationship between delivery delay and review score?",
        "Do customers in São Paulo spend significantly more than those in Rio?",
    ]
    ml_queries = [
        "Segment customers based on recency, frequency and monetary value.",
        "Forecast monthly order volume for the next 3 months.",
    ]
    multi = [
        "Which customer segments are most valuable and how have their orders changed over time?",
    ]

    with st.expander("📊 Simple SQL"):
        for q in simple_sql:
            if st.button(q, key=f"s_{q[:15]}", use_container_width=True):
                st.session_state["prefill"] = q

    with st.expander("📐 SQL + Statistics"):
        for q in stat_queries:
            if st.button(q, key=f"st_{q[:15]}", use_container_width=True):
                st.session_state["prefill"] = q

    with st.expander("🤖 SQL + ML"):
        for q in ml_queries:
            if st.button(q, key=f"ml_{q[:15]}", use_container_width=True):
                st.session_state["prefill"] = q

    with st.expander("🔀 Multi-Agent"):
        for q in multi:
            if st.button(q, key=f"ma_{q[:15]}", use_container_width=True):
                st.session_state["prefill"] = q

    st.divider()
    st.caption("Built with LangGraph · Groq · DuckDB · Streamlit")

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []
if "prefill" not in st.session_state:
    st.session_state["prefill"] = ""

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Query", "📊 Analytics", "🔬 Evaluation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Query Interface
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Ask a Business Question")

    query = st.text_area(
        "Natural-language business question:",
        value=st.session_state.get("prefill", ""),
        height=80,
        placeholder=(
            "e.g. Which customer segments are most valuable "
            "and what drives their purchasing behaviour?"
        ),
        key="query_input",
    )

    col_run, col_clear = st.columns([1, 6])
    with col_run:
        run_btn = st.button("▶ Run", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑 Clear history"):
            st.session_state["history"] = []
            st.rerun()

    if run_btn and query.strip():
        st.session_state["prefill"] = ""
        from pipeline import run_query

        status_box  = st.empty()
        trace_box   = st.empty()
        trace_lines = []

        def log(msg: str) -> None:
            trace_lines.append(f"• {msg}")
            trace_box.markdown("**Agent Trace:**\n" + "\n".join(trace_lines[-12:]))

        log("Supervisor: classifying query and generating plan…")
        status_box.info("Pipeline running — this takes 20–90 seconds")

        t0     = time.time()
        result = run_query(query)
        elapsed = time.time() - t0

        # Build post-hoc trace from timings
        for agent, t in result.get("agent_timings", {}).items():
            log(f"{agent}: {t:.1f}s")
        log(f"Total: {elapsed:.1f}s | LLM calls: {result.get('llm_calls',0)}")

        status = result.get("pipeline_status", "?")
        if status == "SUCCESS":
            status_box.success(f"✅ Pipeline complete in {elapsed:.1f}s")
        elif status == "FAILED":
            status_box.error(f"❌ Pipeline failed after {result.get('iteration',0)} retries")
        else:
            status_box.warning(f"⚠ Pipeline status: {status}")

        st.session_state["history"].append(result)

    # ── Display most recent result ────────────────────────────────────────────
    if st.session_state["history"]:
        result = st.session_state["history"][-1]
        st.divider()

        # Agent plan
        plan = result.get("plan", [])
        if plan:
            with st.expander(f"📋 Agent Plan ({result.get('query_complexity','?')})",
                             expanded=True):
                icons = {"sql": "🗄️", "eda_stats": "📈",
                         "ml": "🤖", "business_analyst": "💼"}
                for step in plan:
                    st.markdown(
                        f"**Step {step['step']}** "
                        f"{icons.get(step.get('agent',''), '•')} "
                        f"`{step.get('agent','?')}` — {step.get('task','?')}"
                    )

        # Critic result
        cr = result.get("critic_result") or {}
        c1, c2, c3 = st.columns(3)
        if cr.get("passed"):
            c1.success("✅ Critic: PASS")
        elif result.get("pipeline_status") == "FAILED":
            c1.error("❌ Pipeline FAILED (max retries)")
        else:
            c1.warning(f"⚠ Critic: {cr.get('failed_component','?')}")

        df = result.get("sql_result")
        if df is not None and hasattr(df, "shape"):
            c2.info(f"📊 {df.shape[0]:,} rows × {df.shape[1]} cols")
        c3.info(f"🔁 {result.get('iteration',0)} retries | "
                f"🤖 {result.get('llm_calls',0)} LLM calls")

        # SQL
        if result.get("sql_query"):
            with st.expander("🗄️ Generated SQL"):
                st.code(result["sql_query"], language="sql")
                meta = result.get("sql_meta", {})
                if meta:
                    st.caption(
                        f"Executed in {meta.get('execution_time_s',0):.2f}s · "
                        f"{meta.get('rows',0):,} rows · "
                        f"{meta.get('retries',0)} SQL retries"
                    )

        # Data preview
        if df is not None and hasattr(df, "head") and not df.empty:
            with st.expander("🔍 Data Preview (first 20 rows)"):
                st.dataframe(df.head(20), use_container_width=True)

        # Statistics
        stat = result.get("stat_result")
        if stat and stat.get("test") not in (None, "none", ""):
            with st.expander("📐 Statistical Test"):
                display = {k: v for k, v in stat.items()
                           if k not in ("research_question",)}
                col_s1, col_s2 = st.columns(2)
                col_s1.metric("Test", display.get("test", "N/A"))
                col_s1.metric("p-value", str(display.get("p_value", "N/A")))
                col_s2.metric(
                    "Significant",
                    "Yes ✅" if display.get("significant") else "No ❌"
                )
                effect_key = next(
                    (k for k in display if "effect" in k.lower()), None
                )
                if effect_key:
                    col_s2.metric("Effect size", str(display.get(effect_key, "N/A")))
                st.markdown(f"**Interpretation:** {display.get('interpretation', '')}")

        # Segmentation
        seg = result.get("segment_result")
        if seg and "error" not in (seg or {}):
            with st.expander("👥 Segmentation Results"):
                cm1, cm2, cm3 = st.columns(3)
                cm1.metric("Customers", f"{seg.get('n_customers',0):,}")
                cm2.metric("Clusters (k)", seg.get("best_k", "?"))
                cm3.metric("Silhouette", seg.get("silhouette_score", "?"))
                st.caption(f"Label source: {seg.get('label_source','?')}")
                profiles = seg.get("cluster_profiles", [])
                if profiles:
                    st.dataframe(pd.DataFrame(profiles), use_container_width=True)

        # Forecast
        fc = result.get("forecast_result")
        if fc and "error" not in (fc or {}):
            with st.expander("📅 Forecast Results"):
                st.markdown(f"**Best model:** {fc.get('best_model','?')}")
                metrics = fc.get("model_metrics", {})
                if metrics:
                    rows = []
                    for model, m in metrics.items():
                        if isinstance(m, dict) and "RMSE" in m:
                            rows.append({"Model": model, **m})
                    if rows:
                        st.dataframe(pd.DataFrame(rows).set_index("Model"),
                                     use_container_width=True)
                st.caption(fc.get("note", ""))

        # Execution timings
        timings = result.get("agent_timings", {})
        if timings:
            with st.expander("⏱ Execution Timings"):
                df_t = pd.DataFrame(
                    [(k, v) for k, v in timings.items()],
                    columns=["Agent", "Time (s)"]
                ).set_index("Agent")
                st.dataframe(df_t, use_container_width=True)

        # Evidence trace
        evidence = result.get("evidence", [])
        if evidence:
            with st.expander("🔗 Evidence Registry"):
                for ev in evidence:
                    st.markdown(
                        f"**[{ev['id']}]** `{ev['agent']}` — {ev['summary']}"
                    )

        # Final report
        report = result.get("report", "")
        if report:
            st.divider()
            st.subheader("📄 Business Intelligence Report")
            st.markdown(report)
            st.download_button(
                "⬇ Download Report (.md)",
                data=report,
                file_name="bi_report.md",
                mime="text/markdown",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Generated Charts")

    figs = []
    for r in st.session_state.get("history", []):
        figs.extend(r.get("figures", []))

    if not figs:
        st.info("Run a query first — charts appear here.")
    else:
        st.caption(f"{len(figs)} chart(s) generated this session.")
        for path in reversed(figs):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    html = f.read()
                st.components.v1.html(html, height=450, scrolling=True)
                st.caption(f"📁 {os.path.basename(path)}")
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Evaluation Dashboard")

    eval_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluation", "results.json"
    )

    if not os.path.exists(eval_path):
        st.info(
            "No evaluation results yet.\n\n"
            "Run: `python evaluation/evaluate.py`\n\n"
            "Results will appear here automatically."
        )
    else:
        with open(eval_path, "r") as f:
            ev = json.load(f)

        # SQL
        st.subheader("SQL Agent — Gold-Query Correctness")
        sql_r = ev.get("sql", {})
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Execution success", f"{sql_r.get('execution_success_rate_pct', 0):.1f}%",
                   help="Did generated SQL execute without error?")
        sc2.metric("Schema accuracy", f"{sql_r.get('schema_accuracy_pct', 0):.1f}%",
                   help="Did result contain all expected columns?")
        sc3.metric("Overall accuracy", f"{sql_r.get('overall_accuracy_pct', 0):.1f}%",
                   help="All checks passed: schema + row count + value + sort")
        sc4, sc5, sc6 = st.columns(3)
        sc4.metric("Row count accuracy", f"{sql_r.get('row_count_accuracy_pct', 0):.1f}%")
        sc5.metric("Value accuracy", f"{sql_r.get('value_accuracy_pct', 0):.1f}%",
                   help="Gold-value comparison ± tolerance")
        sc6.metric("Avg retries", sql_r.get("avg_retries", 0))

        details = sql_r.get("details", [])
        if details:
            with st.expander("SQL test details"):
                st.dataframe(pd.DataFrame(details), use_container_width=True)

        # Critic
        st.subheader("Critic Agent")
        cr = ev.get("critic", {})
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("TP (caught bad)", cr.get("TP", 0))
        cc2.metric("TN (passed good)", cr.get("TN", 0))
        cc3.metric("Precision", f"{cr.get('precision_pct', 0):.1f}%")
        cc4.metric("Recall", f"{cr.get('recall_pct', 0):.1f}%")
        st.caption(
            "Precision = TP/(TP+FP) — what fraction of flagged outputs were truly bad. "
            "Recall = TP/(TP+FN) — what fraction of bad outputs were caught."
        )

        # End-to-end
        st.subheader("End-to-End (LLM-as-Judge)")
        e2e = ev.get("end_to_end", [])
        if e2e:
            df_e2e = pd.DataFrame(e2e)
            st.dataframe(df_e2e, use_container_width=True)
            if "score" in df_e2e.columns:
                st.metric("Mean report quality (0–10)",
                          f"{df_e2e['score'].mean():.1f}")
            st.caption(
                "Scores are from an LLM judge, not ground-truth accuracy. "
                "They reflect perceived completeness, clarity, and actionability."
            )

        st.caption(f"Results from: {eval_path}")
