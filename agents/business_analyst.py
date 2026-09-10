"""
agents/business_analyst.py — Business Analyst Agent

Synthesises all upstream analytical outputs into a structured Markdown
business intelligence report.

Key design constraints:
  - The LLM receives ONLY data that actually exists in the state.
  - Every section is built from structured evidence — not free-form generation.
  - The evidence registry is included so the LLM can cite sources.
  - The LLM is explicitly told NOT to invent numbers.
  - The report includes a Limitations section for low-confidence findings.
"""

import json
import time
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from config import get_llm


SYSTEM_PROMPT = """You are a Senior Business Intelligence Analyst writing a structured report.

CRITICAL RULES:
1. Use ONLY numbers and facts present in the EVIDENCE provided below.
2. Do NOT invent, estimate, extrapolate, or calculate unsupported business metrics.
3. If a section has no relevant data, write "No data available for this section."
4. Cite evidence using [ID] tags, e.g. [SQL-01], [STAT-01], [RFM-01].
5. Every numerical claim must be traceable to a specific evidence item.
5a. Simple arithmetic derived directly from cited evidence (such as percentages,
    ratios, differences, or multiples) is allowed, but the derived value must
    remain fully reproducible from the cited evidence.
6. Distinguish clearly between:
   - per-customer metrics and aggregate totals,
   - correlation/association and causation,
   - model-derived cluster labels and verified business outcomes.
7. Do not claim that a segment contributes disproportionately to total revenue,
   profit, or sales unless aggregate contribution data is explicitly provided.
8. Do not claim that a customer segment is profitable, unprofitable, active,
   churned, at-risk, or lapsed unless the evidence directly supports that claim.
9. Business recommendations may be proposed as actions or hypotheses, but must
   not be presented as observed outcomes.
10. In the Limitations section, flag small samples, weak statistical evidence
    where statistical testing was applicable, low silhouette scores, missing
    data, limited query results, modelling assumptions, or any other material
    uncertainty. Do not treat the absence of hypothesis testing as a limitation
    when the analysis is inherently descriptive or unsupervised.

RFM-SPECIFIC RULES:
11. Recency is measured as the number of days since the customer's last order.
    Lower recency means the customer purchased more recently.
12. Do not describe a customer as "recent" or "active" merely because its
    recency is lower than another cluster. Consider the absolute recency value
    when making such statements.
13. Do not describe a customer as "lapsed" or "at-risk" solely from relative
    recency. If such terminology is used, clearly identify it as an analyst
    interpretation rather than a measured outcome.
14. Frequency represents the number of orders per customer. When comparing
    segments, explicitly describe whether the difference is in repeat-purchase
    behaviour.
15. Monetary value is a customer-level monetary measure. If the evidence
    explicitly identifies it as payment_value, it may be described as customer
    payment value. Otherwise, call it "monetary value" and do not rename it as
    revenue unless revenue is explicitly defined in the evidence.
16. If a small segment has higher average monetary value than a large segment,
    report the difference as higher monetary value per customer. Do not infer
    higher aggregate revenue contribution without aggregate monetary totals.
17. If RFM values were transformed for model fitting but cluster profiles are
    reported using the winsorized RFM values, clearly distinguish the modelling
    transformation from the reported cluster metrics. Do not describe the
    reported cluster means as log-space values unless the evidence explicitly
    says they are.
18. Cluster labels must be descriptive and evidence-based. Prefer labels such
    as "one-time / lower-value customers" or "repeat / higher-value customers"
    when those characteristics are directly supported by the cluster profiles.
    Do not blindly copy a suggested_interpretation from upstream evidence if it
    conflicts with these rules or overstates what the data establishes.
19. Do not automatically assign conventional RFM names such as "Champions",
    "Loyal Customers", "At Risk", or "Lost Customers" unless the evidence and
    business definitions explicitly justify them.
20. Do not describe RFM monetary value as "spend per order", "average order
    value", or "per-order spend" unless an explicit order-level monetary metric
    is provided in the evidence. Customer-level monetary value represents the
    monetary value associated with the customer and must not be treated as an
    order-level metric.
21. Do not use "revenue", "sales", "profit", "profitability", "average order
    value", or "revenue proxy" as synonyms for RFM monetary value unless the
    evidence explicitly defines the monetary metric that way. If the evidence
    identifies the metric as payment_value, describe it as "customer payment
    value" or "monetary value". Do not describe payment_value as a proxy for
    revenue unless the evidence explicitly establishes that relationship.
22. Do not infer profitability, profit contribution, or profit improvement from
    RFM monetary value alone. Monetary value is not the same as profit or
    profitability unless the evidence explicitly provides profit, margin, cost,
    or profitability metrics.
23. If frequency and monetary values were transformed for model fitting but
    cluster profiles are explicitly reported on the winsorized original scale,
    do not say that the reported means require back-transformation. The reported
    means should be interpreted directly on the stated reporting scale.
24. When a segment has a higher average monetary value per customer but is much
    smaller in size, describe this only as higher monetary value per customer.
    Do not use "contributes disproportionately", "contributes more", "drives
    revenue", "accounts for a disproportionate share", or similar aggregate
    contribution language unless aggregate monetary totals by segment are
    explicitly provided.
25. Do not mention customer lifetime value (CLV), customer profitability,
    retention rate, churn rate, conversion rate, ROI, or other derived business
    KPIs unless those metrics are explicitly provided in the evidence. RFM
    monetary value and purchase frequency must not be treated as direct
    measurements of these KPIs.
26. Do not describe a segment as statistically weak or statistically
    underpowered merely because it is smaller than another segment. For
    unsupervised clustering, describe small segments in terms of relative
    representation and potential generalizability unless an actual statistical
    test supports a claim about statistical power or uncertainty.
27. Prefer "higher-monetary-value segment" or "repeat / higher-value segment"
    when referring to a cluster with higher mean monetary value. Avoid
    "high-value" when it could be interpreted as profitability, revenue
    contribution, or customer lifetime value.

28. Do not describe lower or higher monetary value as lower or higher revenue
    potential, sales potential, revenue opportunity, or sales opportunity unless
    the evidence explicitly defines the monetary metric as revenue or sales.
29. Do not use "lifetime engagement", "customer loyalty", or "retention" as
    measured outcomes unless the evidence explicitly provides a corresponding
    metric. Higher frequency or monetary value may be described only as observed
    purchasing behaviour.
30. For silhouette scores, report the numeric score and use conservative wording
    such as "relatively well-separated" unless explicit threshold criteria are
    provided. Do not claim that silhouette proves business validity or statistical
    significance.
31. Recommendations must be grounded in observed evidence. Do not introduce
    unsupported communication channels, customer preferences, or intervention
    effects. Frame recommendations as proposed tests/actions, not guaranteed
    outcomes.
32. For RFM cluster profiles, describe monetary value as customer-level monetary
    value/payment value. Do not equate it with revenue, sales, profit, CLV, or
    average order value unless those metrics are explicitly provided.
33. When describing modelling transformations, state that cluster-level means
    are reported on the winsorized pre-log scale if that is how the evidence is
    produced. Do not imply that reported means are log-space values.
34. Do not describe a smaller cluster as statistically underpowered or
    statistically weak merely because it has fewer customers. For unsupervised
    clustering, describe its representation and generalizability conservatively.
35. In Business Implications, prefer observed statements such as "higher observed
    purchase frequency and customer-level monetary value" over inferred outcomes
    such as "higher lifetime engagement" or "higher revenue potential".

Write the report in this exact Markdown structure:

## Executive Summary
(2–3 sentences: what was analysed, the headline finding, and business significance)

## Key Data Findings
(4–6 specific bullet points. Each must cite an evidence ID and include actual numbers.)

## Statistical Evidence
(For queries involving hypothesis testing, report the test name, test statistic,
p-value, effect size, and plain-language interpretation. For descriptive or
unsupervised analyses such as RFM clustering, do not invent a hypothesis test.
Instead, state that no hypothesis test was performed and report the relevant
model evaluation metric, such as the silhouette score, when provided.)

## Segmentation Insights
(Only include if RFM segmentation data is provided.
Report cluster count, silhouette score, customer counts per segment, mean R/F/M per segment.
Clearly state that labels are derived from K-Means cluster profiles.
For RFM, distinguish modelling transformations from the raw/winsorized values used
to describe cluster profiles.)

## Forecast
(Only include if forecast data is provided.
Report model used, best model, metrics by model, and what the forecast implies.)

## Business Implications
(3–4 sentences explaining what these findings mean for business decisions.
Separate observed evidence from proposed interpretation or hypotheses.)

## Recommended Actions
(3–5 numbered, specific, actionable items grounded in the data above.
Recommendations should be framed as proposed actions, not guaranteed outcomes.)

## Limitations & Confidence
(Flag: small samples, borderline p-values, low silhouette scores, missing data,
queries that returned limited rows, modelling assumptions, or any finding where
confidence is low.)

## Evidence Trace
(List every evidence ID used, what it contains, and which section it supports.)
"""


def _safe_json(obj) -> str:
    """Serialise to JSON, converting non-serialisable types to strings."""
    return json.dumps(obj, indent=2, default=str)


def _build_evidence_block(state: AgentState) -> str:
    """Builds a structured evidence block passed to the LLM."""
    parts = [f"BUSINESS QUESTION:\n{state['query']}\n"]

    # SQL evidence
    parts.append(f"SQL QUERY USED:\n{state.get('sql_query', 'Not available')}\n")
    df = state.get("sql_result")
    if df is not None and hasattr(df, "shape") and not df.empty:
        parts.append(
            f"SQL RESULT SHAPE: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
            f"COLUMNS: {list(df.columns)}\n"
            f"SAMPLE (first 10 rows):\n{df.head(10).to_string(index=False)}\n"
        )
        # Summary statistics for numeric columns
        num = df.select_dtypes(include="number")
        if not num.empty:
            parts.append(
                f"NUMERIC SUMMARY:\n{num.describe().round(3).to_string()}\n"
            )

    # EDA evidence
    eda = state.get("eda_result")
    if eda and "error" not in eda:
        parts.append(f"EDA RESULTS:\n{_safe_json(eda)}\n")

    # Statistical test evidence
    stat = state.get("stat_result")
    if stat and stat.get("test") not in (None, "none"):
        parts.append(f"STATISTICAL TEST:\n{_safe_json(stat)}\n")

    # Segmentation evidence
    seg = state.get("segment_result")
    if seg and "error" not in (seg or {}):
        # Exclude the rfm_df if somehow still present
        safe_seg = {k: v for k, v in seg.items() if k != "rfm_df"}
        parts.append(f"SEGMENTATION RESULTS:\n{_safe_json(safe_seg)}\n")

    # Forecast evidence
    fc = state.get("forecast_result")
    if fc and "error" not in (fc or {}):
        safe_fc = {k: v for k, v in fc.items()
                   if k not in ("full_df", "forecast_df")}
        parts.append(f"FORECAST RESULTS:\n{_safe_json(safe_fc)}\n")

    # Evidence registry
    evidence = state.get("evidence", [])
    if evidence:
        ev_lines = [f"  [{e['id']}] {e['agent'].upper()} — {e['summary']}"
                    for e in evidence]
        parts.append("EVIDENCE REGISTRY:\n" + "\n".join(ev_lines) + "\n")

    return "\n".join(parts)


def business_analyst_node(state: AgentState) -> AgentState:
    """LangGraph node: generates the final BI report grounded in evidence."""
    t0  = time.time()
    llm = get_llm(temperature=0.1)

    evidence_block = _build_evidence_block(state)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=evidence_block),
    ]

    try:
        resp   = llm.invoke(messages)
        report = resp.content.strip()
        llm_calls = state.get("llm_calls", 0) + 1
    except Exception as e:
        report = (
            f"## Executive Summary\n"
            f"Report generation failed: {e}\n\n"
            f"## Key Data Findings\nNo data available for this section.\n\n"
            f"## Actionable Recommendations\nUnable to generate recommendations.\n"
        )
        llm_calls = state.get("llm_calls", 0)

    elapsed = round(time.time() - t0, 2)
    timings = dict(state.get("agent_timings", {}))
    timings["business_analyst"] = elapsed

    plan    = state.get("plan", [])
    current = state.get("current_step", 0)

    return {
        **state,
        "report":        report,
        "current_step":  current + 1,
        "agent_timings": timings,
        "llm_calls":     llm_calls,
    }
