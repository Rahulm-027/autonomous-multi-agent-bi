"""
agents/eda_stats_agent.py — EDA and Statistics Agent

Performs:
  1. Descriptive statistics + outlier detection + correlation
  2. Hypothesis testing — test is determined by the plan step, NOT by guessing.

The Supervisor's plan step specifies:
  - stat_question, null_hypothesis, stat_test, group_column, value_column

If not specified, the agent selects the most appropriate test based on
data structure — with documented logic, not arbitrary column picking.

All results include: test name, statistic, p-value, effect size,
sample sizes, significance level, and plain-language interpretation.
"""

import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from langchain_core.messages import HumanMessage

from state import AgentState
from config import get_llm, SIGNIFICANCE_LEVEL, MIN_SAMPLE_SIZE
from tools.plot_utils import plot_histogram, plot_heatmap, plot_bar, plot_box, plot_scatter

warnings.filterwarnings("ignore")


# ── Descriptive statistics ────────────────────────────────────────────────────
def _descriptive(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include="number")
    if num.empty:
        return {}
    desc = num.describe().round(4).to_dict()
    return desc


def _outliers(df: pd.DataFrame) -> dict:
    out = {}
    for col in df.select_dtypes(include="number").columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n > 0:
            out[col] = {"n_outliers": n,
                        "lower_fence": round(float(lo), 4),
                        "upper_fence": round(float(hi), 4)}
    return out


def _correlation(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return {}
    return num.corr(method="pearson").round(4).to_dict()


# ── Cohen's d (effect size for t-test) ───────────────────────────────────────
def _cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1 - 1) * g1.std() ** 2 + (n2 - 1) * g2.std() ** 2) / (n1 + n2 - 2)
    )
    return float((g1.mean() - g2.mean()) / pooled_std) if pooled_std > 0 else 0.0


def _effect_magnitude(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# ── Cramér's V (effect size for chi-squared) ─────────────────────────────────
def _cramers_v(ct: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim))) if n > 0 and min_dim > 0 else 0.0


# ── Spearman rho (effect size for correlation) ────────────────────────────────
def _interpret_correlation(rho: float) -> str:
    a = abs(rho)
    if a < 0.1:
        return "negligible"
    elif a < 0.3:
        return "weak"
    elif a < 0.5:
        return "moderate"
    elif a < 0.7:
        return "strong"
    else:
        return "very strong"


# ── Hypothesis tests ──────────────────────────────────────────────────────────
def _run_ttest(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
    if len(groups) != 2:
        return {"test": "ttest", "error": f"Expected 2 groups, found {len(groups)}"}
    g1, g2 = groups[0], groups[1]
    if len(g1) < MIN_SAMPLE_SIZE or len(g2) < MIN_SAMPLE_SIZE:
        return {"test": "ttest", "error":
                f"Insufficient sample sizes: {len(g1)}, {len(g2)} (need ≥ {MIN_SAMPLE_SIZE})"}

    stat, p = stats.ttest_ind(g1, g2, equal_var=False)
    d = _cohens_d(g1, g2)
    sig = p < SIGNIFICANCE_LEVEL
    return {
        "test": "Welch t-test",
        "group_column": group_col,
        "value_column": value_col,
        "n_group1": int(len(g1)),
        "n_group2": int(len(g2)),
        "mean_group1": round(float(g1.mean()), 4),
        "mean_group2": round(float(g2.mean()), 4),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(sig),
        "effect_size_cohens_d": round(d, 4),
        "effect_magnitude": _effect_magnitude(d),
        "significance_level": SIGNIFICANCE_LEVEL,
        "interpretation": (
            f"There IS a statistically significant difference in '{value_col}' "
            f"between groups (p={p:.4f}, Cohen's d={d:.2f} — {_effect_magnitude(d)} effect)."
            if sig else
            f"No statistically significant difference in '{value_col}' "
            f"between groups (p={p:.4f}). Fail to reject H0."
        ),
    }


def _run_mannwhitney(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
    if len(groups) != 2:
        return {"test": "mannwhitney", "error": f"Expected 2 groups, found {len(groups)}"}
    g1, g2 = groups[0], groups[1]
    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    sig = p < SIGNIFICANCE_LEVEL
    # Rank-biserial correlation as effect size
    n1, n2 = len(g1), len(g2)
    r = 1 - (2 * stat) / (n1 * n2)
    return {
        "test": "Mann-Whitney U",
        "group_column": group_col,
        "value_column": value_col,
        "n_group1": int(n1),
        "n_group2": int(n2),
        "median_group1": round(float(g1.median()), 4),
        "median_group2": round(float(g2.median()), 4),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(sig),
        "effect_size_r": round(float(r), 4),
        "significance_level": SIGNIFICANCE_LEVEL,
        "interpretation": (
            f"Significant difference in '{value_col}' distribution between groups "
            f"(p={p:.4f}, r={r:.2f})."
            if sig else
            f"No significant difference in '{value_col}' distribution (p={p:.4f})."
        ),
    }


def _run_anova(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    groups = [
        g[value_col].dropna()
        for _, g in df.groupby(group_col)
        if len(g) >= MIN_SAMPLE_SIZE
    ]
    if len(groups) < 3:
        return {"test": "anova",
                "error": f"Need ≥ 3 groups with ≥ {MIN_SAMPLE_SIZE} rows each"}
    stat, p = stats.f_oneway(*groups)
    sig = p < SIGNIFICANCE_LEVEL
    return {
        "test": "One-way ANOVA",
        "group_column": group_col,
        "value_column": value_col,
        "n_groups": len(groups),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(sig),
        "significance_level": SIGNIFICANCE_LEVEL,
        "interpretation": (
            f"Significant variation in '{value_col}' across groups (p={p:.4f}). "
            f"Post-hoc testing recommended."
            if sig else
            f"No significant variation in '{value_col}' across groups (p={p:.4f})."
        ),
    }


def _run_chisquare(df: pd.DataFrame, col1: str, col2: str) -> dict:
    ct = pd.crosstab(df[col1], df[col2])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"test": "chisquare", "error": "Contingency table too small"}
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    sig = p < SIGNIFICANCE_LEVEL
    v = _cramers_v(ct)
    return {
        "test": "Chi-squared",
        "col1": col1,
        "col2": col2,
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p), 6),
        "dof": int(dof),
        "significant": bool(sig),
        "effect_size_cramers_v": round(v, 4),
        "significance_level": SIGNIFICANCE_LEVEL,
        "interpretation": (
            f"Significant association between '{col1}' and '{col2}' "
            f"(χ²={chi2:.2f}, p={p:.4f}, V={v:.2f})."
            if sig else
            f"No significant association between '{col1}' and '{col2}' (p={p:.4f})."
        ),
    }


def _run_correlation(df: pd.DataFrame, col1: str, col2: str) -> dict:
    data = df[[col1, col2]].dropna()
    if len(data) < MIN_SAMPLE_SIZE:
        return {"test": "correlation",
                "error": f"Insufficient data: {len(data)} rows (need ≥ {MIN_SAMPLE_SIZE})"}
    rho, p = stats.spearmanr(data[col1], data[col2])
    sig = p < SIGNIFICANCE_LEVEL
    mag = _interpret_correlation(rho)
    return {
        "test": "Spearman correlation",
        "col1": col1,
        "col2": col2,
        "n": int(len(data)),
        "rho": round(float(rho), 4),
        "p_value": round(float(p), 6),
        "significant": bool(sig),
        "effect_magnitude": mag,
        "significance_level": SIGNIFICANCE_LEVEL,
        "interpretation": (
            f"{mag.capitalize()} {'positive' if rho > 0 else 'negative'} correlation "
            f"between '{col1}' and '{col2}' (ρ={rho:.3f}, p={p:.4f})."
            if sig else
            f"No significant correlation between '{col1}' and '{col2}' (p={p:.4f})."
        ),
    }


# ── Auto test selector (fallback when plan doesn't specify) ───────────────────
def _auto_select_test(df: pd.DataFrame, task: str) -> dict:
    """
    Selects the most appropriate test based on data structure.
    Documents the selection logic explicitly.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if not num_cols:
        return {"test": "none", "reason": "No numeric columns available"}

    # Two numeric columns → correlation
    if len(num_cols) >= 2 and not cat_cols:
        return _run_correlation(df, num_cols[0], num_cols[1])

    # One categorical (2 groups) + one numeric → t-test
    if cat_cols and num_cols:
        n_groups = df[cat_cols[0]].nunique()
        if n_groups == 2:
            return _run_ttest(df, cat_cols[0], num_cols[0])
        elif 3 <= n_groups <= 10:
            return _run_anova(df, cat_cols[0], num_cols[0])

    # Two categorical columns → chi-squared
    if len(cat_cols) >= 2:
        return _run_chisquare(df, cat_cols[0], cat_cols[1])

    return {"test": "none", "reason": "Could not determine appropriate test"}


# ── Main node ─────────────────────────────────────────────────────────────────
def eda_stats_node(state: AgentState) -> AgentState:
    """LangGraph node: descriptive EDA + hypothesis testing."""
    t0 = time.time()

    df = state.get("sql_result")
    if df is None or (hasattr(df, "empty") and df.empty):
        elapsed = round(time.time() - t0, 2)
        timings = dict(state.get("agent_timings", {}))
        timings["eda_stats"] = elapsed
        return {
            **state,
            "eda_result": {"error": "No data received from SQL agent"},
            "stat_result": {"test": "none", "reason": "No data"},
            "current_step": state.get("current_step", 0) + 1,
            "agent_timings": timings,
        }

    plan    = state.get("plan", [])
    current = state.get("current_step", 0)
    step    = plan[current] if current < len(plan) else {}
    figures = list(state.get("figures", []))
    evidence = list(state.get("evidence", []))

    # Descriptive stats
    desc    = _descriptive(df)
    outliers = _outliers(df)
    corr    = _correlation(df)

    eda_result = {
        "shape":         list(df.shape),
        "descriptive":   desc,
        "outliers":      outliers,
        "correlation":   corr,
        "missing_values": {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
    }

    # Hypothesis test — use plan-specified parameters if available
    stat_test  = step.get("stat_test")
    group_col  = step.get("group_column")
    value_col  = step.get("value_column")
    stat_q     = step.get("stat_question", "")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Validate plan-specified columns exist in the data
    if group_col and group_col not in df.columns:
        group_col = None
    if value_col and value_col not in df.columns:
        value_col = None

    if stat_test and group_col and value_col:
        if stat_test == "ttest":
            stat_result = _run_ttest(df, group_col, value_col)
        elif stat_test == "mannwhitney":
            stat_result = _run_mannwhitney(df, group_col, value_col)
        elif stat_test == "anova":
            stat_result = _run_anova(df, group_col, value_col)
        elif stat_test == "chisquare" and len(cat_cols) >= 2:
            stat_result = _run_chisquare(df, cat_cols[0], cat_cols[1])
        elif stat_test == "correlation" and len(num_cols) >= 2:
            c1 = group_col if group_col in num_cols else num_cols[0]
            c2 = value_col if value_col in num_cols else num_cols[1]
            stat_result = _run_correlation(df, c1, c2)
        else:
            stat_result = _auto_select_test(df, stat_q or step.get("task", ""))
    else:
        stat_result = _auto_select_test(df, stat_q or step.get("task", ""))

    if stat_q:
        stat_result["research_question"] = stat_q
    if step.get("null_hypothesis"):
        stat_result["null_hypothesis"] = step["null_hypothesis"]

    # Charts
    if num_cols:
        figures.append(plot_histogram(df, num_cols[0],
                                      title=f"Distribution: {num_cols[0]}"))
    if cat_cols and num_cols:
        figures.append(plot_box(df, cat_cols[0], num_cols[0],
                                title=f"{num_cols[0]} by {cat_cols[0]}"))
    if len(num_cols) >= 2:
        corr_df = df[num_cols].corr()
        figures.append(plot_heatmap(corr_df, "Correlation Matrix"))

    evidence.append({
        "id":       f"STAT-{len(evidence)+1:02d}",
        "agent":    "eda_stats",
        "type":     "statistical_test",
        "summary":  f"{stat_result.get('test','N/A')} — p={stat_result.get('p_value','N/A')}",
        "artifact": "stat_result",
    })

    elapsed = round(time.time() - t0, 2)
    timings = dict(state.get("agent_timings", {}))
    timings["eda_stats"] = elapsed

    return {
        **state,
        "eda_result":   eda_result,
        "stat_result":  stat_result,
        "figures":      figures,
        "evidence":     evidence,
        "current_step": current + 1,
        "agent_timings": timings,
        "llm_calls":    state.get("llm_calls", 0),
        "critic_result": None,
    }
