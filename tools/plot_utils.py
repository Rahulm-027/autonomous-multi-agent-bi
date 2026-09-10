"""
tools/plot_utils.py — Plotly chart helpers.

All charts are saved as self-contained HTML files in figures/.
Returns the file path so agents can register them in state["figures"].
"""

import os
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import FIGURES_DIR


def _save(fig: go.Figure, prefix: str) -> str:
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.html"
    path = os.path.join(FIGURES_DIR, name)
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str = "") -> str:
    fig = px.bar(df, x=x, y=y, title=title, template="plotly_white")
    fig.update_layout(xaxis_tickangle=-35)
    return _save(fig, "bar")


def plot_line(df: pd.DataFrame, x: str, y: str, title: str = "") -> str:
    fig = px.line(df, x=x, y=y, title=title, template="plotly_white", markers=True)
    return _save(fig, "line")


def plot_histogram(df: pd.DataFrame, column: str, title: str = "") -> str:
    fig = px.histogram(df, x=column, title=title, template="plotly_white",
                       marginal="box")
    return _save(fig, "hist")


def plot_scatter(df: pd.DataFrame, x: str, y: str,
                 color: str = None, title: str = "") -> str:
    fig = px.scatter(df, x=x, y=y, color=color,
                     title=title, template="plotly_white",
                     trendline="ols" if color is None else None)
    return _save(fig, "scatter")


def plot_heatmap(corr_df: pd.DataFrame, title: str = "Correlation Matrix") -> str:
    fig = px.imshow(corr_df, text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title=title, template="plotly_white")
    return _save(fig, "heatmap")


def plot_box(df: pd.DataFrame, x: str, y: str, title: str = "") -> str:
    top = df[x].value_counts().nlargest(15).index
    fig = px.box(df[df[x].isin(top)], x=x, y=y,
                 title=title, template="plotly_white")
    fig.update_layout(xaxis_tickangle=-35)
    return _save(fig, "box")


def plot_forecast(full_df: pd.DataFrame, title: str = "Forecast") -> str:
    """
    Expects columns: ds, yhat, yhat_lower, yhat_upper, and optionally y (actuals).
    """
    fig = go.Figure()

    if "y" in full_df.columns:
        actual = full_df[full_df["y"].notna()]
        fig.add_trace(go.Scatter(
            x=actual["ds"], y=actual["y"],
            mode="lines+markers", name="Actual",
            line=dict(color="royalblue", width=2)
        ))

    fig.add_trace(go.Scatter(
        x=full_df["ds"], y=full_df["yhat"],
        mode="lines", name="Prophet Forecast",
        line=dict(color="firebrick", dash="dash", width=2)
    ))

    # Confidence interval as shaded area
    x_fill = list(full_df["ds"]) + list(full_df["ds"])[::-1]
    y_fill = list(full_df["yhat_upper"]) + list(full_df["yhat_lower"])[::-1]
    fig.add_trace(go.Scatter(
        x=x_fill, y=y_fill,
        fill="toself",
        fillcolor="rgba(220,80,80,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% CI"
    ))

    fig.update_layout(title=title, template="plotly_white",
                      xaxis_title="Date", yaxis_title="Value",
                      legend=dict(orientation="h", y=-0.15))
    return _save(fig, "forecast")


def plot_segments(rfm_df: pd.DataFrame, title: str = "Customer Segments") -> str:
    """RFM scatter coloured by K-Means cluster ID."""
    color_col = "cluster"   # always use numeric cluster — labels are in profile table
    size_col  = "frequency" if "frequency" in rfm_df.columns else None

    fig = px.scatter(
        rfm_df.sample(min(2000, len(rfm_df)), random_state=42),
        x="recency", y="monetary",
        color=color_col,
        size=size_col,
        title=title,
        template="plotly_white",
        labels={"recency": "Recency (days since last order)",
                "monetary": "Total Spend (R$)"},
        hover_data=["frequency"] if "frequency" in rfm_df.columns else None,
    )
    return _save(fig, "segments")


def plot_cluster_profile(profile_df: pd.DataFrame,
                         title: str = "Cluster Profiles") -> str:
    """Grouped bar chart of mean R/F/M values per cluster."""
    mean_cols = [c for c in ["mean_recency", "mean_frequency", "mean_monetary"]
                 if c in profile_df.columns]
    if not mean_cols:
        return ""
    df = profile_df.copy()
    df["cluster_id"] = df["cluster"].astype(str).apply(lambda c: f"Cluster {c}")
    melted = df.melt(
        id_vars=["cluster_id"],
        value_vars=mean_cols,
        var_name="metric",
        value_name="mean_value",
    )
    fig = px.bar(
        melted,
        x="metric", y="mean_value",
        color="cluster_id",
        barmode="group",
        title=title,
        template="plotly_white",
        labels={"metric": "RFM Dimension", "mean_value": "Mean Value (winsorized)"},
    )
    return _save(fig, "cluster_profile")
