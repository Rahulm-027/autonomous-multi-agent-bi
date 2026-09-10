"""
agents/ml_agent.py — ML Agent (Segmentation + Forecasting)

SEGMENTATION
─────────────
RFM is built directly from the Olist transactional tables — NOT from
arbitrary numeric columns of whatever the SQL agent returned.

Business definitions:
  Recency  = days between the customer's most recent DELIVERED order
              and a reference date (max order date in the dataset).
  Frequency = count of DISTINCT delivered orders per customer.
  Monetary  = sum of payment_value from order_payments for those orders.

Pipeline:
  DB query → RFM DataFrame → outlier clipping (99th percentile) →
  optional log transform → StandardScaler → KMeans (best k by silhouette) →
  cluster profiling → business interpretation

Labels are derived from K-Means cluster profiles (quantitative),
NOT from arbitrary hand-coded RFM score thresholds.

FORECASTING
────────────
Target column and frequency are taken from the plan step (set by Supervisor).
Uses chronological holdout (NOT random split).
Compares Prophet against two baselines:
  1. Naive forecast (last observed value carried forward)
  2. Seasonal naive (value from same period last cycle)
Metrics: MAE, RMSE, WAPE, sMAPE
"""

import time
import warnings
import numpy as np
import pandas as pd
import duckdb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from state import AgentState
from config import (
    get_llm, DB_PATH, RANDOM_STATE,
    N_CLUSTERS_MIN, N_CLUSTERS_MAX,
    MIN_SILHOUETTE, MIN_CLUSTER_SIZE,
    MIN_TS_OBSERVATIONS, FORECAST_HORIZON,
)
from tools.plot_utils import (
    plot_segments, plot_cluster_profile, plot_bar, plot_forecast
)

SILHOUETTE_SAMPLE_SIZE = 5000

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION — RFM from database
# ═══════════════════════════════════════════════════════════════════════════════

RFM_SQL = """
WITH delivered_orders AS (
    SELECT
        c.customer_unique_id,
        o.order_id,
        o.order_purchase_timestamp
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
),

customer_payments AS (
    SELECT
        d.customer_unique_id,
        SUM(p.payment_value) AS monetary,
        COUNT(DISTINCT d.order_id) AS frequency,
        MAX(CAST(d.order_purchase_timestamp AS DATE)) AS last_order_date
    FROM delivered_orders d
    JOIN payments p
        ON d.order_id = p.order_id
    WHERE p.payment_value > 0
    GROUP BY d.customer_unique_id
),

reference AS (
    SELECT
        MAX(CAST(order_purchase_timestamp AS DATE)) AS ref_date
    FROM orders
)

SELECT
    cp.customer_unique_id,
    DATEDIFF(
        'day',
        cp.last_order_date,
        r.ref_date
    ) AS recency,
    cp.frequency,
    ROUND(cp.monetary, 2) AS monetary
FROM customer_payments cp
CROSS JOIN reference r
WHERE cp.monetary > 0
  AND cp.frequency > 0
"""


def _build_rfm_from_db() -> pd.DataFrame:
    """
    Builds RFM directly from the Olist database using the canonical
    business definitions. Returns a DataFrame with columns:
    customer_unique_id, recency (days), frequency (orders), monetary (R$).
    """
    conn = duckdb.connect(DB_PATH, read_only=True)
    rfm  = conn.execute(RFM_SQL).df()
    conn.close()
    return rfm


def _describe_rfm_level(value: float, low_thresh: float, high_thresh: float,
                        invert: bool = False) -> str:
    """Returns 'Low' / 'Medium' / 'High' relative to dataset thresholds.
    invert=True is used for Recency (lower = more recent = better)."""
    if invert:
        if value <= low_thresh:
            return "High (recent)"
        elif value >= high_thresh:
            return "Low (lapsed)"
        return "Medium"
    else:
        if value >= high_thresh:
            return "High"
        elif value <= low_thresh:
            return "Low"
        return "Medium"


def _profile_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a cluster profile table with quantitative R/F/M descriptions.

    DESIGN DECISION:
    We do not assign conventional business labels such as Champions,
    At-Risk, Loyal Customers, or Lapsed Customers because cluster-level
    RFM values alone do not establish those business states.

    Instead we provide:
      - Mean R/F/M values on the winsorized scale
      - Neutral descriptions of each RFM dimension
      - A descriptive interpretation based only on observed
        frequency and monetary behaviour

    Business interpretation remains the responsibility of the
    Business Analyst and must not overstate what the data establishes.
    """

    profile = (
        rfm.groupby("cluster")
        .agg(
            n_customers=("customer_unique_id", "count"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
        )
        .round(2)
        .reset_index()
    )

    # Thresholds for Low/Medium/High descriptions (33rd and 67th percentiles)
    r_lo, r_hi = (
        profile["mean_recency"].quantile(0.33),
        profile["mean_recency"].quantile(0.67),
    )
    f_lo, f_hi = (
        profile["mean_frequency"].quantile(0.33),
        profile["mean_frequency"].quantile(0.67),
    )
    m_lo, m_hi = (
        profile["mean_monetary"].quantile(0.33),
        profile["mean_monetary"].quantile(0.67),
    )

    # Neutral recency descriptions.
    # Lower recency = purchased more recently.
    def _describe_recency(value: float) -> str:
        if value <= r_lo:
            return "Lower (more recent)"
        elif value >= r_hi:
            return "Higher (less recent)"
        return "Medium"

    def _suggest(row: pd.Series) -> str:
        f = _describe_rfm_level(
            row["mean_frequency"], f_lo, f_hi
        )
        m = _describe_rfm_level(
            row["mean_monetary"], m_lo, m_hi
        )

        # Evidence-based descriptive labels only.
        if f == "High" and m == "High":
            interpretation = "Repeat / higher-value customers"
        elif f == "Low" and m == "Low":
            interpretation = "One-time / lower-value customers"
        elif f == "High":
            interpretation = "Repeat-purchase customers"
        elif m == "High":
            interpretation = "Higher-value customers"
        elif f == "Low":
            interpretation = "Lower-frequency customers"
        elif m == "Low":
            interpretation = "Lower-value customers"
        else:
            interpretation = "Mixed RFM characteristics"

        return (
            f"Recency: {_describe_recency(row['mean_recency'])} | "
            f"Frequency: {f} | "
            f"Monetary: {m} "
            f"— (suggested) {interpretation}"
        )

    profile["recency_level"] = profile["mean_recency"].apply(
        _describe_recency
    )

    profile["frequency_level"] = profile["mean_frequency"].apply(
        lambda v: _describe_rfm_level(v, f_lo, f_hi)
    )

    profile["monetary_level"] = profile["mean_monetary"].apply(
        lambda v: _describe_rfm_level(v, m_lo, m_hi)
    )

    profile["suggested_interpretation"] = profile.apply(
        _suggest,
        axis=1
    )

    return profile


def _segmentation(plan_step: dict) -> dict:
    """Full RFM segmentation pipeline."""
    # 1. Build RFM from database
    rfm = _build_rfm_from_db()

    if len(rfm) < 50:
        return {"error": f"Insufficient customers for segmentation: {len(rfm)}"}

    # 2. Clip extreme outliers at 99th percentile (per column)
    for col in ["recency", "frequency", "monetary"]:
        cap = rfm[col].quantile(0.99)
        rfm[col] = rfm[col].clip(upper=cap)

    # 3. Log-transform skewed distributions (monetary and frequency are typically right-skewed)
    rfm["log_frequency"] = np.log1p(rfm["frequency"])
    rfm["log_monetary"]  = np.log1p(rfm["monetary"])

    features = ["recency", "log_frequency", "log_monetary"]

    # 4. Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features])

    # 5. K-Means with silhouette selection
    best_k     = N_CLUSTERS_MIN
    best_score = -1.0
    scores     = {}

    for k in range(N_CLUSTERS_MIN, min(N_CLUSTERS_MAX + 1, len(rfm) // MIN_CLUSTER_SIZE + 1)):
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sizes  = pd.Series(labels).value_counts()
        if sizes.min() < MIN_CLUSTER_SIZE:
            continue
        
        if len(X) > SILHOUETTE_SAMPLE_SIZE:
            rng = np.random.RandomState(RANDOM_STATE)
            sample_idx = rng.choice(
                len(X),
                size=SILHOUETTE_SAMPLE_SIZE,
                replace=False
            )
            X_sil = X[sample_idx]
            labels_sil = labels[sample_idx]
        else:
            X_sil = X
            labels_sil = labels

        s = silhouette_score(X_sil, labels_sil)
        scores[k] = round(float(s), 4)
        if s > best_score:
            best_score = s
            best_k     = k

    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    rfm["cluster"] = km_final.fit_predict(X)

    # 6. Profile clusters
    profile = _profile_clusters(rfm)

    return {
        "method":            "RFM + K-Means (silhouette-selected k)",
        "rfm_definition": {
            "recency":   "Days since last DELIVERED order (reference = max order date in dataset). Lower = more recent.",
            "frequency": "Count of distinct DELIVERED orders per customer.",
            "monetary":  "Sum of payment_value (R$) across all delivered orders.",
        },
        "data_notes": {
            "winsorization": (
                "R/F/M values are clipped at the 99th percentile before clustering. "
                "Reported mean_recency / mean_frequency / mean_monetary reflect "
                "winsorized (capped) values, not raw customer totals."
            ),
            "log_transform": (
                "Log1p transformation applied to frequency and monetary before "
                "K-Means to reduce skewness. Clustering is performed in log space."
            ),
            "cluster_labels": (
                "No named business labels (Champions, At-Risk, etc.) are assigned "
                "automatically. Each cluster has a quantitative profile and a "
                "suggested_interpretation. Final business interpretation is left "
                "to the analyst."
            ),
        },
        "n_customers":       int(len(rfm)),
        "best_k":            int(best_k),
        "silhouette_score":  round(float(best_score), 4),
        "silhouette_by_k":   scores,
        "cluster_profiles":  profile.to_dict(orient="records"),
        "rfm_df":            rfm,   # kept for plotting, stripped before JSON
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FORECASTING — chronological split, baselines, multiple metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """sMAPE safe against zero denominators."""
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask  = denom > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def _safe_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    total = np.abs(actual).sum()
    return float(np.abs(actual - predicted).sum() / total * 100) if total > 0 else 0.0


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae   = float(np.mean(np.abs(actual - predicted)))
    rmse  = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    wape  = _safe_wape(actual, predicted)
    smape = _safe_smape(actual, predicted)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "WAPE_%": round(wape, 2), "sMAPE_%": round(smape, 2)}


def _naive_forecast(train: pd.DataFrame, n_periods: int) -> np.ndarray:
    """Naive: carry last observed value forward."""
    return np.full(n_periods, train["y"].iloc[-1])


def _seasonal_naive(train: pd.DataFrame, n_periods: int,
                    season_length: int = 12) -> np.ndarray:
    """Seasonal naive: repeat last full season."""
    y = train["y"].values
    if len(y) < season_length:
        return _naive_forecast(train, n_periods)
    preds = []
    for i in range(n_periods):
        preds.append(y[-(season_length - (i % season_length))])
    return np.array(preds)


def _forecasting(plan_step: dict, df: pd.DataFrame) -> dict:
    """
    Runs Prophet + baselines on the provided time-series DataFrame.
    The plan step MUST specify forecast_target and forecast_freq.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return {"error": "Prophet not installed. Run: pip install prophet"}

    target_col = plan_step.get("forecast_target")
    freq       = plan_step.get("forecast_freq", "M")
    horizon    = int(plan_step.get("forecast_horizon", FORECAST_HORIZON))

    # Identify date and value columns
    date_col  = None
    value_col = None

    # Find date column
    for col in df.columns:
        if "datetime" in str(df[col].dtype).lower():
            date_col = col
            break
        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col].dropna().iloc[0])
                date_col = col
                break
            except Exception:
                continue

    # Fail explicitly if Supervisor did not specify a target — never guess
    if not target_col:
        return {
            "error": (
                "Supervisor did not specify 'forecast_target' in the plan step. "
                "The forecasting agent requires an explicit target column name."
            )
        }
    if target_col not in df.columns:
        available = df.select_dtypes(include="number").columns.tolist()
        return {
            "error": (
                f"Forecast target '{target_col}' not found in SQL result. "
                f"Available numeric columns: {available}"
            )
        }

    value_col = target_col

    if not date_col:
        return {"error": "No date column found in the data for forecasting."}

    ts = df[[date_col, value_col]].copy()
    ts.columns = ["ds", "y"]
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts = ts.dropna().sort_values("ds")
    ts = ts.groupby("ds")["y"].sum().reset_index()

    if len(ts) < MIN_TS_OBSERVATIONS:
        return {
            "error": f"Insufficient time-series data: {len(ts)} points "
                     f"(need ≥ {MIN_TS_OBSERVATIONS})."
        }

    # Chronological split — last `horizon` periods as test
    split = len(ts) - horizon
    if split < MIN_TS_OBSERVATIONS // 2:
        split = max(MIN_TS_OBSERVATIONS // 2, len(ts) - 3)

    train = ts.iloc[:split].copy()
    test  = ts.iloc[split:].copy()
    actual_test = test["y"].values

    results = {}

    # Baseline 1: Naive
    naive_pred  = _naive_forecast(train, len(test))
    results["Naive"] = _metrics(actual_test, naive_pred)

    # Baseline 2: Seasonal Naive (season = 12 for monthly)
    season_len = 12 if freq == "M" else 4 if freq == "Q" else 7
    sn_pred    = _seasonal_naive(train, len(test), season_len)
    results["Seasonal Naive"] = _metrics(actual_test, sn_pred)

    # Prophet
    m = Prophet(
        yearly_seasonality=(freq in ("M", "Q", "W")),
        weekly_seasonality=(freq == "W"),
        daily_seasonality=False,
    )
    try:
        m.fit(train)
    except Exception as e:
        return {"error": f"Prophet training failed: {e}"}

    future   = m.make_future_dataframe(periods=horizon + len(test), freq=freq)
    forecast = m.predict(future)

    # Extract test-period predictions
    fc_test = forecast[forecast["ds"].isin(test["ds"])]
    if len(fc_test) == 0:
        # If dates don't match exactly, take the last len(test) rows
        fc_test = forecast.tail(horizon + len(test)).head(len(test))

    prophet_pred = fc_test["yhat"].values[:len(actual_test)]
    if len(prophet_pred) == len(actual_test):
        results["Prophet"] = _metrics(actual_test, prophet_pred)
    else:
        results["Prophet"] = {"error": "Prediction length mismatch"}

    # Best model by RMSE
    valid_models = {
        k: v for k, v in results.items()
        if isinstance(v, dict) and "RMSE" in v
    }
    best_model = min(valid_models, key=lambda k: valid_models[k]["RMSE"]) \
        if valid_models else "Unknown"

    # Future forecast
    future_only = m.make_future_dataframe(periods=horizon, freq=freq)
    future_fc   = m.predict(future_only)

    # Full DataFrame for plotting
    full_df = forecast.merge(ts, on="ds", how="left")[
        ["ds", "y", "yhat", "yhat_lower", "yhat_upper"]
    ]

    return {
        "method":          "Prophet + Baselines",
        "target_column":   value_col,
        "date_column":     date_col,
        "frequency":       freq,
        "horizon":         horizon,
        "n_train":         int(len(train)),
        "n_test":          int(len(test)),
        "model_metrics":   results,
        "best_model":      best_model,
        "note": (
            "Baselines: Naive (last value carried forward), "
            "Seasonal Naive (same period last season). "
            "Best model selected by lowest RMSE on chronological holdout."
        ),
        "full_df":         full_df,   # for plotting, stripped before JSON
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph node
# ═══════════════════════════════════════════════════════════════════════════════
def ml_agent_node(state: AgentState) -> AgentState:
    """LangGraph node: runs segmentation or forecasting per plan instructions."""
    t0 = time.time()

    plan    = state.get("plan", [])
    current = state.get("current_step", 0)
    step    = plan[current] if current < len(plan) else {}
    ml_task = step.get("ml_task", "segmentation")
    figures = list(state.get("figures", []))
    evidence = list(state.get("evidence", []))

    # Critic retry override
    cr = state.get("critic_result")
    if cr and not cr.get("passed", True) and cr.get("retry_agent") == "ml":
        ml_task = step.get("ml_task", "segmentation")

    if ml_task == "forecasting":
        df = state.get("sql_result")
        if df is None or df.empty:
            result = {"error": "No data from SQL agent for forecasting"}
        else:
            result = _forecasting(step, df)

        full_df = result.pop("full_df", None)
        fc_result = {k: v for k, v in result.items() if k != "full_df"}

        if full_df is not None and "error" not in result:
            fig = plot_forecast(full_df, title="Order Volume Forecast")
            figures.append(fig)
            evidence.append({
                "id":       f"FORECAST-{len(evidence)+1:02d}",
                "agent":    "ml",
                "type":     "forecast",
                "summary":  (
                    f"Best model: {result.get('best_model','?')} | "
                    f"Metrics: {result.get('model_metrics',{}).get(result.get('best_model',''),{})}"
                ),
                "artifact": "forecast_result",
            })

        elapsed = round(time.time() - t0, 2)
        timings = dict(state.get("agent_timings", {}))
        timings["ml"] = elapsed

        return {
            **state,
            "forecast_result": fc_result,
            "figures":         figures,
            "evidence":        evidence,
            "current_step":    current + 1,
            "agent_timings":   timings,
            "critic_result":   None,
        }

    else:
        # Segmentation — always built from DB, ignores sql_result columns
        result  = _segmentation(step)
        rfm_df  = result.pop("rfm_df", None)

        if rfm_df is not None and "error" not in result:
            # Cluster profile
            profile_df = pd.DataFrame(result.get("cluster_profiles", []))
            if not profile_df.empty:
                fig1 = plot_segments(rfm_df, title="Customer Segments (RFM by cluster)")
                figures.append(fig1)
                fig2 = plot_cluster_profile(profile_df, title="Cluster Mean RFM Profiles")
                figures.append(fig2)
                counts = profile_df[["cluster", "n_customers"]].copy()
                counts["cluster"] = counts["cluster"].astype(str).apply(lambda c: f"Cluster {c}")
                fig3   = plot_bar(counts, x="cluster", y="n_customers",
                                  title="Customers per Cluster")
                figures.append(fig3)

            # Build traceable evidence summary including cluster profiles
            profile_lines = []

            for p in result.get("cluster_profiles", []):
                profile_lines.append(
                    f"Cluster {p.get('cluster', '?')}: "
                    f"n={p.get('n_customers', '?')} customers; "
                    f"mean_recency={p.get('mean_recency', '?')} days; "
                    f"mean_frequency={p.get('mean_frequency', '?')} orders; "
                    f"mean_monetary=R${p.get('mean_monetary', '?')}; "
                    f"interpretation={p.get('suggested_interpretation', '?')}"
                )

            profile_summary = "\n".join(profile_lines)

            evidence.append({
                "id":       f"RFM-{len(evidence)+1:02d}",
                "agent":    "ml",
                "type":     "segmentation",
                "summary":  (
                    f"K-Means RFM segmentation: "
                    f"k={result.get('best_k', '?')}, "
                    f"silhouette={result.get('silhouette_score', '?')}, "
                    f"n={result.get('n_customers', '?')} customers.\n"
                    f"Cluster profiles:\n{profile_summary}"
                ),
                "artifact": "segment_result",
            })

        elapsed = round(time.time() - t0, 2)
        timings = dict(state.get("agent_timings", {}))
        timings["ml"] = elapsed

        return {
            **state,
            "segment_result": result,
            "figures":        figures,
            "evidence":       evidence,
            "current_step":   current + 1,
            "agent_timings":  timings,
            "critic_result":  None,
        }
