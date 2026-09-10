"""
pipeline.py — LangGraph StateGraph

Architecture:
  START → supervisor → [sql | eda_stats | ml | business_analyst] → critic
                                                                      │
                                                              ┌───────┴───────┐
                                                           PASS             FAIL
                                                              │               │
                                                             END     targeted retry agent
                                                                      OR "failed" if max retries
                                                                              │
                                                                             END

Fail-closed: once MAX_RETRIES is exhausted, routes to "failed" node which
writes an honest failure message. Never continues to business_analyst after
repeated validation failures.
"""

from langgraph.graph import StateGraph, START, END

from state import AgentState
from agents.supervisor       import supervisor_node, route_next
from agents.sql_agent        import sql_agent_node
from agents.eda_stats_agent  import eda_stats_node
from agents.ml_agent         import ml_agent_node
from agents.business_analyst import business_analyst_node
from agents.critic           import critic_node, failed_node


def _critic_router(state: AgentState) -> str:
    """Routes after critic: END on pass, targeted agent on fail, 'failed' on max retries."""
    cr = state.get("critic_result", {}) or {}
    if cr.get("passed", False):
        return END

    # Delegate to supervisor's route_next for retry logic
    return route_next(state)


def build_graph():
    """Builds and compiles the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("supervisor",       supervisor_node)
    graph.add_node("sql",              sql_agent_node)
    graph.add_node("eda_stats",        eda_stats_node)
    graph.add_node("ml",               ml_agent_node)
    graph.add_node("business_analyst", business_analyst_node)
    graph.add_node("critic",           critic_node)
    graph.add_node("failed",           failed_node)      # fail-closed terminal

    # ── Entry ──────────────────────────────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Supervisor routes to first planned agent ────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "sql":              "sql",
            "eda_stats":        "eda_stats",
            "ml":               "ml",
            "business_analyst": "business_analyst",
            "critic":           "critic",
            "failed":           "failed",
        },
    )

    # ── Worker nodes route to next planned agent ────────────────────────────────
    for node in ["sql", "eda_stats", "ml"]:
        graph.add_conditional_edges(
            node,
            route_next,
            {
                "sql":              "sql",
                "eda_stats":        "eda_stats",
                "ml":               "ml",
                "business_analyst": "business_analyst",
                "critic":           "critic",
                "failed":           "failed",
            },
        )

    # ── business_analyst always goes to critic ──────────────────────────────────
    graph.add_edge("business_analyst", "critic")

    # ── Critic: pass → END, fail → targeted retry or failed ────────────────────
    graph.add_conditional_edges(
        "critic",
        _critic_router,
        {
            END:                END,
            "sql":              "sql",
            "eda_stats":        "eda_stats",
            "ml":               "ml",
            "business_analyst": "business_analyst",
            "failed":           "failed",
        },
    )

    # ── Failed terminal ─────────────────────────────────────────────────────────
    graph.add_edge("failed", END)

    return graph.compile()


def _initial_state(query: str) -> AgentState:
    return {
        "query":             query,
        "query_complexity":  "SIMPLE_SQL",
        "plan":              [],
        "current_step":      0,
        "pipeline_status":   "RUNNING",
        "sql_query":         None,
        "sql_result":        None,
        "sql_meta":          None,
        "sql_error":         None,
        "eda_result":        None,
        "stat_result":       None,
        "segment_result":    None,
        "forecast_result":   None,
        "evidence":          [],
        "figures":           [],
        "report":            None,
        "critic_result":     None,
        "iteration":         0,
        "agent_timings":     {},
        "llm_calls":         0,
        "total_retries":     0,
    }


def run_query(query: str) -> AgentState:
    """
    Convenience wrapper: runs the full pipeline for a natural-language query.
    Returns the final AgentState.
    """
    graph = build_graph()
    return graph.invoke(_initial_state(query))
