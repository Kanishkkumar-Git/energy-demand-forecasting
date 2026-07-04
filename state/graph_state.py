"""
Shared state schema for the Energy Demand Forecasting multi-agent pipeline.

Pattern: Pipeline (sequential handoff)
State strategy: Shared global state -- all agents read/write the same
TypedDict object as it flows through the LangGraph graph. Each agent is
only responsible for writing its own designated keys, to avoid agents
overwriting each other's work (see Week 4 theory: "be deliberate about
which keys each agent is allowed to touch").

Flow:
    Data Agent          -> writes: raw_data_summary, historical_window
    Forecasting Agent    -> writes: forecast_mw, forecast_method_used
    Anomaly Agent         -> writes: anomaly_flags, anomaly_notes
    Report Agent          -> writes: final_report

    Every agent may also write to `errors` (for failure handling / retries)
    and `status` (to track pipeline progress).
"""

from typing import TypedDict, List, Optional


class GraphState(TypedDict, total=False):
    # ---- Input (set before the graph runs) ----
    target_datetime: str          # ISO datetime string we want a forecast for, e.g. "2026-01-15 18:00:00"
    data_path: str                # path to the historical demand CSV

    # ---- Written by Data Agent ----
    raw_data_summary: dict        # e.g. {"rows_loaded": 17544, "date_range": [...], "recent_avg_mw": ...}
    historical_window: list       # list of relevant historical records the forecasting agent needs

    # ---- Written by Forecasting Agent ----
    forecast_mw: Optional[float]      # the predicted demand value
    forecast_method_used: str         # "weighted_average" or "fallback_naive_average"

    # ---- Written by Anomaly/Context Agent ----
    anomaly_flags: List[str]      # e.g. ["holiday", "weekend", "seasonal_extreme"]
    anomaly_notes: str            # short LLM-generated explanation of context

    # ---- Written by Report Agent ----
    final_report: str             # the final plain-English grid management report

    # ---- Cross-cutting: failure handling & tracking ----
    errors: List[str]             # any errors/warnings raised by agents, for retry logic
    status: str                   # tracks pipeline progress, e.g. "data_loaded", "forecast_done", "complete"
