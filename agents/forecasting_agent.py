"""
Forecasting Agent

Role: produces the numeric demand forecast from the historical windows
prepared by the Data Agent. Calls the forecast_tool MCP tool
(compute_forecast). Deterministic, tool-executing agent.

Failure handling (Week 4 requirement): if compute_forecast fails
(e.g. insufficient recent data), this agent retries once with a
relaxed assumption, and if that still fails, falls back to
naive_average_forecast rather than letting the pipeline die or
silently produce a garbage value.

Writes to shared state: forecast_mw, forecast_method_used, status, errors
"""

from mcp_tools.forecast_tool import compute_forecast, naive_average_forecast
from state.graph_state import GraphState


def forecasting_agent(state: GraphState) -> GraphState:
    errors = list(state.get("errors", []))
    window = state.get("historical_window", {})
    summary = state.get("raw_data_summary", {})
    overall_avg = summary.get("overall_avg_mw")

    recent = window.get("recent_weekly_window", [])
    seasonal = window.get("seasonal_window", [])

    # Primary attempt
    try:
        result = compute_forecast(recent, seasonal, overall_avg)
        return {
            "forecast_mw": result["forecast_mw"],
            "forecast_method_used": result["method"],
            "status": "forecast_done",
            "errors": errors,
        }
    except Exception as e:
        errors.append(f"forecasting_agent (primary attempt): {str(e)}")

    # Retry: relax the requirement by treating seasonal window as the recent
    # window if recent is what's missing (handles edge case near dataset start)
    try:
        if not recent and seasonal:
            result = compute_forecast(seasonal, [], overall_avg)
            errors.append("forecasting_agent: retried using seasonal window as substitute.")
            return {
                "forecast_mw": result["forecast_mw"],
                "forecast_method_used": result["method"] + "_retry",
                "status": "forecast_done",
                "errors": errors,
            }
    except Exception as e:
        errors.append(f"forecasting_agent (retry attempt): {str(e)}")

    # Fallback: naive average, guarantees the pipeline degrades gracefully
    # instead of failing silently or crashing.
    if overall_avg is None:
        overall_avg = 30000.0  # last-resort safety default
        errors.append("forecasting_agent: overall_avg_mw missing, used hardcoded safety default.")

    fallback = naive_average_forecast(overall_avg)
    errors.append("forecasting_agent: used fallback naive average forecast.")
    return {
        "forecast_mw": fallback["forecast_mw"],
        "forecast_method_used": fallback["method"],
        "status": "forecast_done_via_fallback",
        "errors": errors,
    }
