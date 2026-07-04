"""
MCP Tool: Forecasting Calculation

Exposes the demand forecasting logic as an MCP tool. We deliberately use
a simple, explainable weighted-average method (no external ML library)
so the "smarts" of the project stay in the agent reasoning and
orchestration, not a black-box model:

    forecast = 0.5 * avg(recent same-weekday-same-hour values)
             + 0.3 * avg(seasonal same-date-window values)
             + 0.2 * overall average (as a stabilizer)

If the recent/seasonal windows are empty or the calculation otherwise
fails, `compute_forecast` raises -- the calling agent is expected to
catch this and fall back to `naive_average_forecast`. This is the
project's designed failure-handling path (retry -> fallback), not a
silent bug.
"""

from typing import List, Dict
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("energy-forecast-tool")


def _avg(records: List[Dict]) -> float:
    values = [r["demand_mw"] for r in records]
    if not values:
        raise ValueError("Empty window: no historical records to average.")
    return sum(values) / len(values)


@mcp.tool()
def compute_forecast(
    recent_weekly_window: List[Dict],
    seasonal_window: List[Dict],
    overall_avg_mw: float,
) -> dict:
    """
    Compute a weighted-average demand forecast from historical windows.

    Weights: 50% recent same-weekday/hour pattern, 30% seasonal pattern,
    20% overall average (stabilizer against outlier weeks).

    Raises ValueError if recent_weekly_window is empty (not enough
    recent data to trust) -- caller should fall back to
    naive_average_forecast in that case.
    """
    if not recent_weekly_window:
        raise ValueError("No recent weekly data available for forecasting.")

    recent_avg = _avg(recent_weekly_window)

    # Seasonal window may legitimately be empty (e.g. early in the dataset);
    # in that case redistribute its weight to the recent window instead of failing.
    if seasonal_window:
        seasonal_avg = _avg(seasonal_window)
        forecast = 0.5 * recent_avg + 0.3 * seasonal_avg + 0.2 * overall_avg_mw
    else:
        forecast = 0.7 * recent_avg + 0.3 * overall_avg_mw

    return {
        "forecast_mw": round(forecast, 1),
        "method": "weighted_average",
        "components": {
            "recent_avg_mw": round(recent_avg, 1),
            "seasonal_avg_mw": round(seasonal_avg, 1) if seasonal_window else None,
            "overall_avg_mw": overall_avg_mw,
        },
    }


@mcp.tool()
def naive_average_forecast(overall_avg_mw: float) -> dict:
    """
    Fallback forecast: simply returns the overall historical average.
    Used when compute_forecast fails due to insufficient windowed data.
    This is the pipeline's designed graceful-degradation path.
    """
    return {
        "forecast_mw": round(overall_avg_mw, 1),
        "method": "fallback_naive_average",
        "components": {"overall_avg_mw": overall_avg_mw},
    }


if __name__ == "__main__":
    mcp.run()
