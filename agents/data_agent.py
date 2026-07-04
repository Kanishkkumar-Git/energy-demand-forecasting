"""
Data Agent

Role: retrieves and prepares the historical demand data needed for
forecasting. Calls the data_tool MCP tool (load_data_summary,
get_historical_window). This is a deterministic, tool-executing agent
-- it doesn't need LLM reasoning, its job is data retrieval.

Writes to shared state: raw_data_summary, historical_window, status, errors
"""

from mcp_tools.data_tool import load_data_summary, get_historical_window
from state.graph_state import GraphState


def data_agent(state: GraphState) -> GraphState:
    errors = list(state.get("errors", []))
    data_path = state.get("data_path", "data/energy_demand.csv")
    target_datetime = state["target_datetime"]

    try:
        summary = load_data_summary(data_path)
        window = get_historical_window(target_datetime, data_path)

        return {
            "raw_data_summary": summary,
            "historical_window": window,
            "status": "data_loaded",
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"data_agent: {str(e)}")
        return {
            "raw_data_summary": {},
            "historical_window": {"recent_weekly_window": [], "seasonal_window": []},
            "status": "data_load_failed",
            "errors": errors,
        }
