"""
LangGraph Pipeline Orchestration

Orchestration pattern: PIPELINE (sequential, strictly ordered handoff)
Rationale: this task cannot be reordered -- you can't check anomalies
before a forecast exists, and can't write a report before both the
forecast and anomaly context exist. Pipeline is the correct pattern
per the Week 4 theory ("Pipeline if your task is strictly ordered").

State strategy: SHARED GLOBAL STATE (see state/graph_state.py). Each
node only writes to its own designated keys, so agents never overwrite
each other's work.

Flow:
    Data Agent -> Forecasting Agent -> Anomaly Agent -> Report Agent -> END
"""

from langgraph.graph import StateGraph, END

from state.graph_state import GraphState
from agents.data_agent import data_agent
from agents.forecasting_agent import forecasting_agent
from agents.anomaly_agent import anomaly_agent
from agents.report_agent import report_agent


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("data_agent", data_agent)
    graph.add_node("forecasting_agent", forecasting_agent)
    graph.add_node("anomaly_agent", anomaly_agent)
    graph.add_node("report_agent", report_agent)

    graph.set_entry_point("data_agent")
    graph.add_edge("data_agent", "forecasting_agent")
    graph.add_edge("forecasting_agent", "anomaly_agent")
    graph.add_edge("anomaly_agent", "report_agent")
    graph.add_edge("report_agent", END)

    return graph.compile()


if __name__ == "__main__":
    # quick manual smoke test
    app = build_graph()
    result = app.invoke({
        "target_datetime": "2025-07-04 14:00:00",
        "data_path": "data/energy_demand.csv",
        "errors": [],
    })
    print("Final status:", result["status"])
    print("Forecast:", result["forecast_mw"], "MW")
    print("Anomaly flags:", result["anomaly_flags"])
    print()
    print("Final report:")
    print(result["final_report"])
