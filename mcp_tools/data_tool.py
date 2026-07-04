"""
MCP Tool: Data Access

Exposes historical energy demand data as an MCP tool so the Data Agent
can query it the same way it would query any external MCP server
(learned in Week 3: MCP, LangChain, LangGraph).

Two tools are exposed:
  - load_data_summary: quick stats about the dataset (rows, date range, avg load)
  - get_historical_window: pulls the relevant historical records needed
    to forecast a given target datetime (same hour, past N days/weeks)
"""

import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("energy-data-tool")

DEFAULT_DATA_PATH = "data/energy_demand.csv"


def _load_df(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["Datetime"])
    return df


@mcp.tool()
def load_data_summary(data_path: str = DEFAULT_DATA_PATH) -> dict:
    """
    Load the historical demand dataset and return summary statistics:
    number of rows, date range, and recent average demand.
    """
    df = _load_df(data_path)
    return {
        "rows_loaded": int(len(df)),
        "date_range": [str(df["Datetime"].min()), str(df["Datetime"].max())],
        "recent_avg_mw": round(float(df["Demand_MW"].tail(24 * 30).mean()), 1),
        "overall_avg_mw": round(float(df["Demand_MW"].mean()), 1),
    }


@mcp.tool()
def get_historical_window(target_datetime: str, data_path: str = DEFAULT_DATA_PATH) -> dict:
    """
    Given a target datetime string (e.g. "2026-01-15 18:00:00"), pull the
    historical records the Forecasting Agent needs:
      - same hour-of-day, same day-of-week, over the last 8 weeks
      - same hour-of-day, same date (+/- 3 days), over the last 2 years (seasonal)

    Returns a dict with both windows as lists of {datetime, demand_mw, is_holiday}.
    """
    df = _load_df(data_path)
    target = pd.Timestamp(target_datetime)

    # Recent same-weekday, same-hour window (captures weekly pattern)
    same_hour = df[df["Datetime"].dt.hour == target.hour]
    same_weekday = same_hour[same_hour["Datetime"].dt.dayofweek == target.dayofweek]
    recent_weekly = same_weekday[same_weekday["Datetime"] < target].tail(8)

    # Seasonal window (captures yearly pattern): same month/day (+/- 3 days), same hour, prior years
    seasonal_mask = (
        (same_hour["Datetime"].dt.month == target.month)
        & (same_hour["Datetime"].dt.day.between(target.day - 3, target.day + 3))
        & (same_hour["Datetime"].dt.year < target.year)
    )
    seasonal = same_hour[seasonal_mask]

    def _to_records(sub_df):
        return [
            {
                "datetime": str(row["Datetime"]),
                "demand_mw": float(row["Demand_MW"]),
                "is_holiday": bool(row["Is_Holiday"]),
            }
            for _, row in sub_df.iterrows()
        ]

    return {
        "target_datetime": str(target),
        "recent_weekly_window": _to_records(recent_weekly),
        "seasonal_window": _to_records(seasonal),
    }


if __name__ == "__main__":
    mcp.run()
