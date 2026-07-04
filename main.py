"""
Energy Demand Forecasting Multi-Agent System -- CLI Entry Point

Usage:
    python main.py                              # forecasts for 24 hours from now
    python main.py --datetime "2026-01-15 18:00:00"   # forecasts for a specific date/time

Requires a GOOGLE_API_KEY (Gemini) set in a .env file (see .env.example)
for the Anomaly Agent and Report Agent's LLM reasoning. If missing/invalid,
the pipeline still runs to completion using its built-in fallback paths
(see README for details on the failure-handling design).
"""

import argparse
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from graph import build_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Run the energy demand forecasting multi-agent pipeline.")
    parser.add_argument(
        "--datetime",
        type=str,
        default=None,
        help='Target datetime to forecast for, format: "YYYY-MM-DD HH:MM:SS". Defaults to 24 hours from now.',
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/energy_demand.csv",
        help="Path to the historical demand CSV.",
    )
    return parser.parse_args()


def print_banner(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    args = parse_args()
    target_dt = args.datetime or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

    print_banner("ENERGY DEMAND FORECASTING -- MULTI-AGENT PIPELINE")
    print(f"Target datetime : {target_dt}")
    print(f"Data source     : {args.data_path}")
    print("Orchestration   : Pipeline pattern (Data -> Forecast -> Anomaly -> Report)")

    app = build_graph()
    result = app.invoke({
        "target_datetime": target_dt,
        "data_path": args.data_path,
        "errors": [],
    })

    print_banner("PIPELINE RESULTS")
    print(f"Status              : {result.get('status')}")
    print(f"Forecasted demand   : {result.get('forecast_mw')} MW")
    print(f"Forecast method     : {result.get('forecast_method_used')}")
    print(f"Anomaly flags       : {result.get('anomaly_flags') or 'none'}")

    print_banner("FINAL REPORT")
    print(result.get("final_report"))

    if result.get("errors"):
        print_banner("PIPELINE LOG (retries / fallbacks triggered)")
        for e in result["errors"]:
            print(f" - {e}")

    print()


if __name__ == "__main__":
    main()
