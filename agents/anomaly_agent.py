"""
Anomaly / Context Agent

Role: an LLM-reasoning agent (Week 1 prompting skill) that examines the
target datetime and the forecast produced by the Forecasting Agent, and
flags any context the grid operator should know about before acting on
the raw number -- holidays, weekends, seasonal extremes, or a forecast
that deviates sharply from the historical average.

Unlike the Data/Forecasting agents (deterministic, tool-executing),
this agent's job genuinely benefits from LLM reasoning: judging whether
a date is "notable" and explaining *why* in plain language isn't a
simple lookup, it's contextual judgment.

Writes to shared state: anomaly_flags, anomaly_notes, status, errors
"""

import json
import re
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from state.graph_state import GraphState

# Known fixed-date holidays used when generating the synthetic dataset.
# Real deployment could swap this for a holiday-calendar API/tool.
KNOWN_HOLIDAYS = {
    "01-01": "New Year's Day",
    "07-04": "Independence Day",
    "12-25": "Christmas Day",
}

SYSTEM_PROMPT = """You are a grid demand anomaly-detection assistant for an energy utility.
Given a target date/time, a demand forecast, and recent historical averages,
identify any contextual factors a grid operator should be aware of.

Consider:
- Is the date a public holiday or a weekend? (demand is typically lower)
- Is the forecast far above or below the recent/overall historical average? (>10% deviation is notable)
- Is this a season where extreme demand is typical (summer cooling, winter heating)?

Respond with ONLY a valid JSON object, no other text, in this exact format:
{
  "anomaly_flags": ["flag_one", "flag_two"],
  "anomaly_notes": "One or two sentence plain-English explanation of the context, written for a grid operator."
}
If nothing notable is found, return an empty anomaly_flags list and a brief note saying the forecast looks routine.
"""


def _rule_based_flags(target_dt: datetime, forecast_mw: float, recent_avg: float, overall_avg: float) -> list:
    """Deterministic pre-check, passed to the LLM as supporting context
    rather than replacing its reasoning -- keeps the LLM grounded."""
    flags = []
    date_key = target_dt.strftime("%m-%d")
    if date_key in KNOWN_HOLIDAYS:
        flags.append(f"holiday:{KNOWN_HOLIDAYS[date_key]}")
    if target_dt.weekday() >= 5:
        flags.append("weekend")
    if overall_avg and abs(forecast_mw - overall_avg) / overall_avg > 0.10:
        direction = "above" if forecast_mw > overall_avg else "below"
        flags.append(f"deviation_{direction}_average")
    if target_dt.month in (6, 7, 8):
        flags.append("summer_season")
    elif target_dt.month in (12, 1, 2):
        flags.append("winter_season")
    return flags


def anomaly_agent(state: GraphState) -> GraphState:
    errors = list(state.get("errors", []))
    target_dt = datetime.fromisoformat(state["target_datetime"])
    forecast_mw = state.get("forecast_mw")
    summary = state.get("raw_data_summary", {})
    recent_avg = summary.get("recent_avg_mw", forecast_mw)
    overall_avg = summary.get("overall_avg_mw", forecast_mw)

    precheck_flags = _rule_based_flags(target_dt, forecast_mw, recent_avg, overall_avg)

    user_prompt = f"""Target datetime: {state['target_datetime']}
Forecasted demand: {forecast_mw} MW
Recent 30-day average demand: {recent_avg} MW
Overall historical average demand: {overall_avg} MW
Rule-based pre-check flags found: {precheck_flags if precheck_flags else "none"}

Analyze this and return the JSON object as instructed."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            max_output_tokens=500,
            temperature=0,
            max_retries=1,
            thinking_budget=0,
        )
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        raw_text = response.content.strip()

        # Defensive parsing: Gemini/Claude sometimes wrap JSON in markdown
        # fences or add stray preamble/trailing text. Extract the outermost
        # {...} block rather than assuming the whole string is clean JSON.
        raw_text = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in LLM response: {raw_text[:200]}")
        parsed = json.loads(match.group(0))

        return {
            "anomaly_flags": parsed.get("anomaly_flags", precheck_flags),
            "anomaly_notes": parsed.get("anomaly_notes", ""),
            "status": "anomaly_check_done",
            "errors": errors,
        }

    except Exception as e:
        # Failure handling: fall back to the deterministic rule-based flags
        # with a generic note, so the pipeline still produces something useful.
        errors.append(f"anomaly_agent: LLM call/parsing failed ({str(e)}), used rule-based fallback.")
        fallback_note = (
            f"Automated context check unavailable; rule-based flags: {precheck_flags}"
            if precheck_flags else "No notable context flags detected (rule-based fallback)."
        )
        return {
            "anomaly_flags": precheck_flags,
            "anomaly_notes": fallback_note,
            "status": "anomaly_check_done_via_fallback",
            "errors": errors,
        }
