"""
Report Agent

Role: the final agent in the pipeline. Synthesizes the forecast
(Forecasting Agent) and context flags (Anomaly Agent) into a clear,
plain-English grid management report -- grounded in retrieved grid
management guidelines via RAG (Week 2 skill), rather than generating
generic, ungrounded LLM advice.

RAG flow:
  1. Build a retrieval query from the forecast + anomaly flags
  2. Retrieve top-k relevant guideline documents from the knowledge base
  3. Pass retrieved guidelines as grounding context to the LLM prompt
  4. LLM synthesizes the final report, citing which guideline informed
     each recommendation

Writes to shared state: final_report, status, errors
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from rag.retriever import KnowledgeBaseRetriever
from state.graph_state import GraphState

SYSTEM_PROMPT = """You are a grid management reporting assistant. You write clear,
plain-English reports for grid operators based on a demand forecast, contextual
anomaly flags, and relevant internal guidelines.

Your report must:
- State the forecast value and what date/time it's for
- Explain any notable context (from the anomaly flags/notes) in plain language
- Give a concrete recommended action, grounded in the provided guidelines
- Reference which guideline(s) informed the recommendation, by name
- Be no more than 150 words, written for a busy operator, not a technical audience

If no guidelines are relevant or no anomalies were found, state that the
forecast is routine and no special action is needed.
"""

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeBaseRetriever()
    return _retriever


def report_agent(state: GraphState) -> GraphState:
    errors = list(state.get("errors", []))
    target_dt = state["target_datetime"]
    forecast_mw = state.get("forecast_mw")
    forecast_method = state.get("forecast_method_used", "unknown")
    anomaly_flags = state.get("anomaly_flags", [])
    anomaly_notes = state.get("anomaly_notes", "")

    # Build retrieval query from the forecast context
    query_terms = " ".join(anomaly_flags) if anomaly_flags else "routine forecast normal demand"
    query = f"{query_terms} forecast {forecast_mw} MW"

    try:
        retriever = _get_retriever()
        retrieved_docs = retriever.retrieve(query, k=3)
    except Exception as e:
        errors.append(f"report_agent: retrieval failed ({str(e)}), proceeding without grounding.")
        retrieved_docs = []

    guidelines_context = "\n\n".join(
        f"[{d['source']}]\n{d['text']}" for d in retrieved_docs
    ) or "No specific guidelines retrieved."

    user_prompt = f"""Target datetime: {target_dt}
Forecasted demand: {forecast_mw} MW (method: {forecast_method})
Anomaly flags: {anomaly_flags if anomaly_flags else "none"}
Anomaly context notes: {anomaly_notes if anomaly_notes else "none"}

Relevant internal guidelines (retrieved via RAG):
{guidelines_context}

Write the grid management report now."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            max_output_tokens=600,
            temperature=0.3,
            max_retries=1,
            thinking_budget=0,
        )
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        report_text = response.content.strip()

        return {
            "final_report": report_text,
            "status": "complete",
            "errors": errors,
        }

    except Exception as e:
        # Failure handling: if the LLM call fails, build a basic templated
        # report directly from state so the pipeline still ends with a
        # usable output instead of nothing at all.
        errors.append(f"report_agent: LLM call failed ({str(e)}), used templated fallback report.")
        flags_text = ", ".join(anomaly_flags) if anomaly_flags else "none"
        fallback_report = (
            f"[Fallback Report - LLM unavailable]\n"
            f"Forecast for {target_dt}: {forecast_mw} MW (method: {forecast_method}).\n"
            f"Anomaly flags: {flags_text}.\n"
            f"Notes: {anomaly_notes or 'No additional context available.'}\n"
            f"Please consult the relevant guideline documents manually: "
            f"{', '.join(d['source'] for d in retrieved_docs) if retrieved_docs else 'none retrieved'}."
        )
        return {
            "final_report": fallback_report,
            "status": "complete_via_fallback",
            "errors": errors,
        }
