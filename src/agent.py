"""LangGraph triage -> report -> alert. LLM output never controls policy."""
import logging
import os
from typing import TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from .schemas import Alert, Incident

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    incident: Incident
    severity: str
    action: str
    report: str
    report_source: str
    alert: Alert


class IncidentAgent:
    def __init__(self) -> None:
        graph = StateGraph(AgentState)
        graph.add_node("triage", self.triage)
        graph.add_node("report", self.report)
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "triage")
        graph.add_edge("triage", "report")
        graph.add_edge("report", "finalize")
        graph.add_edge("finalize", END)
        self.graph = graph.compile()

    @staticmethod
    def triage(state: AgentState) -> dict:
        urgent = len(state["incident"].detections) >= 3
        return {"severity": "critical" if urgent else "warning",
                "action": "request_urgent_review" if urgent else "notify_operator"}

    @staticmethod
    async def report(state: AgentState) -> dict:
        incident = state["incident"]
        report = (f"Restricted-zone incident: {len(incident.detections)} person detection(s) "
                  f"on frame {incident.frame_id}. Backend={incident.backend}. "
                  f"Action: {state['action']}. Operator verification required.")
        source = "mock"
        if os.getenv("LLM_MODE", "mock") == "openai":
            try:
                base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.post(base + "/chat/completions", headers={
                        "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY", "local")}, json={
                        "model": os.getenv("LLM_MODEL", "local-model"),
                        "max_tokens": 300,
                        "messages": [{"role": "system", "content":
                            "Write a concise incident report from detection metadata. Do not invent "
                            "identities, intent, or visual details. This may be a simulation. "
                            "Recommend human review; no enforcement decisions."},
                            {"role": "user", "content": incident.model_dump_json()}]})
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Empty report")
                    report, source = content[:8000], "llm"
            except Exception as exc:
                logger.warning("LLM failed (%s); using mock report", type(exc).__name__)
                source = "mock_fallback"
        return {"report": report, "report_source": source}

    @staticmethod
    def finalize(state: AgentState) -> dict:
        alert = Alert(incident=state["incident"], severity=state["severity"],
                      action=state["action"], report=state["report"], report_source=state["report_source"])
        logger.info("AGENT %s: %s", alert.severity, alert.report)
        return {"alert": alert}

    async def run(self, incident: Incident) -> Alert:
        result = await self.graph.ainvoke({"incident": incident})
        return result["alert"]
