from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

from .utils import BaseAgent

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


class Orchestrator(BaseAgent):
    """
    Orchestrator agent: Routes queries to the appropriate agent based on intent.
    Heuristic-first routing:
      - form_agent when the user provides/asks to provide personal contact info (email, name)
      - faq_agent only when there's a strong FAQ match (agent decides)
      - retrieval_agent by default for ESILV info questions
    """

    def __init__(self, name: str, llm_client: Any, agents: List[BaseAgent], vector_store_path: str):
        super().__init__(name, llm_client, tools=[], memory_path=None)
        self.agents = {agent.name: agent for agent in agents}

    def get_system_prompt(self) -> str:
        return "You are the orchestrator. Route queries to retrieval, FAQ, or form agents."

    def _route_by_heuristics(self, query: str) -> Optional[str]:
        q = (query or "").strip().lower()

        # Lead capture / personal info
        if EMAIL_RE.search(query):
            return "form_agent"
        if any(k in q for k in ["my name is", "i am ", "je m'appelle", "je suis"]):
            return "form_agent"
        if any(k in q for k in ["contact", "email", "apply", "application"]):
            # Only treat as form if user intent is clearly to provide details
            if any(k in q for k in ["here is", "it's", "it is", ":", "reach me", "call me"]):
                return "form_agent"

        return None

    def classify_query(self, query: str) -> str:
        """
        LLM-based fallback classifier (only used when heuristics are inconclusive).
        Returns one of: retrieval_agent, form_agent.
        """
        prompt = (
            "Classify the following user query into EXACTLY one of these categories. Respond with ONLY the category name.\n\n"
            "- 'retrieval_agent': Questions about ESILV programs, courses, admissions, rules, calendars, internships, procedures.\n"
            "- 'form_agent': ONLY if the user is providing/asking to store personal contact info (name/email/contact/application details).\n\n"
            f"Query: {query}\n\nCategory:"
        )

        response = self.generate_response(prompt=prompt, context="", model=None)
        response_clean = (response or "").strip().lower()

        if "form_agent" in response_clean:
            return "form_agent"
        return "retrieval_agent"

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # 1) Heuristics first
        routed = self._route_by_heuristics(query)

        # 2) If not decided, prefer retrieval (and only use LLM classifier optionally)
        if routed is None:
            # If Ollama is down, this keeps the system functional.
            try:
                routed = self.classify_query(query)
            except Exception:
                routed = "retrieval_agent"

        # 3) Route to selected agent; use FAQ only as a fallback when it can actually answer
        if routed == "retrieval_agent":
            agent = self.agents.get("retrieval_agent")
            if agent is None:
                return {"answer": "Routing error: retrieval agent not configured.", "sources": [], "action": "error"}
            return agent.process(query, context=context)

        if routed == "form_agent":
            agent = self.agents.get("form_agent")
            if agent is None:
                return {"answer": "Routing error: form agent not configured.", "sources": [], "action": "error"}
            return agent.process(query, context=context)

        # Optional FAQ fallback: try it, but if it returns the generic "don't have that" message,
        # fall back to retrieval.
        faq = self.agents.get("faq_agent")
        if faq is not None:
            faq_out = faq.process(query, context=context)
            if isinstance(faq_out, dict) and "sources" in faq_out and faq_out.get("sources"):
                return faq_out

        retrieval = self.agents.get("retrieval_agent")
        if retrieval is None:
            return {"answer": "Routing error: retrieval agent not configured.", "sources": [], "action": "error"}
        return retrieval.process(query, context=context)