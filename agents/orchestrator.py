from __future__ import annotations

import os
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

    def route_query(self, query: str, *, faq_min_score: float = 0.6) -> str:
        """Single source of truth for routing.

        Deterministic rule-first routing:
        1) form_agent for emails / explicit personal info
        2) faq_agent when FAQAgent would confidently answer (score >= faq_min_score)
        3) retrieval_agent by default

        Modes:
        - ESILV_ORCHESTRATION_MODE=rules (default): deterministic rule-first
        - ESILV_ORCHESTRATION_MODE=llm: use an LLM to pick the route, with guardrails
        """

        mode = os.getenv("ESILV_ORCHESTRATION_MODE", "rules").strip().lower()

        # --- 1) Form (highest priority): lead capture / personal info ---
        q = (query or "").strip().lower()
        if EMAIL_RE.search(query or ""):
            return "form_agent"
        if any(k in q for k in ["my name is", "i am ", "je m'appelle", "je suis"]):
            return "form_agent"
        if any(k in q for k in ["contact", "email", "apply", "application"]):
            # Only treat as form if user intent is clearly to provide details
            if any(k in q for k in ["here is", "it's", "it is", ":", "reach me", "call me"]):
                return "form_agent"

        faq = self.agents.get("faq_agent")

        def _matches_faq() -> bool:
            """Check whether FAQAgent would answer confidently.

            Current FAQAgent implementation signals a match by returning non-empty `sources`.
            """
            if faq is None:
                return False
            try:
                faq_out = faq.process(query, context=None)
                return isinstance(faq_out, dict) and bool(faq_out.get("sources"))
            except Exception:
                return False

        # --- 2a) LLM mode: let the model choose, then validate ---
        if mode == "llm":
            # Classifier is allowed to decide between: retrieval_agent, form_agent, faq_agent.
            # Guardrails:
            # - form_agent is already handled above (email/personal info)
            # - faq_agent is only accepted if it actually matches
            try:
                routed = self.classify_query(query)
            except Exception:
                routed = "retrieval_agent"

            if routed == "faq_agent":
                # Only route to FAQ if it's truly a curated FAQ match.
                return "faq_agent" if _matches_faq() else "retrieval_agent"

            if routed == "form_agent":
                # In LLM mode, we still allow form_agent if the LLM is confident.
                return "form_agent"

            return "retrieval_agent"

        # --- 2b) Rules mode: FAQ if it looks like a strong FAQ match ---
        if _matches_faq():
            return "faq_agent"

        # --- 3) Default: retrieval ---
        return "retrieval_agent"

    def classify_query(self, query: str) -> str:
        """
        LLM-based fallback classifier (only used when heuristics are inconclusive).
        Returns one of: retrieval_agent, form_agent, faq_agent.
        """
        prompt = (
            "Classify the following user query into EXACTLY one of these categories. Respond with ONLY the category name.\n\n"
            "- 'retrieval_agent': Questions about ESILV programs, courses, admissions, rules, calendars, internships, procedures.\n"
            "- 'form_agent': ONLY if the user is providing/asking to store personal contact info (name/email/contact/application details).\n\n"
            "- 'faq_agent': ONLY if the question is a simple, frequently-asked question that can be answered from a curated FAQ.\n"
            f"Query: {query}\n\nCategory:"
        )

        response = self.generate_response(prompt=prompt, context="", model=None)
        response_clean = (response or "").strip().lower()

        if "faq_agent" in response_clean:
            return "faq_agent"
        if "form_agent" in response_clean:
            return "form_agent"
        return "retrieval_agent"

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Deterministic, rule-first routing
        routed = self.route_query(query)

        # Light debug trace: helps verify routing in logs and API response.
        self.log_action(f"Routing decision: {routed}")

        # 3) Route to selected agent; use FAQ only as a fallback when it can actually answer
        if routed == "retrieval_agent":
            agent = self.agents.get("retrieval_agent")
            if agent is None:
                return {"answer": "Routing error: retrieval agent not configured.", "sources": [], "action": "error", "routed_agent": routed}
            out = agent.process(query, context=context)
            if isinstance(out, dict):
                out.setdefault("routed_agent", routed)
            return out

        if routed == "form_agent":
            agent = self.agents.get("form_agent")
            if agent is None:
                return {"answer": "Routing error: form agent not configured.", "sources": [], "action": "error", "routed_agent": routed}
            out = agent.process(query, context=context)
            if isinstance(out, dict):
                out.setdefault("routed_agent", routed)
            return out

        if routed == "faq_agent":
            faq = self.agents.get("faq_agent")
            if faq is None:
                return {"answer": "Routing error: FAQ agent not configured.", "sources": [], "action": "error", "routed_agent": routed}

            faq_out = faq.process(query, context=context)

            # If FAQ can't answer (no sources / generic fallback), fall back to retrieval.
            if isinstance(faq_out, dict):
                faq_out.setdefault("routed_agent", routed)
                has_sources = bool(faq_out.get("sources"))
                if has_sources:
                    return faq_out

            retrieval = self.agents.get("retrieval_agent")
            if retrieval is None:
                return {"answer": "Routing error: retrieval agent not configured.", "sources": [], "action": "error", "routed_agent": "retrieval_agent"}
            out = retrieval.process(query, context=context)
            if isinstance(out, dict):
                out.setdefault("routed_agent", "retrieval_agent")
            return out

        retrieval = self.agents.get("retrieval_agent")
        if retrieval is None:
            return {"answer": "Routing error: retrieval agent not configured.", "sources": [], "action": "error", "routed_agent": "retrieval_agent"}
        out = retrieval.process(query, context=context)
        if isinstance(out, dict):
            out.setdefault("routed_agent", "retrieval_agent")
        return out