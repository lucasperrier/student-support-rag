from __future__ import annotations

from typing import Dict, Any, List, Optional

from .utils import BaseAgent


class Orchestrator(BaseAgent):
    """
    Orchestrator agent: Routes queries to the appropriate agent based on intent.
    Inherits from BaseAgent for consistency, but acts as the central dispatcher.
    """

    def __init__(self, name: str, llm_client: Any, agents: List[BaseAgent], vector_store_path: str):
        super().__init__(name, llm_client, tools=[], memory_path=None)
        self.agents = {agent.name: agent for agent in agents}

    def get_system_prompt(self) -> str:
        return "You are the orchestrator. Classify queries and route to agents: retrieval for info, form for leads."

    def classify_query(self, query: str) -> str:
        """
        Classify the query using Ollama for intent detection.
        Returns the agent name key from self.agents.
        """
        prompt = (
            "Classify the following user query into EXACTLY one of these categories. Respond with ONLY the category name.\n\n"
            "- 'retrieval_agent': For questions about ESILV programs, courses, admissions, rules, calendars, or general information.\n"
            "- 'form_agent': ONLY for queries explicitly involving providing or collecting personal info like names, emails, contacts, or applications.\n\n"
            f"Query: {query}\n\nCategory:"
        )

        response = self.generate_response(prompt=prompt, context="", model="llama2")
        response_clean = response.strip().lower()

        if "form_agent" in response_clean:
            return "form_agent"
        if "retrieval_agent" in response_clean:
            return "retrieval_agent"
        if "faq_agent" in self.agents:
            # Optional safe fallback if you want
            return "faq_agent"
        return "retrieval_agent"

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        agent_name = self.classify_query(query)

        if agent_name not in self.agents:
            return {
                "answer": f"Routing error: unknown agent '{agent_name}'.",
                "sources": [],
                "action": "error",
            }

        agent = self.agents[agent_name]
        return agent.process(query, context=context)