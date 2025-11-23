# Chooses which agent handles each query
from typing import Dict, Any, List, Optional
from .utils import BaseAgent

class Orchestrator(BaseAgent):
    """
    Orchestrator agent: Routes queries to the appropriate agent based on intent.
    Inherits from BaseAgent for consistency, but acts as the central dispatcher.
    """

    def __init__(self, name: str, llm_client: Any, agents: List[BaseAgent], vector_store_path: str):
        """
        Initialize the orchestrator.
        :param name: 'orchestrator'
        :param llm_client: LLM for advanced classification if needed.
        :param agents: List of agent instances (e.g., RetrievalAgent, FormAgent).
        """
        super().__init__(name, llm_client, tools=[], memory_path=None)  # No tools or memory for orchestrator
        self.agents = {agent.name: agent for agent in agents}
        if "retrieval_agent" not in self.agents:
            from .retrieval_agent import RetrievalAgent
            self.agents["retrieval_agent"] = RetrievalAgent("retrieval_agent", llm_client, vector_store_path)

    def get_system_prompt(self) -> str:
        """System prompt for orchestrator (minimal, as it's a dispatcher)."""
        return "You are the orchestrator. Classify queries and route to agents: retrieval for info, form for leads."

    def classify_query(self, query: str) -> str:
        """
        Classify the query to select an agent.
        Simple rule-based for demo; replace with LLM for advanced intent detection.
        """
        query_lower = query.lower()
        if any(word in query_lower for word in ["name", "email", "contact", "apply", "admission"]):
            return "form_agent"
        elif any(word in query_lower for word in ["program", "course", "admission", "esilv"]):
            return "retrieval_agent"
        else:
            return "retrieval_agent"  # Default fallback

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry: Classify query, route to agent, return response.
        :param query: User input.
        :param context: Optional context.
        :return: Agent's response dict.
        """
        agent_name = self.classify_query(query)
        if agent_name not in self.agents:
            return {"answer": "No suitable agent found.", "sources": [], "action": "error"}
        
        agent = self.agents[agent_name]
        self.log_action(f"Routing to {agent_name}")
        return agent.process(query, context)