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

    def get_system_prompt(self) -> str:
        """System prompt for orchestrator (minimal, as it's a dispatcher)."""
        return "You are the orchestrator. Classify queries and route to agents: retrieval for info, form for leads."

    def classify_query(self, query: str) -> str:
        """
        Classify the query using Ollama for intent detection.
        """
        prompt = (
            "Classify the following user query into EXACTLY one of these categories. Respond with ONLY the category name.\n\n"
            "- 'retrieval_agent': For questions about ESILV programs, courses, admissions, history, or general information (e.g., 'What is ESILV?', 'Tell me about programs').\n"
            "- 'form_agent': ONLY for queries explicitly involving providing or collecting personal info like names, emails, contacts, or applications (e.g., 'My name is John', 'I want to apply').\n\n"
            f"Query: {query}\n\nCategory:"
        )
        response = self.generate_response(prompt=prompt, context="", model="llama2")
        print(f"DEBUG: Ollama classification response: '{response}'")  # Add this line
        # Clean and parse the response
        response_clean = response.strip().lower()
        if "retrieval_agent" in response_clean:
            return "retrieval_agent"
        elif "form_agent" in response_clean:
            return "form_agent"
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