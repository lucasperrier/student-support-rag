# Agent communication helpers

# Agent communication helpers
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import ollama
import requests  

class BaseAgent(ABC):
    """
    Base class for agents in the agentic RAG pipeline.
    Provides common interface for input/output, decision-making, tools, memory, and orchestration.
    All agents inherit from this and implement specific logic.
    """

    def __init__(self, name: str, llm_client: Any, tools: List[Dict[str, Any]], memory_path: Optional[Path] = None):
        """
        Initialize the agent.
        :param name: Agent name (e.g., 'retrieval_agent').
        :param llm_client: LLM client (e.g., OpenAI or Ollama instance) for reasoning/generation.
        :param tools: List of tools (dicts with 'name', 'description', 'function').
        :param memory_path: Path to persist memory/state (e.g., JSON file).
        """
        self.name = name
        self.llm_client = llm_client
        self.tools = tools  # e.g., [{'name': 'search', 'function': lambda q: ...}]
        self.memory_path = memory_path or Path(f"data/{name}_memory.json")
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = self._load_memory()  # Dict for conversation history, state

    def _load_memory(self) -> Dict[str, Any]:
        """Load persistent memory from file."""
        if self.memory_path.exists():
            try:
                return json.loads(self.memory_path.read_text())
            except Exception:
                pass
        return {"history": [], "state": {}}

    def _save_memory(self):
        """Save memory to file."""
        self.memory_path.write_text(json.dumps(self.memory, ensure_ascii=False, indent=2))

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent's system prompt defining its role and instructions."""
        pass

    @abstractmethod
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point: Process a query and return a response.
        :param query: User input or message.
        :param context: Additional context (e.g., from orchestrator).
        :return: Dict with 'answer', 'sources', 'action', etc.
        """
        pass

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool by name with arguments."""
        for tool in self.tools:
            if tool['name'] == tool_name:
                return tool['function'](**args)
        raise ValueError(f"Tool '{tool_name}' not found.")

    def update_memory(self, key: str, value: Any):
        """Update internal memory (e.g., add to history)."""
        if key == "history":
            self.memory["history"].append(value)
        else:
            self.memory["state"][key] = value
        self._save_memory()

    def get_memory(self, key: str) -> Any:
        """Retrieve from memory."""
        if key == "history":
            return self.memory.get("history", [])
        return self.memory.get("state", {}).get(key)

    def generate_response(self, prompt: str, context: str = "", model: str = "llama2") -> str:
        """Helper: Call local Ollama LLM to generate a response."""
        full_prompt = f"{self.get_system_prompt()}\n\nContext: {context}\n\nQuery: {prompt}"
        try:
            response = ollama.generate(model=model, prompt=full_prompt, options={"timeout": 300})  # Increased timeout to 300 seconds
            return response.get("response", "No response from Ollama.")
        except Exception as e:
            return f"LLM error: {str(e)}"
        
    def log_action(self, action: str):
        """Log agent actions for monitoring."""
        print(f"[{self.name}] {action}")  # Replace with proper logging