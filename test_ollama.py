# filepath: /home/lucas-perrier/Documents/ESILV/S9/gen_ai/esilv_smart_assistant/test_ollama.py
from agents.utils import BaseAgent
from typing import Dict, Any, List, Optional
from pathlib import Path

class DummyAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return "You are a test agent."

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"answer": "Not implemented", "sources": [], "action": "test"}

# Test the method
agent = DummyAgent(name="test", llm_client=None, tools=[])
response = agent.generate_response(prompt="Hello, what is AI?", context="Context about AI.")
print("Response:", response)