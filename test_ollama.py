from agents.utils import BaseAgent
from typing import Dict, Any, List, Optional
from pathlib import Path
import ollama  # Add this import

class OllamaAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return "You are a helpful AI assistant."

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Assuming llama2 is pulled and Ollama is running
        response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': query}])
        return {"answer": response['message']['content'], "sources": [], "action": "ollama_response"}

# Test the method
agent = OllamaAgent(name="test", llm_client=None, tools=[])
response = agent.generate_response(prompt="Hello, what is AI?", context="Context about AI.")
print("Response:", response)