# Vector DB search agent

from typing import Dict, Any, Optional
from .utils import BaseAgent

class RetrievalAgent(BaseAgent):
    def __init__(self, name: str, llm_client: Any, vector_store_path: str):
        super().__init__(name, llm_client, tools=[], memory_path=None)
        self.vector_store_path = vector_store_path  # Path to vector DB; Person A configures this.

    def get_system_prompt(self) -> str:
        return "You are a retrieval agent. Use the vector store to find relevant documents and generate grounded answers."

    def answer(self, query: str) -> str:
        """
        Stub: Person A must implement this.
        Generates a RAG-based answer using vector_store.search.
        :param query: User query.
        :return: Grounded answer string.
        """
        raise NotImplementedError("Person A: Implement RAG answer generation. Call vector_store.search(query, top_k=5), then use LLM to generate answer from results.")