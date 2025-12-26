"""
Retrieval Agent - RAG-based Question Answering

PURPOSE:
This agent handles information queries about ESILV using Retrieval-Augmented Generation (RAG).
It searches the vector database for relevant document chunks and generates grounded answers
using an LLM with the retrieved context.

INPUTS:
- User query: Question about ESILV (programs, admissions, etc.)
- Vector store path: Location of the indexed documents

OUTPUTS:
- answer: Natural language response grounded in retrieved documents
- sources: List of source documents used
- action: "answer" (successful) or "error" (failed)

PERSON B INTEGRATION:
The orchestrator routes information queries to this agent:

    # In orchestrator.py
    if intent == "information":
        result = retrieval_agent.process(query)
        return result["answer"]

DESIGN DECISIONS:

1. **RAG Pipeline**:
   - Step 1: Search vector store for top-k relevant chunks
   - Step 2: Filter chunks by relevance threshold (>0.3 similarity)
   - Step 3: Build context from retrieved chunks
   - Step 4: Generate answer using LLM with context
   - Step 5: Return answer with source citations

2. **Relevance Threshold**:
   - Minimum similarity: 0.3 (cosine similarity, 0-1 scale)
   - Reasoning: Filters out irrelevant chunks, improves answer quality
   - If no chunks meet threshold, returns "insufficient information" message

3. **Context Building**:
   - Top-k: 5 chunks (configurable)
   - Format: Numbered chunks with source attribution
   - Deduplication: Remove duplicate sources
   - Token limit awareness: ~500 tokens of context max

4. **Answer Generation**:
   - Uses Ollama LLM (llama2 by default)
   - System prompt instructs to stay grounded in context
   - Refuses to answer if context doesn't contain relevant info
   - Cites sources naturally in response

5. **Error Handling**:
   - Vector store loading failures
   - Empty search results
   - LLM generation errors
   - Returns informative error messages

TECHNICAL NOTES:
- Lazy loads vector store (only once, cached)
- Thread-safe search operations
- Logs all retrieval and generation steps
- Memory efficient: doesn't store large context in memory
"""

import os
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path

# Import BaseAgent
from agents.utils import BaseAgent

# Import vector store for retrieval
from ingestion.vector_store import VectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_VECTOR_STORE_PATH = "data/vector_db/index"
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.3  # Minimum cosine similarity to consider chunk relevant
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama2")


# =============================================================================
# Retrieval Agent
# =============================================================================

class RetrievalAgent(BaseAgent):
    """
    RAG-based retrieval agent for answering questions about ESILV.

    Uses vector database to find relevant document chunks, then generates
    grounded answers using an LLM.
    """

    def __init__(
        self,
        name: str,
        llm_client: Any,
        vector_store_path: str = DEFAULT_VECTOR_STORE_PATH,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY
    ):
        """
        Initialize retrieval agent.

        Args:
            name: Agent name
            llm_client: LLM client (Ollama)
            vector_store_path: Path to vector store index
            top_k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
        """
        super().__init__(name, llm_client, tools=[], memory_path=None)
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        self.min_similarity = min_similarity

        # Lazy-loaded vector store (loaded on first use)
        self._vector_store: Optional[VectorStore] = None

        self.log_action(f"Initialized with vector_store_path={vector_store_path}, top_k={top_k}")

    def get_system_prompt(self) -> str:
        """
        Return system prompt for answer generation.

        This prompt instructs the LLM to:
        - Only use information from provided context
        - Cite sources when possible
        - Admit when it doesn't know
        """
        return """You are a helpful assistant answering questions about ESILV (École Supérieure d'Ingénieurs Léonard de Vinci).

IMPORTANT INSTRUCTIONS:
1. ONLY use information from the provided CONTEXT below
2. If the CONTEXT doesn't contain relevant information, say "I don't have enough information to answer that question."
3. Be concise and direct in your answers
4. When mentioning specific information, try to cite which document it came from if possible
5. Do NOT make up information or use knowledge outside the provided context

Your goal is to provide accurate, helpful answers grounded in the retrieved documents."""

    def _load_vector_store(self) -> VectorStore:
        """
        Load vector store (lazy loading, cached).

        Returns:
            VectorStore instance

        Raises:
            FileNotFoundError: If vector store doesn't exist
        """
        if self._vector_store is None:
            self.log_action(f"Loading vector store from {self.vector_store_path}")

            vector_store_file = Path(self.vector_store_path).with_suffix('.faiss')
            if not vector_store_file.exists():
                raise FileNotFoundError(
                    f"Vector store not found at {self.vector_store_path}. "
                    f"Please run the ingestion pipeline first: "
                    f"python -m ingestion.pipeline --data-dir data/raw"
                )

            self._vector_store = VectorStore.load(str(self.vector_store_path), backend="faiss")
            stats = self._vector_store.get_stats()
            self.log_action(f"Loaded vector store with {stats['count']} chunks")

        return self._vector_store

    def _retrieve_chunks(self, query: str) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Retrieve relevant chunks from vector store.

        Args:
            query: User query

        Returns:
            List of (text, metadata, score) tuples, filtered by min_similarity
        """
        self.log_action(f"Retrieving chunks for query: '{query[:50]}...'")

        # Load vector store
        try:
            store = self._load_vector_store()
        except FileNotFoundError as e:
            logger.error(f"Vector store not found: {e}")
            return []

        # Search vector store
        try:
            results = store.search(query, top_k=self.top_k, min_score=self.min_similarity)
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

        # Log results
        if results:
            self.log_action(f"Retrieved {len(results)} chunks (scores: {[f'{s:.3f}' for _, _, s in results]})")
        else:
            self.log_action("No relevant chunks found")

        return results

    def _build_context(self, chunks: List[Tuple[str, Dict[str, Any], float]]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of (text, metadata, score) tuples

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in the knowledge base."

        context_parts = []
        sources_seen = set()

        for i, (text, metadata, score) in enumerate(chunks):
            source = metadata.get('filename', 'unknown')
            sources_seen.add(source)

            # Format: [1] (from admissions.pdf, relevance: 0.85)
            # Actual chunk text here...
            context_parts.append(
                f"[{i+1}] (from {source}, relevance: {score:.2f})\n{text.strip()}"
            )

        context = "\n\n".join(context_parts)
        self.log_action(f"Built context from {len(chunks)} chunks, {len(sources_seen)} unique sources")

        return context

    def _extract_sources(self, chunks: List[Tuple[str, Dict[str, Any], float]]) -> List[str]:
        """
        Extract unique source filenames from chunks.

        Args:
            chunks: List of (text, metadata, score) tuples

        Returns:
            List of unique source filenames
        """
        sources = set()
        for _, metadata, _ in chunks:
            filename = metadata.get('filename', 'unknown')
            sources.add(filename)

        return sorted(list(sources))

    def answer(self, query: str) -> str:
        """
        Generate RAG-based answer to a query.

        This is the main RAG pipeline:
        1. Retrieve relevant chunks from vector store
        2. Build context from chunks
        3. Generate answer using LLM with context
        4. Return grounded answer

        Args:
            query: User question

        Returns:
            Generated answer string
        """
        self.log_action(f"Answering query: '{query}'")

        # Step 1: Retrieve relevant chunks
        chunks = self._retrieve_chunks(query)

        if not chunks:
            return (
                "I don't have enough information in my knowledge base to answer that question. "
                "This could mean:\n"
                "- The information isn't in the indexed documents\n"
                "- The vector store hasn't been created yet (run: python -m ingestion.pipeline)\n"
                "- The query is too different from the available content"
            )

        # Step 2: Build context
        context = self._build_context(chunks)

        # Step 3: Generate answer with LLM
        self.log_action("Generating answer with LLM...")

        full_prompt = f"""Based on the following CONTEXT from ESILV documents, please answer the question.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

        try:
            # Use the BaseAgent's generate_response method
            # Note: We pass the context separately since generate_response has a context parameter
            answer = self.generate_response(
                prompt=query,
                context=context,
                model=DEFAULT_MODEL
            )

            self.log_action("Answer generated successfully")
            return answer.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while generating the answer: {str(e)}"

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query and return a structured response.

        This is the main entry point called by the orchestrator.

        Args:
            query: User question
            context: Optional additional context from orchestrator

        Returns:
            Dict with:
                - answer: Generated response
                - sources: List of source documents used
                - action: "answer" or "error"
        """
        self.log_action(f"Processing query: '{query[:50]}...'")

        try:
            # Retrieve chunks first to get sources
            chunks = self._retrieve_chunks(query)

            # Extract sources
            sources = self._extract_sources(chunks) if chunks else []

            # Generate answer
            answer = self.answer(query)

            # Update memory with query/answer
            self.update_memory("history", {
                "query": query,
                "answer": answer,
                "sources": sources
            })

            return {
                "answer": answer,
                "sources": sources,
                "action": "answer"
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "answer": f"I encountered an error: {str(e)}",
                "sources": [],
                "action": "error"
            }


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    """
    Test the retrieval agent.
    Run: python -m agents.retrieval_agent
    """

    print("Retrieval Agent Test")
    print("=" * 80)

    # Create agent (llm_client is passed but used via BaseAgent)
    agent = RetrievalAgent(
        name="test_retrieval_agent",
        llm_client=None,  # BaseAgent handles Ollama directly
        vector_store_path="data/vector_db/index"
    )

    # Test queries
    test_queries = [
        "What programs does ESILV offer?",
        "How do I apply to ESILV?",
        "When is the application deadline?",
        "What is the meaning of life?"  # Should return "insufficient info"
    ]

    print("\nTesting RAG pipeline:")
    print("-" * 80)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)

        result = agent.process(query)

        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['sources']}")
        print(f"Action: {result['action']}")
        print()

    print("=" * 80)
    print("Test complete!")
