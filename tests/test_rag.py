"""
Tests for the RAG (Retrieval-Augmented Generation) system.

Tests the retrieval agent and end-to-end RAG pipeline:
- Retrieval agent initialization
- Vector store integration
- Chunk retrieval
- Context building
- Source extraction

Note: These tests focus on retrieval functionality without requiring Ollama.
To test LLM answer generation, ensure Ollama is running with llama2 model.

Run with: pytest tests/test_rag.py -v
"""

import pytest
from pathlib import Path

from agents.retrieval_agent import RetrievalAgent


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def vector_store_path():
    """Return path to test vector store."""
    # Use the production index created by pipeline
    return "data/vector_db/index"


@pytest.fixture
def retrieval_agent(vector_store_path):
    """Create a retrieval agent instance."""
    agent = RetrievalAgent(
        name="test_retrieval_agent",
        llm_client=None,
        vector_store_path=vector_store_path,
        top_k=5,
        min_similarity=0.3
    )
    return agent


# =============================================================================
# Test Retrieval Agent Initialization
# =============================================================================

class TestRetrievalAgentInit:
    """Test retrieval agent initialization and configuration."""

    def test_agent_initialization(self, retrieval_agent):
        """Test agent initializes correctly."""
        assert retrieval_agent is not None
        assert retrieval_agent.name == "test_retrieval_agent"
        assert retrieval_agent.vector_store_path == "data/vector_db/index"
        assert retrieval_agent.top_k == 5
        assert retrieval_agent.min_similarity == 0.3

    def test_system_prompt(self, retrieval_agent):
        """Test system prompt is properly defined."""
        prompt = retrieval_agent.get_system_prompt()

        assert "ESILV" in prompt
        assert "context" in prompt.lower()
        assert "grounded" in prompt.lower() or "information" in prompt.lower()

    def test_agent_configuration(self):
        """Test agent with custom configuration."""
        agent = RetrievalAgent(
            name="custom_agent",
            llm_client=None,
            vector_store_path="data/vector_db/index",
            top_k=3,
            min_similarity=0.5
        )

        assert agent.top_k == 3
        assert agent.min_similarity == 0.5


# =============================================================================
# Test Vector Store Integration
# =============================================================================

class TestVectorStoreIntegration:
    """Test retrieval agent's vector store integration."""

    def test_load_vector_store(self, retrieval_agent):
        """Test loading vector store."""
        store = retrieval_agent._load_vector_store()

        assert store is not None
        stats = store.get_stats()
        assert stats['count'] > 0
        assert stats['dimension'] == 384
        assert stats['backend'] == 'faiss'

    def test_vector_store_caching(self, retrieval_agent):
        """Test vector store is cached after first load."""
        # First load
        store1 = retrieval_agent._load_vector_store()

        # Second load (should return cached instance)
        store2 = retrieval_agent._load_vector_store()

        assert store1 is store2  # Same object instance

    def test_vector_store_stats(self, retrieval_agent):
        """Test getting vector store statistics."""
        store = retrieval_agent._load_vector_store()
        stats = store.get_stats()

        assert 'count' in stats
        assert 'dimension' in stats
        assert 'backend' in stats
        assert stats['count'] >= 2  # At least test documents


# =============================================================================
# Test Chunk Retrieval
# =============================================================================

class TestChunkRetrieval:
    """Test chunk retrieval functionality."""

    def test_retrieve_chunks_basic(self, retrieval_agent):
        """Test basic chunk retrieval."""
        chunks = retrieval_agent._retrieve_chunks("What programs does ESILV offer?")

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check chunk structure
        for chunk in chunks:
            assert len(chunk) == 3  # (text, metadata, score)
            text, metadata, score = chunk
            assert isinstance(text, str)
            assert isinstance(metadata, dict)
            assert isinstance(score, float)

    def test_retrieve_chunks_relevance(self, retrieval_agent):
        """Test retrieved chunks are relevant."""
        chunks = retrieval_agent._retrieve_chunks("engineering programs")

        assert len(chunks) > 0

        # Top chunk should have reasonable similarity score
        text, metadata, score = chunks[0]
        assert score >= 0.3  # Above minimum threshold

    def test_retrieve_chunks_sorting(self, retrieval_agent):
        """Test chunks are sorted by relevance."""
        chunks = retrieval_agent._retrieve_chunks("ESILV programs")

        if len(chunks) > 1:
            scores = [score for _, _, score in chunks]
            # Scores should be in descending order
            assert scores == sorted(scores, reverse=True)

    def test_retrieve_chunks_different_queries(self, retrieval_agent):
        """Test retrieval with different query types."""
        queries = [
            "What programs does ESILV offer?",
            "How do I apply?",
            "admission requirements",
            "engineering school Paris"
        ]

        for query in queries:
            chunks = retrieval_agent._retrieve_chunks(query)
            # Should retrieve at least something for each query
            assert isinstance(chunks, list)

    def test_retrieve_chunks_irrelevant_query(self, retrieval_agent):
        """Test retrieval with irrelevant query."""
        chunks = retrieval_agent._retrieve_chunks("cooking recipes for pasta")

        # Should return empty list or very low relevance
        if chunks:
            # If any results, they should have low scores
            scores = [score for _, _, score in chunks]
            assert all(score < 0.5 for score in scores)

    def test_retrieve_chunks_respects_top_k(self, retrieval_agent):
        """Test that retrieval respects top_k parameter."""
        chunks = retrieval_agent._retrieve_chunks("ESILV")

        # Should return at most top_k chunks
        assert len(chunks) <= retrieval_agent.top_k

    def test_retrieve_chunks_min_similarity(self):
        """Test minimum similarity threshold filtering."""
        agent = RetrievalAgent(
            name="strict_agent",
            llm_client=None,
            vector_store_path="data/vector_db/index",
            min_similarity=0.8  # Very high threshold
        )

        chunks = agent._retrieve_chunks("random unrelated query")

        # With high threshold, might get no results
        if chunks:
            for _, _, score in chunks:
                assert score >= 0.8


# =============================================================================
# Test Context Building
# =============================================================================

class TestContextBuilding:
    """Test context building from retrieved chunks."""

    def test_build_context_basic(self, retrieval_agent):
        """Test basic context building."""
        chunks = retrieval_agent._retrieve_chunks("ESILV programs")
        context = retrieval_agent._build_context(chunks)

        assert isinstance(context, str)
        assert len(context) > 0

    def test_build_context_structure(self, retrieval_agent):
        """Test context has proper structure."""
        chunks = retrieval_agent._retrieve_chunks("ESILV")
        context = retrieval_agent._build_context(chunks)

        # Context should include chunk numbering
        assert "[1]" in context

        # Should include source information
        assert "from" in context

        # Should include relevance scores
        assert "relevance:" in context

    def test_build_context_empty_chunks(self, retrieval_agent):
        """Test context building with empty chunks."""
        context = retrieval_agent._build_context([])

        assert isinstance(context, str)
        assert "no relevant" in context.lower()

    def test_build_context_includes_text(self, retrieval_agent):
        """Test context includes chunk text."""
        chunks = retrieval_agent._retrieve_chunks("programs")
        context = retrieval_agent._build_context(chunks)

        # Context should contain text from chunks
        for text, _, _ in chunks:
            # At least part of the text should be in context
            assert text[:50] in context or text in context


# =============================================================================
# Test Source Extraction
# =============================================================================

class TestSourceExtraction:
    """Test source filename extraction."""

    def test_extract_sources_basic(self, retrieval_agent):
        """Test basic source extraction."""
        chunks = retrieval_agent._retrieve_chunks("ESILV")
        sources = retrieval_agent._extract_sources(chunks)

        assert isinstance(sources, list)
        assert len(sources) > 0
        assert all(isinstance(s, str) for s in sources)

    def test_extract_sources_unique(self, retrieval_agent):
        """Test sources are unique."""
        chunks = retrieval_agent._retrieve_chunks("ESILV programs")
        sources = retrieval_agent._extract_sources(chunks)

        # No duplicates
        assert len(sources) == len(set(sources))

    def test_extract_sources_sorted(self, retrieval_agent):
        """Test sources are sorted."""
        chunks = retrieval_agent._retrieve_chunks("ESILV")
        sources = retrieval_agent._extract_sources(chunks)

        # Should be alphabetically sorted
        assert sources == sorted(sources)

    def test_extract_sources_empty(self, retrieval_agent):
        """Test source extraction with empty chunks."""
        sources = retrieval_agent._extract_sources([])

        assert isinstance(sources, list)
        assert len(sources) == 0


# =============================================================================
# Test Process Method
# =============================================================================

class TestProcessMethod:
    """Test the main process method (without LLM)."""

    def test_process_returns_structure(self, retrieval_agent):
        """Test process method returns correct structure."""
        # Note: This will fail to generate answer without Ollama,
        # but we can test the structure
        chunks = retrieval_agent._retrieve_chunks("ESILV programs")
        sources = retrieval_agent._extract_sources(chunks)

        # Verify we can get chunks and sources
        assert len(chunks) > 0
        assert len(sources) > 0

    def test_process_source_tracking(self, retrieval_agent):
        """Test that process tracks sources correctly."""
        chunks = retrieval_agent._retrieve_chunks("programs")
        sources = retrieval_agent._extract_sources(chunks)

        # Sources should match chunks
        assert len(sources) > 0

        # Each source should appear in chunk metadata
        chunk_sources = {metadata.get('filename') for _, metadata, _ in chunks}
        assert set(sources).issubset(chunk_sources)


# =============================================================================
# Integration Tests
# =============================================================================

class TestRAGIntegration:
    """Test end-to-end RAG integration."""

    def test_end_to_end_retrieval_flow(self, retrieval_agent):
        """Test complete retrieval flow."""
        query = "What programs does ESILV offer?"

        # Step 1: Retrieve chunks
        chunks = retrieval_agent._retrieve_chunks(query)
        assert len(chunks) > 0

        # Step 2: Build context
        context = retrieval_agent._build_context(chunks)
        assert len(context) > 0

        # Step 3: Extract sources
        sources = retrieval_agent._extract_sources(chunks)
        assert len(sources) > 0

        # Verify context contains chunk information
        for text, metadata, score in chunks[:2]:  # Check first 2
            assert metadata['filename'] in sources

    def test_multiple_queries_consistency(self, retrieval_agent):
        """Test retrieval consistency across multiple queries."""
        query = "ESILV engineering programs"

        # Run same query multiple times
        results1 = retrieval_agent._retrieve_chunks(query)
        results2 = retrieval_agent._retrieve_chunks(query)

        # Should return same results
        assert len(results1) == len(results2)

        # Scores should be identical
        scores1 = [s for _, _, s in results1]
        scores2 = [s for _, _, s in results2]
        assert scores1 == scores2

    def test_different_queries_different_results(self, retrieval_agent):
        """Test different queries return different results."""
        query1 = "programs"
        query2 = "admissions"

        chunks1 = retrieval_agent._retrieve_chunks(query1)
        chunks2 = retrieval_agent._retrieve_chunks(query2)

        # Top results should likely be different
        if chunks1 and chunks2:
            text1, _, _ = chunks1[0]
            text2, _, _ = chunks2[0]

            # Texts should be different (or at least scores different)
            scores1 = [s for _, _, s in chunks1]
            scores2 = [s for _, _, s in chunks2]
            assert text1 != text2 or scores1 != scores2


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
