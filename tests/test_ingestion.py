"""
Tests for the ingestion pipeline.

Tests all components of the document ingestion system:
- Document loading (PDF, HTML, TXT)
- Text cleaning
- Chunking
- Embedding generation
- Vector store operations
- End-to-end pipeline

Run with: pytest tests/test_ingestion.py -v
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from ingestion.loader import load_document, load_documents_from_directory, Document
from ingestion.text_cleaning import clean_text, clean_document, clean_documents
from ingestion.chunker import (
    chunk_text, chunk_document, chunk_documents,
    Chunk, get_chunk_stats, RecursiveCharacterTextSplitter
)
from ingestion.embedder import (
    EmbeddingGenerator, embed_chunks, embed_texts, embed_query,
    get_embedding_stats, cosine_similarity, batch_cosine_similarity
)
from ingestion.vector_store import VectorStore, FAISSVectorStore
from ingestion.pipeline import run_pipeline, PipelineConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("ESILV is a leading French engineering school in Paris.")
    return file_path


@pytest.fixture
def sample_html_file(temp_dir):
    """Create a sample HTML file."""
    file_path = temp_dir / "test.html"
    html_content = """
    <html>
        <head><title>ESILV Info</title></head>
        <body>
            <h1>ESILV Programs</h1>
            <p>We offer programs in computer science and engineering.</p>
        </body>
    </html>
    """
    file_path.write_text(html_content)
    return file_path


@pytest.fixture
def sample_documents(temp_dir):
    """Create multiple sample documents."""
    # Text file
    (temp_dir / "doc1.txt").write_text(
        "ESILV is located in Paris La Defense. "
        "The school offers engineering programs.",
        encoding='utf-8'
    )

    # HTML file
    (temp_dir / "doc2.html").write_text(
        "<html><body><p>Applications open in January.</p></body></html>",
        encoding='utf-8'
    )

    return temp_dir


# =============================================================================
# Test Loader
# =============================================================================

class TestLoader:
    """Test document loading functionality."""

    def test_load_text_file(self, sample_text_file):
        """Test loading a text file."""
        doc = load_document(str(sample_text_file))

        assert doc is not None
        assert isinstance(doc, Document)
        assert "ESILV" in doc.text
        assert doc.metadata["file_type"] == "text"
        assert doc.metadata["filename"] == "test.txt"

    def test_load_html_file(self, sample_html_file):
        """Test loading an HTML file."""
        doc = load_document(str(sample_html_file))

        assert doc is not None
        assert "ESILV Programs" in doc.text
        assert doc.metadata["file_type"] == "html"

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        doc = load_document("nonexistent.txt")
        assert doc is None

    def test_load_directory(self, sample_documents):
        """Test loading all documents from a directory."""
        docs = load_documents_from_directory(str(sample_documents))

        assert len(docs) == 2
        assert all(isinstance(doc, Document) for doc in docs)

        # Check both file types are loaded
        file_types = {doc.metadata["file_type"] for doc in docs}
        assert "text" in file_types
        assert "html" in file_types


# =============================================================================
# Test Text Cleaning
# =============================================================================

class TestTextCleaning:
    """Test text cleaning functionality."""

    def test_clean_text_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    world\n\n\nTest"
        cleaned = clean_text(text)

        assert "    " not in cleaned  # Multiple spaces removed
        assert "\n\n\n" not in cleaned  # Multiple newlines reduced

    def test_clean_text_unicode(self):
        """Test Unicode normalization."""
        text = "caf�"  # Contains accented character
        cleaned = clean_text(text)

        assert cleaned is not None
        assert len(cleaned) > 0

    def test_clean_document(self, sample_text_file):
        """Test cleaning a Document object."""
        doc = load_document(str(sample_text_file))
        cleaned = clean_document(doc)

        assert isinstance(cleaned, str)
        assert "ESILV" in cleaned

    def test_clean_documents_batch(self, sample_documents):
        """Test batch cleaning of documents."""
        docs = load_documents_from_directory(str(sample_documents))
        cleaned = clean_documents(docs)

        assert len(cleaned) == len(docs)
        assert all(isinstance(text, str) for text in cleaned)


# =============================================================================
# Test Chunker
# =============================================================================

class TestChunker:
    """Test text chunking functionality."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "A" * 1000  # Long text
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        # Chunks may slightly exceed target due to overlap
        assert all(chunk.char_count <= 250 for chunk in chunks)

    def test_chunk_text_with_metadata(self):
        """Test chunking preserves metadata."""
        text = "Test text"
        metadata = {"source": "test.txt"}
        chunks = chunk_text(text, metadata=metadata)

        assert all(chunk.metadata["source"] == "test.txt" for chunk in chunks)

    def test_chunk_document(self):
        """Test chunking a document with metadata."""
        text = "ESILV is a great school. " * 50  # Make it long enough to chunk
        metadata = {"filename": "test.txt", "file_type": "text"}

        chunks = chunk_document(text, metadata, chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 0
        assert all(chunk.metadata["filename"] == "test.txt" for chunk in chunks)

    def test_chunk_stats(self):
        """Test chunk statistics calculation."""
        chunks = [
            Chunk(text="A" * 100, metadata={}),
            Chunk(text="B" * 200, metadata={}),
            Chunk(text="C" * 150, metadata={})
        ]

        stats = get_chunk_stats(chunks)

        assert stats["count"] == 3
        assert stats["avg_chars"] == 150
        assert stats["min_chars"] == 100
        assert stats["max_chars"] == 200

    def test_recursive_text_splitter(self):
        """Test RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )

        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)


# =============================================================================
# Test Embedder
# =============================================================================

class TestEmbedder:
    """Test embedding generation functionality."""

    def test_embedding_generator_init(self):
        """Test initializing embedding generator."""
        embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

        assert embedder.embedding_dim == 384
        assert embedder.model is not None

    def test_embed_single_text(self):
        """Test embedding a single text."""
        embedder = EmbeddingGenerator()
        embedding = embedder.embed_single("ESILV is great")

        assert embedding.shape == (384,)
        assert isinstance(embedding, np.ndarray)

    def test_embed_batch(self):
        """Test batch embedding."""
        embedder = EmbeddingGenerator()
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedder.embed_batch(texts)

        assert embeddings.shape == (3, 384)
        assert isinstance(embeddings, np.ndarray)

    def test_embed_chunks(self):
        """Test embedding Chunk objects."""
        chunks = [
            Chunk(text="ESILV offers programs", metadata={}),
            Chunk(text="Applications open in January", metadata={})
        ]

        embeddings = embed_chunks(chunks)

        assert embeddings.shape == (2, 384)

    def test_embed_query(self):
        """Test query embedding."""
        query_emb = embed_query("What programs does ESILV offer?")

        assert query_emb.shape == (384,)
        assert isinstance(query_emb, np.ndarray)

    def test_embedding_normalization(self):
        """Test embeddings are L2 normalized."""
        embedder = EmbeddingGenerator(normalize=True)
        embedding = embedder.embed_single("Test text")

        # L2 norm should be ~1.0
        norm = np.linalg.norm(embedding)
        assert 0.99 < norm < 1.01

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        emb1 = embed_query("engineering school")
        emb2 = embed_query("engineering program")
        emb3 = embed_query("cooking recipe")

        sim_12 = cosine_similarity(emb1, emb2)
        sim_13 = cosine_similarity(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13

    def test_batch_cosine_similarity(self):
        """Test batch similarity computation."""
        query_emb = embed_query("engineering")

        chunks = [
            Chunk(text="engineering program", metadata={}),
            Chunk(text="cooking recipe", metadata={})
        ]
        chunk_embs = embed_chunks(chunks)

        similarities = batch_cosine_similarity(query_emb, chunk_embs)

        assert len(similarities) == 2
        assert similarities[0] > similarities[1]  # First chunk more relevant

    def test_embedding_stats(self):
        """Test embedding statistics."""
        embeddings = embed_texts(["Text 1", "Text 2"])
        stats = get_embedding_stats(embeddings)

        assert stats["count"] == 2
        assert stats["dimension"] == 384
        assert 0.99 < stats["mean_norm"] < 1.01  # Should be normalized


# =============================================================================
# Test Vector Store
# =============================================================================

class TestVectorStore:
    """Test vector store functionality."""

    def test_vector_store_init(self, temp_dir):
        """Test initializing vector store."""
        store = VectorStore(dimension=384, backend="faiss")

        assert store.backend == "faiss"
        assert isinstance(store.store, FAISSVectorStore)

    def test_add_and_search(self, temp_dir):
        """Test adding chunks and searching."""
        # Create sample chunks
        chunks = [
            Chunk(text="ESILV offers engineering programs", metadata={"source": "doc1"}),
            Chunk(text="Applications open in January", metadata={"source": "doc2"}),
            Chunk(text="The school is in Paris", metadata={"source": "doc3"})
        ]

        # Generate embeddings
        embeddings = embed_chunks(chunks)

        # Create and populate store
        store = VectorStore(dimension=384, backend="faiss")
        store.add_chunks(chunks, embeddings)

        # Search
        results = store.search("engineering programs", top_k=2)

        assert len(results) <= 2
        assert all(len(r) == 3 for r in results)  # (text, metadata, score) tuples

        # Check top result is relevant
        top_text, top_metadata, top_score = results[0]
        assert "engineering" in top_text.lower()

    def test_save_and_load(self, temp_dir):
        """Test saving and loading vector store."""
        # Create and populate store
        chunks = [Chunk(text="Test text", metadata={"source": "test"})]
        embeddings = embed_chunks(chunks)

        store = VectorStore(dimension=384, backend="faiss")
        store.add_chunks(chunks, embeddings)

        # Save
        save_path = temp_dir / "test_index"
        store.save(str(save_path))

        # Check files exist
        assert (temp_dir / "test_index.faiss").exists()
        assert (temp_dir / "test_index_metadata.json").exists()

        # Load
        loaded_store = VectorStore.load(str(save_path), backend="faiss")

        # Verify loaded store works
        results = loaded_store.search("test", top_k=1)
        assert len(results) == 1

    def test_vector_store_stats(self):
        """Test getting vector store statistics."""
        chunks = [
            Chunk(text="A" * 100, metadata={}),
            Chunk(text="B" * 200, metadata={})
        ]
        embeddings = embed_chunks(chunks)

        store = VectorStore(dimension=384, backend="faiss")
        store.add_chunks(chunks, embeddings)

        stats = store.get_stats()

        assert stats["count"] == 2
        assert stats["dimension"] == 384
        assert stats["backend"] == "faiss"


# =============================================================================
# Test Pipeline
# =============================================================================

class TestPipeline:
    """Test end-to-end pipeline."""

    def test_pipeline_config_validation(self, sample_documents):
        """Test pipeline configuration validation."""
        config = PipelineConfig(
            data_dir=str(sample_documents),
            chunk_size=800,
            chunk_overlap=150
        )

        assert config.validate() is True

    def test_pipeline_config_invalid_dir(self):
        """Test pipeline config with invalid directory."""
        config = PipelineConfig(data_dir="/nonexistent/path")

        with pytest.raises(ValueError, match="does not exist"):
            config.validate()

    def test_pipeline_config_invalid_chunk_size(self, sample_documents):
        """Test pipeline config with invalid chunk size."""
        config = PipelineConfig(
            data_dir=str(sample_documents),
            chunk_size=50  # Too small
        )

        with pytest.raises(ValueError, match="too small"):
            config.validate()

    def test_run_pipeline_end_to_end(self, sample_documents, temp_dir):
        """Test running the complete pipeline."""
        output_path = temp_dir / "test_index"

        stats = run_pipeline(
            data_dir=str(sample_documents),
            output_path=str(output_path),
            chunk_size=500,
            chunk_overlap=100,
            backend="faiss"
        )

        assert stats["success"] is True
        assert stats["total_docs"] == 2
        assert stats["total_chunks"] > 0
        assert stats["total_embeddings"] > 0

        # Check output files exist
        assert (temp_dir / "test_index.faiss").exists()
        assert (temp_dir / "test_index_metadata.json").exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Test integration between components."""

    def test_full_ingestion_workflow(self, sample_documents, temp_dir):
        """Test the complete ingestion workflow."""
        # 1. Load documents
        docs = load_documents_from_directory(str(sample_documents))
        assert len(docs) > 0

        # 2. Clean texts
        cleaned_texts = clean_documents(docs)
        assert len(cleaned_texts) == len(docs)

        # 3. Chunk documents
        metadata_list = [doc.metadata for doc in docs]
        chunks = chunk_documents(cleaned_texts, metadata_list, chunk_size=300)
        assert len(chunks) > 0

        # 4. Generate embeddings
        embeddings = embed_chunks(chunks)
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == 384

        # 5. Create vector store
        store = VectorStore(dimension=384, backend="faiss")
        store.add_chunks(chunks, embeddings)

        # 6. Test search
        results = store.search("ESILV programs", top_k=2)
        assert len(results) > 0

        # 7. Save and load
        save_path = temp_dir / "integration_test"
        store.save(str(save_path))

        loaded_store = VectorStore.load(str(save_path), backend="faiss")
        loaded_results = loaded_store.search("ESILV", top_k=1)
        assert len(loaded_results) > 0

    def test_semantic_search_quality(self):
        """Test that semantic search returns relevant results."""
        # Create documents with different topics
        chunks = [
            Chunk(text="ESILV offers computer science and engineering programs",
                  metadata={"topic": "programs"}),
            Chunk(text="Applications are due in March for fall admission",
                  metadata={"topic": "admissions"}),
            Chunk(text="The campus is located in Paris La D�fense",
                  metadata={"topic": "location"}),
        ]

        embeddings = embed_chunks(chunks)
        store = VectorStore(dimension=384, backend="faiss")
        store.add_chunks(chunks, embeddings)

        # Search for program information
        results = store.search("What programs does ESILV offer?", top_k=1)
        top_text, top_metadata, top_score = results[0]

        # Should retrieve the programs chunk
        assert top_metadata["topic"] == "programs"
        assert "programs" in top_text.lower()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
