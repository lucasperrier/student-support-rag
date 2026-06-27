"""
Embeddings Module

PURPOSE:
This module converts text chunks into dense vector embeddings for semantic search.
It's the fourth stage in the ingestion pipeline, taking Chunk objects and producing
numerical vector representations that capture semantic meaning.

INPUTS:
- List of Chunk objects from chunker.py
- OR raw text strings
- Configuration: model name, batch size, device (CPU/GPU)

OUTPUTS:
- numpy arrays of embeddings with shape (num_chunks, embedding_dim)
  - Default embedding_dim = 384 (for all-MiniLM-L6-v2)
- Compatible with FAISS and ChromaDB vector stores
- Each embedding is a dense vector that captures semantic meaning

INTEGRATION:
This module is used internally by the ingestion pipeline. The flow is:
  loader.py → text_cleaning.py → chunker.py → embedder.py → vector_store.py

EXAMPLE USAGE:
    from ingestion.chunker import chunk_text
    from ingestion.embedder import EmbeddingGenerator, embed_chunks

    # Generate chunks
    chunks = chunk_text("ESILV is a leading engineering school...")

    # Option 1: Using the class (for batch processing)
    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    embeddings = embedder.embed_batch([c.text for c in chunks])

    # Option 2: Using the convenience function
    embeddings = embed_chunks(chunks)

    # Result: numpy array of shape (num_chunks, 384)
    print(embeddings.shape)  # (5, 384) for 5 chunks

DESIGN DECISIONS:

1. **Embedding Model: all-MiniLM-L6-v2**
   - Embedding dimension: 384
   - Reasoning: Balanced speed/quality. Small model (80MB) runs fast on CPU.
     Quality is excellent for semantic search (MTEB score: 58.8).
   - Alternative: all-mpnet-base-v2 (768-dim, slower but higher quality)
   - Future: Can add Ollama nomic-embed-text support

2. **Batch Processing**:
   - Default batch size: 32 chunks
   - Reasoning: Balance memory usage vs. speed. 32 chunks fit easily in RAM
     while providing good throughput (~10x faster than one-at-a-time).
   - GPU support: Automatically uses GPU if available (torch.cuda)

3. **Normalization**:
   - L2 normalization applied to all embeddings
   - Reasoning: Required for cosine similarity in vector search. Ensures all
     vectors have unit length, making dot product = cosine similarity.

4. **Model Caching**:
   - Model loaded once per EmbeddingGenerator instance
   - Reasoning: Loading model is slow (~1-2 seconds). Reusing the same instance
     for multiple embed calls saves time.

5. **Error Handling**:
   - Empty chunks are skipped with warnings
   - Failed embeddings return zero vectors (logged)
   - Continues processing despite individual failures

TECHNICAL NOTES:
- Uses sentence-transformers library (HuggingFace)
- Model is downloaded automatically on first use (~80MB)
- Output is numpy arrays for compatibility with FAISS/ChromaDB
- Supports both CPU and GPU (auto-detected)
- Memory usage: ~200MB model + ~1KB per chunk embedding
"""

import logging
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass

# sentence-transformers for embedding generation
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Import Chunk dataclass from chunker
from ingestion.chunker import Chunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Alternative high-quality model: "all-mpnet-base-v2" (768-dim, slower)

# Batch processing size
DEFAULT_BATCH_SIZE = 32

# Device selection (auto-detect GPU)
# Will be set by EmbeddingGenerator based on torch.cuda.is_available()


# =============================================================================
# Core Embedding Generator
# =============================================================================

class EmbeddingGenerator:
    """
    Generates dense vector embeddings for text using sentence-transformers.

    This class handles model loading, batch processing, and normalization.
    Create one instance and reuse it for multiple embed calls to avoid
    reloading the model.

    Example:
        >>> embedder = EmbeddingGenerator()
        >>> texts = ["Hello world", "ESILV is great"]
        >>> embeddings = embedder.embed_batch(texts)
        >>> embeddings.shape
        (2, 384)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of sentence-transformers model to use
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            normalize: Whether to L2-normalize embeddings (required for cosine similarity)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.normalize = normalize

        # Auto-detect device if not specified
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading embedding model: {model_name} on {self.device}")

        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar (useful for large batches)

        Returns:
            numpy array of shape (len(texts), embedding_dim)

        Example:
            >>> embedder = EmbeddingGenerator()
            >>> texts = ["Text 1", "Text 2", "Text 3"]
            >>> embeddings = embedder.embed_batch(texts)
            >>> embeddings.shape
            (3, 384)
        """
        if not texts:
            logger.warning("Empty text list provided to embed_batch()")
            return np.array([])

        # Filter out empty texts (keep track of indices)
        valid_indices = []
        valid_texts = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(idx)
                valid_texts.append(text.strip())
            else:
                logger.warning(f"Skipping empty text at index {idx}")

        if not valid_texts:
            logger.warning("All texts are empty after filtering")
            # Return zero embeddings for all texts
            return np.zeros((len(texts), self.embedding_dim))

        # Generate embeddings
        try:
            logger.info(f"Embedding {len(valid_texts)} texts (batch_size={batch_size})")

            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            )

            # If we filtered out any empty texts, insert zero vectors at those positions
            if len(valid_indices) < len(texts):
                full_embeddings = np.zeros((len(texts), self.embedding_dim))
                full_embeddings[valid_indices] = embeddings
                embeddings = full_embeddings

            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.embedding_dim))

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        For efficiency, prefer embed_batch() when processing multiple texts.

        Args:
            text: Text string to embed

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to embed_single()")
            return np.zeros(self.embedding_dim)

        # Use embed_batch with single item
        embeddings = self.embed_batch([text.strip()], batch_size=1)
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        return self.embedding_dim


# =============================================================================
# Convenience Functions
# =============================================================================

# Global embedder instance (lazy loaded)
_global_embedder: Optional[EmbeddingGenerator] = None


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingGenerator:
    """
    Get or create global embedding generator instance.

    This avoids reloading the model for every embed call.
    The first call loads the model, subsequent calls reuse it.

    Args:
        model_name: Embedding model to use

    Returns:
        Shared EmbeddingGenerator instance
    """
    global _global_embedder

    if _global_embedder is None or _global_embedder.model_name != model_name:
        logger.info("Initializing global embedder")
        _global_embedder = EmbeddingGenerator(model_name=model_name)

    return _global_embedder


def embed_chunks(
    chunks: List[Chunk],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = False
) -> np.ndarray:
    """
    Generate embeddings for a list of Chunk objects.

    This is the main function for the ingestion pipeline.
    Takes Chunk objects from chunker.py and returns embeddings.

    Args:
        chunks: List of Chunk objects from chunker
        model_name: Embedding model to use
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar

    Returns:
        numpy array of shape (len(chunks), embedding_dim)

    Example:
        >>> from ingestion.chunker import chunk_text
        >>> from ingestion.embedder import embed_chunks
        >>>
        >>> chunks = chunk_text("Long document text...")
        >>> embeddings = embed_chunks(chunks)
        >>> print(f"Generated {len(embeddings)} embeddings")
    """
    if not chunks:
        logger.warning("Empty chunks list provided to embed_chunks()")
        return np.array([])

    # Extract text from chunks
    texts = [chunk.text for chunk in chunks]

    # Get global embedder
    embedder = get_embedder(model_name)

    # Generate embeddings
    embeddings = embedder.embed_batch(texts, batch_size=batch_size, show_progress=show_progress)

    logger.info(f"Embedded {len(chunks)} chunks into {embeddings.shape}")

    return embeddings


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = False
) -> np.ndarray:
    """
    Generate embeddings for a list of text strings.

    Convenience function for embedding raw text without Chunk objects.

    Args:
        texts: List of text strings
        model_name: Embedding model to use
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    if not texts:
        logger.warning("Empty texts list provided to embed_texts()")
        return np.array([])

    # Get global embedder
    embedder = get_embedder(model_name)

    # Generate embeddings
    return embedder.embed_batch(texts, batch_size=batch_size, show_progress=show_progress)


def embed_query(
    query: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    """
    Generate embedding for a search query.

    Used by retrieval agent for query-to-chunk similarity search.

    Args:
        query: Search query string
        model_name: Embedding model to use (must match chunks' model)

    Returns:
        numpy array of shape (embedding_dim,)

    Example:
        >>> from ingestion.embedder import embed_query
        >>> query_embedding = embed_query("What are ESILV admission requirements?")
        >>> query_embedding.shape
        (384,)
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to embed_query()")
        embedder = get_embedder(model_name)
        return np.zeros(embedder.embedding_dim)

    # Get global embedder
    embedder = get_embedder(model_name)

    # Generate embedding
    return embedder.embed_single(query.strip())


# =============================================================================
# Utility Functions
# =============================================================================

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    If embeddings are L2-normalized (default), this is just the dot product.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Similarity score between -1 and 1 (higher = more similar)

    Example:
        >>> emb1 = embed_query("ESILV engineering program")
        >>> emb2 = embed_query("Engineering school in Paris")
        >>> similarity = cosine_similarity(emb1, emb2)
        >>> print(f"Similarity: {similarity:.3f}")
    """
    # If already normalized (which our embedder does), dot product = cosine similarity
    # Otherwise: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.dot(embedding1, embedding2)


def batch_cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple embeddings.

    Optimized for comparing one query against many chunks.

    Args:
        query_embedding: Query embedding of shape (embedding_dim,)
        embeddings: Matrix of embeddings with shape (num_chunks, embedding_dim)

    Returns:
        Array of similarity scores with shape (num_chunks,)

    Example:
        >>> query_emb = embed_query("What is ESILV?")
        >>> chunk_embs = embed_chunks(chunks)
        >>> similarities = batch_cosine_similarity(query_emb, chunk_embs)
        >>> top_indices = np.argsort(similarities)[::-1][:5]  # Top 5 most similar
    """
    # Matrix multiplication: (num_chunks, dim) @ (dim,) -> (num_chunks,)
    return np.dot(embeddings, query_embedding)


def get_embedding_stats(embeddings: np.ndarray) -> dict:
    """
    Get statistics about a batch of embeddings.

    Useful for debugging and quality assurance.

    Args:
        embeddings: Matrix of embeddings with shape (num_embeddings, embedding_dim)

    Returns:
        Dict with statistics (shape, mean_norm, std_norm, etc.)
    """
    if embeddings.size == 0:
        return {
            'count': 0,
            'dimension': 0,
            'mean_norm': 0,
            'std_norm': 0
        }

    # Compute norms (should be ~1.0 if normalized)
    norms = np.linalg.norm(embeddings, axis=1)

    return {
        'count': embeddings.shape[0],
        'dimension': embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings),
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'min_norm': float(np.min(norms)),
        'max_norm': float(np.max(norms))
    }


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    """
    Quick test/demo of embedding generation.
    Run this file directly to test: python -m ingestion.embedder
    """

    print("Embeddings Module Test")
    print("=" * 80)

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\nERROR: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        exit(1)

    # Test 1: Basic embedding
    print("\n1. Basic Text Embedding:")
    embedder = EmbeddingGenerator()

    test_texts = [
        "ESILV is a leading French engineering school.",
        "The school specializes in digital engineering.",
        "Students can apply through Parcoursup."
    ]

    embeddings = embedder.embed_batch(test_texts)
    print(f"Embedded {len(test_texts)} texts")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.embedding_dim}")

    # Test 2: Embedding statistics
    print("\n2. Embedding Statistics:")
    stats = get_embedding_stats(embeddings)
    print(f"  Count: {stats['count']}")
    print(f"  Dimension: {stats['dimension']}")
    print(f"  Mean norm: {stats['mean_norm']:.4f} (should be ~1.0 if normalized)")
    print(f"  Std norm: {stats['std_norm']:.4f}")

    # Test 3: Similarity computation
    print("\n3. Cosine Similarity Test:")
    query = "engineering school in Paris"
    query_emb = embedder.embed_single(query)

    similarities = batch_cosine_similarity(query_emb, embeddings)
    print(f"Query: '{query}'")
    print(f"Similarities with test texts:")
    for i, (text, sim) in enumerate(zip(test_texts, similarities)):
        print(f"  [{i}] {sim:.3f} - {text}")

    # Test 4: Chunk embedding
    print("\n4. Chunk Embedding Test:")
    from ingestion.chunker import chunk_text

    long_text = """
    ESILV is a leading French engineering school located in Paris La Défense.
    The school specializes in digital engineering and computer science.

    Our programs include a 5-year engineering degree, specialized masters,
    and international exchange programs. We have strong industry partnerships
    with companies like Amazon, Google, and Capgemini.
    """

    chunks = chunk_text(long_text, chunk_size=150, chunk_overlap=30)
    chunk_embeddings = embed_chunks(chunks, show_progress=True)

    print(f"\nChunked text into {len(chunks)} chunks")
    print(f"Generated embeddings with shape: {chunk_embeddings.shape}")

    # Test 5: Real documents (if available)
    print("\n5. Testing with real documents from data/raw/:")
    from pathlib import Path
    from ingestion.loader import load_documents_from_directory
    from ingestion.text_cleaning import clean_documents
    from ingestion.chunker import chunk_documents

    raw_dir = Path(__file__).parent.parent / "data" / "raw"

    if raw_dir.exists() and any(raw_dir.iterdir()):
        docs = load_documents_from_directory(str(raw_dir))
        if docs:
            print(f"Loaded {len(docs)} documents")

            # Clean and chunk
            cleaned_texts = clean_documents(docs)
            metadata_list = [doc.metadata for doc in docs]
            all_chunks = chunk_documents(cleaned_texts, metadata_list, chunk_size=500, chunk_overlap=100)

            print(f"Created {len(all_chunks)} chunks")

            # Embed all chunks
            print("Generating embeddings for all chunks...")
            all_embeddings = embed_chunks(all_chunks, show_progress=True)

            print(f"\nEmbedding complete!")
            print(f"  Total embeddings: {all_embeddings.shape[0]}")
            print(f"  Embedding dimension: {all_embeddings.shape[1]}")

            # Show statistics
            stats = get_embedding_stats(all_embeddings)
            print(f"  Mean norm: {stats['mean_norm']:.4f}")

            # Test semantic search
            print("\n6. Semantic Search Demo:")
            test_query = "What programs does ESILV offer?"
            query_emb = embed_query(test_query)

            similarities = batch_cosine_similarity(query_emb, all_embeddings)
            top_k = 3
            top_indices = np.argsort(similarities)[::-1][:top_k]

            print(f"\nQuery: '{test_query}'")
            print(f"Top {top_k} most relevant chunks:")
            for rank, idx in enumerate(top_indices):
                chunk = all_chunks[idx]
                score = similarities[idx]
                print(f"\n  Rank {rank + 1} (similarity: {score:.3f})")
                print(f"  Source: {chunk.metadata.get('filename', 'unknown')}")
                print(f"  Text: {chunk.text[:150]}...")
        else:
            print("No documents found in data/raw/")
    else:
        print(f"Directory not found or empty: {raw_dir}")

    print("\n" + "=" * 80)
    print("Test complete!")
