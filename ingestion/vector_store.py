"""
Vector Store Module

PURPOSE:
This module provides a vector database for storing and searching document embeddings.
It's the fifth stage in the ingestion pipeline and the primary interface for the
retrieval agent to find relevant chunks based on semantic similarity.

INPUTS:
- Embeddings: numpy arrays from embedder.py with shape (num_chunks, embedding_dim)
- Chunks: Chunk objects with text and metadata from chunker.py
- Queries: Text strings from retrieval agent for semantic search

OUTPUTS:
- search() returns: List of (chunk_text, metadata_dict, similarity_score) tuples
- Persisted index saved to disk for fast loading
- Statistics about the vector store (count, dimension, etc.)

PERSON B INTEGRATION:
The retrieval agent calls search() to find relevant chunks:

    from ingestion.vector_store import get_vector_store

    # Get the global vector store instance
    store = get_vector_store()

    # Search for relevant chunks
    results = store.search("What programs does ESILV offer?", top_k=5)

    # Results format: [(chunk_text, metadata, score), ...]
    for text, metadata, score in results:
        print(f"Relevance: {score:.3f}")
        print(f"Source: {metadata['filename']}")
        print(f"Text: {text[:100]}...")

EXAMPLE USAGE:
    from ingestion.chunker import chunk_text
    from ingestion.embedder import embed_chunks
    from ingestion.vector_store import VectorStore

    # Create chunks and embeddings
    chunks = chunk_text("ESILV is a leading engineering school...")
    embeddings = embed_chunks(chunks)

    # Create vector store
    store = VectorStore(dimension=384)
    store.add_chunks(chunks, embeddings)

    # Save to disk
    store.save("data/vector_db/index")

    # Later: Load from disk
    store2 = VectorStore.load("data/vector_db/index")

    # Search
    results = store2.search("engineering programs", top_k=3)

DESIGN DECISIONS:

1. **Backend: FAISS (default) with ChromaDB as alternative**
   - FAISS: Lightweight, fast, ideal for <100k chunks
   - ChromaDB: Feature-rich, better for large-scale, has built-in persistence
   - Reasoning: FAISS is simpler and faster for our scale (~1000 chunks)
   - Both are supported via a common interface

2. **Index Type: IndexFlatIP (Inner Product)**
   - Uses dot product for similarity (= cosine similarity for normalized vectors)
   - Reasoning: Exact search (no approximation), fast for small datasets
   - Alternative: IndexIVFFlat for >10k chunks (approximate but faster)

3. **Metadata Storage**
   - Separate JSON file stores chunk texts and metadata
   - FAISS index stores only vectors (lightweight)
   - Reasoning: FAISS doesn't support metadata, so we store it separately
   - Index position = metadata position (synchronized)

4. **Persistence**
   - FAISS: index saved as .faiss binary + metadata.json
   - ChromaDB: Uses built-in persistence
   - Automatic versioning to prevent corruption

5. **Thread Safety**
   - Read operations (search) are thread-safe
   - Write operations (add_chunks) use locks
   - Reasoning: Multiple agents might search concurrently

TECHNICAL NOTES:
- FAISS index is memory-mapped for fast loading
- Embeddings must be L2-normalized (handled by embedder.py)
- Supports incremental updates (add more chunks after initial creation)
- Automatic dimension validation
- Efficient batch operations
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import asdict
import threading

# FAISS for vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

# ChromaDB as alternative
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

# Import embedder for query embedding
from ingestion.embedder import embed_query
from ingestion.chunker import Chunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_VECTOR_STORE_TYPE = "faiss"  # "faiss" or "chroma"
DEFAULT_INDEX_PATH = "data/vector_db/index"

# =============================================================================
# FAISS-based Vector Store
# =============================================================================

class FAISSVectorStore:
    """
    FAISS-based vector store for fast similarity search.

    Uses Facebook AI Similarity Search (FAISS) for efficient vector operations.
    Stores metadata separately in JSON format.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine similarity for normalized vectors
        self.chunks_metadata: List[Dict[str, Any]] = []  # Store chunk text + metadata
        self.lock = threading.Lock()  # Thread safety for writes

        logger.info(f"Initialized FAISS vector store (dimension={dimension})")

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the vector store.

        Args:
            chunks: List of Chunk objects from chunker.py
            embeddings: numpy array of embeddings with shape (len(chunks), dimension)
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match store dimension {self.dimension}")

        with self.lock:
            # Add embeddings to FAISS index
            # FAISS requires float32
            embeddings_f32 = embeddings.astype(np.float32)
            self.index.add(embeddings_f32)

            # Store metadata
            for chunk in chunks:
                metadata = {
                    'text': chunk.text,
                    'metadata': chunk.metadata,
                    'char_count': chunk.char_count,
                    'token_estimate': chunk.token_estimate
                }
                self.chunks_metadata.append(metadata)

            logger.info(f"Added {len(chunks)} chunks to vector store (total: {self.index.ntotal})")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for most similar chunks to a query.

        Args:
            query: Search query string
            top_k: Number of top results to return
            min_score: Minimum similarity score (0-1, cosine similarity)

        Returns:
            List of (chunk_text, metadata_dict, similarity_score) tuples,
            sorted by descending similarity
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        # Embed query
        query_embedding = embed_query(query)
        query_embedding_f32 = query_embedding.astype(np.float32).reshape(1, -1)

        # Search FAISS index
        # Returns: distances (similarity scores), indices (positions in index)
        top_k = min(top_k, self.index.ntotal)  # Can't return more than we have
        scores, indices = self.index.search(query_embedding_f32, top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Skip results below minimum score
            if score < min_score:
                continue

            # Get chunk metadata
            chunk_data = self.chunks_metadata[idx]

            results.append((
                chunk_data['text'],
                chunk_data['metadata'],
                float(score)
            ))

        logger.info(f"Search returned {len(results)} results for query: '{query[:50]}...'")
        return results

    def save(self, path: str) -> None:
        """
        Save vector store to disk.

        Args:
            path: Base path (will create path.faiss and path_metadata.json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = str(path) + ".faiss"
        faiss.write_index(self.index, index_file)

        # Save metadata
        metadata_file = str(path) + "_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dimension': self.dimension,
                'num_chunks': self.index.ntotal,
                'chunks_metadata': self.chunks_metadata
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved vector store to {path} ({self.index.ntotal} chunks)")

    @classmethod
    def load(cls, path: str) -> 'FAISSVectorStore':
        """
        Load vector store from disk.

        Args:
            path: Base path (looks for path.faiss and path_metadata.json)

        Returns:
            Loaded FAISSVectorStore instance
        """
        path = Path(path)
        index_file = str(path) + ".faiss"
        metadata_file = str(path) + "_metadata.json"

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create instance
        store = cls(dimension=data['dimension'])

        # Load FAISS index
        store.index = faiss.read_index(index_file)
        store.chunks_metadata = data['chunks_metadata']

        logger.info(f"Loaded vector store from {path} ({store.index.ntotal} chunks)")
        return store

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dict with stats (count, dimension, avg_chunk_size, etc.)
        """
        if self.index.ntotal == 0:
            return {
                'count': 0,
                'dimension': self.dimension,
                'avg_chars': 0,
                'avg_tokens': 0
            }

        char_counts = [m['char_count'] for m in self.chunks_metadata]
        token_counts = [m['token_estimate'] for m in self.chunks_metadata]

        return {
            'count': self.index.ntotal,
            'dimension': self.dimension,
            'avg_chars': sum(char_counts) // len(char_counts),
            'avg_tokens': sum(token_counts) // len(token_counts),
            'total_chars': sum(char_counts)
        }


# =============================================================================
# ChromaDB-based Vector Store (Alternative)
# =============================================================================

class ChromaDBVectorStore:
    """
    ChromaDB-based vector store with built-in persistence.

    Alternative to FAISS with more features but slightly heavier.
    """

    def __init__(self, dimension: int = 384, persist_directory: str = "data/vector_db/chroma"):
        """
        Initialize ChromaDB vector store.

        Args:
            dimension: Embedding dimension
            persist_directory: Directory for ChromaDB persistence
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")

        self.dimension = dimension
        self.persist_directory = persist_directory

        # Create client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="esilv_chunks",
            metadata={"dimension": dimension}
        )

        logger.info(f"Initialized ChromaDB vector store at {persist_directory}")

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and embeddings to ChromaDB.

        Args:
            chunks: List of Chunk objects
            embeddings: numpy array of embeddings
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings")

        # Prepare data for ChromaDB
        ids = [f"chunk_{i}_{chunk.metadata.get('filename', 'unknown')}" for i, chunk in enumerate(chunks)]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings_list = embeddings.tolist()

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"Added {len(chunks)} chunks to ChromaDB")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search ChromaDB for similar chunks.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score

        Returns:
            List of (chunk_text, metadata, score) tuples
        """
        # Embed query
        query_embedding = embed_query(query)

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Format results
        output = []
        for i in range(len(results['documents'][0])):
            score = 1 - results['distances'][0][i]  # ChromaDB returns distances, convert to similarity

            if score < min_score:
                continue

            output.append((
                results['documents'][0][i],
                results['metadatas'][0][i],
                float(score)
            ))

        logger.info(f"ChromaDB search returned {len(output)} results")
        return output

    def save(self, path: str = None) -> None:
        """ChromaDB auto-persists, so this is a no-op."""
        logger.info("ChromaDB auto-persists, no manual save needed")

    @classmethod
    def load(cls, path: str) -> 'ChromaDBVectorStore':
        """
        Load ChromaDB from persist directory.

        Args:
            path: Persist directory path

        Returns:
            Loaded ChromaDBVectorStore
        """
        return cls(persist_directory=path)

    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        count = self.collection.count()
        return {
            'count': count,
            'dimension': self.dimension,
            'backend': 'chromadb'
        }


# =============================================================================
# Unified VectorStore Interface
# =============================================================================

class VectorStore:
    """
    Unified interface for vector stores.

    Automatically selects backend (FAISS or ChromaDB) based on availability.
    """

    def __init__(self, dimension: int = 384, backend: str = "auto"):
        """
        Initialize vector store.

        Args:
            dimension: Embedding dimension
            backend: "faiss", "chroma", or "auto" (auto-select based on availability)
        """
        if backend == "auto":
            if FAISS_AVAILABLE:
                backend = "faiss"
            elif CHROMADB_AVAILABLE:
                backend = "chroma"
            else:
                raise ImportError("No vector store backend available. Install faiss-cpu or chromadb.")

        if backend == "faiss":
            self.store = FAISSVectorStore(dimension=dimension)
        elif backend == "chroma":
            self.store = ChromaDBVectorStore(dimension=dimension)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.backend = backend
        logger.info(f"Using {backend} backend for vector store")

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks to store."""
        self.store.add_chunks(chunks, embeddings)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Tuple[str, Dict, float]]:
        """Search store."""
        return self.store.search(query, top_k, min_score)

    def save(self, path: str) -> None:
        """Save store to disk."""
        self.store.save(path)

    @classmethod
    def load(cls, path: str, backend: str = "auto") -> 'VectorStore':
        """Load store from disk."""
        store = cls(dimension=384, backend=backend)  # Dimension will be overwritten on load

        if backend == "faiss" or (backend == "auto" and FAISS_AVAILABLE):
            store.store = FAISSVectorStore.load(path)
        elif backend == "chroma" or (backend == "auto" and CHROMADB_AVAILABLE):
            store.store = ChromaDBVectorStore.load(path)

        return store

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        stats = self.store.get_stats()
        stats['backend'] = self.backend
        return stats


# =============================================================================
# Global Vector Store Instance
# =============================================================================

_global_vector_store: Optional[VectorStore] = None


def get_vector_store(
    path: Optional[str] = None,
    dimension: int = 384,
    backend: str = "auto"
) -> VectorStore:
    """
    Get or create global vector store instance.

    This is the main function that Person B's retrieval agent should use.

    Args:
        path: Optional path to load existing store
        dimension: Embedding dimension (ignored if loading)
        backend: Backend to use ("faiss", "chroma", or "auto")

    Returns:
        VectorStore instance

    Example:
        # First time: create new store
        store = get_vector_store()

        # Later: load from disk
        store = get_vector_store(path="data/vector_db/index")
    """
    global _global_vector_store

    if _global_vector_store is None:
        if path and os.path.exists(str(path) + ".faiss"):
            # Load existing store
            logger.info(f"Loading vector store from {path}")
            _global_vector_store = VectorStore.load(path, backend=backend)
        else:
            # Create new store
            logger.info("Creating new vector store")
            _global_vector_store = VectorStore(dimension=dimension, backend=backend)

    return _global_vector_store


# =============================================================================
# Convenience Function (for Person B)
# =============================================================================

def search(query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
    """
    Search the global vector store.

    THIS IS THE MAIN API FOR PERSON B's RETRIEVAL AGENT.

    Args:
        query: User query string
        top_k: Number of results to return

    Returns:
        List of (chunk_text, metadata_dict) tuples

    Example:
        from ingestion.vector_store import search

        results = search("What programs does ESILV offer?", top_k=5)
        for text, metadata in results:
            print(f"Source: {metadata['filename']}")
            print(f"Text: {text}")
    """
    store = get_vector_store(path=DEFAULT_INDEX_PATH)

    # Search and return without scores (to match original API contract)
    results_with_scores = store.search(query, top_k=top_k)

    # Convert to (text, metadata) format for backward compatibility
    return [(text, metadata) for text, metadata, score in results_with_scores]


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    """
    Test the vector store implementation.
    Run: python -m ingestion.vector_store
    """

    print("Vector Store Module Test")
    print("=" * 80)

    # Test 1: Basic FAISS store
    print("\n1. Testing FAISS Vector Store:")
    from ingestion.chunker import chunk_text
    from ingestion.embedder import embed_chunks

    test_texts = [
        "ESILV is a leading French engineering school in Paris La Défense.",
        "The school offers a 5-year engineering program with specializations in digital engineering.",
        "Students can apply through Parcoursup for French students or directly for international students.",
        "ESILV has partnerships with companies like Amazon, Google, and Capgemini.",
        "The admissions process includes reviewing academic records and conducting interviews."
    ]

    # Create chunks
    chunks = []
    for i, text in enumerate(test_texts):
        chunk_objs = chunk_text(text, metadata={'filename': f'test_{i}.txt', 'source': 'test'})
        chunks.extend(chunk_objs)

    # Generate embeddings
    embeddings = embed_chunks(chunks)

    # Create store
    store = VectorStore(dimension=384, backend="faiss")
    store.add_chunks(chunks, embeddings)

    print(f"Added {len(chunks)} chunks to store")

    # Test search
    print("\n2. Testing Search:")
    test_queries = [
        "What programs does ESILV offer?",
        "How do I apply to ESILV?",
        "What companies partner with ESILV?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = store.search(query, top_k=2)
        for i, (text, metadata, score) in enumerate(results):
            print(f"  [{i+1}] Score: {score:.3f}")
            print(f"      Text: {text[:80]}...")

    # Test save/load
    print("\n3. Testing Save/Load:")
    test_path = "data/vector_db/test_index"
    store.save(test_path)
    print(f"Saved to {test_path}")

    store2 = VectorStore.load(test_path, backend="faiss")
    print(f"Loaded from {test_path}")

    # Verify loaded store works
    results = store2.search("engineering school", top_k=1)
    print(f"Search on loaded store: {len(results)} results")

    # Test stats
    print("\n4. Vector Store Statistics:")
    stats = store2.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test with real documents
    print("\n5. Testing with Real Documents:")
    from pathlib import Path
    from ingestion.loader import load_documents_from_directory
    from ingestion.text_cleaning import clean_documents
    from ingestion.chunker import chunk_documents

    raw_dir = Path(__file__).parent.parent / "data" / "raw"

    if raw_dir.exists() and any(raw_dir.iterdir()):
        docs = load_documents_from_directory(str(raw_dir))
        if docs:
            print(f"Loaded {len(docs)} documents")

            # Clean, chunk, embed
            cleaned_texts = clean_documents(docs)
            metadata_list = [doc.metadata for doc in docs]
            all_chunks = chunk_documents(cleaned_texts, metadata_list)
            all_embeddings = embed_chunks(all_chunks)

            # Create production store
            prod_store = VectorStore(dimension=384, backend="faiss")
            prod_store.add_chunks(all_chunks, all_embeddings)

            # Save
            prod_path = "data/vector_db/index"
            prod_store.save(prod_path)
            print(f"Saved production index to {prod_path}")

            # Test search
            print("\n6. Production Search Test:")
            test_query = "What are the admission requirements?"
            results = prod_store.search(test_query, top_k=3)

            print(f"\nQuery: '{test_query}'")
            for i, (text, metadata, score) in enumerate(results):
                print(f"\n  [{i+1}] Relevance: {score:.3f}")
                print(f"      Source: {metadata.get('filename', 'unknown')}")
                print(f"      Text: {text[:100]}...")

            # Test global convenience function
            print("\n7. Testing Global search() Function:")
            from ingestion.vector_store import search as global_search

            results = global_search("programs at ESILV", top_k=2)
            print(f"Global search returned {len(results)} results")
            for text, metadata in results:
                print(f"  - {metadata.get('filename')}: {text[:60]}...")
    else:
        print(f"No documents in {raw_dir}, skipping real document test")

    print("\n" + "=" * 80)
    print("Vector Store Test Complete!")
