"""
Ingestion Pipeline Orchestrator

PURPOSE:
This module orchestrates the complete end-to-end document ingestion pipeline.
It's the main entry point for processing raw documents into a searchable vector store.

INPUTS:
- Raw documents directory (PDFs, HTML, TXT files)
- Configuration: chunk size, embedding model, output path
- Optional: existing vector store to update incrementally

OUTPUTS:
- Vector store index saved to disk (data/vector_db/index.faiss + metadata.json)
- Processing statistics and logs
- Ready-to-use vector database for retrieval agent

PERSON B INTEGRATION:
Person B doesn't need to call this directly - it's a one-time setup or periodic update.
After running the pipeline, the retrieval agent can use the vector store:

    # Person A runs this once to ingest documents:
    python -m ingestion.pipeline --data-dir data/raw

    # Then Person B's retrieval agent uses the resulting index:
    from ingestion.vector_store import search
    results = search("What programs does ESILV offer?")

EXAMPLE USAGE:
    # Basic usage - ingest all documents in data/raw/
    python -m ingestion.pipeline --data-dir data/raw

    # Custom configuration
    python -m ingestion.pipeline \
        --data-dir data/raw \
        --output data/vector_db/index \
        --chunk-size 800 \
        --chunk-overlap 150 \
        --embedding-model all-MiniLM-L6-v2

    # Incremental update (add new docs to existing index)
    python -m ingestion.pipeline \
        --data-dir data/raw/new_docs \
        --output data/vector_db/index \
        --incremental

    # From Python code
    from ingestion.pipeline import run_pipeline

    stats = run_pipeline(
        data_dir="data/raw",
        output_path="data/vector_db/index"
    )
    print(f"Processed {stats['total_chunks']} chunks from {stats['total_docs']} documents")

DESIGN DECISIONS:

1. **Pipeline Stages**:
   - Stage 1: Load documents (loader.py)
   - Stage 2: Clean text (text_cleaning.py)
   - Stage 3: Chunk documents (chunker.py)
   - Stage 4: Generate embeddings (embedder.py)
   - Stage 5: Store in vector DB (vector_store.py)

2. **Error Handling**:
   - Individual document failures don't stop the pipeline
   - Failed documents are logged with reasons
   - Pipeline continues and processes successful documents

3. **Progress Reporting**:
   - Show progress at each stage
   - Log statistics (files processed, chunks created, etc.)
   - Final summary with timing and counts

4. **Incremental Updates**:
   - Can load existing vector store and add new documents
   - Avoids reprocessing entire corpus for updates
   - Useful for adding new documents periodically

5. **CLI Interface**:
   - Argparse for command-line arguments
   - Sensible defaults from CLAUDE.md guidance
   - Help text for all options

TECHNICAL NOTES:
- Uses pathlib for cross-platform path handling
- Logs to both console and optional log file
- Memory efficient: processes in batches
- Validates inputs before starting pipeline
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import all pipeline components
from ingestion.loader import load_documents_from_directory, Document
from ingestion.text_cleaning import clean_documents
from ingestion.chunker import chunk_documents, Chunk, get_chunk_stats
from ingestion.embedder import embed_chunks, get_embedding_stats
from ingestion.vector_store import VectorStore, get_vector_store

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class PipelineConfig:
    """Configuration for the ingestion pipeline."""

    def __init__(
        self,
        data_dir: str = "data/raw",
        output_path: str = "data/vector_db/index",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        backend: str = "faiss",
        incremental: bool = False,
        recursive: bool = True
    ):
        """
        Initialize pipeline configuration.

        Args:
            data_dir: Directory containing raw documents
            output_path: Path to save vector store
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            embedding_model: Sentence-transformers model name
            embedding_dimension: Embedding dimension
            backend: Vector store backend ('faiss' or 'chroma')
            incremental: Whether to add to existing index or create new
            recursive: Whether to search subdirectories
        """
        self.data_dir = Path(data_dir)
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.backend = backend
        self.incremental = incremental
        self.recursive = recursive

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        if not self.data_dir.is_dir():
            raise ValueError(f"Data path is not a directory: {self.data_dir}")

        if self.chunk_size < 100:
            raise ValueError(f"Chunk size too small: {self.chunk_size} (minimum: 100)")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"Chunk overlap ({self.chunk_overlap}) must be less than chunk size ({self.chunk_size})")

        if self.incremental and not self.output_path.with_suffix('.faiss').exists():
            logger.warning(f"Incremental mode requested but no existing index found at {self.output_path}")
            logger.warning("Will create new index instead")
            self.incremental = False

        return True


# =============================================================================
# Pipeline Stages
# =============================================================================

def stage_1_load_documents(config: PipelineConfig) -> List[Document]:
    """
    Stage 1: Load documents from directory.

    Args:
        config: Pipeline configuration

    Returns:
        List of loaded Document objects
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: Loading Documents")
    logger.info("=" * 80)

    docs = load_documents_from_directory(
        str(config.data_dir),
        recursive=config.recursive
    )

    if not docs:
        logger.warning(f"No documents found in {config.data_dir}")
        return []

    logger.info(f"✓ Loaded {len(docs)} documents")

    # Log document types
    doc_types = {}
    for doc in docs:
        file_type = doc.metadata.get('file_type', 'unknown')
        doc_types[file_type] = doc_types.get(file_type, 0) + 1

    logger.info("Document types:")
    for file_type, count in doc_types.items():
        logger.info(f"  - {file_type}: {count}")

    return docs


def stage_2_clean_documents(docs: List[Document]) -> List[str]:
    """
    Stage 2: Clean document text.

    Args:
        docs: List of Document objects

    Returns:
        List of cleaned text strings
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("STAGE 2: Cleaning Text")
    logger.info("=" * 80)

    cleaned_texts = clean_documents(docs)

    # Calculate cleaning statistics
    original_chars = sum(len(doc.text) for doc in docs)
    cleaned_chars = sum(len(text) for text in cleaned_texts)
    reduction = ((original_chars - cleaned_chars) / original_chars * 100) if original_chars > 0 else 0

    logger.info(f"✓ Cleaned {len(cleaned_texts)} documents")
    logger.info(f"  Original: {original_chars:,} chars")
    logger.info(f"  Cleaned: {cleaned_chars:,} chars")
    logger.info(f"  Reduction: {reduction:.1f}%")

    return cleaned_texts


def stage_3_chunk_documents(
    cleaned_texts: List[str],
    docs: List[Document],
    config: PipelineConfig
) -> List[Chunk]:
    """
    Stage 3: Chunk documents into smaller pieces.

    Args:
        cleaned_texts: List of cleaned text strings
        docs: Original Document objects (for metadata)
        config: Pipeline configuration

    Returns:
        List of Chunk objects
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("STAGE 3: Chunking Documents")
    logger.info("=" * 80)

    metadata_list = [doc.metadata for doc in docs]

    chunks = chunk_documents(
        cleaned_texts,
        metadata_list,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )

    # Get chunk statistics
    stats = get_chunk_stats(chunks)

    logger.info(f"✓ Created {len(chunks)} chunks")
    logger.info(f"  Chunk size: {config.chunk_size} chars (overlap: {config.chunk_overlap})")
    logger.info(f"  Avg chunk: {stats['avg_chars']} chars (~{stats['avg_tokens']} tokens)")
    logger.info(f"  Min/Max: {stats['min_chars']}/{stats['max_chars']} chars")

    return chunks


def stage_4_generate_embeddings(
    chunks: List[Chunk],
    config: PipelineConfig
) -> Any:  # numpy.ndarray
    """
    Stage 4: Generate embeddings for chunks.

    Args:
        chunks: List of Chunk objects
        config: Pipeline configuration

    Returns:
        numpy array of embeddings
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("STAGE 4: Generating Embeddings")
    logger.info("=" * 80)

    logger.info(f"Model: {config.embedding_model}")
    logger.info(f"Processing {len(chunks)} chunks...")

    embeddings = embed_chunks(
        chunks,
        model_name=config.embedding_model,
        show_progress=True
    )

    # Get embedding statistics
    stats = get_embedding_stats(embeddings)

    logger.info(f"✓ Generated {stats['count']} embeddings")
    logger.info(f"  Dimension: {stats['dimension']}")
    logger.info(f"  Mean norm: {stats['mean_norm']:.4f} (should be ~1.0 if normalized)")

    return embeddings


def stage_5_create_vector_store(
    chunks: List[Chunk],
    embeddings: Any,  # numpy.ndarray
    config: PipelineConfig
) -> VectorStore:
    """
    Stage 5: Create and populate vector store.

    Args:
        chunks: List of Chunk objects
        embeddings: numpy array of embeddings
        config: Pipeline configuration

    Returns:
        Populated VectorStore instance
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("STAGE 5: Creating Vector Store")
    logger.info("=" * 80)

    # Load existing store if incremental, otherwise create new
    if config.incremental:
        logger.info(f"Loading existing vector store from {config.output_path}")
        try:
            store = VectorStore.load(str(config.output_path), backend=config.backend)
            logger.info(f"Loaded existing store with {store.get_stats()['count']} chunks")
        except Exception as e:
            logger.warning(f"Failed to load existing store: {e}")
            logger.info("Creating new vector store instead")
            store = VectorStore(dimension=config.embedding_dimension, backend=config.backend)
    else:
        logger.info("Creating new vector store")
        store = VectorStore(dimension=config.embedding_dimension, backend=config.backend)

    # Add chunks
    logger.info(f"Adding {len(chunks)} chunks to vector store...")
    store.add_chunks(chunks, embeddings)

    # Save to disk
    logger.info(f"Saving vector store to {config.output_path}")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(str(config.output_path))

    # Get final stats
    stats = store.get_stats()
    logger.info(f"✓ Vector store created and saved")
    logger.info(f"  Total chunks: {stats['count']}")
    logger.info(f"  Backend: {stats.get('backend', 'unknown')}")
    logger.info(f"  Output: {config.output_path}")

    return store


# =============================================================================
# Main Pipeline Function
# =============================================================================

def run_pipeline(
    data_dir: str = "data/raw",
    output_path: str = "data/vector_db/index",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    embedding_model: str = "all-MiniLM-L6-v2",
    backend: str = "faiss",
    incremental: bool = False,
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Run the complete ingestion pipeline.

    This is the main function to call from Python code.

    Args:
        data_dir: Directory containing raw documents
        output_path: Path to save vector store
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        embedding_model: Sentence-transformers model name
        backend: Vector store backend ('faiss' or 'chroma')
        incremental: Add to existing index or create new
        recursive: Search subdirectories

    Returns:
        Dictionary with pipeline statistics

    Example:
        >>> from ingestion.pipeline import run_pipeline
        >>> stats = run_pipeline(data_dir="data/raw")
        >>> print(f"Processed {stats['total_docs']} documents")
    """
    start_time = datetime.now()

    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "ESILV DOCUMENT INGESTION PIPELINE" + " " * 25 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    # Create configuration
    config = PipelineConfig(
        data_dir=data_dir,
        output_path=output_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        backend=backend,
        incremental=incremental,
        recursive=recursive
    )

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {'error': str(e)}

    # Run pipeline stages
    try:
        # Stage 1: Load documents
        docs = stage_1_load_documents(config)
        if not docs:
            logger.error("No documents to process. Exiting.")
            return {'error': 'No documents found'}

        # Stage 2: Clean text
        cleaned_texts = stage_2_clean_documents(docs)

        # Stage 3: Chunk documents
        chunks = stage_3_chunk_documents(cleaned_texts, docs, config)

        # Stage 4: Generate embeddings
        embeddings = stage_4_generate_embeddings(chunks, config)

        # Stage 5: Create vector store
        store = stage_5_create_vector_store(chunks, embeddings, config)

        # Calculate timing
        duration = datetime.now() - start_time

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Processed {len(docs)} documents in {duration.total_seconds():.2f}s")
        logger.info(f"✓ Created {len(chunks)} chunks")
        logger.info(f"✓ Generated {embeddings.shape[0]} embeddings")
        logger.info(f"✓ Vector store saved to {config.output_path}")
        logger.info("")

        # Return statistics
        return {
            'success': True,
            'total_docs': len(docs),
            'total_chunks': len(chunks),
            'total_embeddings': embeddings.shape[0],
            'output_path': str(config.output_path),
            'duration_seconds': duration.total_seconds(),
            'chunk_stats': get_chunk_stats(chunks),
            'vector_store_stats': store.get_stats()
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return {'error': str(e)}


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for the ingestion pipeline."""

    parser = argparse.ArgumentParser(
        description="ESILV Document Ingestion Pipeline - Process documents into searchable vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m ingestion.pipeline --data-dir data/raw

  # Custom chunk size
  python -m ingestion.pipeline --data-dir data/raw --chunk-size 1000

  # Incremental update (add new docs to existing index)
  python -m ingestion.pipeline --data-dir data/new --incremental

  # Use ChromaDB instead of FAISS
  python -m ingestion.pipeline --data-dir data/raw --backend chroma
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing documents to ingest (default: data/raw)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/vector_db/index',
        help='Output path for vector store (default: data/vector_db/index)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=800,
        help='Target chunk size in characters (default: 800)'
    )

    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=150,
        help='Overlap between chunks in characters (default: 150)'
    )

    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence-transformers model name (default: all-MiniLM-L6-v2)'
    )

    parser.add_argument(
        '--backend',
        type=str,
        choices=['faiss', 'chroma', 'auto'],
        default='faiss',
        help='Vector store backend (default: faiss)'
    )

    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Add to existing index instead of creating new one'
    )

    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories'
    )

    args = parser.parse_args()

    # Run pipeline
    stats = run_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        backend=args.backend,
        incremental=args.incremental,
        recursive=not args.no_recursive
    )

    # Exit with appropriate code
    if 'error' in stats:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
