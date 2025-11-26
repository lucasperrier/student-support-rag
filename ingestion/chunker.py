"""
Chunking Module

PURPOSE:
This module splits cleaned text into smaller, semantically meaningful chunks optimized
for RAG retrieval. It's the third stage in the ingestion pipeline, taking cleaned text
and producing fixed-size chunks with overlap for better retrieval quality.

INPUTS:
- Cleaned text string from text_cleaning.py
- OR Document object with cleaned text
- Configuration: chunk_size, overlap, separators

OUTPUTS:
- List of Chunk objects, each containing:
  - chunk_text: The actual text content
  - metadata: Source file, page, chunk index, etc.
  - char_count: Number of characters
- Ready for embedding and vector storage

PERSON B INTEGRATION:
Person B doesn't directly call this module - it's used internally by the ingestion
pipeline (pipeline.py). The flow is:
  loader.py (Document) → text_cleaning.py (clean text) → chunker.py (chunks) → embedder.py

EXAMPLE USAGE:
    from ingestion.loader import load_document
    from ingestion.text_cleaning import clean_document
    from ingestion.chunker import chunk_document

    # Load and clean document
    doc = load_document("data/raw/esilv_brochure.pdf")
    cleaned_text = clean_document(doc)

    # Chunk the cleaned text
    chunks = chunk_document(cleaned_text, doc.metadata)

    # Result: List of Chunk objects ready for embedding
    for chunk in chunks[:3]:
        print(f"Chunk {chunk.metadata['chunk_index']}: {len(chunk.text)} chars")

DESIGN DECISIONS:

1. **Chunk Size**:
   - Default: 800 characters (~200 tokens)
   - Reasoning: Balances context vs. precision. Too small = missing context,
     too large = irrelevant info mixed in.
   - Most embedding models handle 512 tokens, we stay comfortably below that.

2. **Overlap**:
   - Default: 150 characters (~40 tokens)
   - Reasoning: Prevents losing context at chunk boundaries. If a sentence is split
     across chunks, overlap ensures both chunks contain the full sentence.

3. **Recursive Splitting**:
   - Try to split on natural boundaries: paragraphs → sentences → words → chars
   - Preserves semantic coherence (don't split mid-sentence if possible)

4. **Metadata Tracking**:
   - Each chunk knows its source file, page number, and position in document
   - Critical for citations: "This info comes from admissions.pdf, page 3"

5. **Flexible Separators**:
   - French and English text have different punctuation (« », etc.)
   - Support custom separators for different languages

TECHNICAL NOTES:
- Uses recursive splitting algorithm similar to LangChain's RecursiveCharacterTextSplitter
- Preserves document structure while meeting size constraints
- Handles edge cases: empty text, very short text, etc.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Chunk:
    """
    Represents a single text chunk ready for embedding.

    This is the output format of the chunker. Each chunk is a semantically
    coherent piece of text with metadata for tracking and citation.

    Attributes:
        text: The actual chunk content
        metadata: Dict with source file, page, chunk_index, etc.
        char_count: Number of characters in the chunk
        token_estimate: Rough estimate of tokens (~char_count / 4)
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    char_count: int = 0
    token_estimate: int = 0

    def __post_init__(self):
        """Calculate char_count and token_estimate after initialization."""
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.token_estimate:
            # Rough estimate: 1 token ≈ 4 characters for English
            # This is approximate but good enough for planning
            self.token_estimate = self.char_count // 4

    def __repr__(self):
        """Pretty representation for debugging."""
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        source = self.metadata.get('filename', 'unknown')
        chunk_idx = self.metadata.get('chunk_index', '?')
        return f"Chunk(source={source}, index={chunk_idx}, chars={self.char_count}, preview='{preview}')"


# =============================================================================
# Core Chunking Algorithm
# =============================================================================

class RecursiveCharacterTextSplitter:
    """
    Splits text recursively on multiple separators to create semantic chunks.

    Algorithm:
    1. Try to split on paragraphs (\\n\\n)
    2. If chunks still too large, split on sentences (\\n or .)
    3. If still too large, split on words (spaces)
    4. Last resort: split on characters

    This preserves semantic boundaries as much as possible.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separator strings to try (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators: try natural boundaries first
        if separators is None:
            self.separators = [
                "\n\n",  # Paragraph breaks (highest priority)
                "\n",    # Line breaks / sentence breaks
                ". ",    # Sentence endings (period + space)
                "! ",    # Exclamation sentence endings
                "? ",    # Question sentence endings
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Word boundaries (spaces)
                ""       # Character-level (last resort)
            ]
        else:
            self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive algorithm.

        Args:
            text: Text to split

        Returns:
            List of text chunks (strings)
        """
        if not text or len(text) == 0:
            return []

        # If text is already small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]

        # Try each separator in order of preference
        for separator in self.separators:
            if separator == "":
                # Last resort: character-level splitting
                return self._split_by_characters(text)

            # Try splitting on this separator
            chunks = self._split_with_separator(text, separator)

            # If all chunks are acceptable size, we're done
            if all(len(chunk) <= self.chunk_size for chunk in chunks):
                return self._merge_small_chunks(chunks)

        # Fallback: character-level split
        return self._split_by_characters(text)

    def _split_with_separator(self, text: str, separator: str) -> List[str]:
        """
        Split text on a separator and recursively split oversized chunks.

        Args:
            text: Text to split
            separator: Separator string to split on

        Returns:
            List of text chunks
        """
        # Split on separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # Character-level

        # Add separator back (except for last split)
        # This preserves the original text structure
        chunks = []
        for i, split in enumerate(splits):
            if i < len(splits) - 1:
                chunks.append(split + separator)
            else:
                chunks.append(split)

        # Merge small chunks and add overlap
        return self._merge_chunks_with_overlap(chunks)

    def _merge_chunks_with_overlap(self, chunks: List[str]) -> List[str]:
        """
        Merge small chunks together and add overlap between chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Merged chunks with overlap
        """
        if not chunks:
            return []

        merged = []
        current_chunk = ""

        for chunk in chunks:
            # If adding this chunk would exceed size, start new chunk
            if len(current_chunk) + len(chunk) > self.chunk_size and current_chunk:
                merged.append(current_chunk)

                # Start new chunk with overlap from previous chunk
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + chunk
            else:
                # Add to current chunk
                current_chunk += chunk

        # Add final chunk
        if current_chunk:
            merged.append(current_chunk)

        return merged

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge very small chunks together to avoid tiny fragments.

        Args:
            chunks: List of chunks that might be too small

        Returns:
            Merged chunks
        """
        if not chunks:
            return []

        # Minimum chunk size: 1/3 of target size
        min_size = self.chunk_size // 3

        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            # If current chunk is too small and merging won't exceed limit
            if len(current) < min_size and len(current) + len(chunk) <= self.chunk_size:
                current += chunk
            else:
                merged.append(current)
                current = chunk

        # Add final chunk
        if current:
            merged.append(current)

        return merged

    def _split_by_characters(self, text: str) -> List[str]:
        """
        Last resort: split by fixed character count.

        Args:
            text: Text to split

        Returns:
            Character-split chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Add overlap if not first chunk
            if start > 0:
                start = max(0, start - self.chunk_overlap)

            chunks.append(text[start:end])
            start = end

        return chunks


# =============================================================================
# Main Chunking Functions
# =============================================================================

def chunk_text(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None
) -> List[Chunk]:
    """
    Chunk a text string into Chunk objects.

    This is the main chunking function for raw text strings.

    Args:
        text: Cleaned text to chunk
        metadata: Optional metadata to attach to all chunks
        chunk_size: Target chunk size in characters (default: 800)
        chunk_overlap: Overlap between chunks in characters (default: 150)
        separators: Custom separators (default: None, uses standard set)

    Returns:
        List of Chunk objects

    Example:
        >>> text = "Long document text here..."
        >>> chunks = chunk_text(text, metadata={'source': 'doc.pdf'})
        >>> len(chunks)
        5
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to chunk_text()")
        return []

    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    # Split text
    text_chunks = splitter.split_text(text)

    # Convert to Chunk objects with metadata
    chunks = []
    for idx, chunk_text in enumerate(text_chunks):
        # Merge base metadata with chunk-specific metadata
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata['chunk_index'] = idx
        chunk_metadata['total_chunks'] = len(text_chunks)

        chunk = Chunk(
            text=chunk_text.strip(),
            metadata=chunk_metadata
        )
        chunks.append(chunk)

    logger.info(f"Chunked text into {len(chunks)} chunks (avg {sum(c.char_count for c in chunks) // len(chunks)} chars/chunk)")

    return chunks


def chunk_document(
    cleaned_text: str,
    doc_metadata: Dict[str, Any],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Chunk]:
    """
    Chunk a cleaned document with metadata preservation.

    This is a convenience wrapper for chunking documents from the pipeline.
    Takes cleaned text and document metadata, returns Chunk objects.

    Args:
        cleaned_text: Cleaned text from text_cleaning.py
        doc_metadata: Metadata dict from Document object (filename, file_type, etc.)
        chunk_size: Target chunk size (default: 800)
        chunk_overlap: Overlap size (default: 150)

    Returns:
        List of Chunk objects

    Example:
        >>> from ingestion.loader import load_document
        >>> from ingestion.text_cleaning import clean_document
        >>> doc = load_document("esilv.pdf")
        >>> cleaned = clean_document(doc)
        >>> chunks = chunk_document(cleaned, doc.metadata)
    """
    if not cleaned_text or not cleaned_text.strip():
        logger.warning(f"Empty text for document: {doc_metadata.get('filename', 'unknown')}")
        return []

    logger.info(f"Chunking document: {doc_metadata.get('filename', 'unknown')} ({len(cleaned_text)} chars)")

    return chunk_text(
        text=cleaned_text,
        metadata=doc_metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


# =============================================================================
# Batch Processing
# =============================================================================

def chunk_documents(
    cleaned_texts: List[str],
    metadata_list: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Chunk]:
    """
    Chunk multiple documents in batch.

    Args:
        cleaned_texts: List of cleaned text strings
        metadata_list: List of metadata dicts (same length as cleaned_texts)
        chunk_size: Target chunk size
        chunk_overlap: Overlap size

    Returns:
        Flattened list of all chunks from all documents

    Example:
        >>> from ingestion.loader import load_documents_from_directory
        >>> from ingestion.text_cleaning import clean_documents
        >>>
        >>> docs = load_documents_from_directory("data/raw")
        >>> cleaned_texts = clean_documents(docs)
        >>> metadata_list = [doc.metadata for doc in docs]
        >>> all_chunks = chunk_documents(cleaned_texts, metadata_list)
    """
    if len(cleaned_texts) != len(metadata_list):
        raise ValueError(f"Mismatch: {len(cleaned_texts)} texts but {len(metadata_list)} metadata dicts")

    all_chunks = []

    for text, metadata in zip(cleaned_texts, metadata_list):
        try:
            chunks = chunk_document(text, metadata, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to chunk document {metadata.get('filename', 'unknown')}: {e}")
            continue

    logger.info(f"Chunked {len(metadata_list)} documents into {len(all_chunks)} total chunks")

    return all_chunks


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Rough approximation: 1 token ≈ 4 characters for English/French.
    This is not exact but good enough for planning chunk sizes.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def get_chunk_stats(chunks: List[Chunk]) -> Dict[str, Any]:
    """
    Get statistics about a list of chunks.

    Useful for debugging and quality assurance.

    Args:
        chunks: List of Chunk objects

    Returns:
        Dict with statistics (count, avg_size, min_size, max_size, etc.)
    """
    if not chunks:
        return {
            'count': 0,
            'avg_chars': 0,
            'min_chars': 0,
            'max_chars': 0,
            'avg_tokens': 0,
            'total_chars': 0
        }

    char_counts = [c.char_count for c in chunks]
    token_counts = [c.token_estimate for c in chunks]

    return {
        'count': len(chunks),
        'avg_chars': sum(char_counts) // len(char_counts),
        'min_chars': min(char_counts),
        'max_chars': max(char_counts),
        'avg_tokens': sum(token_counts) // len(token_counts),
        'total_chars': sum(char_counts)
    }


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    """
    Quick test/demo of chunking.
    Run this file directly to test: python -m ingestion.chunker
    """

    print("Chunking Module Test")
    print("=" * 80)

    # Test 1: Basic chunking
    print("\n1. Basic Text Chunking:")
    test_text = """
    ESILV is a leading French engineering school located in Paris La Défense.
    The school specializes in digital engineering and computer science.

    Our programs include a 5-year engineering degree, specialized masters,
    and international exchange programs. We have strong industry partnerships
    with companies like Amazon, Google, and Capgemini.

    Admissions are competitive and based on academic excellence. Students
    can apply through the Parcoursup platform or directly for international
    students. The application deadline is typically in March for fall admission.
    """

    chunks = chunk_text(test_text, metadata={'source': 'test'}, chunk_size=200, chunk_overlap=50)
    print(f"Created {len(chunks)} chunks from {len(test_text)} chars")
    for chunk in chunks:
        print(f"\n  Chunk {chunk.metadata['chunk_index']}: {chunk.char_count} chars, ~{chunk.token_estimate} tokens")
        print(f"  Preview: {chunk.text[:100]}...")

    # Test 2: Chunk statistics
    print("\n2. Chunk Statistics:")
    stats = get_chunk_stats(chunks)
    print(f"  Total chunks: {stats['count']}")
    print(f"  Avg size: {stats['avg_chars']} chars (~{stats['avg_tokens']} tokens)")
    print(f"  Min size: {stats['min_chars']} chars")
    print(f"  Max size: {stats['max_chars']} chars")

    # Test 3: Real documents from data/raw
    print("\n3. Testing with real documents from data/raw/:")
    from pathlib import Path
    from ingestion.loader import load_documents_from_directory
    from ingestion.text_cleaning import clean_documents

    raw_dir = Path(__file__).parent.parent / "data" / "raw"

    if raw_dir.exists():
        docs = load_documents_from_directory(str(raw_dir))
        if docs:
            print(f"Loaded {len(docs)} documents")

            # Clean and chunk
            cleaned_texts = clean_documents(docs)
            metadata_list = [doc.metadata for doc in docs]
            all_chunks = chunk_documents(cleaned_texts, metadata_list, chunk_size=500, chunk_overlap=100)

            print(f"\nChunked into {len(all_chunks)} total chunks")

            # Show stats per document
            for doc in docs:
                doc_chunks = [c for c in all_chunks if c.metadata.get('filename') == doc.metadata['filename']]
                stats = get_chunk_stats(doc_chunks)
                print(f"\n  {doc.metadata['filename']}:")
                print(f"    Chunks: {stats['count']}")
                print(f"    Avg: {stats['avg_chars']} chars (~{stats['avg_tokens']} tokens)")

                # Show first chunk as example
                if doc_chunks:
                    print(f"    First chunk preview: {doc_chunks[0].text[:100]}...")
        else:
            print("No documents found in data/raw/")
    else:
        print(f"Directory not found: {raw_dir}")
