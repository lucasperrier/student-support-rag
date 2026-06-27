"""
Text Cleaning Module

PURPOSE:
This module cleans and normalizes text extracted from documents (PDFs, HTML, TXT).
It's the second stage in the ingestion pipeline, preparing raw extracted text for
chunking and embedding by removing noise, fixing encoding issues, and normalizing
whitespace while preserving semantic structure.

INPUTS:
- Document object from loader.py containing raw extracted text
- OR raw text string

OUTPUTS:
- Cleaned text string ready for chunking
- Preserves paragraph structure and sentence boundaries
- Removes noise (extra whitespace, control chars, artifacts)
- Fixes common PDF extraction issues (hyphenation, line breaks)

INTEGRATION:
This module is used internally by the ingestion pipeline (pipeline.py). The flow is:
  loader.py (Document) → text_cleaning.py (clean text) → chunker.py (chunks)

EXAMPLE USAGE:
    from ingestion.loader import load_document
    from ingestion.text_cleaning import clean_document, clean_text

    # Clean a Document object
    doc = load_document("data/raw/esilv_brochure.pdf")
    cleaned_text = clean_document(doc)

    # Or clean raw text directly
    raw_text = "Some   text   with    extra spaces\\n\\nand\\tweird\\twhitespace"
    cleaned = clean_text(raw_text)

DESIGN DECISIONS:
1. Non-destructive: We clean aggressively but preserve semantic content (sentences,
   paragraphs). Better to keep some noise than lose meaningful structure.

2. PDF-Aware: PDFs often have hyphenated words split across lines ("compu-\\nter").
   We detect and fix these patterns.

3. Unicode Normalization: Convert all text to NFC form to handle accents consistently.
   Important for French text (ESILV is in France).

4. Smart Whitespace: Collapse multiple spaces but preserve paragraph breaks.
   Two+ newlines = paragraph boundary (keep), single newline = join.

5. Configurable: Main function has options for aggressive cleaning if needed,
   but defaults are tuned for RAG quality.
"""

import re
import unicodedata
import logging
from typing import Optional

# Import Document class from loader
from .loader import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Core Cleaning Functions
# =============================================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to consistent form.

    Uses NFC (Canonical Decomposition, followed by Canonical Composition).
    This ensures characters like 'é' are represented consistently, which is
    important for French text and for embedding models.

    Args:
        text: Raw text with potentially inconsistent unicode

    Returns:
        Text with normalized unicode characters

    Example:
        >>> normalize_unicode("café")  # Even if 'é' was decomposed
        'café'
    """
    if not text:
        return text

    # NFC = composed form (é instead of e + combining accent)
    return unicodedata.normalize('NFC', text)


def remove_control_characters(text: str) -> str:
    """
    Remove control characters and non-printable characters.

    Keeps: letters, numbers, punctuation, spaces, newlines, tabs
    Removes: null bytes, form feeds, vertical tabs, etc.

    Args:
        text: Text potentially containing control characters

    Returns:
        Text with control characters removed

    Technical Notes:
        - Preserves \n (newline) and \t (tab) as they're semantically meaningful
        - Removes other control chars in Unicode category 'Cc' (Other, control)
    """
    if not text:
        return text

    # Keep common whitespace (\n, \t, \r, space) but remove other control chars
    cleaned = []
    for char in text:
        # Allow normal whitespace
        if char in '\n\r\t ':
            cleaned.append(char)
        # Allow printable characters (not in control category)
        elif not unicodedata.category(char).startswith('C'):
            cleaned.append(char)
        # Skip control characters

    return ''.join(cleaned)


def fix_pdf_hyphenation(text: str) -> str:
    """
    Fix hyphenated words broken across lines in PDFs.

    PDFs often split words at line boundaries:
        "compu-\nter science" → "computer science"
        "re-\nsearch" → "research"

    Args:
        text: Text with potential hyphenated line breaks

    Returns:
        Text with hyphenated words rejoined

    Technical Notes:
        - Only joins if hyphen is at end of line followed by newline
        - Preserves intentional hyphens ("state-of-the-art" stays intact)
        - Pattern: word char + hyphen + optional spaces + newline + word char
    """
    if not text:
        return text

    # Match: letter/number, then hyphen, then newline, then letter
    # Group 1: text before hyphen, Group 2: text after newline
    pattern = r'(\w)-\s*\n\s*(\w)'

    # Replace "word-\npart" with "wordpart"
    # Use lambda to join the captured groups
    cleaned = re.sub(pattern, r'\1\2', text)

    return cleaned


def normalize_whitespace(text: str, preserve_paragraphs: bool = True) -> str:
    """
    Normalize whitespace while preserving document structure.

    - Converts tabs to spaces
    - Collapses multiple spaces into single space
    - Collapses multiple newlines into double newline (paragraph break)
    - Removes leading/trailing whitespace from lines

    Args:
        text: Text with irregular whitespace
        preserve_paragraphs: If True, keep paragraph breaks (double newlines).
                           If False, collapse all to single spaces.

    Returns:
        Text with normalized whitespace

    Example:
        >>> normalize_whitespace("Hello    world\\n\\n\\nNew paragraph")
        "Hello world\\n\\nNew paragraph"
    """
    if not text:
        return text

    # Step 1: Convert tabs to spaces
    text = text.replace('\t', ' ')

    # Step 2: Remove carriage returns (Windows line endings)
    text = text.replace('\r', '')

    # Step 3: Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    # Step 4: Collapse multiple spaces into single space
    text = re.sub(r' +', ' ', text)

    if preserve_paragraphs:
        # Step 5: Normalize paragraph breaks (3+ newlines → 2 newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Step 6: Remove single newlines (join sentences in same paragraph)
        # But preserve double newlines (paragraph boundaries)
        # Strategy: temporarily replace double newlines with placeholder
        text = text.replace('\n\n', '<<<PARAGRAPH_BREAK>>>')
        text = text.replace('\n', ' ')  # Single newlines become spaces
        text = text.replace('<<<PARAGRAPH_BREAK>>>', '\n\n')  # Restore paragraphs
    else:
        # Collapse all newlines to spaces
        text = text.replace('\n', ' ')
        text = re.sub(r' +', ' ', text)  # Clean up double spaces

    # Final trim
    return text.strip()


def remove_page_markers(text: str) -> str:
    """
    Remove page markers inserted by PDF loaders.

    The loader.py module adds markers like "[Page 1]" to PDFs.
    These are useful for citations but should be removed before chunking.

    Args:
        text: Text potentially containing [Page N] markers

    Returns:
        Text with page markers removed

    Note:
        Page information is preserved in metadata, so removing these
        markers doesn't lose information.
    """
    if not text:
        return text

    # Remove [Page N] markers
    text = re.sub(r'\[Page \d+\]', '', text)

    # Clean up any extra whitespace left behind
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def remove_urls_and_emails(text: str, remove: bool = False) -> str:
    """
    Optionally remove URLs and email addresses.

    Args:
        text: Text potentially containing URLs/emails
        remove: If True, remove them. If False, leave them.

    Returns:
        Text with URLs/emails removed (if remove=True)

    Note:
        Default is False because URLs and emails might be important
        for ESILV content (contact info, application links, etc.).
    """
    if not text or not remove:
        return text

    # Remove URLs (http, https, www)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+', '', text)

    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    return text


def remove_extra_punctuation(text: str) -> str:
    """
    Clean up excessive punctuation while preserving sentence structure.

    - Removes multiple consecutive punctuation marks (e.g., "!!!!" → "!")
    - Preserves ellipsis "..." as single unit
    - Preserves normal sentence punctuation

    Args:
        text: Text with potentially excessive punctuation

    Returns:
        Text with normalized punctuation
    """
    if not text:
        return text

    # Preserve ellipsis first (replace with placeholder)
    text = text.replace('...', '<<<ELLIPSIS>>>')

    # Remove repeated punctuation (!!!! → !, ??? → ?)
    text = re.sub(r'([!?.,;:])\1+', r'\1', text)

    # Restore ellipsis
    text = text.replace('<<<ELLIPSIS>>>', '...')

    return text


# =============================================================================
# Main Cleaning Function
# =============================================================================

def clean_text(
    text: str,
    remove_page_numbers: bool = True,
    preserve_paragraphs: bool = True,
    fix_hyphenation: bool = True,
    remove_urls: bool = False,
    aggressive: bool = False
) -> str:
    """
    Main text cleaning function - applies all cleaning steps.

    This is the primary function to use for cleaning raw text.
    It applies multiple cleaning operations in the correct order.

    Args:
        text: Raw text to clean
        remove_page_numbers: Remove [Page N] markers (default: True)
        preserve_paragraphs: Keep paragraph breaks vs. collapse to single space (default: True)
        fix_hyphenation: Fix PDF hyphenation issues (default: True)
        remove_urls: Remove URLs and emails (default: False, keep them)
        aggressive: If True, apply more aggressive cleaning (default: False)

    Returns:
        Cleaned text ready for chunking

    Example:
        >>> raw = "Some   messy\\n\\n\\ntext  with [Page 1] markers"
        >>> clean_text(raw)
        "Some messy\\n\\ntext with markers"
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to clean_text()")
        return ""

    original_length = len(text)

    # Step 1: Unicode normalization (handle accents, special chars)
    text = normalize_unicode(text)

    # Step 2: Remove control characters
    text = remove_control_characters(text)

    # Step 3: Fix PDF hyphenation issues
    if fix_hyphenation:
        text = fix_pdf_hyphenation(text)

    # Step 4: Remove page markers
    if remove_page_numbers:
        text = remove_page_markers(text)

    # Step 5: Remove URLs/emails if requested
    if remove_urls:
        text = remove_urls_and_emails(text, remove=True)

    # Step 6: Normalize whitespace
    text = normalize_whitespace(text, preserve_paragraphs=preserve_paragraphs)

    # Step 7: Clean up punctuation
    text = remove_extra_punctuation(text)

    # Step 8: Final whitespace cleanup
    text = text.strip()

    # Log cleaning statistics
    cleaned_length = len(text)
    reduction = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
    logger.info(f"Text cleaned: {original_length} → {cleaned_length} chars ({reduction:.1f}% reduction)")

    return text


def clean_document(doc: Document, **kwargs) -> str:
    """
    Clean text from a Document object.

    This is a convenience wrapper around clean_text() that works
    directly with Document objects from loader.py.

    Args:
        doc: Document object from loader.py
        **kwargs: Passed to clean_text() (remove_page_numbers, preserve_paragraphs, etc.)

    Returns:
        Cleaned text string

    Example:
        >>> from ingestion.loader import load_document
        >>> doc = load_document("data/raw/esilv.pdf")
        >>> cleaned = clean_document(doc)
    """
    if doc is None:
        logger.error("Cannot clean None document")
        return ""

    if not doc.text:
        logger.warning(f"Document {doc.metadata.get('filename', 'unknown')} has no text")
        return ""

    logger.info(f"Cleaning document: {doc.metadata.get('filename', 'unknown')}")

    return clean_text(doc.text, **kwargs)


# =============================================================================
# Batch Processing
# =============================================================================

def clean_documents(documents: list[Document], **kwargs) -> list[str]:
    """
    Clean multiple documents in batch.

    Args:
        documents: List of Document objects from loader.py
        **kwargs: Passed to clean_text()

    Returns:
        List of cleaned text strings (same order as input)

    Example:
        >>> from ingestion.loader import load_documents_from_directory
        >>> docs = load_documents_from_directory("data/raw")
        >>> cleaned_texts = clean_documents(docs)
    """
    cleaned = []

    for doc in documents:
        try:
            cleaned_text = clean_document(doc, **kwargs)
            cleaned.append(cleaned_text)
        except Exception as e:
            logger.error(f"Failed to clean document {doc.metadata.get('filename', 'unknown')}: {e}")
            # Append empty string for failed documents to maintain order
            cleaned.append("")

    logger.info(f"Cleaned {len(cleaned)} documents")
    return cleaned


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    """
    Quick test/demo of text cleaning.
    Run this file directly to test: python -m ingestion.text_cleaning
    """

    print("Text Cleaning Test")
    print("=" * 60)

    # Test 1: Basic cleaning
    raw_text = """
    This   is   some    messy  text.


    With    lots of    extra   spaces  and
    newlines.

    [Page 1]

    And page markers!
    """

    print("\n1. Basic Cleaning:")
    print("BEFORE:", repr(raw_text[:100]))
    cleaned = clean_text(raw_text)
    print("AFTER:", repr(cleaned[:100]))
    print(f"Length: {len(raw_text)} → {len(cleaned)}")

    # Test 2: PDF hyphenation
    pdf_text = """
    Computer sci-
    ence is the study of compu-
    tation and information.
    """

    print("\n2. PDF Hyphenation Fix:")
    print("BEFORE:", repr(pdf_text))
    cleaned = clean_text(pdf_text)
    print("AFTER:", repr(cleaned))

    # Test 3: Unicode normalization
    unicode_text = "café résumé naïve"  # French accents
    print("\n3. Unicode Normalization:")
    print("BEFORE:", unicode_text)
    cleaned = clean_text(unicode_text)
    print("AFTER:", cleaned)

    # Test 4: Real document test
    print("\n4. Testing with real documents from data/raw/:")
    from pathlib import Path
    from .loader import load_documents_from_directory

    raw_dir = Path(__file__).parent.parent / "data" / "raw"

    if raw_dir.exists():
        docs = load_documents_from_directory(str(raw_dir))
        if docs:
            print(f"Loaded {len(docs)} documents")
            cleaned_texts = clean_documents(docs)

            for doc, cleaned in zip(docs, cleaned_texts):
                if cleaned:
                    print(f"\n  File: {doc.metadata['filename']}")
                    print(f"  Original: {len(doc.text)} chars")
                    print(f"  Cleaned: {len(cleaned)} chars")
                    print(f"  Preview: {cleaned[:150]}...")
        else:
            print("No documents found in data/raw/")
    else:
        print(f"Directory not found: {raw_dir}")
        print("Add some documents to test.")
