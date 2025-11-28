# PDF/HTML loading : lit les fichiers PDF/HTML et extrait le texte.
"""
Document Loader Module

PURPOSE:
This module loads documents from various file formats (PDF, HTML, TXT) and extracts
their text content. It's the first stage in the ingestion pipeline, converting raw
files into structured Document objects that can be processed by downstream components.

INPUTS:
- file_path: Path to a document file (PDF, HTML, TXT, etc.)
- File must exist and be readable
- Supported formats: .pdf, .html, .htm, .txt, .md

OUTPUTS:
- Document object containing:
  - text: Extracted text content (str)
  - metadata: Dict with filename, file_type, page_count (for PDFs), etc.
  - source: Original file path
- Returns None if file cannot be loaded (with error logged)

PERSON B INTEGRATION:
Person B doesn't directly call this module - it's used internally by the ingestion
pipeline (pipeline.py). However, Person B should know that uploaded files flow through:
  User Upload (UI) → save to data/raw/ → pipeline.py → loader.py → text extraction

The Document objects produced here feed into text_cleaning.py next.

EXAMPLE USAGE:
    from ingestion.loader import load_document

    # Load a PDF
    doc = load_document("data/raw/esilv_brochure.pdf")
    if doc:
        print(f"Loaded {len(doc.text)} characters from {doc.metadata['filename']}")
        print(f"Pages: {doc.metadata.get('page_count', 'N/A')}")

    # Load HTML
    doc = load_document("data/raw/admissions.html")
    if doc:
        print(f"Text preview: {doc.text[:200]}...")

DESIGN DECISIONS:
1. Factory Pattern: Single load_document() entry point that delegates to specialized
   loaders based on file extension. This makes it easy to add new formats later.

2. Document Dataclass: Standardized structure ensures consistent data flow through
   the pipeline. All downstream components can rely on this format.

3. Error Handling: Returns None on failure rather than raising exceptions. This allows
   the pipeline to skip corrupted files and continue processing others.

4. Metadata Preservation: Keeps filename, file type, page numbers, etc. This metadata
   is crucial for citation generation and debugging.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# PDF extraction
try:
    from pypdf import PdfReader
except ImportError:
    # Fallback for older pypdf versions
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

# HTML extraction
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Represents a loaded document with its content and metadata.

    This is the standard format used throughout the ingestion pipeline.
    All loader functions return Document objects.

    Attributes:
        text: Extracted text content (cleaned of extra whitespace)
        metadata: Dict containing filename, file_type, page_count, etc.
        source: Original file path (as string)
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self):
        """Validate document has minimum required data."""
        if not self.text or not self.text.strip():
            logger.warning(f"Document from {self.source} has empty text content")


# =============================================================================
# File Type Detection
# =============================================================================

def _get_file_type(file_path: Path) -> str:
    """
    Determine file type from extension.

    Args:
        file_path: Path object to the file

    Returns:
        File type string (e.g., 'pdf', 'html', 'txt')
    """
    extension = file_path.suffix.lower()

    # Map extensions to file types
    type_mapping = {
        '.pdf': 'pdf',
        '.html': 'html',
        '.htm': 'html',
        '.txt': 'text',
        '.md': 'text',  # Treat markdown as plain text
        '.text': 'text',
    }

    return type_mapping.get(extension, 'unknown')


# =============================================================================
# Specialized Loaders for Each File Type
# =============================================================================

def _load_pdf(file_path: Path) -> Optional[Document]:
    """
    Load and extract text from a PDF file.

    Uses pypdf library to read PDF pages and extract text. Preserves page
    information in metadata for citation purposes.

    Args:
        file_path: Path to PDF file

    Returns:
        Document object with extracted text and metadata, or None on failure

    Technical Notes:
        - Some PDFs are scanned images without OCR → will return empty text
        - Complex layouts (tables, multi-column) may have garbled text order
        - For production, consider adding OCR support (tesseract) for scanned PDFs
    """
    if PdfReader is None:
        logger.error("pypdf/PyPDF2 not installed. Cannot load PDF files.")
        return None

    try:
        reader = PdfReader(str(file_path))

        # Extract text from all pages
        text_parts = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Add page marker for reference (useful for citations)
                    text_parts.append(f"[Page {page_num}]\n{page_text}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num} of {file_path.name}: {e}")
                continue

        # Combine all pages
        full_text = "\n\n".join(text_parts)

        # Build metadata
        metadata = {
            'filename': file_path.name,
            'file_type': 'pdf',
            'page_count': len(reader.pages),
            'file_size_bytes': file_path.stat().st_size,
        }

        # Add PDF metadata if available (title, author, etc.)
        if reader.metadata:
            try:
                if reader.metadata.title:
                    metadata['title'] = reader.metadata.title
                if reader.metadata.author:
                    metadata['author'] = reader.metadata.author
            except Exception:
                pass  # Some PDFs have malformed metadata

        logger.info(f"✓ Loaded PDF: {file_path.name} ({len(reader.pages)} pages, {len(full_text)} chars)")

        return Document(
            text=full_text,
            metadata=metadata,
            source=str(file_path)
        )

    except Exception as e:
        logger.error(f"✗ Failed to load PDF {file_path.name}: {e}")
        return None


def _load_html(file_path: Path) -> Optional[Document]:
    """
    Load and extract text from an HTML file.

    Uses BeautifulSoup to parse HTML and extract clean text content.
    Removes scripts, styles, and other non-content elements.

    Args:
        file_path: Path to HTML file

    Returns:
        Document object with extracted text and metadata, or None on failure

    Technical Notes:
        - Strips all JavaScript, CSS, and HTML tags
        - Preserves text structure (paragraphs separated by newlines)
        - For web scraping, use requests + BeautifulSoup instead of file reading
    """
    try:
        # Read HTML file with encoding detection
        try:
            html_content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            html_content = file_path.read_text(encoding='latin-1')

        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove script and style elements (they contain code, not content)
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract text
        text = soup.get_text(separator='\n', strip=True)

        # Clean up excessive newlines (HTML often has lots of whitespace)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)

        # Build metadata
        metadata = {
            'filename': file_path.name,
            'file_type': 'html',
            'file_size_bytes': file_path.stat().st_size,
        }

        # Try to extract HTML title tag
        if soup.title and soup.title.string:
            metadata['title'] = soup.title.string.strip()

        logger.info(f"✓ Loaded HTML: {file_path.name} ({len(clean_text)} chars)")

        return Document(
            text=clean_text,
            metadata=metadata,
            source=str(file_path)
        )

    except Exception as e:
        logger.error(f"✗ Failed to load HTML {file_path.name}: {e}")
        return None


def _load_text(file_path: Path) -> Optional[Document]:
    """
    Load a plain text file.

    Handles .txt, .md, and other plain text formats.

    Args:
        file_path: Path to text file

    Returns:
        Document object with text content and metadata, or None on failure

    Technical Notes:
        - Tries UTF-8 first, falls back to latin-1 for legacy files
        - Preserves original formatting (newlines, spacing)
        - Markdown is treated as plain text (not rendered to HTML)
    """
    try:
        # Read text file with encoding detection
        try:
            text = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {file_path.name}, trying latin-1")
            text = file_path.read_text(encoding='latin-1')

        # Build metadata
        metadata = {
            'filename': file_path.name,
            'file_type': 'text',
            'file_size_bytes': file_path.stat().st_size,
        }

        logger.info(f"✓ Loaded text file: {file_path.name} ({len(text)} chars)")

        return Document(
            text=text,
            metadata=metadata,
            source=str(file_path)
        )

    except Exception as e:
        logger.error(f"✗ Failed to load text file {file_path.name}: {e}")
        return None


# =============================================================================
# Main Entry Point
# =============================================================================

def load_document(file_path: str) -> Optional[Document]:
    """
    Load a document from any supported file format.

    This is the main entry point for document loading. It automatically detects
    the file type and delegates to the appropriate loader function.

    Args:
        file_path: Path to document file (as string or Path object)

    Returns:
        Document object containing extracted text and metadata
        Returns None if file doesn't exist, isn't supported, or fails to load

    Supported Formats:
        - PDF (.pdf)
        - HTML (.html, .htm)
        - Plain text (.txt, .md)

    Example:
        >>> doc = load_document("data/raw/esilv_brochure.pdf")
        >>> if doc:
        ...     print(f"Loaded {doc.metadata['page_count']} pages")
        ...     print(doc.text[:200])  # Preview first 200 chars
    """
    # Convert to Path object for easier handling
    path = Path(file_path)

    # Validate file exists
    if not path.exists():
        logger.error(f"✗ File not found: {file_path}")
        return None

    if not path.is_file():
        logger.error(f"✗ Not a file: {file_path}")
        return None

    # Detect file type
    file_type = _get_file_type(path)

    # Delegate to appropriate loader
    if file_type == 'pdf':
        return _load_pdf(path)
    elif file_type == 'html':
        return _load_html(path)
    elif file_type == 'text':
        return _load_text(path)
    else:
        logger.error(f"✗ Unsupported file type: {path.suffix} for {path.name}")
        return None


def load_documents_from_directory(directory_path: str, recursive: bool = False) -> list[Document]:
    """
    Load all supported documents from a directory.

    This is a convenience function for batch processing. It scans a directory
    for all supported file types and loads them.

    Args:
        directory_path: Path to directory containing documents
        recursive: If True, search subdirectories recursively

    Returns:
        List of Document objects (successfully loaded files only)
        Skips files that fail to load (logs errors but continues)

    Example:
        >>> docs = load_documents_from_directory("data/raw")
        >>> print(f"Loaded {len(docs)} documents")
        >>> for doc in docs:
        ...     print(f"- {doc.metadata['filename']}: {len(doc.text)} chars")
    """
    directory = Path(directory_path)

    if not directory.exists() or not directory.is_dir():
        logger.error(f"✗ Directory not found: {directory_path}")
        return []

    # Find all supported files
    supported_extensions = ['.pdf', '.html', '.htm', '.txt', '.md']
    documents = []

    # Choose search pattern based on recursive flag
    pattern = '**/*' if recursive else '*'

    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            doc = load_document(str(file_path))
            if doc:
                documents.append(doc)

    logger.info(f"✓ Loaded {len(documents)} documents from {directory_path}")
    return documents


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    """
    Quick test/demo of the document loader.
    Run this file directly to test: python -m ingestion.loader
    """
    import sys

    print("Document Loader Test")
    print("=" * 60)

    # Test with data/raw directory if it exists
    raw_dir = Path(__file__).parent.parent / "data" / "raw"

    if raw_dir.exists():
        print(f"\nLoading documents from: {raw_dir}")
        docs = load_documents_from_directory(str(raw_dir))

        if docs:
            print(f"\n✓ Successfully loaded {len(docs)} documents:")
            for doc in docs:
                print(f"\n  File: {doc.metadata['filename']}")
                print(f"  Type: {doc.metadata['file_type']}")
                print(f"  Size: {len(doc.text)} characters")
                if 'page_count' in doc.metadata:
                    print(f"  Pages: {doc.metadata['page_count']}")
                print(f"  Preview: {doc.text[:150]}...")
        else:
            print("\n✗ No documents loaded. Add some files to data/raw/ and try again.")
    else:
        print(f"\n✗ Directory not found: {raw_dir}")
        print("  Create it and add some documents to test.")

