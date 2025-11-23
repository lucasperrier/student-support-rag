# FAISS/Chroma wrapper
from typing import List, Tuple, Dict

def search(query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
    """
    Stub: Person A must implement this.
    Searches the vector DB for relevant chunks.
    :param query: User query string.
    :param top_k: Number of top results.
    :return: List of (chunk_text, metadata_dict) tuples.
    """
    raise NotImplementedError("Person A: Implement vector DB search (e.g., using FAISS/Chroma). Return list of (chunk_text, metadata).")