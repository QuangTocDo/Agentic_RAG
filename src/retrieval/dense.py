"""
Dense retriever — semantic search via ChromaDB embeddings.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings


def dense_search(query: str, k: int | None = None) -> list[dict]:
    """Run dense (vector) similarity search."""
    from src.indexing.chroma_store import similarity_search

    if k is None:
        k = settings.retrieval_k
    results = similarity_search(query, k=k)
    # Convert LangChain Documents to dicts
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in results
    ]
