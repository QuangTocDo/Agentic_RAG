"""
ChromaDB vector store — persistent local storage for legal document embeddings.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings


def get_chroma_store():
    """Create or open a persistent Chroma vector store."""
    from langchain_community.vectorstores import Chroma
    from src.indexing.embeddings import get_embedding_function

    persist_dir = settings.chroma_persist_dir
    collection = settings.chroma_collection

    return Chroma(
        collection_name=collection,
        embedding_function=get_embedding_function(),
        persist_directory=persist_dir,
    )


def add_documents(chunks: list[dict]) -> None:
    """Add chunked documents to ChromaDB.
    Each chunk is a dict with 'page_content' and 'metadata'.
    """
    from langchain_core.documents import Document

    store = get_chroma_store()
    docs = [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in chunks
    ]
    store.add_documents(docs)
    print(f"  ✅ Added {len(docs)} chunks to ChromaDB")


def similarity_search(query: str, k: int | None = None) -> list:
    """Search ChromaDB for similar documents."""
    if k is None:
        k = settings.retrieval_k
    store = get_chroma_store()
    return store.similarity_search(query, k=k)
