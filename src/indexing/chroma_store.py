"""
ChromaDB vector store — persistent local storage for legal document embeddings.
"""
from __future__ import annotations
import hashlib
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings


_chroma_store = None


def get_chroma_store():
    """Create or return a cached persistent Chroma vector store."""
    global _chroma_store
    if _chroma_store is not None:
        return _chroma_store

    from langchain_chroma import Chroma
    from src.indexing.embeddings import get_embedding_function

    persist_dir = settings.chroma_persist_dir
    collection = settings.chroma_collection

    _chroma_store = Chroma(
        collection_name=collection,
        embedding_function=get_embedding_function(),
        persist_directory=persist_dir,
    )
    return _chroma_store


def reset_collection() -> None:
    """Delete all documents in the current Chroma collection so a full ingest starts cleanly."""
    store = get_chroma_store()
    try:
        data = store.get()
        ids = data.get("ids", [])
        if ids:
            # Delete documents in batches to avoid exceeding limits
            batch_size = 4000
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                store.delete(ids=batch_ids)
            print(f"  🧹 Deleted {len(ids)} existing documents from Chroma collection.")
    except Exception as e:
        print(f"  ⚠️  Could not reset Chroma collection ({e})")


def add_documents(chunks: list[dict], ids: list[str] | None = None) -> None:
    """Add chunked documents to ChromaDB in batches to prevent exceeding max batch size."""
    from langchain_core.documents import Document

    store = get_chroma_store()
    docs = [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in chunks
    ]
    if ids is None:
        ids = [c.get("metadata", {}).get("chunk_id") or _stable_chunk_id(c, i) for i, c in enumerate(chunks)]
    
    # ChromaDB has a maximum batch size limit of 5461. Using 4000 for safety.
    batch_size = 4000
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        store.add_documents(batch_docs, ids=batch_ids)
        
    print(f"  ✅ Added {len(docs)} chunks to ChromaDB in batches of {batch_size}")


def similarity_search(query: str, k: int | None = None) -> list:
    """Search ChromaDB for similar documents."""
    if k is None:
        k = settings.retrieval_k
    store = get_chroma_store()
    return store.similarity_search(query, k=k)


def _stable_chunk_id(chunk: dict, index: int) -> str:
    """Build a repeatable Chroma ID from source metadata and chunk content."""
    meta = chunk.get("metadata", {})
    identity = "|".join(
        [
            str(meta.get("source", "")),
            str(meta.get("filename", "")),
            str(meta.get("article", "")),
            str(meta.get("sub_chunk", index)),
            chunk.get("page_content", ""),
        ]
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()
