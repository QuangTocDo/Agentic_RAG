"""
Reranker — optional cross-encoder reranking of retrieval results.
Uses a lightweight cross-encoder model.
"""
from __future__ import annotations

_reranker_model = None


def rerank(query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
    """
    Rerank documents using cross-encoder similarity.
    Falls back to returning documents as-is if model is unavailable.
    """
    try:
        model = _get_reranker_model()

        pairs = [(query, doc["page_content"]) for doc in documents]
        scores = model.predict(pairs)

        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    except Exception as e:
        print(f"  ⚠️  Reranker unavailable ({e}), returning original order")
        return documents[:top_k]


def _get_reranker_model():
    """Return a cached cross-encoder model."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model
