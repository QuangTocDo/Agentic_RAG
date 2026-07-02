"""
Reranker — optional cross-encoder reranking of retrieval results.
Uses a lightweight cross-encoder model.
"""
from __future__ import annotations


def rerank(query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
    """
    Rerank documents using cross-encoder similarity.
    Falls back to returning documents as-is if model is unavailable.
    """
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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
