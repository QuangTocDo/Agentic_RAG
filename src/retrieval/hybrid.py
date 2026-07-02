"""
Hybrid retriever — combines Dense (vector) + BM25 (keyword) via Reciprocal Rank Fusion.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings


def hybrid_search(query: str, k: int | None = None) -> list[dict]:
    """
    Run both dense and BM25 search, then fuse results with RRF.
    """
    from src.retrieval.dense import dense_search
    from src.indexing.bm25_index import BM25Index

    if k is None:
        k = settings.retrieval_k
    rrf_k = settings.rrf_k

    # Dense results
    dense_results = dense_search(query, k=k * 2)

    # BM25 results
    bm25 = BM25Index()
    try:
        bm25.load()
        bm25_results = bm25.search(query, top_k=k * 2)
    except FileNotFoundError:
        print("  ⚠️  BM25 index not found, using dense search only")
        bm25_results = []

    # Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=rrf_k)
    return fused[:k]


def reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Combine multiple ranked result lists using RRF.
    
    RRF score = Σ 1 / (k + rank_i)
    
    Documents are identified by their page_content.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc["page_content"][:200]  # Use first 200 chars as ID
            if key not in doc_map:
                doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    # Sort by RRF score descending
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)

    return [
        {**doc_map[key], "rrf_score": scores[key]}
        for key in sorted_keys
    ]
