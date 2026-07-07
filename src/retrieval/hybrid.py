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
    from src.indexing.bm25_index import get_bm25_index

    if k is None:
        k = settings.retrieval_k
    rrf_k = settings.rrf_k

    # Dense results
    dense_results = dense_search(query, k=k * 2)

    # BM25 results
    bm25 = get_bm25_index()
    if bm25.bm25 is not None:
        bm25_results = bm25.search(query, top_k=k * 2)
    else:
        print("  ⚠️  BM25 index not loaded or empty, using dense search only")
        bm25_results = []

    # Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=rrf_k)
    expanded = _expand_with_graph(fused[:k], max_hops=settings.graph_max_hops)
    combined = _deduplicate(fused + expanded)

    if settings.use_reranker and combined:
        from src.retrieval.reranker import rerank

        return rerank(query, combined, top_k=k)
    return combined[:k]


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


def _expand_with_graph(seed_docs: list[dict], max_hops: int) -> list[dict]:
    """Add related legal articles from the persisted cross-reference graph."""
    if not seed_docs or max_hops <= 0:
        return []
    try:
        from src.retrieval.graph import load_graph

        graph = load_graph()
        return graph.get_related(seed_docs, max_hops=max_hops)
    except FileNotFoundError:
        print("  ⚠️  Legal graph not found, skipping graph expansion")
        return []
    except ImportError as e:
        print(f"  ⚠️  Graph retrieval unavailable ({e})")
        return []


def _deduplicate(documents: list[dict]) -> list[dict]:
    """Keep first occurrence of each retrieved chunk."""
    seen = set()
    unique = []
    for doc in documents:
        meta = doc.get("metadata", {})
        key = (
            meta.get("source"),
            meta.get("article"),
            meta.get("sub_chunk"),
            doc.get("page_content", "")[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique
