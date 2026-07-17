"""
Hybrid retriever — combines Dense (vector) + BM25 (keyword) via Reciprocal Rank Fusion.
"""
from __future__ import annotations
import re
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings

_curated_chunks: list[dict] | None = None

_STOPWORDS = {
    "anh", "chị", "cho", "có", "của", "cần", "các", "cái", "căn", "cứ",
    "gì", "hay", "khi", "là", "mà", "một", "nào", "như", "những", "này",
    "pháp", "quy", "ra", "sao", "sau", "theo", "thì", "thế", "trong",
    "tại", "việt", "và", "về", "được", "định", "để",
}


def hybrid_search(query: str, k: int | None = None) -> list[dict]:
    """
    Run exact legal lookup, dense and BM25 search, then fuse results with RRF.
    """
    from src.retrieval.dense import dense_search
    from src.indexing.bm25_index import get_bm25_index

    if k is None:
        k = settings.retrieval_k
    rrf_k = settings.rrf_k
    candidate_k = max(k * 4, 12)

    # Curated local laws are small but high quality; keep them in the candidate pool.
    curated_results = _curated_local_matches(query, limit=k)

    # Dense results
    try:
        dense_results = dense_search(query, k=candidate_k)
    except Exception as e:
        print(f"  ⚠️  Dense search unavailable ({e}), using curated/BM25 search only")
        dense_results = []

    # BM25 results
    bm25 = get_bm25_index()
    if bm25.bm25 is not None:
        bm25_results = bm25.search(query, top_k=candidate_k)
        exact_results = _exact_legal_matches(query, bm25.corpus_chunks, limit=k)
    else:
        print("  ⚠️  BM25 index not loaded or empty, using dense search only")
        bm25_results = []
        exact_results = []

    # Exact article/law matches are intentionally ranked first. RRF handles the rest.
    fused = reciprocal_rank_fusion(curated_results, exact_results, dense_results, bm25_results, k=rrf_k)

    priority_results = curated_results + exact_results
    if priority_results:
        priority_ids = {_document_key(doc) for doc in priority_results}
        fused = sorted(
            fused,
            key=lambda d: (
                _document_key(d) not in priority_ids,
                -d.get("rrf_score", 0.0),
            ),
        )
    elif fused:
        top_score = fused[0].get("rrf_score", 1.0)
        min_score = top_score * 0.15
        fused = [d for d in fused if d.get("rrf_score", 0) >= min_score]

    expanded = _expand_with_graph(fused[:k], settings.graph_max_hops)
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
    
    Documents are identified by chunk_id when available, then by source/article,
    then by content.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = _document_key(doc)
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
    """Keep first occurrence of each retrieved chunk using stable document keys."""
    seen = set()
    unique = []
    for doc in documents:
        key = _document_key(doc)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _exact_legal_matches(query: str, corpus_chunks: list[dict], limit: int) -> list[dict]:
    """Find chunks that exactly match cited article numbers and optional law hints."""
    article_refs = _extract_article_refs(query)
    if not article_refs:
        return []

    law_hints = _extract_law_hints(query)
    matches = []
    for chunk in corpus_chunks:
        meta = chunk.get("metadata", {})
        article = str(meta.get("article", "")).lower()
        if article not in article_refs:
            continue
        if law_hints and not _matches_law_hint(chunk, law_hints):
            continue
        doc = chunk.copy()
        doc["exact_match_score"] = 1.0
        matches.append(doc)

    # If the user only mentions "Điều X" without a law name, keep all matching laws.
    return matches[:limit]


def _curated_local_matches(query: str, limit: int) -> list[dict]:
    """Search the small curated local law files and boost them above noisy bulk data."""
    chunks = _load_curated_chunks()
    if not chunks:
        return []

    article_refs = _extract_article_refs(query)
    law_hints = _extract_law_hints(query)
    query_tokens = _query_tokens(query)
    query_phrases = _query_phrases(query_tokens)
    scored = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        article = str(meta.get("article", "")).lower()
        if article_refs and article not in article_refs:
            continue
        if law_hints and not _matches_law_hint(chunk, law_hints):
            continue

        text = " ".join(
            [
                chunk.get("page_content", ""),
                str(meta.get("law_name", "")),
                str(meta.get("filename", "")),
            ]
        )
        text_tokens = _query_tokens(text)
        text_phrases = _query_phrases(text_tokens)
        overlap = len(query_tokens & text_tokens)
        phrase_overlap = len(query_phrases & text_phrases)
        article_bonus = 4 if article_refs and article in article_refs else 0
        law_bonus = 2 if law_hints and _matches_law_hint(chunk, law_hints) else 0
        heading_bonus = _heading_bonus(query_tokens, chunk.get("page_content", ""))
        score = overlap + phrase_overlap * 3 + heading_bonus + article_bonus + law_bonus
        if score <= 0:
            continue

        doc = chunk.copy()
        doc["curated_match_score"] = float(score)
        scored.append(doc)

    scored.sort(key=lambda d: d.get("curated_match_score", 0.0), reverse=True)
    return scored[:limit]


def _load_curated_chunks() -> list[dict]:
    """Load curated local legal docs once per process."""
    global _curated_chunks
    if _curated_chunks is not None:
        return _curated_chunks

    try:
        from pathlib import Path
        from src.ingestion.cleaner import clean_text
        from src.ingestion.chunker import chunk_document
        from src.ingestion.loader import load_directory

        project_root = Path(__file__).resolve().parents[2]
        docs_dir = project_root / "data" / "legal_docs"
        docs = load_directory(str(docs_dir))
        chunks = []
        for doc in docs:
            doc["page_content"] = clean_text(doc["page_content"])
            chunks.extend(chunk_document(doc))
        _curated_chunks = chunks
    except Exception as e:
        print(f"  ⚠️  Could not load curated local laws ({e})")
        _curated_chunks = []
    return _curated_chunks


def _extract_article_refs(query: str) -> set[str]:
    """Extract article numbers from Vietnamese legal queries."""
    return {
        match.lower()
        for match in re.findall(r"(?:điều|dieu)\s+(\d+[a-zA-Z]?)", query, flags=re.IGNORECASE)
    }


def _extract_law_hints(query: str) -> set[str]:
    """Detect broad law names from the query for lightweight metadata filtering."""
    q = query.lower()
    hints = set()
    if any(term in q for term in ["lao động", "hop dong lao dong", "hợp đồng lao động"]):
        hints.add("lao động")
    if any(term in q for term in ["dân sự", "bo luat dan su", "bộ luật dân sự", "bồi thường"]):
        hints.add("dân sự")
    if any(term in q for term in ["hôn nhân", "hon nhan", "vợ chồng", "ly hôn", "kết hôn"]):
        hints.add("hôn nhân")
    return hints


def _matches_law_hint(doc: dict, hints: set[str]) -> bool:
    meta = doc.get("metadata", {})
    haystack = " ".join(
        str(meta.get(key, ""))
        for key in ("law_name", "title", "filename", "source")
    ).lower()
    haystack = f"{haystack} {doc.get('page_content', '')[:300].lower()}"
    return any(hint in haystack for hint in hints)


def _query_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w]+", text.lower(), re.UNICODE)
        if len(token) > 1 and token not in _STOPWORDS
    }


def _query_phrases(tokens: set[str]) -> set[str]:
    ordered = sorted(tokens)
    phrases = set()
    for size in (2, 3):
        for idx in range(0, max(len(ordered) - size + 1, 0)):
            phrases.add(" ".join(ordered[idx : idx + size]))
    return phrases


def _heading_bonus(query_tokens: set[str], content: str) -> int:
    heading = content.splitlines()[0] if content else ""
    heading_tokens = _query_tokens(heading)
    return len(query_tokens & heading_tokens) * 2


def _document_key(doc: dict) -> str:
    """Return a stable key for ranking/dedup across retrievers."""
    meta = doc.get("metadata", {})
    chunk_id = meta.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"

    source = meta.get("source") or meta.get("filename")
    article = meta.get("article")
    sub_chunk = meta.get("sub_chunk")
    if source and article is not None:
        return f"meta:{source}:{article}:{sub_chunk}"

    return f"content:{doc.get('page_content', '')[:500]}"
