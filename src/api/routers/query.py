import time
import sys
import os
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from fastapi import APIRouter, Query
from configs.setting import settings
from src.retrieval.graph import graph_search, load_graph
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.indexing.chroma_store import get_chroma_store

router = APIRouter(prefix="/query", tags=["query"])


def _get_store():
    return get_chroma_store()


def _get_graph():
    return load_graph()


def _agentic_retrieve(question: str, k: int) -> tuple[list[dict], float]:
    # Keyword-based routing for Vietnamese legal queries
    t0 = time.perf_counter()
    q_lower = question.lower()

    if any(kw in q_lower for kw in ["sửa đổi", "thay thế", "bãi bỏ", "tham chiếu"]):
        docs = graph_search(_get_store(), _get_graph(), question, k=k, initial_k=3, max_hops=settings.graph_max_hops)
    elif any(kw in q_lower for kw in ["còn hiệu lực", "hết hiệu lực", "sau năm", "trước năm"]):
        docs = hybrid_search(question, k=k)
    else:
        candidates = hybrid_search(question, k=k * 2)
        docs = rerank(question, candidates, top_k=k)

    latency_ms = (time.perf_counter() - t0) * 1000
    
    # Format return elements to match d.page_content and d.metadata
    formatted_docs = []
    for d in docs:
        if hasattr(d, "page_content"):
            formatted_docs.append({
                "page_content": d.page_content,
                "metadata": d.metadata
            })
        else:
            formatted_docs.append({
                "page_content": d.get("page_content", ""),
                "metadata": d.get("metadata", {})
            })
            
    return formatted_docs, latency_ms


@router.get("")
def query_endpoint(
    q: str = Query(..., description="Câu hỏi pháp luật"), 
    k: int = Query(5, description="Số lượng kết quả")
):
    docs, latency = _agentic_retrieve(q, k)
    return {
        "latency_ms": latency,
        "results": docs
    }
