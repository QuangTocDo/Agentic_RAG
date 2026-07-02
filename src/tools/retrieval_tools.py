"""
LangChain tools wrapping retrieval operations for the RAG agent.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.tools import tool


@tool
def search_legal_documents(query: str) -> str:
    """Tìm kiếm văn bản pháp luật Việt Nam liên quan đến câu hỏi.
    Sử dụng công cụ này khi cần tra cứu điều luật, quy định pháp lý,
    hoặc tìm thông tin trong Bộ luật Dân sự, Luật Lao động, 
    Luật Hôn nhân và Gia đình, và các văn bản pháp luật khác.
    
    Args:
        query: Câu hỏi hoặc từ khóa tìm kiếm pháp luật bằng tiếng Việt.
    
    Returns:
        Nội dung các điều luật liên quan được tìm thấy.
    """
    from src.retrieval.hybrid import hybrid_search
    from configs.setting import settings

    results = hybrid_search(query, k=settings.retrieval_k)

    if not results:
        return "Không tìm thấy văn bản pháp luật nào liên quan đến câu hỏi."

    # Format results for the agent
    output_parts = []
    for i, doc in enumerate(results, 1):
        meta = doc.get("metadata", {})
        source = meta.get("law_name", meta.get("filename", "N/A"))
        article = meta.get("article", "N/A")
        content = doc["page_content"]

        output_parts.append(
            f"--- Kết quả {i} ---\n"
            f"📜 Nguồn: {source}\n"
            f"📌 Điều: {article}\n"
            f"📄 Nội dung:\n{content}\n"
        )

    return "\n".join(output_parts)


def get_retrieval_tools() -> list:
    """Return all retrieval-related tools."""
    return [search_legal_documents]
