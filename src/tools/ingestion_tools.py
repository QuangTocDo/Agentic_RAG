"""
LangChain tools wrapping ingestion operations.
Allows the agent to trigger document ingestion on-the-fly.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.tools import tool


@tool
def ingest_document(file_path: str) -> str:
    """Nạp một tài liệu pháp luật mới vào hệ thống.
    
    Args:
        file_path: Đường dẫn tới file .txt cần nạp.
    
    Returns:
        Thông báo kết quả nạp dữ liệu.
    """
    from src.ingestion.loader import load_text_file
    from src.ingestion.cleaner import clean_text
    from src.ingestion.chunker import chunk_document
    from src.indexing.chroma_store import add_documents

    try:
        doc = load_text_file(file_path)
        doc["page_content"] = clean_text(doc["page_content"])
        chunks = chunk_document(doc)
        add_documents(chunks)
        return f"✅ Đã nạp thành công {len(chunks)} phân đoạn từ file: {file_path}"
    except Exception as e:
        return f"❌ Lỗi khi nạp tài liệu: {str(e)}"


def get_ingestion_tools() -> list:
    """Return all ingestion-related tools."""
    return [ingest_document]
