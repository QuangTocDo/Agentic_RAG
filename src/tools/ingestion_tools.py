"""
LangChain tools wrapping ingestion operations.
Allows the agent to run the ingestion pipeline step-by-step.
"""
from __future__ import annotations
import sys, os
import json
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from langchain_core.tools import tool


@tool
def load_dataset_tool(source: str, path: str = "data/legal_docs", ids: list[int] | None = None) -> str:
    """Tải tài liệu pháp luật thô từ thư mục cục bộ (source='local') hoặc từ Hugging Face Dataset (source='hf').
    
    Args:
        source: Nguồn dữ liệu ('local' hoặc 'hf').
        path: Đường dẫn thư mục hoặc tệp tin cục bộ (mặc định 'data/legal_docs').
        ids: Danh sách ID tệp tin cần tải từ Hugging Face (ví dụ [36870, 95942, 27615]).
        
    Returns:
        JSON danh sách tài liệu thô được đọc.
    """
    from src.ingestion.loader import load_hf_dataset, load_directory, load_text_file
    
    try:
        if source == "hf":
            if not ids:
                return json.dumps({"error": "Cần truyền danh sách ID khi source là 'hf'"}, ensure_ascii=False)
            print(f"📥 [Ingestion Tool] Đang tải từ Hugging Face cho các ID: {ids}")
            docs = load_hf_dataset(ids)
        else:
            print(f"📥 [Ingestion Tool] Đang đọc từ thư mục cục bộ: {path}")
            if os.path.isdir(path):
                docs = load_directory(path)
            else:
                docs = [load_text_file(path)]
        return json.dumps(docs, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Lỗi khi đọc tài liệu: {str(e)}"}, ensure_ascii=False)


@tool
def clean_docs_tool(docs_json: str) -> str:
    """Làm sạch nội dung HTML/văn bản thô của các tài liệu pháp lý.
    
    Args:
        docs_json: Chuỗi JSON danh sách các tài liệu thô.
        
    Returns:
        JSON danh sách tài liệu đã làm sạch văn bản.
    """
    from src.ingestion.cleaner import clean_text
    
    try:
        docs = json.loads(docs_json)
        if "error" in docs:
            return docs_json
            
        print(f"🧹 [Ingestion Tool] Đang làm sạch {len(docs)} tài liệu...")
        for doc in docs:
            doc["page_content"] = clean_text(doc["page_content"])
        return json.dumps(docs, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Lỗi khi làm sạch tài liệu: {str(e)}"}, ensure_ascii=False)


@tool
def chunk_docs_tool(docs_json: str) -> str:
    """Chia nhỏ các tài liệu pháp luật đã làm sạch thành các phân đoạn (chunks) đồng nhất.
    
    Args:
        docs_json: Chuỗi JSON danh sách tài liệu đã làm sạch.
        
    Returns:
        JSON danh sách các phân đoạn (chunks) kèm metadata.
    """
    from src.ingestion.chunker import chunk_document
    
    try:
        docs = json.loads(docs_json)
        if "error" in docs:
            return docs_json
            
        print(f"✂️  [Ingestion Tool] Đang chia nhỏ {len(docs)} tài liệu...")
        all_chunks = []
        for doc in docs:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
        return json.dumps(all_chunks, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Lỗi khi chia nhỏ tài liệu: {str(e)}"}, ensure_ascii=False)


@tool
def build_index_tool(chunks_json: str, reset: bool = False) -> str:
    """Xây dựng và cập nhật chỉ mục tìm kiếm (ChromaDB, BM25, LegalGraph) từ danh sách các phân đoạn.
    
    Args:
        chunks_json: Chuỗi JSON danh sách các phân đoạn (chunks).
        reset: Nếu True, xóa dữ liệu cũ và xây dựng lại từ đầu. Mặc định là False.
        
    Returns:
        Thông điệp kết quả nạp chỉ mục thành công.
    """
    from src.indexing.chroma_store import add_documents, reset_collection
    from src.indexing.bm25_index import BM25Index
    from src.retrieval.graph import build_graph, append_graph
    
    try:
        chunks = json.loads(chunks_json)
        if isinstance(chunks, dict) and "error" in chunks:
            return chunks_json
            
        print(f"💾 [Ingestion Tool] Đang cập nhật chỉ mục tìm kiếm cho {len(chunks)} phân đoạn...")
        
        # 1. ChromaDB
        if reset:
            reset_collection()
        add_documents(chunks)
        
        # 2. BM25 Index
        bm25 = BM25Index()
        if reset:
            bm25.build(chunks)
        else:
            bm25.append_and_build(chunks)
        bm25.save()
        
        # 3. Legal Graph
        if reset:
            build_graph(chunks)
        else:
            append_graph(chunks)
            
        return f"✅ Đã nạp và lập chỉ mục thành công {len(chunks)} phân đoạn vào ChromaDB, BM25 và Đồ thị Pháp lý!"
    except Exception as e:
        return f"❌ Lỗi khi xây dựng chỉ mục: {str(e)}"


def get_ingestion_tools() -> list:
    """Return all ingestion-related tools."""
    return [load_dataset_tool, clean_docs_tool, chunk_docs_tool, build_index_tool]
