"""
LangChain tools wrapping retrieval operations for the RAG agent.
"""
from __future__ import annotations
import sys, os
import json
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings
from langchain_core.tools import tool


def _get_store():
    from src.indexing.chroma_store import get_chroma_store
    return get_chroma_store()


def _get_graph():
    from src.retrieval.graph import load_graph
    return load_graph()


def _doc_to_dict(d: Any) -> dict:
    if isinstance(d, str):
        return {"page_content": d, "metadata": {}}
    if hasattr(d, "page_content"):
        return {"page_content": d.page_content, "metadata": d.metadata}
    if isinstance(d, dict):
        return {"page_content": d.get("page_content", ""), "metadata": d.get("metadata", {})}
    return {"page_content": str(d), "metadata": {}}


def _format_docs_for_llm(docs: list) -> str:
    """Format retrieved docs into clear, structured text so small LLMs can read them easily."""
    if not docs:
        return "Không tìm thấy tài liệu liên quan."
    
    parts = []
    for i, doc in enumerate(docs, 1):
        d = _doc_to_dict(doc)
        meta = d.get("metadata", {})
        law_name = meta.get("law_name", "")
        so_ky_hieu = meta.get("so_ky_hieu", "")
        article = meta.get("article", "")
        hieu_luc = meta.get("tinh_trang_hieu_luc", "")
        content = d.get("page_content", "")
        
        # Build header with all available metadata
        header_parts = []
        if law_name:
            header_parts.append(law_name)
        if so_ky_hieu:
            header_parts.append(so_ky_hieu)
        if article and article not in ("none", "header", "?"):
            header_parts.append(f"Điều {article}")
        if hieu_luc:
            header_parts.append(hieu_luc)
        
        header = " — ".join(header_parts) if header_parts else f"Tài liệu {i}"
        parts.append(f"=== [{i}] {header} ===\n{content}")
    
    return "\n\n".join(parts)


@tool
def dense_search_tool(query: str, k: int = settings.retrieval_k) -> str:
    """Tìm kiếm thông tin pháp lý bằng dense embeddings (vector search).
    Sử dụng công cụ này khi cần tìm các điều luật có ý nghĩa ngữ nghĩa tương tự hoặc diễn đạt khác đi với câu hỏi.
    """
    from src.retrieval.dense import dense_search
    docs = dense_search(query, k=k)
    return json.dumps([_doc_to_dict(d) for d in docs], ensure_ascii=False)


@tool
def hybrid_search_tool(query: str, k: int = settings.retrieval_k) -> str:
    """Tìm kiếm thông tin pháp lý kết hợp cả dense embeddings (vector search) và BM25 (keyword search).
    Sử dụng công cụ này cho các câu hỏi thông thường, tìm kiếm kết hợp cả từ khóa chính xác và ngữ nghĩa.
    Kết quả bao gồm tên luật, số hiệu, điều khoản và tình trạng hiệu lực.
    """
    from src.retrieval.hybrid import hybrid_search
    docs = hybrid_search(query, k=k)
    return _format_docs_for_llm(docs)


@tool
def graph_traverse_tool(
    query: str,
    k: int = settings.retrieval_k,
    initial_k: int = 3,
    max_hops: int = settings.graph_max_hops,
) -> str:
    """Retrieve documents using graph-guided multi-hop retrieval.
    Seed with dense search, then expand via relationship graph edges.
    Use this for multi-hop questions about document history,
    amendments, replacements or legal hierarchy."""
    from src.retrieval.graph import graph_search
    docs = graph_search(
        _get_store(), _get_graph(), query,
        k=k, initial_k=initial_k, max_hops=max_hops,
    )
    return _format_docs_for_llm(docs)


@tool
def generate_answer_tool(query: str, context_json: str) -> str:
    """Tạo câu trả lời pháp lý hoàn chỉnh từ danh sách tài liệu tìm kiếm được (context).
    Sử dụng công cụ này sau khi đã thực hiện tra cứu tài liệu bằng các công cụ tìm kiếm và cần tổng hợp câu trả lời chính xác, đáng tin cậy.
    """
    from src.llm import get_llm
    from langchain_core.messages import HumanMessage
    
    try:
        contexts = json.loads(context_json)
    except Exception:
        # Fallback if the input is not a JSON list (e.g. raw text)
        contexts = [context_json]
        
    formatted_contexts = []
    if not isinstance(contexts, list):
        contexts = [contexts]
        
    for idx, doc in enumerate(contexts, 1):
        if isinstance(doc, str):
            content = doc
            source = "N/A"
            article = "N/A"
        elif isinstance(doc, dict):
            content = doc.get("page_content", "")
            meta = doc.get("metadata", {})
            source = meta.get("law_name", meta.get("filename", "N/A"))
            article = meta.get("article", "N/A")
        else:
            content = str(doc)
            source = "N/A"
            article = "N/A"
            
        formatted_contexts.append(f"Tài liệu {idx} (Nguồn: {source}, Điều: {article}):\n{content}")
        
    context_str = "\n\n".join(formatted_contexts)
    prompt = (
        f"Bạn là trợ lý pháp lý AI chuyên giải đáp câu hỏi dựa trên dữ liệu luật pháp Việt Nam.\n\n"
        f"Hãy trả lời câu hỏi sau bằng tiếng Việt chuẩn, rõ ràng và chính xác. "
        f"Chỉ trả lời dựa trên thông tin có trong phần Tài liệu tham khảo dưới đây. Tuyệt đối không tự suy diễn hoặc bịa đặt điều luật hay số Điều/Khoản.\n\n"
        f"Câu hỏi: {query}\n\n"
        f"Tài liệu tham khảo:\n{context_str}\n\n"
        f"Câu trả lời:"
    )
    
    llm = get_llm()
    res = llm.invoke([HumanMessage(content=prompt)])
    if hasattr(res, "content"):
        return str(res.content).strip()
    return str(res).strip()


def grade_documents(query: str, documents: list[dict]) -> list[dict]:
    """Grade retrieved documents using the LLM in a single batch to save API requests (Corrective RAG)."""
    if not documents:
        return []

    if not settings.use_llm_grader:
        return documents

    from src.llm import get_llm
    from langchain_core.messages import HumanMessage

    llm = get_llm()
    relevant_docs = []

    print(f"\n🧠 [Corrective RAG] Đang đánh giá độ liên quan của {len(documents)} tài liệu (Chế độ Batch)...")
    
    # Format all documents into a single prompt
    doc_prompts = []
    for idx, doc in enumerate(documents):
        content = doc["page_content"]
        source = doc.get("metadata", {}).get("law_name", "Văn bản")
        article = doc.get("metadata", {}).get("article", "")
        doc_prompts.append(
            f"--- Tài liệu {idx} ({source} - Điều {article}) ---\n"
            f"{content}"
        )
    
    prompt = (
        f"Bạn là kiểm duyệt viên tài liệu pháp lý chuyên nghiệp.\n"
        f"Nhiệm vụ: Hãy đánh giá từng đoạn văn bản pháp luật dưới đây xem nó có chứa thông tin hữu ích giúp trả lời trực tiếp cho câu hỏi sau hay không.\n\n"
        f"Câu hỏi: {query}\n\n"
        f"{chr(10).join(doc_prompts)}\n\n"
        f"Hãy trả về kết quả dưới dạng một mảng JSON chứa các giá trị boolean (true nếu có liên quan, false nếu không). "
        f"Ví dụ: [true, false, true, false, false]. "
        f"Không giải thích, không viết thêm từ nào ngoài mảng JSON hợp lệ."
    )
    
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        res_text = ""
        if hasattr(res, "content"):
            if isinstance(res.content, list) and res.content and isinstance(res.content[0], dict):
                res_text = res.content[0].get("text", "")
            else:
                res_text = res.content
        elif isinstance(res, list) and res and isinstance(res[0], dict):
            res_text = res[0].get("text", "")
        
        # Parse JSON
        res_text = str(res_text).strip()
        # Clean potential markdown block formatting like ```json ... ```
        if res_text.startswith("```"):
            lines = res_text.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            res_text = "\n".join(lines).strip()
            
        grades = json.loads(res_text)
        if isinstance(grades, list) and len(grades) == len(documents):
            for doc, grade in zip(documents, grades):
                source = doc.get("metadata", {}).get("law_name", "Văn bản")
                article = doc.get("metadata", {}).get("article", "")
                if grade is True:
                    relevant_docs.append(doc)
                    print(f"    ✅ Giữ lại: Điều {article} ({source}) - LLM Grader: YES")
                else:
                    print(f"    ❌ Loại bỏ: Điều {article} ({source}) - LLM Grader: NO")
        else:
            raise ValueError("Mảng kết quả không hợp lệ hoặc lệch số lượng.")
    except Exception as e:
        # Fallback to keep everything on error or rate limit
        print(f"    ⚠️ Lỗi khi chấm điểm tài liệu theo lô ({e}), giữ lại toàn bộ làm dự phòng.")
        relevant_docs = documents

    print(f"👉 Chấm điểm hoàn tất: Giữ lại {len(relevant_docs)} / {len(documents)} tài liệu phù hợp.\n")
    return relevant_docs


def get_retrieval_tools() -> list:
    """Return all retrieval-related tools."""
    return [dense_search_tool, hybrid_search_tool, graph_traverse_tool, generate_answer_tool]
