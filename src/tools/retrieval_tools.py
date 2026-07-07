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
    results = grade_documents(query, results)

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


def grade_documents(query: str, documents: list[dict]) -> list[dict]:
    """Grade retrieved documents using the LLM in a single batch to save API requests (Corrective RAG)."""
    if not documents:
        return []

    from src.llm import get_llm
    from langchain_core.messages import HumanMessage
    import json

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
    return [search_legal_documents]
