"""
Legal RAG Agent — ReAct agent for Vietnamese legal Q&A.
Uses Gemini LLM with retrieval tools to answer legal questions.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý AI chuyên giải đáp câu hỏi dựa trên dữ liệu luật pháp Việt Nam được tìm kiếm.

## QUY TẮC BẮT BUỘC:
1. KHÔNG tự trả lời bằng kiến thức của bạn. Bạn BẮT BUỘC phải dùng công cụ `search_legal_documents` để tìm kiếm thông tin trước.
2. CHỈ trả lời dựa vào nội dung văn bản luật tìm được từ công cụ tìm kiếm.
3. Nếu kết quả tìm kiếm chứa nhiều văn bản từ các chủ đề khác nhau (ví dụ: vừa có Luật Lao động vừa có Luật Giao thông), bạn chỉ được sử dụng tài liệu liên quan trực tiếp đến câu hỏi của người dùng. TUYỆT ĐỐI không được trộn lẫn, kết hợp các thông tin từ các chủ đề không liên quan vào cùng một câu trả lời.
4. BẮT BUỘC viết câu trả lời bằng tiếng Việt chuẩn thuần túy. Tuyệt đối không sử dụng chữ Hán (như 动, 骑), không dịch thô từ tiếng Trung sang (ví dụ: không dùng từ 'cưỡi xe', 'tải nhân', phải dùng 'đi xe máy', 'chở người').
5. Nếu nội dung tìm được KHÔNG chứa câu trả lời cho câu hỏi, bạn PHẢI trả lời ngay lập tức: "Tôi không tìm thấy thông tin hoặc điều luật liên quan trong cơ sở dữ liệu pháp luật hiện tại." và TUYỆT ĐỐI không được viết thêm bất kỳ thông tin, suy đoán hay lời khuyên nào khác.
6. Tuyệt đối KHÔNG tự bịa đặt số Điều, số Khoản hoặc tên Luật. Tất cả thông tin pháp lý đưa ra phải có nguồn từ kết quả tìm kiếm của công cụ.

## Hướng dẫn sử dụng Công cụ tìm kiếm:
- Tự động trích xuất các từ khóa pháp lý chính (ví dụ: "ly hôn đơn phương", "độ tuổi kết hôn") để tìm kiếm.
- KHÔNG đưa các từ ngữ xưng hô cá nhân (anh A, chị B, tôi, bạn) vào ô tìm kiếm.
"""

_agent = None


def create_agent():
    """Create the Legal RAG ReAct agent."""
    from langgraph.prebuilt import create_react_agent
    from src.llm import get_llm
    from src.tools.retrieval_tools import get_retrieval_tools

    llm = get_llm()
    tools = get_retrieval_tools()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )
    return agent


def get_agent():
    """Return a cached Legal RAG agent."""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


def ask(question: str, history: list | None = None) -> str:
    """Ask the legal agent a question and return the answer."""
    agent = get_agent()
    messages = _format_messages(question, history)

    result = agent.invoke(
        {"messages": messages}
    )

    # Extract the final AI message
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = None
        if hasattr(msg, "content") and msg.content:
            if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                content = msg.content
        elif isinstance(msg, dict) and msg.get("content"):
            if not msg.get("tool_calls"):
                content = msg["content"]

        if content is not None:
            # Handle list content (often returned by langchain-google-genai)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "".join(text_parts).strip()
            return str(content).strip()

    return "Xin lỗi, tôi không thể trả lời câu hỏi này. Vui lòng thử lại."


def _format_messages(question: str, history: list | None = None) -> list[dict]:
    """Convert Gradio history variants into LangGraph message dicts."""
    messages: list[dict] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg, assistant_msg = item[0], item[1]
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": question})
    return messages
