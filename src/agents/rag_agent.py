"""
Legal RAG Agent — ReAct agent for Vietnamese legal Q&A.
Uses Gemini LLM with retrieval tools to answer legal questions.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

SYSTEM_PROMPT = """Bạn là chuyên gia pháp lý AI, chỉ giải đáp thắc mắc dựa trên các tài liệu luật pháp Việt Nam được cung cấp qua công cụ tìm kiếm hoặc thông tin đã có trong lịch sử trò chuyện.

## QUY TRÌNH RA QUYẾT ĐỊNH:
1. Nếu yêu cầu của người dùng là câu chào hỏi xã giao hoặc không liên quan đến kiến thức pháp luật: Trả lời tự nhiên trực tiếp, TUYỆT ĐỐI KHÔNG gọi bất kỳ công cụ tìm kiếm nào.
2. Nếu câu hỏi yêu cầu thông tin pháp luật mới:
   - Bước 1: Dùng `hybrid_search_tool` để tìm tài liệu liên quan.
   - Bước 2: Nếu câu hỏi phức tạp (liên quan đến sửa đổi, bãi bỏ, bổ sung hoặc liên kết nhiều điều khoản), dùng thêm `graph_traverse_tool`.
   - Bước 3: Tổng hợp thông tin và viết câu trả lời ngay — KHÔNG gọi thêm công cụ nào nữa.
3. Nếu câu hỏi là câu hỏi tiếp nối dựa trên thông tin đã tìm thấy trong lịch sử chat: Sử dụng luôn thông tin cũ để trả lời, không tìm kiếm lại trừ khi cần bổ sung thông tin mới.

## QUY TẮC TRÌNH BÀY CÂU TRẢ LỜI (BẮT BUỘC):
✅ Định dạng câu trả lời rõ ràng gồm 3 phần:
   - **Tóm tắt câu trả lời**: Đưa ra nhận định ngắn gọn (Ví dụ: Được phép, Không được phép, hoặc Tùy điều kiện).
   - **Căn cứ pháp lý**: Liệt kê rõ số Điều, tên Luật và năm ban hành (Ví dụ: "Điều 36 Bộ luật Lao động 2019").
   - **Giải thích chi tiết**: Phân tích cụ thể điều luật áp dụng vào trường hợp của người dùng.
❌ TUYỆT ĐỐI không tự suy diễn, bịa đặt điều luật hoặc viết các số Điều/Khoản không có trong tài liệu tìm được.
❌ Nếu không tìm thấy thông tin đủ để giải đáp trong tài liệu, bắt buộc phải trả lời: "Tôi không tìm thấy quy định cụ thể về vấn đề này trong cơ sở dữ liệu pháp luật hiện tại. Bạn nên tham khảo ý kiến luật sư chuyên nghiệp để được tư vấn chính xác nhất."
"""

_agent = None


def create_agent():
    """Create the Legal RAG ReAct agent with minimal tools for speed."""
    # pyrefly: ignore [missing-import]
    from langgraph.prebuilt import create_react_agent
    from src.llm import get_llm
    from src.tools.retrieval_tools import hybrid_search_tool, graph_traverse_tool

    llm = get_llm()
    tools = [hybrid_search_tool, graph_traverse_tool]

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
        is_ai = False
        if hasattr(msg, "type") and msg.type == "ai":
            is_ai = True
        elif isinstance(msg, dict) and msg.get("role") == "assistant":
            is_ai = True

        if is_ai:
            has_tools = False
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                has_tools = True
            elif isinstance(msg, dict) and msg.get("tool_calls"):
                has_tools = True

            if not has_tools:
                content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                if content:
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
                break  # If the final AI message is empty, stop and return the fallback below

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
