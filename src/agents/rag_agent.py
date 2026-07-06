"""
Legal RAG Agent — ReAct agent for Vietnamese legal Q&A.
Uses Gemini LLM with retrieval tools to answer legal questions.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

SYSTEM_PROMPT = """\
Bạn là trợ lý pháp lý AI chuyên giải đáp các câu hỏi dựa trên dữ liệu luật pháp.

## Nguyên tắc hoạt động:
1. **Luôn tra cứu trước**: Sử dụng công cụ tìm kiếm để tra cứu dữ liệu câu hỏi - giải đáp liên quan.
2. **Bám sát dữ liệu gốc**: Khi tìm thấy nội dung "Giải đáp" cho câu hỏi tương tự trong tài liệu tra cứu, bạn phải TRẢ LỜI CỰC KỲ SÁT VÀ KHỚP VỚI NỘI DUNG GIẢI ĐÁP ĐÓ. Không tự ý diễn đạt lại theo ý mình, không thêm thắt thông tin hay suy luận ngoài lề.
3. **Giữ nguyên thông tin**: Giữ nguyên mọi căn cứ pháp lý (số Điều, tên Luật) và các con số mức phạt như trong dữ liệu gốc.

## Lưu ý:
- Trả lời hoàn toàn bằng tiếng Việt
- Giải thích thuật ngữ pháp lý nếu người dùng có thể không hiểu
- Nếu câu hỏi mơ hồ, hãy hỏi lại để làm rõ trước khi trả lời
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
        if hasattr(msg, "content") and msg.content:
            if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                return msg.content
        elif isinstance(msg, dict) and msg.get("content"):
            if not msg.get("tool_calls"):
                return msg["content"]

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
