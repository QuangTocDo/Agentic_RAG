"""
Legal RAG Agent — ReAct agent for Vietnamese legal Q&A.
Uses Gemini LLM with retrieval tools to answer legal questions.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

SYSTEM_PROMPT = """\
Bạn là một trợ lý pháp lý AI chuyên về luật pháp Việt Nam. Nhiệm vụ của bạn là trả lời
các câu hỏi pháp lý bằng tiếng Việt một cách chính xác, dễ hiểu và có căn cứ pháp luật.

## Nguyên tắc làm việc:

1. **Luôn tra cứu trước khi trả lời**: Sử dụng công cụ tìm kiếm văn bản pháp luật
   để tìm các điều khoản liên quan trước khi đưa ra câu trả lời.

2. **Trích dẫn cụ thể**: Khi trả lời, LUÔN trích dẫn số Điều, tên Luật/Bộ luật cụ thể.
   Ví dụ: "Theo Điều 8 Luật Hôn nhân và Gia đình 2014, ..."

3. **Trả lời rõ ràng**: Tổ chức câu trả lời theo cấu trúc:
   - Trả lời ngắn gọn câu hỏi chính
   - Giải thích chi tiết dựa trên luật
   - Trích dẫn điều khoản cụ thể

4. **Trung thực**: Nếu không tìm thấy thông tin trong cơ sở dữ liệu,
   hãy nói rõ và khuyên người dùng tham khảo luật sư chuyên nghiệp.

5. **Không tư vấn thay luật sư**: Nhấn mạnh rằng đây chỉ là thông tin tham khảo,
   không thay thế tư vấn pháp lý chuyên nghiệp.

## Lưu ý:
- Trả lời hoàn toàn bằng tiếng Việt
- Giải thích thuật ngữ pháp lý nếu người dùng có thể không hiểu
- Nếu câu hỏi mơ hồ, hãy hỏi lại để làm rõ trước khi trả lời
"""


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


def ask(question: str) -> str:
    """Ask the legal agent a question and return the answer."""
    agent = create_agent()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
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
