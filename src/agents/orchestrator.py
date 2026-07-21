"""
Orchestrator — entry point for the Legal RAG system.
Manages two sub-agents: Legal RAG Agent and Ingestion Agent.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.llm import get_llm

# Import tools
from src.tools.retrieval_tools import (
    dense_search_tool,
    hybrid_search_tool,
    graph_traverse_tool,
    generate_answer_tool,
)
from src.tools.ingestion_tools import (
    load_dataset_tool,
    clean_docs_tool,
    chunk_docs_tool,
    build_index_tool,
)

# Define prompts
ORCHESTRATOR_PROMPT = """Bạn là Orchestrator - Agent điều phối trung tâm của hệ thống Legal RAG.
Nhiệm vụ của bạn là nhận yêu cầu của người dùng, phân loại và gọi sub-agent tương ứng để xử lý thông qua công cụ:
1. Nếu yêu cầu là câu hỏi pháp lý Việt Nam, tìm kiếm văn bản luật hoặc điều khoản liên quan -> Gọi `legal_rag_agent`.
2. Nếu yêu cầu liên quan đến nạp tài liệu mới, cập nhật cơ sở dữ liệu, tải dataset, làm sạch hoặc xây dựng chỉ mục (ingest/index) -> Gọi `ingestion_agent`.

Hãy chuyển tiếp yêu cầu của người dùng nguyên vẹn cho sub-agent tương ứng và trả về câu trả lời cuối cùng từ sub-agent đó cho người dùng.
"""

LEGAL_RAG_PROMPT = """Bạn là Legal RAG Agent chuyên giải đáp câu hỏi pháp lý dựa trên dữ liệu luật Việt Nam.
Bạn có quyền sử dụng các công cụ tìm kiếm và duyệt đồ thị sau:
- `dense_search_tool`: Dùng khi cần tìm kiếm các phân đoạn có sự tương đồng về mặt ngữ nghĩa (khi dùng từ đồng nghĩa hoặc mô tả chung).
- `hybrid_search_tool`: Tìm kiếm kết hợp từ khóa chính xác và ngữ nghĩa (phù hợp cho hầu hết các câu hỏi thông thường).
- `graph_traverse_tool`: Duyệt liên kết chéo Điều luật (phù hợp với các câu hỏi phức tạp, yêu cầu sâu chuỗi nhiều Điều luật, lịch sử sửa đổi, bãi bỏ, bổ sung hoặc hướng dẫn thi hành).
- `generate_answer_tool`: Sử dụng sau khi đã tìm đủ tài liệu để tổng hợp câu trả lời chuẩn xác nhất.

QUY TẮC HOẠT ĐỘNG:
- BẮT BUỘC phải thực hiện tìm kiếm tài liệu trước tiên nếu lịch sử trò chuyện chưa có thông tin.
- Chỉ trả lời dựa trên thông tin tìm thấy, tuyệt đối không tự suy diễn điều luật.
"""

INGESTION_PROMPT = """Bạn là Ingestion Agent chịu trách nhiệm chạy quy trình nạp dữ liệu (Ingestion Pipeline).
Khi nhận được yêu cầu nạp tài liệu hoặc tải dataset, bạn BẮT BUỘC phải gọi tuần tự cả 4 công cụ này để hoàn tất toàn bộ quy trình:
1. `load_dataset_tool` để tải tài liệu thô.
2. `clean_docs_tool` để làm sạch nội dung HTML/văn bản thô của kết quả bước 1.
3. `chunk_docs_tool` để chia nhỏ văn bản từ kết quả bước 2 thành các phân đoạn (chunks).
4. `build_index_tool` để xây dựng và cập nhật chỉ mục từ kết quả bước 3.

Hãy thực hiện đầy đủ 4 bước trên và báo cáo lại kết quả cụ thể cho người dùng.
"""


class DeepAgent:
    def __init__(self, agent_graph, name: str, description: str):
        self.agent_graph = agent_graph
        self.name = name
        self.description = description

    def invoke(self, input_data: dict) -> dict:
        return self.agent_graph.invoke(input_data)

    def as_tool(self):
        # pyrefly: ignore [missing-import]
        from langchain_core.tools import Tool
        
        def run_agent(query: str) -> str:
            print(f"\n🤖 [Orchestrator] Đang chuyển tiếp yêu cầu tới sub-agent: {self.name}...")
            res = self.agent_graph.invoke({"messages": [("user", query)]})
            messages = res.get("messages", [])
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
                        return str(msg.content if hasattr(msg, "content") else msg.get("content", "")).strip()
            return "Sub-agent không có câu trả lời."

        return Tool(
            name=self.name,
            description=self.description,
            func=run_agent
        )


def create_deep_agent(model, tools, system_prompt: str) -> DeepAgent:
    # pyrefly: ignore [missing-import]
    from langgraph.prebuilt import create_react_agent
    
    if "rag" in system_prompt.lower() or "pháp lý" in system_prompt.lower():
        name = "legal_rag_agent"
        description = "Dùng công cụ này để giải đáp câu hỏi pháp lý Việt Nam, tìm kiếm văn bản luật hoặc duyệt đồ thị dẫn chiếu."
    elif "ingest" in system_prompt.lower() or "nạp" in system_prompt.lower():
        name = "ingestion_agent"
        description = "Dùng công cụ này để nạp tài liệu mới vào cơ sở dữ liệu, tải dataset, làm sạch, chia phân đoạn và lập chỉ mục."
    else:
        name = "orchestrator_agent"
        description = "Orchestrator coordinator agent."
        
    agent_graph = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt
    )
    return DeepAgent(agent_graph, name, description)


# Initialize the two sub-agents
legal_rag_agent = create_deep_agent(
    model=get_llm(),
    tools=[
        dense_search_tool,
        hybrid_search_tool,
        graph_traverse_tool,
        generate_answer_tool,
    ],
    system_prompt=LEGAL_RAG_PROMPT,
)

ingestion_agent = create_deep_agent(
    model=get_llm(),
    tools=[
        load_dataset_tool,
        clean_docs_tool,
        chunk_docs_tool,
        build_index_tool,
    ],
    system_prompt=INGESTION_PROMPT,
)

# Initialize the main Orchestrator agent
agent = create_deep_agent(
    model=get_llm(),
    tools=[legal_rag_agent.as_tool(), ingestion_agent.as_tool()],
    system_prompt=ORCHESTRATOR_PROMPT,
)


def answer_question(question: str) -> str:
    """Main entry point: takes a user question, runs the agent, and returns an answer."""
    return chat(question)


def chat(question: str, history: list[dict] | None = None) -> str:
    """
    Chat-compatible entry point for Gradio.
    Accepts question + conversation history and returns answer string.
    """
    from src.agents.rag_agent import _format_messages
    messages = _format_messages(question, history)
    
    res = agent.invoke({"messages": messages})
    messages = res.get("messages", [])
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
                return str(msg.content if hasattr(msg, "content") else msg.get("content", "")).strip()
    return "Không thể nhận diện câu trả lời từ Orchestrator."
