"""
Gradio Chat UI for Vietnamese Legal Q&A System.
"""
import sys
import os

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import gradio as gr


def respond(message: str, history: list) -> str:
    """Handle user message and return agent response."""
    if not message.strip():
        return "Vui lòng nhập câu hỏi pháp lý."

    try:
        from src.agents.orchestrator import chat

        answer = chat(message, history)
        return answer
    except ValueError as e:
        return str(e)
    except Exception as e:
        return (
            f"❌ Đã xảy ra lỗi: {str(e)}\n\n"
            "Vui lòng kiểm tra:\n"
            "1. File .env đã cấu hình GOOGLE_API_KEY\n"
            "2. Đã chạy `python scripts/ingest.py` để nạp dữ liệu\n"
            "3. Đã cài đặt đầy đủ thư viện: `pip install -r requirements.txt`"
        )


EXAMPLES = [
    "Điều kiện kết hôn theo pháp luật Việt Nam là gì?",
    "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động khi nào?",
    "Mức lương tối thiểu được quy định như thế nào?",
    "Khi nào được bồi thường thiệt hại theo Bộ luật Dân sự?",
    "Tài sản chung của vợ chồng được chia như thế nào khi ly hôn?",
    "Trợ cấp thôi việc được tính như thế nào?",
]

# Build the UI
demo = gr.ChatInterface(
    fn=respond,
    title="🏛️ Trợ lý Pháp lý AI Việt Nam",
    description=(
        "Hỏi đáp về pháp luật Việt Nam dựa trên **Bộ luật Dân sự 2015**, "
        "**Bộ luật Lao động 2019**, **Luật Hôn nhân và Gia đình 2014**.\n\n"
        "⚠️ *Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp.*"
    ),
    examples=EXAMPLES,
    fill_height=True,
    chatbot=gr.Chatbot(
        height=500,
        placeholder=(
            "<center><h2>🏛️ Trợ lý Pháp lý AI</h2>"
            "<p>Hãy đặt câu hỏi về pháp luật Việt Nam!</p></center>"
        ),
    ),
    textbox=gr.Textbox(
        placeholder="Nhập câu hỏi pháp lý tại đây...",
        scale=7,
    ),
    save_history=True,
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
