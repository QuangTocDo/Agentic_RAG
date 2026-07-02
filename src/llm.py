"""
Central LLM initialization — Google Gemini via LangChain.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.setting import settings

_llm = None


def get_llm():
    """Return a singleton ChatGoogleGenerativeAI instance."""
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = settings.google_api_key
        if not api_key or api_key == "your_google_api_key_here":
            raise ValueError(
                "❌ GOOGLE_API_KEY chưa được cấu hình!\n"
                "   Hãy thêm API key vào file .env:\n"
                "   GOOGLE_API_KEY=your_actual_key_here\n"
                "   Lấy key tại: https://aistudio.google.com/apikey"
            )

        _llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=api_key,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
        )
    return _llm
