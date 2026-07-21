"""
Central LLM initialization — Google Gemini via LangChain.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.setting import settings

import requests
from typing import Any

_llm = None

class MockModel:
    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path

class AltaCloudPipeline:
    def __init__(self, model_name: str, temperature: float, max_new_tokens: int):
        self.task = "text-generation"
        self.model = MockModel(model_name)
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.api_url = "https://router.huggingface.co/v1/chat/completions"

    def __call__(self, prompt: Any, **kwargs: Any) -> list:
        if isinstance(prompt, list):
            results = []
            for p in prompt:
                generated = self._call_api(p)
                results.append({"generated_text": generated})
            return results
        else:
            generated = self._call_api(prompt)
            return [{"generated_text": generated}]

    def _call_api(self, prompt: str) -> str:
        token = settings.hf_token
        if not token or token == "your_huggingface_token_here":
            raise ValueError("Không tìm thấy HF_TOKEN trong biến môi trường hoặc file .env")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
            )
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise ValueError(f"API Error: {data['error']}")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print("Error calling API:", e)
            raise RuntimeError(f"Lỗi khi gọi API LLM: {e}")

def get_llm():
    """Return a singleton ChatGoogleGenerativeAI, ChatOpenAI (for vLLM), or HuggingFacePipeline instance."""
    global _llm
    if _llm is None:
        provider = settings.llm_provider

        if provider in ("vllm", "ollama"):
            # pyrefly: ignore [missing-import]
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": settings.llm_model,
                "openai_api_key": "none-needed-for-vllm",
                "openai_api_base": settings.vllm_api_base,
                "temperature": settings.temperature,
                "max_tokens": settings.max_output_tokens,
            }
            # For Ollama or Ngrok-tunnelled Ollama endpoints, pass context length configurations
            if provider == "ollama" or (settings.vllm_api_base and ("ollama" in settings.vllm_api_base or "ngrok" in settings.vllm_api_base)):
                kwargs["extra_body"] = {"options": {"num_ctx": 8192}}

            _llm = ChatOpenAI(**kwargs)
        elif provider == "huggingface":
            # pyrefly: ignore [missing-import]
            from langchain_openai import ChatOpenAI

            token = settings.hf_token
            if not token or token == "your_huggingface_token_here":
                raise ValueError("Không tìm thấy HF_TOKEN trong biến môi trường hoặc file .env")

            # Hugging Face's Serverless Router is 100% OpenAI-compatible.
            # Using ChatOpenAI gives us native tool-calling (bind_tools) support for Qwen models.
            _llm = ChatOpenAI(
                model=settings.llm_model,
                openai_api_key=token,
                openai_api_base="https://router.huggingface.co/v1",
                temperature=settings.temperature,
                max_tokens=settings.max_output_tokens,
                frequency_penalty=0.5,
            )

        else:
            # pyrefly: ignore [missing-import]
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


