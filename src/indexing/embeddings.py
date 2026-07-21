"""
Embedding manager — wraps HuggingFace sentence-transformers for Vietnamese.
Uses singleton pattern to avoid reloading the model.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings

_embedding_fn = None


def get_embedding_function():
    """Return a LangChain-compatible embedding function (singleton)."""
    global _embedding_fn
    if _embedding_fn is None:
        model_name = settings.embedding_model
        
        # Detect if model is an Ollama model
        is_ollama = ":" in model_name or "qwen3-embedding" in model_name.lower()

        if is_ollama:
            from langchain_community.embeddings import OllamaEmbeddings
            base_url = os.environ.get("OLLAMA_HOST") or settings.vllm_api_base.replace("/v1", "")
            print(f"✨ Khởi tạo mô hình Embedding sử dụng Ollama: {model_name} tại {base_url}")
            _embedding_fn = OllamaEmbeddings(
                model=model_name,
                base_url=base_url,
            )
        else:
            # pyrefly: ignore [missing-import]
            from langchain_huggingface import HuggingFaceEmbeddings
            # pyrefly: ignore [missing-import]
            import torch

            # Detect GPU acceleration device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            print(f"✨ Khởi tạo mô hình Embedding sử dụng thiết bị: {device.upper()}")

            _embedding_fn = HuggingFaceEmbeddings(
                model=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
    return _embedding_fn
