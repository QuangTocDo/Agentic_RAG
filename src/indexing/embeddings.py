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
    """Return a LangChain-compatible embedding function (singleton) with GPU acceleration if available."""
    global _embedding_fn
    if _embedding_fn is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        import torch

        # Detect GPU acceleration device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"✨ Khởi tạo mô hình Embedding sử dụng thiết bị: {device.upper()}")

        model_name = settings.embedding_model
        _embedding_fn = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_fn
