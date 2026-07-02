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
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = settings.embedding_model
        _embedding_fn = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_fn
