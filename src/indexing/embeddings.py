"""
Embedding manager — wraps HuggingFace sentence-transformers for Vietnamese.
Uses singleton pattern to avoid reloading the model.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings

import requests
from langchain_core.embeddings import Embeddings

_embedding_fn = None


class OllamaBatchEmbeddings(Embeddings):
    """Custom LangChain Embeddings class that batches embedding requests using Ollama's native batch /api/embed API."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents in batches of 100 to reduce HTTP overhead."""
        url = f"{self.base_url}/api/embed"
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {
                "model": self.model,
                "input": batch,
            }
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings", [])
                all_embeddings.extend(embeddings)
            except Exception as e:
                # Fallback to sequential embedding if batch API fails
                print(f"⚠️ Batch embedding failed ({e}). Falling back to sequential embedding...")
                for text in batch:
                    all_embeddings.append(self.embed_query(text))

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text,
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except Exception as e:
            # Try /api/embed as a secondary fallback
            try:
                embed_url = f"{self.base_url}/api/embed"
                embed_payload = {"model": self.model, "input": [text]}
                response = requests.post(embed_url, json=embed_payload, timeout=30)
                response.raise_for_status()
                return response.json().get("embeddings", [])[0]
            except Exception as nested_e:
                raise RuntimeError(f"Failed to embed query: {e} | {nested_e}")


def get_embedding_function():
    """Return a LangChain-compatible embedding function (singleton)."""
    global _embedding_fn
    if _embedding_fn is None:
        model_name = settings.embedding_model
        
        # Detect if model is an Ollama model
        is_ollama = ":" in model_name or "qwen3-embedding" in model_name.lower()

        if is_ollama:
            base_url = os.environ.get("OLLAMA_HOST") or settings.vllm_api_base.replace("/v1", "")
            print(f"✨ Khởi tạo mô hình Embedding sử dụng Ollama Batching: {model_name} tại {base_url}")
            _embedding_fn = OllamaBatchEmbeddings(
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
