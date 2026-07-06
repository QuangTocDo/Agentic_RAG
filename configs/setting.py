"""
Settings module — load from config.yaml + .env file.
Environment variables always override YAML values.
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings:
    """Singleton settings loader from config.yaml with env-var overrides."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str | None = None):
        if hasattr(self, "_loaded"):
            return
        if config_path is None:
            config_path = _PROJECT_ROOT / "configs" / "config.yaml"
        self.config_path = Path(config_path)
        self.config: dict = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        self._loaded = True

    # ---- helpers ----
    def get(self, key_path: str, default=None):
        """Get config value using dot notation, e.g., 'chunking.chunk_size'.
        Environment variables are checked first (dots replaced with underscores, uppercased).
        """
        env_var = key_path.replace(".", "_").upper()
        env_val = os.environ.get(env_var)
        if env_val is not None:
            # Try to cast to int/float if possible
            try:
                return int(env_val)
            except ValueError:
                try:
                    return float(env_val)
                except ValueError:
                    return env_val

        keys = key_path.split(".")
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    # ---- convenience properties ----
    @property
    def google_api_key(self) -> str:
        return os.environ.get("GOOGLE_API_KEY", "")

    @property
    def hf_token(self) -> str:
        return os.environ.get("HF_TOKEN", "")


    @property
    def llm_model(self) -> str:
        return os.environ.get("LLM_MODEL", "gemini-1.5-flash")

    @property
    def llm_provider(self) -> str:
        return os.environ.get("LLM_PROVIDER", "gemini")

    @property
    def vllm_api_base(self) -> str:
        return os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")


    @property
    def embedding_model(self) -> str:
        return os.environ.get(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

    @property
    def chroma_persist_dir(self) -> str:
        return os.environ.get("CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "data" / "chroma_db"))

    @property
    def chroma_collection(self) -> str:
        return os.environ.get("CHROMA_COLLECTION", "legal_documents")

    @property
    def bm25_index_path(self) -> str:
        return os.environ.get("BM25_INDEX_PATH", str(_PROJECT_ROOT / "data" / "bm25_index.pkl"))

    @property
    def graph_index_path(self) -> str:
        return os.environ.get("GRAPH_INDEX_PATH", str(_PROJECT_ROOT / "data" / "legal_graph.pkl"))

    @property
    def chunk_size(self) -> int:
        return self.get("chunking.chunk_size", 1000)

    @property
    def chunk_overlap(self) -> int:
        return self.get("chunking.chunk_overlap", 200)

    @property
    def retrieval_k(self) -> int:
        return self.get("retrieval.k", 5)

    @property
    def rrf_k(self) -> int:
        return self.get("retrieval.rrf_k", self.get("retrieval.rff_k", 60))

    @property
    def graph_max_hops(self) -> int:
        return self.get("retrieval.graph_max_hops", 2)

    @property
    def use_reranker(self) -> bool:
        value = self.get("retrieval.use_reranker", True)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def temperature(self) -> float:
        return self.get("generation.temperature", 0.0)

    @property
    def max_output_tokens(self) -> int:
        return self.get("generation.max_output_tokens", 2048)


# Global settings instance
settings = Settings()
