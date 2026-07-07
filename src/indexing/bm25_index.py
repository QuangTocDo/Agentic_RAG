"""
BM25 keyword index — build, save, load and search.
"""
from __future__ import annotations
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings


class BM25Index:
    """BM25 index over pre-chunked legal documents."""

    def __init__(self):
        self.bm25 = None
        self.corpus_chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.corpus_chunks = chunks
        tokenized = [_tokenize(c["page_content"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  ✅ BM25 index built with {len(chunks)} documents")

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return top-k matching chunks by BM25 score."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() or load() first.")
        if top_k is None:
            top_k = settings.retrieval_k

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.corpus_chunks[idx].copy()
                chunk["bm25_score"] = float(scores[idx])
                results.append(chunk)
        return results

    def save(self, path: str | None = None) -> None:
        """Save index to disk."""
        if path is None:
            path = settings.bm25_index_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "corpus": self.corpus_chunks}, f)
        print(f"  ✅ BM25 index saved to {path}")

    def load(self, path: str | None = None) -> None:
        """Load index from disk."""
        if path is None:
            path = settings.bm25_index_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index not found at: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.corpus_chunks = data["corpus"]
        print(f"  ✅ BM25 index loaded ({len(self.corpus_chunks)} docs)")

    def append_and_build(self, chunks: list[dict]) -> None:
        """Load existing index, append new chunks, deduplicate, and build."""
        try:
            self.load()
        except FileNotFoundError:
            self.corpus_chunks = []
        
        # Deduplicate to avoid adding the same chunks twice
        existing_contents = {c["page_content"] for c in self.corpus_chunks}
        added_count = 0
        for chunk in chunks:
            if chunk["page_content"] not in existing_contents:
                self.corpus_chunks.append(chunk)
                existing_contents.add(chunk["page_content"])
                added_count += 1
        
        print(f"  ➕ Appending {added_count} new chunks to BM25 index (Total: {len(self.corpus_chunks)})")
        self.build(self.corpus_chunks)


# Global cached instance
_bm25_instance: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25Index()
        try:
            _bm25_instance.load()
        except FileNotFoundError:
            print("  ⚠️  BM25 index file not found, initializing empty index.")
    return _bm25_instance


def _tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text for BM25, with a regex fallback."""
    text = text.lower()
    try:
        from underthesea import word_tokenize

        text = word_tokenize(text, format="text")
    except Exception:
        pass
    # Keep Vietnamese characters and digits
    tokens = re.findall(r"[\w]+", text, re.UNICODE)
    # Remove very short tokens (1 char)
    return [t for t in tokens if len(t) > 1]
