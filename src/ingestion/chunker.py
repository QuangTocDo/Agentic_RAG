"""
Text chunker — split legal documents into overlapping chunks
while preserving article (Điều) boundaries when possible.
"""
from __future__ import annotations
import re
import sys, os

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings


def chunk_document(doc: dict, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[dict]:
    """
    Split a document into chunks. Tries to split on article boundaries first,
    then falls back to recursive character splitting.
    
    Returns list of dicts with 'page_content' and 'metadata'.
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    text = doc["page_content"]
    base_meta = doc.get("metadata", {})

    # Split by articles ("Điều X.")
    articles = _split_by_articles(text)

    chunks = []
    for article_text, article_num in articles:
        # If article is small enough, keep as one chunk
        if len(article_text) <= chunk_size:
            meta = {**base_meta, "article": article_num}
            chunks.append({"page_content": article_text.strip(), "metadata": meta})
        else:
            # Split long articles into sub-chunks
            sub_chunks = _recursive_split(article_text, chunk_size, chunk_overlap)
            for i, sc in enumerate(sub_chunks):
                meta = {**base_meta, "article": article_num, "sub_chunk": i}
                chunks.append({"page_content": sc.strip(), "metadata": meta})

    # If no articles found, just split the whole text
    if not chunks:
        sub_chunks = _recursive_split(text, chunk_size, chunk_overlap)
        for i, sc in enumerate(sub_chunks):
            meta = {**base_meta, "sub_chunk": i}
            chunks.append({"page_content": sc.strip(), "metadata": meta})

    return chunks


def _split_by_articles(text: str) -> list[tuple[str, str]]:
    """
    Split text at 'Điều X.' boundaries.
    Returns list of (article_text, article_number).
    """
    pattern = r"(Điều\s+\d+[a-zA-Z]?\.)"
    parts = re.split(pattern, text)

    articles = []
    # parts[0] is text before first Điều (header)
    if parts[0].strip():
        articles.append((parts[0].strip(), "header"))

    # Pair up: parts[1]=heading, parts[2]=body, parts[3]=heading, ...
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        # Extract article number
        num_match = re.search(r"Điều\s+(\d+[a-zA-Z]?)\.", heading)
        num = num_match.group(1) if num_match else "?"
        articles.append((heading + body, num))

    return articles


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Simple recursive character text splitter."""
    separators = ["\n\n", "\n", ". ", ", ", " "]
    return _split_with_separators(text, separators, chunk_size, chunk_overlap)


def _split_with_separators(
    text: str, separators: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Split text recursively using a list of separators, from coarsest to finest."""
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if len(separators) > 1 else []

    if sep and sep in text:
        parts = text.split(sep)
    else:
        if remaining_seps:
            return _split_with_separators(text, remaining_seps, chunk_size, chunk_overlap)
        # Last resort: hard split
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i : i + chunk_size])
        return chunks

    # Merge small parts into chunks
    chunks = []
    current = ""
    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If single part is too long, recurse with finer separator
            if len(part) > chunk_size and remaining_seps:
                sub = _split_with_separators(part, remaining_seps, chunk_size, chunk_overlap)
                chunks.extend(sub)
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Add overlap
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap_text = chunks[i - 1][-chunk_overlap:]
            overlapped.append(overlap_text + chunks[i])
        chunks = overlapped

    return chunks
