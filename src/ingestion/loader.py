"""
Document loader — load legal .txt files from a directory.
Each file is read in full and returned as a Document-like dict.
"""
import os
from pathlib import Path


def load_text_file(file_path: str) -> dict:
    """Load a single .txt file and return a document dict."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "page_content": content,
        "metadata": {
            "source": str(path.resolve()),
            "filename": path.name,
            "law_name": _extract_law_name(content),
        },
    }


def load_directory(dir_path: str, extensions: tuple[str, ...] = (".txt",)) -> list[dict]:
    """Load all supported files from a directory recursively."""
    docs = []
    dir_p = Path(dir_path)
    if not dir_p.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    for root, _, files in os.walk(dir_p):
        for fname in sorted(files):
            if fname.lower().endswith(extensions):
                fpath = os.path.join(root, fname)
                docs.append(load_text_file(fpath))
    return docs


# --------------- helpers ---------------

def _extract_law_name(text: str) -> str:
    """Try to extract the law name from the first non-empty line."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return "Unknown"
