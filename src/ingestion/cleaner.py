"""
Text cleaner — normalize Vietnamese legal text.
"""
import re
import unicodedata


def clean_text(text: str) -> str:
    """Full cleaning pipeline for Vietnamese legal text."""
    text = normalize_unicode(text)
    text = strip_html(text)
    text = collapse_whitespace(text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Normalize to NFC (composed form) for consistent Vietnamese diacritics."""
    return unicodedata.normalize("NFC", text)


def strip_html(text: str) -> str:
    """Remove any HTML tags."""
    return re.sub(r"<[^>]+>", "", text)


def collapse_whitespace(text: str) -> str:
    """Replace multiple spaces / blank lines with single space or newline."""
    # collapse multiple blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse multiple spaces to one (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text
