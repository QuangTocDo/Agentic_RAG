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
    """Remove HTML tags cleanly using BeautifulSoup to preserve spacing and structure."""
    if not text:
        return ""
    # Check if text contains HTML elements
    if "<" in text and ">" in text:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, "html.parser")
            
            # Insert a newline or space around block level tags to prevent text merging
            for tag_name in ["p", "div", "tr", "td", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"]:
                for tag in soup.find_all(tag_name):
                    tag.insert_before("\n")
                    tag.insert_after("\n")
                    
            text = soup.get_text()
        except Exception:
            # Fallback to regex if bs4 is not available or errors out
            text = re.sub(r"<[^>]+>", " ", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Replace multiple spaces / blank lines with single space or newline."""
    # collapse multiple blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse multiple spaces to one (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text
