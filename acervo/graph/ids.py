"""Stable ID generation and text normalization for graph nodes.

These are pure functions with no storage dependency — shared across
TopicGraph (JSON backend) and LadybugGraphStore (LadybugDB backend).
"""

from __future__ import annotations

import re
import unicodedata


def _make_id(name: str) -> str:
    """Convert a name to a stable ASCII node ID. Strips accents."""
    nfkd = unicodedata.normalize("NFKD", name.lower())
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_str).strip("_")


def make_symbol_id(file_path: str, name: str, parent: str | None = None) -> str:
    """Build a stable ID for a structural symbol node.

    Convention:
        file: _make_id(path)
        symbol: _make_id(path) + "__" + _make_id(name)
        nested: _make_id(path) + "__" + _make_id(parent) + "__" + _make_id(name)
    """
    path_part = _make_id(file_path)
    if parent:
        return f"{path_part}__{_make_id(parent)}__{_make_id(name)}"
    return f"{path_part}__{_make_id(name)}"


_DEDUP_STRIP = re.compile(r"[^a-z0-9\s]")
_DEDUP_ARTICLES = re.compile(
    r"\b(el|la|los|las|un|una|de|del|en|the|a|an|of|in)\b"
)


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for dedup comparison: lowercase, no punctuation, no articles."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    no_punct = _DEDUP_STRIP.sub("", ascii_str)
    no_articles = _DEDUP_ARTICLES.sub("", no_punct)
    return " ".join(no_articles.split())
