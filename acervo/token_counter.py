"""Token counting utility.

Uses tiktoken if available, otherwise falls back to a word-based estimate.
"""

from __future__ import annotations


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    return _count(text)


try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def _count(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:
    def _count(text: str) -> int:
        # Rough estimate: ~0.75 tokens per word for English/Spanish
        return max(1, int(len(text.split()) * 1.3))
