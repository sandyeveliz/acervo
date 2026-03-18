"""Private text utilities for acervo."""

from __future__ import annotations

import re

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output.

    Handles both closed blocks and unclosed blocks (truncated by max_tokens).
    """
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()
