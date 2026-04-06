"""LLM port — async chat interface for utility model calls."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    """Async LLM chat interface.

    Messages use the standard OpenAI-compatible format:
        [{"role": "user", "content": "..."}]

    Used for: extraction, planning, topic detection, summarization.
    """

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 500,
        json_mode: bool = False,
    ) -> str:
        """Send messages and return the response text content."""
        ...


# Backward compat alias
LLMClient = LLMPort
