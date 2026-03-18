"""LLM client protocols — the abstractions Acervo needs from the host."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Async LLM chat interface (utility model).

    Messages use the standard OpenAI-compatible format:
        [{"role": "user", "content": "..."}]

    Used for: extraction, planning, topic detection, summarization.
    Any provider (LM Studio, OpenAI, Ollama, etc.) can satisfy this
    protocol with a thin adapter.
    """

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        """Send messages and return the response text content."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Async embedding interface. Optional — used for topic detection L2.

    If not provided, topic detection falls back to keyword matching (L1)
    and LLM classification (L3), skipping the embedding step.
    """

    async def embed(self, text: str) -> list[float]:
        """Return embedding vector for text."""
        ...
