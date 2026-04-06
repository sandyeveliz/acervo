"""Embedder port — async embedding interface for topic detection L2."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedderPort(Protocol):
    """Async embedding interface. Optional — used for topic detection L2.

    If not provided, topic detection falls back to keyword matching (L1)
    and LLM classification (L3), skipping the embedding step.
    """

    async def embed(self, text: str) -> list[float]:
        """Return embedding vector for text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call."""
        return [await self.embed(t) for t in texts]


# Backward compat alias
Embedder = EmbedderPort
