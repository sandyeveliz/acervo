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
        json_mode: bool = False,
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

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call. Default: sequential fallback."""
        return [await self.embed(t) for t in texts]


@runtime_checkable
class VectorStore(Protocol):
    """Async vector store interface for semantic search over facts and file chunks.

    Optional — if not provided, context gathering uses only the graph.
    AVS-Agents provides a ChromaDB implementation.
    """

    async def search(self, query: str, n_results: int = 10) -> list[dict]:
        """Semantic search. Returns list of dicts with 'text', 'node_id', 'source', 'score'."""
        ...

    async def index_facts(self, node_id: str, label: str, facts: list[str]) -> None:
        """Embed and index facts for a node."""
        ...

    async def index_file_chunks(self, file_path: str, chunks: list[str]) -> None:
        """Embed and index chunks from a file."""
        ...

    def remove_node(self, node_id: str) -> None:
        """Remove all indexed facts for a node."""
        ...

    def remove_file(self, file_path: str) -> None:
        """Remove all indexed chunks for a file."""
        ...
