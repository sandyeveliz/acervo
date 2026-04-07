"""Vector store port — async semantic search over facts and file chunks."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStorePort(Protocol):
    """Async vector store interface for semantic search.

    Optional — if not provided, context gathering uses only the graph.
    """

    async def search(self, query: str, n_results: int = 10) -> list[dict]:
        """Semantic search. Returns list of dicts with 'text', 'node_id', 'source', 'score'."""
        ...

    async def search_with_embedding(self, embedding: list[float], n_results: int = 10) -> list[dict]:
        """Search using a pre-computed embedding vector."""
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


# Backward compat alias
VectorStore = VectorStorePort
