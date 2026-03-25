"""ChromaDB vector store — indexes graph facts and file chunks for semantic retrieval.

Satisfies the VectorStore protocol defined in acervo.llm.
chromadb is an optional dependency — only imported when this module is used.
"""

from __future__ import annotations

import logging
from typing import Callable, Awaitable

log = logging.getLogger(__name__)

# Chunk size for splitting long file content
_CHUNK_SIZE = 500  # characters
_CHUNK_OVERLAP = 50


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


class ChromaVectorStore:
    """ChromaDB-backed vector store satisfying acervo's VectorStore protocol."""

    def __init__(
        self,
        persist_path: str,
        embed_fn: Callable[[str], Awaitable[list[float]]],
        embed_batch_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    ) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_path)
        self._facts = self._client.get_or_create_collection(
            "facts", metadata={"hnsw:space": "cosine"},
        )
        self._files = self._client.get_or_create_collection(
            "files", metadata={"hnsw:space": "cosine"},
        )
        self._embed = embed_fn
        self._embed_batch = embed_batch_fn

    async def search(self, query: str, n_results: int = 10) -> list[dict]:
        """Semantic search across both facts and file chunks."""
        embedding = await self._embed(query)
        return self._search_with_vector(embedding, n_results)

    async def search_with_embedding(self, embedding: list[float], n_results: int = 10) -> list[dict]:
        """Semantic search using a pre-computed embedding (avoids redundant embed call)."""
        return self._search_with_vector(embedding, n_results)

    def _search_with_vector(self, embedding: list[float], n_results: int) -> list[dict]:
        """Internal search using an embedding vector."""
        results: list[dict] = []

        # Search facts
        try:
            facts_hits = self._facts.query(
                query_embeddings=[embedding],
                n_results=min(n_results, max(self._facts.count(), 1)),
            )
            if facts_hits and facts_hits["documents"]:
                for i, doc in enumerate(facts_hits["documents"][0]):
                    meta = facts_hits["metadatas"][0][i] if facts_hits["metadatas"] else {}
                    dist = facts_hits["distances"][0][i] if facts_hits["distances"] else 1.0
                    results.append({
                        "text": doc,
                        "node_id": meta.get("node_id", ""),
                        "label": meta.get("label", ""),
                        "source": "fact",
                        "score": 1.0 - dist,  # cosine distance → similarity
                    })
        except Exception as e:
            log.warning("Facts search failed: %s", e)

        # Search files
        try:
            files_hits = self._files.query(
                query_embeddings=[embedding],
                n_results=min(n_results, max(self._files.count(), 1)),
            )
            if files_hits and files_hits["documents"]:
                for i, doc in enumerate(files_hits["documents"][0]):
                    meta = files_hits["metadatas"][0][i] if files_hits["metadatas"] else {}
                    dist = files_hits["distances"][0][i] if files_hits["distances"] else 1.0
                    results.append({
                        "text": doc,
                        "file_path": meta.get("file_path", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "source": "file",
                        "score": 1.0 - dist,
                    })
        except Exception as e:
            log.warning("Files search failed: %s", e)

        # Sort by score descending, return top n
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:n_results]

    async def index_facts(self, node_id: str, label: str, facts: list[str]) -> None:
        """Embed and index facts for a node. Replaces existing facts for this node."""
        if not facts:
            return
        # Remove old facts for this node
        self.remove_node(node_id)

        documents = [f"{label}: {fact}" for fact in facts if fact.strip()]
        if not documents:
            return

        # Batch embed all docs in one HTTP call
        if self._embed_batch:
            embeddings = await self._embed_batch(documents)
        else:
            embeddings = [await self._embed(doc) for doc in documents]

        ids = [f"{node_id}_f{i}" for i in range(len(documents))]
        metadatas = [{"node_id": node_id, "label": label, "fact_index": i} for i in range(len(documents))]

        self._facts.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        log.info("Indexed %d facts for node '%s'", len(ids), label)

    async def index_file_chunks(self, file_path: str, chunks: list[str]) -> None:
        """Embed and index chunks from a file."""
        if not chunks:
            return
        # Remove old chunks for this file
        self.remove_file(file_path)

        documents = [chunk for chunk in chunks if chunk.strip()]
        if not documents:
            return

        normalized = file_path.replace("\\", "/")
        file_id = normalized.replace("/", "_").replace(".", "_")

        # Batch embed all chunks in one HTTP call
        if self._embed_batch:
            embeddings = await self._embed_batch(documents)
        else:
            embeddings = [await self._embed(doc) for doc in documents]

        ids = [f"{file_id}_c{i}" for i in range(len(documents))]
        metadatas = [{"file_path": normalized, "chunk_index": i} for i in range(len(documents))]

        self._files.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        log.info("Indexed %d chunks for file '%s'", len(ids), file_path)

    def remove_node(self, node_id: str) -> None:
        """Remove all indexed facts for a node."""
        try:
            existing = self._facts.get(where={"node_id": node_id})
            if existing and existing["ids"]:
                self._facts.delete(ids=existing["ids"])
        except Exception:
            pass

    def remove_file(self, file_path: str) -> None:
        """Remove all indexed chunks for a file."""
        normalized = file_path.replace("\\", "/")
        try:
            existing = self._files.get(where={"file_path": normalized})
            if existing and existing["ids"]:
                self._files.delete(ids=existing["ids"])
        except Exception:
            pass

    @staticmethod
    def chunk_text(text: str) -> list[str]:
        """Public access to chunking utility."""
        return _chunk_text(text)
