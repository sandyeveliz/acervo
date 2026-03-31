"""ChromaDB vector store — indexes graph facts and file chunks for semantic retrieval.

Satisfies the VectorStore protocol defined in acervo.llm.
chromadb is an optional dependency — only imported when this module is used.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Awaitable

log = logging.getLogger(__name__)

# Chunk size for splitting long file content
_CHUNK_SIZE = 500  # characters
_CHUNK_OVERLAP = 50


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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

    async def index_file_chunks(
        self,
        file_path: str,
        chunks: list[str],
        chunk_ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        extra_metadata: dict | None = None,
    ) -> list[str]:
        """Embed and index chunks from a file.

        Args:
            file_path: Source file path.
            chunks: Chunk text content.
            chunk_ids: Use these as ChromaDB IDs (instead of auto-generated).
            embeddings: Pre-computed embeddings (skip re-embedding).
            extra_metadata: Extra metadata merged into each chunk's metadata.

        Returns:
            List of stored chunk IDs.
        """
        if not chunks:
            return []
        # Remove old chunks for this file
        self.remove_file(file_path)

        documents = [chunk for chunk in chunks if chunk.strip()]
        if not documents:
            return []

        normalized = file_path.replace("\\", "/")

        # Use provided embeddings or compute them
        if embeddings:
            vecs = embeddings
        elif self._embed_batch:
            vecs = await self._embed_batch(documents)
        else:
            vecs = [await self._embed(doc) for doc in documents]

        # Use provided chunk_ids or generate auto IDs
        if chunk_ids:
            ids = chunk_ids[:len(documents)]
        else:
            file_id = normalized.replace("/", "_").replace(".", "_")
            ids = [f"{file_id}_c{i}" for i in range(len(documents))]

        metadatas = [{"file_path": normalized, "chunk_index": i} for i in range(len(documents))]
        if extra_metadata:
            for m in metadatas:
                m.update(extra_metadata)

        self._files.add(
            ids=ids,
            documents=documents,
            embeddings=vecs,
            metadatas=metadatas,
        )
        log.info("Indexed %d chunks for file '%s'", len(ids), file_path)
        return ids

    async def search_by_chunk_ids(
        self,
        chunk_ids: list[str],
        query_embedding: list[float],
        n_results: int = 3,
    ) -> list[dict]:
        """Retrieve chunks by IDs and rank by cosine similarity to query.

        Fetches the specified chunks from ChromaDB, then ranks them against
        the query embedding. Efficient for small sets (typical: 3-30 chunks per node).
        """
        if not chunk_ids:
            return []

        try:
            results = self._files.get(
                ids=chunk_ids,
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception as e:
            log.warning("search_by_chunk_ids get failed: %s", e)
            return []

        if not results or not results.get("documents"):
            return []

        # Rank by cosine similarity
        scored: list[dict] = []
        embeddings = results.get("embeddings")
        metadatas = results.get("metadatas")
        for i, doc in enumerate(results["documents"]):
            meta = metadatas[i] if metadatas is not None else {}
            emb = embeddings[i] if embeddings is not None else None
            score = _cosine_similarity(query_embedding, list(emb)) if emb is not None else 0.0
            scored.append({
                "text": doc,
                "chunk_id": results["ids"][i],
                "file_path": meta.get("file_path", ""),
                "source": "node_scoped_chunk",
                "score": score,
            })

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:n_results]

    def remove_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Remove specific chunks by their IDs."""
        if not chunk_ids:
            return
        try:
            self._files.delete(ids=chunk_ids)
        except Exception as e:
            log.warning("remove_by_chunk_ids failed: %s", e)

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

    # -- Read-only inspection methods (no embedding needed) --

    def get_collection_stats(self) -> dict:
        """Return chunk/fact counts from both collections."""
        return {
            "facts_count": self._facts.count(),
            "files_count": self._files.count(),
        }

    def get_all_file_chunks(self, file_path: str | None = None) -> list[dict]:
        """Retrieve all file chunks, optionally filtered by file_path.

        Returns list of {chunk_id, file_path, chunk_index, content}.
        No embedding needed — uses ChromaDB get().
        """
        kwargs: dict = {"include": ["documents", "metadatas"]}
        if file_path:
            normalized = file_path.replace("\\", "/")
            kwargs["where"] = {"file_path": normalized}

        try:
            results = self._files.get(**kwargs)
        except Exception as e:
            log.warning("get_all_file_chunks failed: %s", e)
            return []

        if not results or not results.get("ids"):
            return []

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            doc = results["documents"][i] if results.get("documents") else ""
            chunks.append({
                "chunk_id": chunk_id,
                "file_path": meta.get("file_path", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "content": doc,
            })

        chunks.sort(key=lambda c: (c["file_path"], c["chunk_index"]))
        return chunks

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Retrieve chunks by their IDs without ranking (no embedding needed).

        Returns list of {chunk_id, file_path, chunk_index, content}.
        """
        if not chunk_ids:
            return []

        try:
            results = self._files.get(
                ids=chunk_ids,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            log.warning("get_chunks_by_ids failed: %s", e)
            return []

        if not results or not results.get("ids"):
            return []

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            doc = results["documents"][i] if results.get("documents") else ""
            chunks.append({
                "chunk_id": chunk_id,
                "file_path": meta.get("file_path", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "content": doc,
            })
        return chunks

    @staticmethod
    def chunk_text(text: str) -> list[str]:
        """Public access to chunking utility."""
        return _chunk_text(text)
