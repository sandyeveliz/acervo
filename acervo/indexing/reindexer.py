"""Reindexer — deferred re-indexing of stale file nodes.

When a file changes on disk after ingestion, its graph nodes become stale
(wrong line ranges, deleted symbols). The Reindexer re-parses stale files
and updates the graph, typically called after the conversation turn completes
to avoid blocking the pipeline.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from acervo.graph import TopicGraph

log = logging.getLogger(__name__)


def hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file's content."""
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class Reindexer:
    """Re-indexes stale file nodes in the graph.

    Usage:
        reindexer = Reindexer(graph, structural_parser, workspace_root)

        # During Gather — check freshness inline (facade does this)
        # After turn — re-index anything marked stale
        reindexed = await reindexer.reindex_stale()
    """

    def __init__(
        self,
        graph: TopicGraph,
        structural_parser: object,  # StructuralParser (avoid circular import)
        workspace_root: Path,
    ) -> None:
        self._graph = graph
        self._parser = structural_parser
        self._workspace_root = workspace_root

    async def reindex_stale(self) -> list[str]:
        """Re-index all file nodes marked stale.

        Re-parses each stale file with StructuralParser, updates graph nodes
        via upsert_file_structure(), and clears the stale flag.

        Returns:
            List of file paths that were successfully re-indexed.
        """
        stale_nodes = self._graph.get_stale_files()
        if not stale_nodes:
            return []

        reindexed: list[str] = []
        for node in stale_nodes:
            file_path = node.get("attributes", {}).get("path")
            if not file_path:
                continue

            full_path = self._workspace_root / file_path
            if not full_path.is_file():
                # File was deleted — remove the file node and its children
                self._graph.remove_node(node["id"])
                log.info("Removed deleted file from graph: %s", file_path)
                reindexed.append(file_path)
                continue

            try:
                structure = self._parser.parse(full_path, self._workspace_root)
                self._graph.upsert_file_structure(
                    structure.file_path,
                    structure.language,
                    structure.units,
                    structure.content_hash,
                )
                # Always clear stale flag after successful re-parse
                node["stale"] = False
                node["stale_since"] = None
                reindexed.append(file_path)
                log.info("Re-indexed stale file: %s (%d symbols)", file_path, len(structure.units))
            except Exception as e:
                log.warning("Failed to re-index %s: %s", file_path, e)

        if reindexed:
            self._graph.save()

        return reindexed

    def check_freshness(self, file_path: str) -> bool:
        """Check if a file node's content hash matches the current file on disk.

        Returns True if fresh (hash matches), False if stale or missing.
        Does NOT modify the graph — caller is responsible for marking stale.
        """
        from acervo.graph import _make_id

        file_id = _make_id(file_path)
        node = self._graph.get_node(file_id)
        if not node or node.get("kind") != "file":
            return False

        full_path = self._workspace_root / file_path
        if not full_path.is_file():
            return False

        stored_hash = node.get("attributes", {}).get("content_hash", "")
        try:
            current_hash = hash_file(full_path)
        except Exception:
            return False

        return current_hash == stored_hash
