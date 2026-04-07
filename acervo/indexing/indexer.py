"""Indexer — full indexation pipeline for a project directory.

Orchestrates two sequential phases:
  Phase 1: Structural analysis (tree-sitter, no LLM, fast)
  Phase 2: Semantic enrichment (embeddings + LLM summaries, parallel)

Then stores results into the Topic Graph and Vector DB.

Usage:
    indexer = Indexer(config, llm=my_llm, embedder=my_embedder, vector_store=my_vs)
    result = await indexer.index(workspace_root)

    # Or from CLI:
    python -m acervo.indexer /path/to/project
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from acervo.dependency_resolver import DependencyEdge, DependencyResolver
from acervo.graph import TopicGraph, _make_id, make_symbol_id
from acervo.llm import LLMClient, Embedder, VectorStore
from acervo.semantic_enricher import (
    ChunkEmbedding,
    EnrichmentResult,
    SemanticEnricher,
    SemanticSummary,
)
from acervo.structural_parser import FileStructure, StructuralParser

log = logging.getLogger(__name__)


# ── Progress events ──


@dataclass
class IndexingStarted:
    total_files: int


@dataclass
class FileAnalyzed:
    file_path: str
    entities: int
    phase: str = "structural"


@dataclass
class FileEnriched:
    file_path: str
    chunks: int
    phase: str = "semantic"


@dataclass
class FileIndexed:
    file_path: str
    nodes: int
    edges: int


@dataclass
class IndexingComplete:
    total_nodes: int
    total_edges: int
    total_files: int
    duration_seconds: float


@dataclass
class IndexingError:
    file_path: str
    error: str


# ── Result ──


@dataclass
class IndexResult:
    """Summary of a complete indexation run."""

    files_analyzed: int = 0
    files_enriched: int = 0
    files_skipped: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    total_chunks: int = 0
    total_summaries: int = 0
    dependency_edges: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ── Default excludes ──

DEFAULT_EXCLUDES: set[str] = {
    "node_modules", "dist", "build", ".git", "__pycache__",
    ".venv", "venv", ".next", "coverage", ".acervo",
}

DEFAULT_EXTENSIONS: set[str] = {
    ".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".html", ".css",
    ".epub", ".txt", ".pdf",
}


# ── Indexer ──


class Indexer:
    """Full indexation pipeline: structural parse → semantic enrich → store."""

    def __init__(
        self,
        graph: TopicGraph,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        on_event: Callable | None = None,
        content_type: str = "auto",
    ) -> None:
        """
        Args:
            graph: TopicGraph to populate with nodes and edges.
            llm: Optional LLM client for semantic summaries (3B model).
            embedder: Optional embedder for chunk embeddings.
            vector_store: Optional vector store for embedding storage.
            on_event: Optional callback for progress events.
            content_type: Content type hint ("auto", "code", "prose").
        """
        self._graph = graph
        self._llm = llm
        self._embedder = embedder
        self._vector_store = vector_store
        self._on_event = on_event
        self._parser = StructuralParser()
        self._enricher = SemanticEnricher(
            llm=llm, embedder=embedder, content_type=content_type,
        )

    def _emit(self, event: object) -> None:
        """Emit a progress event to the callback."""
        if self._on_event:
            try:
                self._on_event(event)
            except Exception as e:
                log.debug("Event callback error: %s", e)

    async def index(
        self,
        workspace_root: Path,
        extensions: set[str] | None = None,
        excludes: set[str] | None = None,
    ) -> IndexResult:
        """Run the full indexation pipeline on a workspace directory.

        Args:
            workspace_root: Root directory to index.
            extensions: File extensions to include (default: see DEFAULT_EXTENSIONS).
            excludes: Directory names to skip (default: see DEFAULT_EXCLUDES).

        Returns:
            IndexResult with summary statistics.
        """
        start_time = time.monotonic()
        extensions = extensions or DEFAULT_EXTENSIONS
        excludes = excludes or DEFAULT_EXCLUDES
        result = IndexResult()

        # Scan for files
        files = self._scan_files(workspace_root, extensions, excludes)
        self._emit(IndexingStarted(total_files=len(files)))
        log.info("Indexing %d files in %s", len(files), workspace_root)

        # ── Phase 1: Structural analysis ──
        structures: list[FileStructure] = []

        for file_path in files:
            try:
                structure = self._parser.parse(file_path, workspace_root)

                # Check if file is unchanged (hash match in graph)
                file_id = _make_id(structure.file_path)
                existing = self._graph.get_node(file_id)
                if existing:
                    old_hash = existing.get("attributes", {}).get("content_hash", "")
                    if old_hash == structure.content_hash:
                        result.files_skipped += 1
                        continue

                structures.append(structure)

                # Upsert structural nodes into graph
                symbols_created = self._graph.upsert_file_structure(
                    structure.file_path,
                    structure.language,
                    structure.units,
                    structure.content_hash,
                )

                self._emit(FileAnalyzed(
                    file_path=structure.file_path,
                    entities=len(structure.units),
                ))
                result.files_analyzed += 1

            except Exception as e:
                err_msg = f"{file_path}: {e}"
                result.errors.append(err_msg)
                self._emit(IndexingError(file_path=str(file_path), error=str(e)))
                log.warning("Phase 1 error: %s", err_msg)

        # Build dependency graph from resolved imports
        dep_resolver = DependencyResolver(workspace_root, structures)
        dep_edges = dep_resolver.resolve()
        self._store_dependency_edges(dep_edges)
        result.dependency_edges = len(dep_edges)

        # Save graph after Phase 1
        self._graph.save()

        # ── Phase 2: Semantic enrichment ──
        if self._llm or self._embedder:
            for structure in structures:
                try:
                    file_path = workspace_root / structure.file_path
                    if structure.full_text:
                        content = structure.full_text
                    else:
                        content = file_path.read_text(encoding="utf-8")

                    enrichment = await self._enricher.enrich_file(structure, content)

                    # Store embeddings in vector DB + link chunk_ids to graph nodes
                    if enrichment.chunks and self._vector_store:
                        await self._store_and_link_chunks(structure, enrichment.chunks)

                    # Attach summaries to graph nodes
                    if enrichment.summaries:
                        self._attach_summaries(structure, enrichment.summaries)

                    # Create semantic edges from implicit relations
                    if enrichment.summaries:
                        self._create_semantic_edges(structure, enrichment.summaries)

                    self._emit(FileEnriched(
                        file_path=structure.file_path,
                        chunks=len(enrichment.chunks),
                    ))
                    result.files_enriched += 1
                    result.total_chunks += len(enrichment.chunks)
                    result.total_summaries += len(enrichment.summaries)

                except Exception as e:
                    err_msg = f"{structure.file_path}: {e}"
                    result.errors.append(err_msg)
                    self._emit(IndexingError(
                        file_path=structure.file_path, error=str(e),
                    ))
                    log.warning("Phase 2 error: %s", err_msg)

        # Phase 3: Cross-file semantic edges from shared topics
        if self._llm:
            self._create_cross_file_edges()

        # Final save
        self._graph.save()

        result.total_nodes = self._graph.node_count
        result.total_edges = self._graph.edge_count
        result.duration_seconds = time.monotonic() - start_time

        self._emit(IndexingComplete(
            total_nodes=result.total_nodes,
            total_edges=result.total_edges,
            total_files=result.files_analyzed,
            duration_seconds=result.duration_seconds,
        ))

        log.info(
            "Indexing complete: %d files, %d nodes, %d edges in %.1fs",
            result.files_analyzed, result.total_nodes,
            result.total_edges, result.duration_seconds,
        )

        return result

    # ── File scanning ──

    def _scan_files(
        self,
        workspace_root: Path,
        extensions: set[str],
        excludes: set[str],
    ) -> list[Path]:
        """Scan workspace for indexable files, respecting excludes."""
        files: list[Path] = []

        for path in sorted(workspace_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in extensions:
                continue
            # Check if any parent directory matches exclude patterns
            parts = path.relative_to(workspace_root).parts
            if any(p in excludes for p in parts):
                continue
            files.append(path)

        return files

    # ── Storage: dependency edges ──

    def _store_dependency_edges(self, edges: list[DependencyEdge]) -> None:
        """Add dependency edges to the graph."""
        for edge in edges:
            source_id = _make_id(edge.source_file)
            target_id = _make_id(edge.target_file)

            # Both file nodes must exist
            if not self._graph.get_node(source_id) or not self._graph.get_node(target_id):
                continue

            self._graph.add_edge(
                source_id=source_id,
                target_id=target_id,
                relation="imports",
                weight=1.0,
                edge_type="structural",
            )

    # ── Storage: embeddings + chunk linkage ──

    async def _store_and_link_chunks(
        self,
        structure: FileStructure,
        chunks: list[ChunkEmbedding],
    ) -> None:
        """Store chunk embeddings in vector DB and link chunk_ids to graph nodes."""
        if not self._vector_store or not chunks:
            return

        file_id = _make_id(structure.file_path)

        try:
            # Store ALL chunks for this file in one call (fixes the bug where
            # per-chunk calls to index_file_chunks removed previous chunks)
            stored_ids = await self._vector_store.index_file_chunks(
                file_path=chunks[0].file_path,
                chunks=[c.content for c in chunks],
                chunk_ids=[c.chunk_id for c in chunks],
                embeddings=[c.embedding for c in chunks],
                extra_metadata={"file_id": file_id},
            )

            # Link ALL chunk_ids to the file node
            all_chunk_ids = [c.chunk_id for c in chunks]
            self._graph.link_chunks(file_id, all_chunk_ids)

            # Link chunk_ids to section/symbol nodes by line range overlap
            for unit in structure.units:
                node_id = make_symbol_id(structure.file_path, unit.name, unit.parent)
                matching_ids = [
                    c.chunk_id for c in chunks
                    if c.line_start >= unit.start_line and c.line_end <= unit.end_line
                ]
                if matching_ids:
                    self._graph.link_chunks(node_id, matching_ids)

            log.info(
                "Stored %d chunks and linked to %s (%d sections)",
                len(stored_ids), structure.file_path, len(structure.units),
            )
        except Exception as e:
            log.warning("Chunk storage/linkage error for %s: %s", structure.file_path, e)

    # ── Storage: semantic summaries ──

    def _attach_summaries(
        self,
        structure: FileStructure,
        summaries: list[SemanticSummary],
    ) -> None:
        """Attach semantic summaries to graph nodes as facts and attributes."""
        # Build a map from chunk_id → summary for lookup
        summary_map = {s.chunk_id: s for s in summaries}

        # Summaries align with structure.units (same ordering from enricher)
        for unit in structure.units:
            node_id = make_symbol_id(structure.file_path, unit.name, unit.parent)
            node = self._graph.get_node(node_id)
            if not node:
                continue

            # Find matching summary (match by entity name)
            matching = None
            for s in summaries:
                if not matching:
                    matching = s
                    summaries = summaries[1:]
                    break

            if matching:
                # Store summary and topics as node attributes
                attrs = node.setdefault("attributes", {})
                attrs["summary"] = matching.summary
                # Sanitize topics: LLM may return dicts instead of strings
                attrs["topics"] = [
                    str(t) if not isinstance(t, str) else t
                    for t in matching.topics
                    if t
                ]
                if matching.implicit_relations:
                    attrs["implicit_relations"] = [
                        str(r) if not isinstance(r, str) else r
                        for r in matching.implicit_relations
                        if r
                    ]

    def _create_semantic_edges(
        self,
        structure: FileStructure,
        summaries: list[SemanticSummary],
    ) -> None:
        """Create semantic 'related_to' edges from topic overlap within a file."""
        # Group nodes by topic
        topic_to_nodes: dict[str, list[str]] = {}

        for unit in structure.units:
            node_id = make_symbol_id(structure.file_path, unit.name, unit.parent)
            node = self._graph.get_node(node_id)
            if not node:
                continue
            topics = node.get("attributes", {}).get("topics", [])
            for topic in topics:
                if not isinstance(topic, str):
                    continue
                topic_to_nodes.setdefault(topic, []).append(node_id)

        # Create edges between nodes that share topics
        for topic, node_ids in topic_to_nodes.items():
            if len(node_ids) < 2:
                continue
            for i, src in enumerate(node_ids):
                for tgt in node_ids[i + 1:]:
                    if src != tgt:
                        self._graph.add_edge(
                            source_id=src,
                            target_id=tgt,
                            relation="related_to",
                            weight=0.6,
                            edge_type="semantic",
                        )

    def _create_cross_file_edges(self) -> None:
        """Create semantic edges between nodes in different files that share topics.

        Runs after all files have been enriched so cross-file relationships
        (e.g., same character appearing in multiple chapters) are connected.
        """
        _MAX_PAIRS_PER_TOPIC = 20

        topic_to_nodes: dict[str, list[str]] = {}

        for node in self._graph.get_all_nodes():
            topics = node.get("attributes", {}).get("topics", [])
            node_id = node.get("id", "")
            if not topics or not node_id:
                continue
            for topic in topics:
                if not isinstance(topic, str):
                    continue
                topic_lower = topic.lower().strip()
                if topic_lower:
                    topic_to_nodes.setdefault(topic_lower, []).append(node_id)

        edges_created = 0
        for topic, node_ids in topic_to_nodes.items():
            if len(node_ids) < 2 or len(node_ids) > 50:
                # Skip topics that appear in too many nodes (too generic)
                continue

            # Group by file to only create cross-file edges
            by_file: dict[str, list[str]] = {}
            for nid in node_ids:
                n = self._graph.get_node(nid)
                if n:
                    fpath = n.get("attributes", {}).get("file_path", n.get("attributes", {}).get("path", ""))
                    by_file.setdefault(fpath, []).append(nid)

            if len(by_file) < 2:
                continue

            # Create edges between nodes from different files
            file_keys = list(by_file.keys())
            pairs = 0
            for i, fk1 in enumerate(file_keys):
                for fk2 in file_keys[i + 1:]:
                    for src in by_file[fk1]:
                        for tgt in by_file[fk2]:
                            if pairs >= _MAX_PAIRS_PER_TOPIC:
                                break
                            self._graph.add_edge(
                                source_id=src,
                                target_id=tgt,
                                relation="related_to",
                                weight=0.5,
                                edge_type="semantic",
                            )
                            pairs += 1
                            edges_created += 1
                        if pairs >= _MAX_PAIRS_PER_TOPIC:
                            break
                    if pairs >= _MAX_PAIRS_PER_TOPIC:
                        break
                if pairs >= _MAX_PAIRS_PER_TOPIC:
                    break

        if edges_created:
            log.info("Created %d cross-file semantic edges", edges_created)
