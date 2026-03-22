"""File ingestor — reads files, optionally parses structure, extracts entities.

Ingestion flow:
1. (Optional) Structural parse — get functions, classes, sections with line ranges
2. Upsert structural nodes into graph (file + symbol nodes)
3. LLM extraction — pull entities, relations, and facts
   - For markdown: per-section extraction for better quality
   - For code: structural info is sufficient (no LLM needed)
4. Link file to each extracted entity node
5. Optionally index into vector store
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from acervo.graph import TopicGraph, _make_id

log = logging.getLogger(__name__)

# Supported extensions for structural parsing
_STRUCTURAL_EXTENSIONS = {".md", ".py", ".ts", ".tsx", ".js", ".jsx"}


@dataclass
class IngestResult:
    """Result of ingesting a file."""

    file: str
    entities: int = 0
    facts: int = 0
    relations: int = 0
    symbols: int = 0
    skipped: bool = False


class FileIngestor:
    """Reads files and extracts knowledge into the graph."""

    def __init__(self, llm, graph: TopicGraph, vector_store=None, structural_parser=None) -> None:
        """Args:
            llm: LLMClient for extraction
            graph: TopicGraph to persist entities/facts
            vector_store: optional VectorStore to index file chunks
            structural_parser: optional StructuralParser for code/markdown analysis
        """
        self._llm = llm
        self._graph = graph
        self._vector_store = vector_store
        self._parser = structural_parser

    async def ingest(self, file_path: Path, workspace_root: Path) -> IngestResult:
        """Ingest a file: structural parse + LLM extraction."""
        from acervo.extractor import TextExtractor

        content = file_path.read_text(encoding="utf-8")
        relative = str(file_path.relative_to(workspace_root)).replace("\\", "/")
        query = file_path.stem.replace("-", " ").replace("_", " ")
        is_markdown = file_path.suffix.lower() == ".md"

        # Phase 1: Structural parsing (if parser available)
        symbols_created = 0
        structure = None
        if self._parser and file_path.suffix.lower() in _STRUCTURAL_EXTENSIONS:
            structure = self._parser.parse(file_path, workspace_root)
            symbols_created = self._graph.upsert_file_structure(
                structure.file_path, structure.language,
                structure.units, structure.content_hash,
            )
            if symbols_created == 0 and structure.units:
                # File unchanged (hash match) — skip re-extraction
                log.info("File unchanged, skipping: %s", relative)
                return IngestResult(file=relative, skipped=True)

        # Phase 2: LLM extraction
        extractor = TextExtractor(self._llm)
        total_entities = 0
        total_facts = 0
        total_relations = 0

        if is_markdown and structure and structure.units:
            # Per-section extraction for markdown (better quality)
            lines = content.split("\n")
            for unit in structure.units:
                section_text = "\n".join(lines[unit.start_line - 1 : unit.end_line])
                if len(section_text.strip()) < 20:
                    continue
                result = await extractor.extract(section_text, query=unit.name)
                if result.entities:
                    self._graph.upsert_entities(
                        entities=[(e.name, e.type) for e in result.entities],
                        relations=[(r.source, r.target, r.relation) for r in result.relations],
                        facts=[(f.entity, f.fact, "file") for f in result.facts],
                        source="file",
                    )
                    for entity in result.entities:
                        node_id = _make_id(entity.name)
                        self._graph.link_file(node_id, relative)
                    total_entities += len(result.entities)
                    total_facts += len(result.facts)
                    total_relations += len(result.relations)
        elif is_markdown:
            # No structural parser — whole-file extraction (original flow)
            result = await extractor.extract(content, query=query)
            if result.entities:
                self._graph.upsert_entities(
                    entities=[(e.name, e.type) for e in result.entities],
                    relations=[(r.source, r.target, r.relation) for r in result.relations],
                    facts=[(f.entity, f.fact, "file") for f in result.facts],
                    source="file",
                )
                for entity in result.entities:
                    node_id = _make_id(entity.name)
                    self._graph.link_file(node_id, relative)
                total_entities = len(result.entities)
                total_facts = len(result.facts)
                total_relations = len(result.relations)
        # Code files: structural info is sufficient — no LLM extraction needed

        self._graph.save()

        # Index file chunks into vector store if available
        if self._vector_store and is_markdown:
            try:
                from providers.vector_store import ChromaVectorStore
                chunks = ChromaVectorStore.chunk_text(content)
                await self._vector_store.index_file_chunks(relative, chunks)
            except Exception as e:
                log.warning("Vector indexing failed for %s: %s", relative, e)

        log.info(
            "Ingested %s: %d entities, %d facts, %d relations, %d symbols",
            relative, total_entities, total_facts, total_relations, symbols_created,
        )

        return IngestResult(
            file=relative,
            entities=total_entities,
            facts=total_facts,
            relations=total_relations,
            symbols=symbols_created,
        )

    async def ingest_all(
        self,
        workspace_root: Path,
        extensions: set[str] | None = None,
    ) -> list[IngestResult]:
        """Ingest files in the workspace directory.

        Args:
            workspace_root: root directory to scan
            extensions: file extensions to include (default: .md only for LLM,
                        plus code extensions if structural parser is available)
        """
        if extensions is None:
            extensions = {".md"}
            if self._parser:
                extensions |= _STRUCTURAL_EXTENSIONS

        results: list[IngestResult] = []
        for ext in sorted(extensions):
            for f in sorted(workspace_root.rglob(f"*{ext}")):
                result = await self.ingest(f, workspace_root)
                results.append(result)
        return results
