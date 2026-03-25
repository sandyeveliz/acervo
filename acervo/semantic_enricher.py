"""Semantic enricher — generates embeddings and LLM summaries for code chunks.

Phase 2 of the indexation pipeline. Takes structural units from Phase 1 and:
1. Creates logical chunks (entities for code, heading sections for markdown)
2. Generates embeddings for each chunk (via Embedder protocol)
3. Generates semantic summaries via LLM 3B (topics, description, implicit relations)

Embedding generation and LLM summarization run in parallel for each chunk.

Usage:
    enricher = SemanticEnricher(llm=my_client, embedder=my_embedder)
    results = await enricher.enrich_file(file_structure, file_content)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from acervo.llm import LLMClient, Embedder
from acervo.structural_parser import FileStructure, StructuralUnit

log = logging.getLogger(__name__)

# Maximum tokens for a single chunk before splitting
_MAX_CHUNK_TOKENS = 500  # approx 500 tokens ~ 2000 chars


@dataclass
class ChunkEmbedding:
    """A chunk with its embedding vector, ready for vector DB storage."""

    chunk_id: str
    file_path: str
    entity_name: str | None
    line_start: int
    line_end: int
    content: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class SemanticSummary:
    """LLM-generated semantic analysis of a code chunk."""

    chunk_id: str
    summary: str
    topics: list[str]
    implicit_relations: list[str]


@dataclass
class EnrichmentResult:
    """Combined result of enriching a single file."""

    file_path: str
    chunks: list[ChunkEmbedding]
    summaries: list[SemanticSummary]


@dataclass
class Chunk:
    """Intermediate chunk before enrichment."""

    chunk_id: str
    file_path: str
    entity_name: str | None
    entity_kind: str
    line_start: int
    line_end: int
    content: str
    language: str
    parent: str | None = None


class SemanticEnricher:
    """Generates embeddings and semantic summaries for file chunks."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        concurrency: int = 4,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._concurrency = concurrency

    async def enrich_file(
        self,
        structure: FileStructure,
        content: str,
    ) -> EnrichmentResult:
        """Enrich a single file: create chunks, generate embeddings + summaries."""
        chunks = self._create_chunks(structure, content)

        if not chunks:
            return EnrichmentResult(
                file_path=structure.file_path, chunks=[], summaries=[],
            )

        # Run embeddings and summaries in parallel
        embed_task = self._generate_embeddings(chunks)
        summary_task = self._generate_summaries(chunks)

        embeddings, summaries = await asyncio.gather(embed_task, summary_task)

        return EnrichmentResult(
            file_path=structure.file_path,
            chunks=embeddings,
            summaries=summaries,
        )

    def _create_chunks(
        self, structure: FileStructure, content: str,
    ) -> list[Chunk]:
        """Create logical chunks from a file's structural units."""
        lines = content.split("\n")

        if structure.language == "markdown":
            return self._chunk_markdown(structure, lines)
        return self._chunk_code(structure, lines)

    def _chunk_code(
        self, structure: FileStructure, lines: list[str],
    ) -> list[Chunk]:
        """Create chunks from code entities (one chunk per structural unit)."""
        chunks: list[Chunk] = []

        for unit in structure.units:
            source = "\n".join(lines[unit.start_line - 1 : unit.end_line])

            # If entity is too large, split at logical boundaries
            if len(source) > _MAX_CHUNK_TOKENS * 4:  # ~4 chars per token
                sub_chunks = self._split_large_entity(
                    unit, source, structure.file_path, structure.language,
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4())[:12],
                    file_path=structure.file_path,
                    entity_name=unit.name,
                    entity_kind=unit.unit_type,
                    line_start=unit.start_line,
                    line_end=unit.end_line,
                    content=source,
                    language=structure.language,
                    parent=unit.parent,
                ))

        return chunks

    def _split_large_entity(
        self,
        unit: StructuralUnit,
        source: str,
        file_path: str,
        language: str,
    ) -> list[Chunk]:
        """Split a large entity (e.g., class with many methods) into sub-chunks."""
        # For classes: each method becomes its own chunk (already handled by
        # StructuralParser which emits methods as separate units).
        # For very large functions: split at blank line boundaries.
        lines = source.split("\n")
        chunk_size = _MAX_CHUNK_TOKENS * 4  # chars

        chunks: list[Chunk] = []
        current_start = 0
        current_text: list[str] = []
        current_len = 0

        for i, line in enumerate(lines):
            current_text.append(line)
            current_len += len(line) + 1

            if current_len >= chunk_size and (not line.strip() or i == len(lines) - 1):
                abs_start = unit.start_line + current_start
                abs_end = unit.start_line + i
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4())[:12],
                    file_path=file_path,
                    entity_name=f"{unit.name}[{len(chunks)}]",
                    entity_kind=unit.unit_type,
                    line_start=abs_start,
                    line_end=abs_end,
                    content="\n".join(current_text),
                    language=language,
                    parent=unit.parent,
                ))
                current_start = i + 1
                current_text = []
                current_len = 0

        # Remaining lines
        if current_text:
            abs_start = unit.start_line + current_start
            abs_end = unit.end_line
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4())[:12],
                file_path=file_path,
                entity_name=f"{unit.name}[{len(chunks)}]",
                entity_kind=unit.unit_type,
                line_start=abs_start,
                line_end=abs_end,
                content="\n".join(current_text),
                language=language,
                parent=unit.parent,
            ))

        return chunks

    def _chunk_markdown(
        self, structure: FileStructure, lines: list[str],
    ) -> list[Chunk]:
        """Create chunks from markdown heading sections."""
        chunks: list[Chunk] = []

        for unit in structure.units:
            section_text = "\n".join(lines[unit.start_line - 1 : unit.end_line])
            if len(section_text.strip()) < 20:
                continue

            # Build heading hierarchy for context
            hierarchy = unit.name
            if unit.parent:
                hierarchy = f"{unit.parent} > {unit.name}"

            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4())[:12],
                file_path=structure.file_path,
                entity_name=hierarchy,
                entity_kind="section",
                line_start=unit.start_line,
                line_end=unit.end_line,
                content=section_text,
                language="markdown",
                parent=unit.parent,
            ))

        return chunks

    # ── Embedding generation ──

    async def _generate_embeddings(self, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        """Generate embeddings for all chunks using the embedder."""
        if not self._embedder:
            return []

        semaphore = asyncio.Semaphore(self._concurrency)
        results: list[ChunkEmbedding | None] = []

        async def embed_one(chunk: Chunk) -> ChunkEmbedding | None:
            async with semaphore:
                try:
                    # Prepend file context for better embeddings
                    embed_text = f"File: {chunk.file_path}\n"
                    if chunk.entity_name:
                        embed_text += f"{chunk.entity_kind}: {chunk.entity_name}\n"
                    embed_text += chunk.content

                    vector = await self._embedder.embed(embed_text)
                    return ChunkEmbedding(
                        chunk_id=chunk.chunk_id,
                        file_path=chunk.file_path,
                        entity_name=chunk.entity_name,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                        content=chunk.content,
                        embedding=vector,
                        metadata={
                            "language": chunk.language,
                            "kind": chunk.entity_kind,
                            "parent": chunk.parent,
                        },
                    )
                except Exception as e:
                    log.warning("Embedding failed for %s:%s: %s",
                                chunk.file_path, chunk.entity_name, e)
                    return None

        tasks = [embed_one(c) for c in chunks]
        raw = await asyncio.gather(*tasks)
        return [r for r in raw if r is not None]

    # ── LLM semantic summaries ──

    async def _generate_summaries(self, chunks: list[Chunk]) -> list[SemanticSummary]:
        """Generate semantic summaries for all chunks using the 3B model."""
        if not self._llm:
            return []

        semaphore = asyncio.Semaphore(self._concurrency)
        results: list[SemanticSummary | None] = []

        async def summarize_one(chunk: Chunk) -> SemanticSummary | None:
            async with semaphore:
                try:
                    return await self._summarize_chunk(chunk)
                except Exception as e:
                    log.warning("Summary failed for %s:%s: %s",
                                chunk.file_path, chunk.entity_name, e)
                    return None

        tasks = [summarize_one(c) for c in chunks]
        raw = await asyncio.gather(*tasks)
        return [r for r in raw if r is not None]

    async def _summarize_chunk(self, chunk: Chunk) -> SemanticSummary:
        """Call the 3B LLM to generate a semantic summary for one chunk."""
        prompt = f"""Analyze this code chunk and provide:
1. A 1-2 sentence summary of what it does
2. A list of semantic topics (e.g., authentication, database, CRUD, validation, routing, UI rendering)
3. Any implicit relationships not visible from imports (e.g., "this middleware protects these routes by convention", "this is called when the user clicks submit")

File: {chunk.file_path}
Language: {chunk.language}
Entity: {chunk.entity_name} ({chunk.entity_kind})

Code:
{chunk.content[:3000]}

Respond in JSON format:
{{"summary": "...", "topics": ["...", "..."], "implicit_relations": ["...", "..."]}}"""

        response = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )

        parsed = self._parse_summary_response(response)
        return SemanticSummary(
            chunk_id=chunk.chunk_id,
            summary=parsed.get("summary", ""),
            topics=parsed.get("topics", []),
            implicit_relations=parsed.get("implicit_relations", []),
        )

    @staticmethod
    def _parse_summary_response(response: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = response.strip()

        # Strip markdown code block if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

            log.debug("Failed to parse LLM summary response: %s", text[:200])
            return {"summary": text[:200], "topics": [], "implicit_relations": []}
