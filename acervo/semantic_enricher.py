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

from acervo._text import strip_think_blocks
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
        content_type: str = "auto",
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._concurrency = concurrency
        self._content_type = content_type

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

        # Resolve content type once per batch (auto-detection uses chunk stats)
        self._effective_content_type = self._resolve_content_type(chunks)

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
        # For binary formats, use full_text (epub/pdf store extracted text there)
        text = structure.full_text if structure.full_text else content
        lines = text.split("\n")

        if structure.language == "markdown":
            return self._chunk_markdown(structure, lines)
        if structure.language in ("epub", "pdf", "plaintext"):
            return self._chunk_prose(structure, lines)
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
        """Create chunks from markdown heading sections.

        Small sections become single chunks. Large sections (>2000 chars) are
        split at paragraph boundaries, same as prose chunking.
        """
        max_chars = _MAX_CHUNK_TOKENS * 4
        chunks: list[Chunk] = []

        for unit in structure.units:
            section_text = "\n".join(lines[unit.start_line - 1 : unit.end_line])
            if len(section_text.strip()) < 20:
                continue

            hierarchy = unit.name
            if unit.parent:
                hierarchy = f"{unit.parent} > {unit.name}"

            # Small section — keep as one chunk
            if len(section_text) <= max_chars:
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
                continue

            # Large section — split at paragraph boundaries
            # Reuse the prose chunking logic via a temporary structure
            temp = FileStructure(
                file_path=structure.file_path,
                language="markdown",
                content_hash="",
                units=[unit],
                total_lines=unit.end_line - unit.start_line + 1,
            )
            sub_chunks = self._chunk_prose(temp, lines)
            # Preserve hierarchy in entity_name
            for sc in sub_chunks:
                sc.entity_name = hierarchy
            chunks.extend(sub_chunks)

        return chunks

    def _chunk_prose(
        self, structure: FileStructure, lines: list[str],
    ) -> list[Chunk]:
        """Split prose content (epub, pdf, txt) into paragraph-cluster chunks.

        Unlike markdown chunking (one chunk per heading section), this splits
        each section into ~500-token chunks at paragraph boundaries (blank lines).
        This fixes the under-chunking problem where entire book chapters became
        single 20k+ char chunks with one embedding vector.
        """
        max_chars = _MAX_CHUNK_TOKENS * 4  # ~2000 chars per chunk
        chunks: list[Chunk] = []

        for unit in structure.units:
            section_lines = lines[unit.start_line - 1 : unit.end_line]
            section_text = "\n".join(section_lines)

            if len(section_text.strip()) < 20:
                continue

            # If section is small enough, keep as one chunk
            if len(section_text) <= max_chars:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4())[:12],
                    file_path=structure.file_path,
                    entity_name=unit.name,
                    entity_kind="section",
                    line_start=unit.start_line,
                    line_end=unit.end_line,
                    content=section_text,
                    language=structure.language,
                    parent=unit.parent,
                ))
                continue

            # Split into paragraphs (blank-line separated), then cluster
            paragraphs: list[tuple[int, int, str]] = []  # (start, end, text)
            para_start = 0
            para_lines: list[str] = []

            for i, line in enumerate(section_lines):
                if not line.strip() and para_lines:
                    para_text = "\n".join(para_lines)
                    if para_text.strip():
                        paragraphs.append((
                            unit.start_line + para_start,
                            unit.start_line + i - 1,
                            para_text,
                        ))
                    para_start = i + 1
                    para_lines = []
                else:
                    para_lines.append(line)

            # Last paragraph
            if para_lines:
                para_text = "\n".join(para_lines)
                if para_text.strip():
                    paragraphs.append((
                        unit.start_line + para_start,
                        unit.end_line,
                        para_text,
                    ))

            # Cluster paragraphs into chunks of ~max_chars
            cluster_paras: list[tuple[int, int, str]] = []
            cluster_chars = 0

            for para_s, para_e, para_t in paragraphs:
                if cluster_chars + len(para_t) > max_chars and cluster_paras:
                    # Flush current cluster
                    c_start = cluster_paras[0][0]
                    c_end = cluster_paras[-1][1]
                    c_text = "\n\n".join(p[2] for p in cluster_paras)
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4())[:12],
                        file_path=structure.file_path,
                        entity_name=unit.name,
                        entity_kind="section",
                        line_start=c_start,
                        line_end=c_end,
                        content=c_text,
                        language=structure.language,
                        parent=unit.parent,
                    ))
                    cluster_paras = []
                    cluster_chars = 0

                cluster_paras.append((para_s, para_e, para_t))
                cluster_chars += len(para_t)

            # Flush remaining
            if cluster_paras:
                c_start = cluster_paras[0][0]
                c_end = cluster_paras[-1][1]
                c_text = "\n\n".join(p[2] for p in cluster_paras)
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4())[:12],
                    file_path=structure.file_path,
                    entity_name=unit.name,
                    entity_kind="section",
                    line_start=c_start,
                    line_end=c_end,
                    content=c_text,
                    language=structure.language,
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

    def _resolve_content_type(self, chunks: list[Chunk]) -> str:
        """Determine content type. If 'auto', detect from chunk languages."""
        if self._content_type != "auto":
            return self._content_type
        if not chunks:
            return "code"
        markdown_count = sum(1 for c in chunks if c.language == "markdown")
        return "prose" if markdown_count / len(chunks) > 0.7 else "code"

    def _build_summary_prompt(self, chunk: Chunk, content_type: str) -> str:
        """Build the LLM prompt for semantic summarization."""
        from acervo.prompts import load_prompt
        if content_type == "prose":
            template = load_prompt("enricher_prose")
        else:
            template = load_prompt("enricher_code")
        return template.format(
            file_path=chunk.file_path,
            entity_name=chunk.entity_name,
            entity_kind=chunk.entity_kind,
            language=getattr(chunk, "language", ""),
            content=chunk.content[:3000],
        )

    _ENRICHER_SYSTEM_PROMPT = ""  # loaded lazily below

    async def _summarize_chunk(self, chunk: Chunk) -> SemanticSummary:
        """Call the LLM to generate a semantic summary for one chunk."""
        from acervo.prompts import load_prompt
        prompt = self._build_summary_prompt(chunk, self._effective_content_type)
        system_prompt = load_prompt("enricher_system")

        response = await self._llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
            json_mode=True,
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
        text = strip_think_blocks(response).strip()

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
