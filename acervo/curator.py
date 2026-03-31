"""Batch Curation — LLM-powered cross-file relationship discovery.

Analyzes groups of indexed file nodes and discovers relationships that
cannot be found from individual files alone: saga ordering, shared
characters, thematic connections, hierarchical grouping.

Usage:
    from acervo.curator import curate_graph

    result = await curate_graph(graph, llm, on_progress=my_callback)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from acervo.graph import TopicGraph, _make_id
from acervo.layers import Layer
from acervo.llm import LLMClient

log = logging.getLogger(__name__)

BATCH_SIZE = 10
MAX_SUMMARY_LEN = 200

_CURATION_PROMPT = """\
You are analyzing a collection of files in a project. For each file I provide:
- File name and path
- Summary (if available)
- Section/chapter titles

Your job: discover meaningful relationships BETWEEN these files. Look for:
1. **Series/saga ordering**: Which files are sequels, prequels, or parts of a series?
2. **Shared elements**: Characters, themes, locations, or concepts that appear across files.
3. **Hierarchical grouping**: Create parent entities (e.g., "Harry Potter Saga", "API Documentation") that group related files.
4. **Authorship**: If you can identify a common author or creator.
5. **Thematic connections**: Files that share topics, genres, or subject matter.

IMPORTANT:
- Only create relationships you are confident about based on the file names, summaries, and section titles.
- Use the node IDs provided (not labels) when referencing existing files.
- For new entities (like a saga or author), provide a descriptive name and appropriate type.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "entities": [
    {{"name": "Entity Name", "type": "concept|document|person|organization|event"}}
  ],
  "relations": [
    {{"source": "node_id_or_new_name", "target": "node_id_or_new_name", "relation": "verb_phrase"}}
  ],
  "facts": [
    {{"entity": "node_id_or_name", "fact": "specific claim", "source": "curation"}}
  ]
}}

Valid relation types: part_of, sequel_of, prequel_of, created_by, shares_characters_with, shares_theme_with, related_to, contains, member_of, located_in, documented_in

FILES:
{files_block}
"""


@dataclass
class CurationResult:
    """Result of a batch curation run."""
    total_relations: int = 0
    total_entities: int = 0
    total_facts: int = 0
    batches_processed: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


async def curate_graph(
    graph: TopicGraph,
    llm: LLMClient,
    on_progress: Callable[[str, dict], None] | None = None,
) -> CurationResult:
    """Run batch curation across all indexed file nodes.

    Groups file nodes into batches, sends each to the LLM for relationship
    discovery, and applies results to the graph.

    Args:
        graph: The knowledge graph to curate.
        llm: LLM client for analysis.
        on_progress: Optional callback(event_name, data) for progress updates.

    Returns:
        CurationResult with totals.
    """
    start = time.time()
    result = CurationResult()

    def emit(event: str, data: dict) -> None:
        if on_progress:
            on_progress(event, data)

    # Collect all file nodes
    file_nodes = [n for n in graph.get_all_nodes() if n.get("kind") == "file"]
    if not file_nodes:
        emit("curation_complete", {"total_relations": 0, "total_entities": 0, "duration_seconds": 0})
        return result

    # Group by directory prefix
    batches = _create_batches(file_nodes, BATCH_SIZE)
    emit("curation_started", {"total_batches": len(batches)})

    for i, batch in enumerate(batches):
        try:
            files_block = _build_files_block(batch, graph)
            prompt = _CURATION_PROMPT.format(files_block=files_block)

            response = await llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )

            parsed = _parse_response(response)
            applied = _apply_curation(graph, parsed)

            result.total_entities += applied["entities"]
            result.total_relations += applied["relations"]
            result.total_facts += applied["facts"]
            result.batches_processed += 1

            emit("batch_processed", {
                "batch": i + 1,
                "relations_found": applied["relations"],
                "entities_created": applied["entities"],
            })

        except Exception as e:
            log.warning("Curation batch %d failed: %s", i + 1, e)
            result.errors.append(f"Batch {i + 1}: {e}")
            result.batches_processed += 1

    graph.save()
    result.duration_seconds = time.time() - start

    emit("curation_complete", {
        "total_relations": result.total_relations,
        "total_entities": result.total_entities,
        "total_facts": result.total_facts,
        "duration_seconds": result.duration_seconds,
    })

    return result


def _create_batches(file_nodes: list[dict], batch_size: int) -> list[list[dict]]:
    """Group file nodes into batches, preferring same-directory grouping."""
    # Group by directory
    by_dir: dict[str, list[dict]] = {}
    for node in file_nodes:
        path = node.get("attributes", {}).get("path", "")
        parts = path.rsplit("/", 1)
        directory = parts[0] if len(parts) > 1 else ""
        by_dir.setdefault(directory, []).append(node)

    # Flatten into batches of ~batch_size, keeping directory groups together
    batches: list[list[dict]] = []
    current: list[dict] = []
    for nodes in by_dir.values():
        if len(current) + len(nodes) > batch_size and current:
            batches.append(current)
            current = []
        current.extend(nodes)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    return batches


def _build_files_block(batch: list[dict], graph: TopicGraph) -> str:
    """Build the FILES section of the curation prompt."""
    parts: list[str] = []
    for node in batch:
        nid = node.get("id", "")
        label = node.get("label", "")
        path = node.get("attributes", {}).get("path", "")
        summary = node.get("attributes", {}).get("summary", "")

        lines = [f"## {label} (id: {nid})"]
        if path:
            lines.append(f"Path: {path}")
        if summary:
            lines.append(f"Summary: {summary[:MAX_SUMMARY_LEN]}")

        # Get child sections via CONTAINS edges
        section_labels: list[str] = []
        for edge in graph.get_edges_for(nid):
            if edge.get("relation") == "contains":
                child_id = edge["target"] if edge["source"] == nid else edge["source"]
                child = graph.get_node(child_id)
                if child and child.get("kind") == "section":
                    section_labels.append(child.get("label", ""))
        if section_labels:
            lines.append("Sections: " + ", ".join(section_labels[:15]))

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _parse_response(raw: str) -> dict:
    """Parse LLM response into structured curation data."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            data = json.loads(match.group())
        else:
            log.warning("Failed to parse curation response")
            return {"entities": [], "relations": [], "facts": []}

    return {
        "entities": data.get("entities", []),
        "relations": data.get("relations", []),
        "facts": data.get("facts", []),
    }


def _apply_curation(graph: TopicGraph, parsed: dict) -> dict:
    """Apply parsed curation results to the graph."""
    counts = {"entities": 0, "relations": 0, "facts": 0}

    entities = parsed.get("entities", [])
    relations = parsed.get("relations", [])
    facts = parsed.get("facts", [])

    # Create new entities
    entity_pairs = []
    for e in entities:
        name = e.get("name", "")
        etype = e.get("type", "concept")
        if name:
            entity_pairs.append((name, etype))

    # Build relations tuples (source_name, target_name, relation)
    relation_tuples = []
    for r in relations:
        src = r.get("source", "")
        tgt = r.get("target", "")
        rel = r.get("relation", "related_to")
        if src and tgt:
            relation_tuples.append((src, tgt, rel))

    # Build facts tuples (entity_name, fact_text, source)
    fact_tuples = []
    for f in facts:
        entity = f.get("entity", "")
        fact_text = f.get("fact", "")
        if entity and fact_text:
            fact_tuples.append((entity, fact_text, "curation"))

    if entity_pairs or relation_tuples or fact_tuples:
        n_nodes, n_edges = graph.upsert_entities(
            entity_pairs,
            relation_tuples if relation_tuples else None,
            fact_tuples if fact_tuples else None,
            layer=Layer.PERSONAL,
            source="curation",
        )
        counts["entities"] = len(entity_pairs)
        counts["relations"] = len(relation_tuples)
        counts["facts"] = len(fact_tuples)

    return counts
