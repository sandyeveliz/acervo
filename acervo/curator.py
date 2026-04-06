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

from acervo._text import strip_think_blocks
from acervo.graph import TopicGraph, _make_id
from acervo.layers import Layer
from acervo.llm import LLMClient
from acervo.ontology import map_extractor_type

log = logging.getLogger(__name__)

BATCH_SIZE = 10
MAX_SUMMARY_LEN = 200

_CURATION_SYSTEM = """\
You are an entity and fact extractor for project files. Output valid JSON only. No explanation."""

_CURATION_PROMPT = """\
You are an entity and fact extractor for project files.
Extract structured knowledge from the project files below.

═══ RULES ═══
- Extract only what is explicitly stated. Do not infer or invent.
- Extract at least 3-5 entities per batch (technologies, people, concepts, locations).
- Use node IDs (provided in parentheses) when referencing existing files.
- Valid types: person, organization, project, technology, place, event, document, concept
- Valid relations: part_of, created_by, uses_technology, depends_on, appears_in, documented_in, related_to

═══ OUTPUT FORMAT ═══
Return ONLY valid JSON:

{{
  "entities": [
    {{
      "name": "exact name from text",
      "type": "person | technology | concept",
      "layer": "UNIVERSAL"
    }}
  ],
  "relations": [
    {{"source": "entity_name_or_node_id", "target": "entity_name_or_node_id", "relation": "verb_phrase"}}
  ],
  "facts": [
    {{"entity": "entity_name_or_node_id", "fact": "specific verifiable claim", "source": "curation"}}
  ]
}}

PROJECT FILES:
{files_block}

JSON:"""


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
                [
                    {"role": "system", "content": _CURATION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
                json_mode=True,
            )

            log.info("Curation batch %d raw response (%d chars): %s",
                     i + 1, len(response), response[:500])

            parsed = _parse_response(response)
            log.info("Curation batch %d parsed: %d entities, %d relations, %d facts",
                     i + 1, len(parsed.get("entities", [])),
                     len(parsed.get("relations", [])),
                     len(parsed.get("facts", [])))

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

        # Get child sections and their summaries via CONTAINS edges
        section_labels: list[str] = []
        child_summaries: list[str] = []
        for edge in graph.get_edges_for(nid):
            if edge.get("relation") == "contains":
                child_id = edge["target"] if edge["source"] == nid else edge["source"]
                child = graph.get_node(child_id)
                if child and child.get("kind") in ("section", "symbol"):
                    child_label = child.get("label", "")
                    section_labels.append(child_label)
                    # Include child summaries for richer context
                    child_summary = child.get("attributes", {}).get("summary", "")
                    if child_summary:
                        child_summaries.append(f"{child_label}: {child_summary[:100]}")
                    # Include facts from children
                    for fact in child.get("facts", [])[:2]:
                        fact_text = fact.get("fact", "") if isinstance(fact, dict) else str(fact)
                        if fact_text:
                            child_summaries.append(f"{child_label}: {fact_text[:100]}")
        if section_labels:
            lines.append("Sections: " + ", ".join(section_labels[:20]))
        if child_summaries:
            lines.append("Details:\n  " + "\n  ".join(child_summaries[:10]))

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _parse_response(raw: str) -> dict:
    """Parse LLM response into structured curation data."""
    # Strip <think>...</think> blocks (Qwen3 reasoning traces)
    cleaned = strip_think_blocks(raw).strip()
    # Strip markdown code fences if present
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
            log.warning("Failed to parse curation response: %s", cleaned[:300])
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

    # Create new entities (map lowercase types to ontology-capitalized)
    entity_pairs = []
    for e in entities:
        name = e.get("name", "")
        etype = map_extractor_type(e.get("type", "concept"))
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
