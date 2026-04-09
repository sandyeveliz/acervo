"""Graph Synthesizer — generates holistic understanding from indexed content.

Post-indexation phase that reads all file summaries, curation entities, and
structural metadata to produce synthesis nodes: project overview, module
summaries, and cross-cutting insights.

These synthesis nodes are stored in the graph and used by the context builder
to provide rich, high-level context that individual file chunks cannot.

Usage:
    from acervo.graph_synthesizer import synthesize_graph

    result = await synthesize_graph(graph, llm, on_progress=my_callback)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from acervo.graph import TopicGraph
from acervo.llm import LLMClient

log = logging.getLogger(__name__)

MAX_SUMMARY_LEN = 200
MAX_FILES_IN_PROMPT = 80
MAX_SECTIONS_PER_FILE = 15


# ── Result ──

@dataclass
class SynthesisResult:
    """Result of a synthesis run."""
    nodes_created: int = 0
    nodes_updated: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ── Prompts ──

from acervo.prompts import load_prompt

_CODE_PROJECT_PROMPT = load_prompt("synthesize_code")
_LITERARY_PROJECT_PROMPT = load_prompt("synthesize_literary")
_MODULE_SUMMARY_PROMPT = load_prompt("synthesize_module")


# ── Main entry point ──

async def synthesize_graph(
    graph: TopicGraph,
    llm: LLMClient,
    project_description: str = "",
    content_type: str = "auto",
    on_progress: Callable[[str, dict], None] | None = None,
) -> SynthesisResult:
    """Run synthesis across the entire graph to produce understanding nodes.

    Args:
        graph: The knowledge graph (post-indexation, post-curation).
        llm: LLM client for generating summaries.
        project_description: Optional project description from the user.
        content_type: "auto", "code", or "prose". Auto-detects from extensions.
        on_progress: Optional callback(event_name, data) for progress updates.

    Returns:
        SynthesisResult with counts.
    """
    start = time.time()
    result = SynthesisResult()

    def emit(event: str, data: dict) -> None:
        if on_progress:
            on_progress(event, data)

    all_nodes = graph.get_all_nodes()
    file_nodes = [n for n in all_nodes if n.get("kind") == "file"]

    if not file_nodes:
        emit("synthesis_complete", {"nodes_created": 0, "duration_seconds": 0})
        return result

    # Detect content type
    effective_type = _detect_content_type(file_nodes, content_type)

    # Collect curation entities for context
    curation_entities = [
        n for n in all_nodes
        if n.get("source") == "curation" and n.get("kind") == "entity"
    ]
    curation_block = _build_curation_block(curation_entities, graph)

    emit("synthesis_started", {
        "file_count": len(file_nodes),
        "content_type": effective_type,
    })

    # ── Tier 1: Project Overview ──
    try:
        overview = await _generate_project_overview(
            llm, file_nodes, graph, project_description,
            effective_type, curation_block,
        )
        if overview:
            _upsert_synthesis_node(
                graph,
                node_id="synthesis:project_overview",
                label="Project Overview",
                node_type="project_overview",
                summary=overview,
                attributes={
                    "content_type": effective_type,
                    "source_files": len(file_nodes),
                },
            )
            result.nodes_created += 1
            emit("overview_generated", {"length": len(overview)})
    except Exception as e:
        log.warning("Project overview synthesis failed: %s", e)
        result.errors.append(f"Project overview: {e}")

    # ── Tier 2: Module/Group Summaries ──
    groups = _group_files_by_directory(file_nodes)
    # Only generate module summaries if there are multiple groups
    if len(groups) > 1:
        for group_name, group_files in groups.items():
            if len(group_files) < 2:
                continue
            try:
                summary = await _generate_module_summary(
                    llm, group_name, group_files, graph, curation_block,
                )
                if summary:
                    safe_id = re.sub(r"[^a-z0-9_]", "_", group_name.lower())
                    _upsert_synthesis_node(
                        graph,
                        node_id=f"synthesis:module_{safe_id}",
                        label=f"Module: {group_name}",
                        node_type="module_summary",
                        summary=summary,
                        attributes={
                            "group_name": group_name,
                            "source_files": len(group_files),
                        },
                    )
                    result.nodes_created += 1
            except Exception as e:
                log.warning("Module summary failed for %s: %s", group_name, e)
                result.errors.append(f"Module {group_name}: {e}")

        emit("modules_generated", {"count": result.nodes_created - 1})

    # Save and finish
    graph.save()
    result.duration_seconds = time.time() - start

    emit("synthesis_complete", {
        "nodes_created": result.nodes_created,
        "nodes_updated": result.nodes_updated,
        "duration_seconds": result.duration_seconds,
    })

    log.info(
        "Synthesis complete: %d nodes created in %.1fs",
        result.nodes_created, result.duration_seconds,
    )

    return result


# ── Generators ──

async def _generate_project_overview(
    llm: LLMClient,
    file_nodes: list[dict],
    graph: TopicGraph,
    description: str,
    content_type: str,
    curation_block: str,
) -> str:
    """Generate a project-level overview using the LLM."""
    file_list = _build_file_list(file_nodes, graph)
    description_block = f"Project description: {description}" if description else ""

    prompt_template = (
        _LITERARY_PROJECT_PROMPT if content_type == "prose"
        else _CODE_PROJECT_PROMPT
    )

    prompt = prompt_template.format(
        description_block=description_block,
        file_count=len(file_nodes),
        file_list=file_list,
        curation_block=curation_block,
    )

    response = await llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )

    return _clean_response(response)


async def _generate_module_summary(
    llm: LLMClient,
    group_name: str,
    file_nodes: list[dict],
    graph: TopicGraph,
    curation_block: str,
) -> str:
    """Generate a summary for a file group/module."""
    file_list = _build_file_list(file_nodes, graph)

    prompt = _MODULE_SUMMARY_PROMPT.format(
        group_name=group_name,
        file_list=file_list,
        curation_block=curation_block,
    )

    response = await llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )

    return _clean_response(response)


# ── Graph storage ──

def _upsert_synthesis_node(
    graph: TopicGraph,
    node_id: str,
    label: str,
    node_type: str,
    summary: str,
    attributes: dict | None = None,
) -> None:
    """Create or update a synthesis node in the graph."""
    from datetime import datetime, timezone

    existing = graph.get_node(node_id)
    attrs = attributes or {}
    attrs["summary"] = summary
    attrs["generated_at"] = datetime.now(timezone.utc).isoformat()

    if existing:
        existing["label"] = label
        existing["type"] = node_type
        existing["attributes"] = {**existing.get("attributes", {}), **attrs}
    else:
        graph._nodes[node_id] = {
            "id": node_id,
            "label": label,
            "type": node_type,
            "kind": "synthesis",
            "source": "synthesis",
            "layer": "PERSONAL",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_active": datetime.now(timezone.utc).isoformat(),
            "session_count": 0,
            "facts": [],
            "attributes": attrs,
        }


# ── Helpers ──

def _detect_content_type(file_nodes: list[dict], configured: str) -> str:
    """Detect whether this is a code or prose/literary project."""
    if configured != "auto":
        return configured

    prose_extensions = {".epub", ".md", ".txt", ".rst"}
    code_extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".html", ".css"}

    prose_count = 0
    code_count = 0
    for node in file_nodes:
        path = node.get("attributes", {}).get("path", "")
        ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
        if ext in prose_extensions:
            prose_count += 1
        elif ext in code_extensions:
            code_count += 1

    return "prose" if prose_count > code_count else "code"


def _build_file_list(file_nodes: list[dict], graph: TopicGraph) -> str:
    """Build a compact file listing with summaries and section titles."""
    lines: list[str] = []

    for node in file_nodes[:MAX_FILES_IN_PROMPT]:
        path = node.get("attributes", {}).get("path", node.get("label", ""))
        summary = node.get("attributes", {}).get("summary", "")

        line = f"- {path}"
        if summary:
            line += f": {summary[:MAX_SUMMARY_LEN]}"

        # Get child sections
        nid = node.get("id", "")
        section_labels: list[str] = []
        for edge in graph.get_edges_for(nid):
            if edge.get("relation") == "contains":
                child_id = edge["target"] if edge["source"] == nid else edge["source"]
                child = graph.get_node(child_id)
                if child and child.get("kind") == "section":
                    section_labels.append(child.get("label", ""))
        if section_labels:
            line += f" [sections: {', '.join(section_labels[:MAX_SECTIONS_PER_FILE])}]"

        lines.append(line)

    if len(file_nodes) > MAX_FILES_IN_PROMPT:
        lines.append(f"... and {len(file_nodes) - MAX_FILES_IN_PROMPT} more files")

    return "\n".join(lines)


def _build_curation_block(curation_entities: list[dict], graph: TopicGraph) -> str:
    """Build a description of curation-discovered entities and relations."""
    if not curation_entities:
        return ""

    lines = ["Discovered relationships:"]
    for entity in curation_entities[:20]:
        label = entity.get("label", "")
        etype = entity.get("type", "")
        facts = entity.get("facts", [])

        line = f"- {label} ({etype})"
        if facts:
            fact_texts = [f.get("fact", "") for f in facts[:3] if f.get("fact")]
            if fact_texts:
                line += ": " + "; ".join(fact_texts)
        lines.append(line)

        # Show relations
        nid = entity.get("id", "")
        for edge in graph.get_edges_for(nid)[:5]:
            other_id = edge["target"] if edge["source"] == nid else edge["source"]
            other = graph.get_node(other_id)
            if other:
                rel = edge.get("relation", "related_to")
                lines.append(f"  → {rel}: {other.get('label', other_id)}")

    return "\n".join(lines)


def _group_files_by_directory(file_nodes: list[dict]) -> dict[str, list[dict]]:
    """Group file nodes by their top-level directory."""
    groups: dict[str, list[dict]] = {}

    for node in file_nodes:
        path = node.get("attributes", {}).get("path", "")
        parts = path.split("/")
        # Use the first directory level as the group name
        group = parts[0] if len(parts) > 1 else "(root)"
        groups.setdefault(group, []).append(node)

    return groups


def _clean_response(raw: str) -> str:
    """Clean LLM response: strip code fences, extra whitespace."""
    text = raw.strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
