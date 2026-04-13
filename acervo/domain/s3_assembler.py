"""S3 Assembler — layer-aware context assembly with compressed format.

Takes the LayeredContext from S2 (hot/warm/cold nodes) and builds
a token-budgeted context block using XML structure:

  <ctx>
  <hot>
    Supabase [Tech] — used by: Checkear, Walletfy
  </hot>
  <warm>
    Checkear [Project]: React + Supabase — gestión de tareas
    Walletfy [Project]: Next.js + Supabase — finanzas personales
  </warm>
  </ctx>

Format principles:
- Level 1 compression (~12tk per node): "Label [Type]: connections — desc"
- Correct relation direction: "Used by: X" not "uses_technology: X"
- XML delimiters: <ctx>, <hot>, <warm> for structured reference
- No "UNVERIFIED" label — graph data is presented as knowledge
- Intent-specific grounding instruction after context block
"""

from __future__ import annotations

import logging
from typing import Any

from acervo.domain.models import LayeredContext, S3Result
from acervo.token_counter import count_tokens

log = logging.getLogger(__name__)

# Budget per intent (tokens)
_BUDGETS = {
    "overview": 300,
    "specific": 600,
    "followup": 400,
    "chat": 100,
}

# Human-readable relation names (outgoing direction)
_RELATION_LABELS = {
    "uses_technology": "Uses",
    "works_at": "Works at",
    "maintains": "Maintains",
    "part_of": "Part of",
    "depends_on": "Depends on",
    "created_by": "Created by",
    "located_in": "Located in",
    "deployed_on": "Deployed on",
    "member_of": "Member of",
    "produces": "Produces",
    "serves": "Serves",
    "documented_in": "Documented in",
    "contains": "Contains",
}

# Reverse relation names (incoming direction)
_REVERSE_LABELS = {
    "uses_technology": "Used by",
    "works_at": "Team member",
    "maintains": "Maintained by",
    "part_of": "Contains",
    "created_by": "Creator",
    "depends_on": "Required by",
    "contains": "Part of",
}


class S3Assembler:
    """Assembles compressed context from layered graph nodes within budget."""

    def run(
        self,
        layered: LayeredContext,
        intent: str,
        graph: Any,
        project_overview: str = "",
        context_index: Any = None,
        history: list[dict] | None = None,
        current_topic: str = "none",
        warm_budget_override: int | None = None,
        query_embedding: list[float] | None = None,
    ) -> S3Result:
        """Execute S3: format nodes by layer, assemble within budget.

        When ``query_embedding`` is provided, Phase 4 MMR reranking runs on
        the WARM and COLD layers before the per-layer truncation. This
        favours diverse results within each layer instead of straight BFS
        order, without changing the HOT/WARM/COLD semantics.
        """
        budget = warm_budget_override or _BUDGETS.get(intent, 400)

        # Phase 4: optional MMR rerank within WARM and COLD layers. HOT is
        # left alone — those are the direct seeds and reordering would hide
        # the most relevant items.
        if query_embedding:
            layered = _mmr_rerank_layers(layered, query_embedding)

        # Build context block from layers
        warm_content = self._build_context_block(
            layered, graph, intent, project_overview, budget,
        )

        # Add grounding instruction
        if warm_content:
            grounding = _grounding_instruction(intent)
            if grounding:
                warm_content = f"{warm_content}\n\n{grounding}"

        warm_source = "graph" if warm_content else ""

        # Build context stack via ContextIndex (if provided)
        if context_index and history is not None:
            context_stack, hot_tk, warm_tk, total_tk = context_index.build_context_stack(
                history, current_topic,
                warm_override=warm_content,
                warm_source=warm_source,
            )
        else:
            warm_tk = count_tokens(warm_content) if warm_content else 0
            hot_tk = 0
            total_tk = warm_tk
            context_stack = []

        has_context = warm_tk > 0

        log.info(
            "[acervo] S3 — warm=%dtk hot=%dtk total=%dtk "
            "layers(hot=%d warm=%d cold=%d) intent=%s",
            warm_tk, hot_tk, total_tk,
            len(layered.hot), len(layered.warm), len(layered.cold),
            intent,
        )

        return S3Result(
            context_stack=context_stack,
            warm_content=warm_content,
            warm_tokens=warm_tk,
            hot_tokens=hot_tk,
            total_tokens=total_tk,
            has_context=has_context,
            needs_tool=not has_context,
        )

    def _build_context_block(
        self,
        layered: LayeredContext,
        graph: Any,
        intent: str,
        project_overview: str,
        budget: int,
    ) -> str:
        """Build compressed XML context block within budget."""
        if not layered.hot and not layered.warm:
            # No graph data — return just overview if present
            return project_overview if project_overview else ""

        inner_parts: list[str] = []
        tokens_used = 0

        # Project overview (before XML tags)
        overview_prefix = ""
        if project_overview:
            ov_tk = count_tokens(project_overview)
            if tokens_used + ov_tk <= budget:
                overview_prefix = project_overview
                tokens_used += ov_tk

        # HOT layer
        hot_lines: list[str] = []
        for node in layered.hot:
            line = _format_hot(node, graph)
            line_tk = count_tokens(line)
            if tokens_used + line_tk <= budget:
                hot_lines.append(line)
                tokens_used += line_tk

        # WARM layer
        warm_lines: list[str] = []
        for node in layered.warm:
            line = _format_warm(node, graph)
            line_tk = count_tokens(line)
            if tokens_used + line_tk <= budget:
                warm_lines.append(line)
                tokens_used += line_tk

        # COLD layer (specific/followup only)
        cold_str = ""
        if intent in ("specific", "followup") and layered.cold:
            cold_items = []
            for node in layered.cold:
                item = f"{node.get('label', '')} [{node.get('type', '')}]"
                item_tk = count_tokens(item)
                if tokens_used + item_tk <= budget:
                    cold_items.append(item)
                    tokens_used += item_tk
            if cold_items:
                cold_str = "  Also: " + ", ".join(cold_items)

        # Assemble XML structure
        parts: list[str] = []
        if overview_prefix:
            parts.append(overview_prefix)

        if hot_lines or warm_lines:
            parts.append("<ctx>")
            if hot_lines:
                parts.append("<hot>")
                parts.extend(hot_lines)
                parts.append("</hot>")
            if warm_lines:
                parts.append("<warm>")
                parts.extend(warm_lines)
                if cold_str:
                    parts.append(cold_str)
                parts.append("</warm>")
            parts.append("</ctx>")

        return "\n".join(parts) if parts else ""


# ── Node formatters (Level 1 compression) ──


def _format_hot(node: dict, graph: Any) -> str:
    """HOT node: full detail — Label [Type] + desc + grouped relations + facts."""
    label = node.get("label", "")
    ntype = node.get("type", "")
    desc = node.get("attributes", {}).get("description", "")
    nid = node.get("id", "")

    # Header line
    header = f"  {label} [{ntype}]"
    if desc:
        header += f" — {desc}"

    lines = [header]

    # Group relations by type with correct direction
    edges = graph.get_edges_for(nid)
    grouped: dict[str, list[str]] = {}
    for edge in edges[:12]:
        if edge["source"] == nid:
            # Outgoing: this node → other
            rel_label = _RELATION_LABELS.get(edge.get("relation", ""), edge.get("relation", ""))
            other = graph.get_node(edge["target"])
        else:
            # Incoming: other → this node — reverse the label
            rel_label = _REVERSE_LABELS.get(edge.get("relation", ""), f"← {edge.get('relation', '')}")
            other = graph.get_node(edge["source"])
        if other:
            grouped.setdefault(rel_label, []).append(other.get("label", ""))

    for rel_name, targets in grouped.items():
        lines.append(f"    {rel_name}: {', '.join(targets)}")

    # Facts (max 3)
    for fact in node.get("facts", [])[:3]:
        fact_text = fact.get("fact", "") if isinstance(fact, dict) else str(fact)
        if fact_text:
            lines.append(f"    * {fact_text}")

    return "\n".join(lines)


def _format_warm(node: dict, graph: Any) -> str:
    """WARM node: single compact line — Label [Type]: connections — desc."""
    label = node.get("label", "")
    ntype = node.get("type", "")
    desc = node.get("attributes", {}).get("description", "")
    nid = node.get("id", "")

    # Collect connected labels (max 4)
    edges = graph.get_edges_for(nid)
    connected: list[str] = []
    for edge in edges[:4]:
        other_id = edge["target"] if edge["source"] == nid else edge["source"]
        other = graph.get_node(other_id)
        if other:
            connected.append(other.get("label", ""))

    conn_str = " + ".join(connected) if connected else ""

    if desc and conn_str:
        return f"  {label} [{ntype}]: {conn_str} — {desc}"
    elif desc:
        return f"  {label} [{ntype}] — {desc}"
    elif conn_str:
        return f"  {label} [{ntype}]: {conn_str}"
    else:
        return f"  {label} [{ntype}]"


# ── Grounding instruction ──


def _grounding_instruction(intent: str) -> str:
    """Intent-specific grounding instruction for the LLM."""
    if intent == "specific":
        return "Answer using the knowledge context above. If the answer is not in the context, say so clearly."
    elif intent == "overview":
        return "Use the knowledge context above to give a complete overview. Include all relevant entities and their relationships."
    elif intent == "followup":
        return "Continue based on the knowledge context and previous conversation."
    return ""  # chat: no grounding


# ── Phase 4: MMR reranking ──

def _node_embedding(node: dict) -> list[float] | None:
    """Extract the name embedding from a graph-store node dict.

    Supports both the LadybugGraphStore layout (top-level
    ``name_embedding`` column, lifted into the dict by _row_to_node) and
    the TopicGraph layout (embedding nested inside ``attributes``).
    Returns None when no embedding is present.
    """
    emb = node.get("name_embedding")
    if emb:
        return list(emb)
    attrs = node.get("attributes") or {}
    emb = attrs.get("name_embedding")
    return list(emb) if emb else None


def _mmr_rerank_layers(
    layered: LayeredContext,
    query_embedding: list[float],
) -> LayeredContext:
    """Reorder WARM and COLD layers via Maximal Marginal Relevance.

    Nodes without an embedding keep their BFS order at the tail of each
    layer (MMR only reorders the ones it can score). HOT is left alone
    intentionally: those are the direct seeds the user explicitly named.
    """
    from acervo.search.fusion import maximal_marginal_relevance

    def _rerank(nodes: list[dict]) -> list[dict]:
        if len(nodes) < 2:
            return nodes
        with_emb: dict[str, list[float]] = {}
        order_by_id: dict[str, dict] = {}
        tail: list[dict] = []
        for n in nodes:
            nid = n.get("id") or n.get("uuid") or ""
            emb = _node_embedding(n)
            if nid and emb:
                with_emb[nid] = emb
                order_by_id[nid] = n
            else:
                tail.append(n)
        if not with_emb:
            return nodes
        ids, _scores = maximal_marginal_relevance(query_embedding, with_emb)
        reordered = [order_by_id[i] for i in ids if i in order_by_id]
        reordered.extend(tail)
        return reordered

    return LayeredContext(
        hot=layered.hot,
        warm=_rerank(layered.warm),
        cold=_rerank(layered.cold),
        seeds_used=layered.seeds_used,
    )
