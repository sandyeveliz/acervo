"""Context Synthesizer — builds a compact context paragraph from graph nodes.

Receives hot+warm nodes and the current message, returns a short paragraph
that the LLM reads as context. No LLM calls — pure template logic.
"""

from __future__ import annotations

import re

from acervo.graph import TopicGraph

_IDENTITY_PATTERNS = re.compile(
    r"(?:my name is|i am|i'm|the user (?:is called|identified as)|user's name is)\s+(\S+)",
    re.IGNORECASE,
)


def synthesize(graph: TopicGraph, user_message: str) -> str:
    """Build a compact context paragraph from hot and warm graph nodes.

    Hot nodes always included. Warm nodes only if relevant to the message.
    Returns an empty string if there's nothing relevant to inject.
    """
    parts: list[str] = []

    # Check for user identity across all nodes
    identity = _find_user_identity(graph)
    if identity:
        parts.append(f"Note: in previous sessions the user identified as {identity}.")

    hot_nodes = graph.get_nodes_by_status("hot")
    warm_nodes = graph.get_nodes_by_status("warm")

    if not hot_nodes and not warm_nodes and not parts:
        return ""

    sections: list[str] = []
    included_ids: set[str] = set()
    msg_words = set(user_message.lower().split())

    # Hot nodes: only include if they have facts or are mentioned in the message
    for node in hot_nodes:
        label = node.get("label", "").lower()
        has_facts = bool(node.get("facts"))
        mentioned = label in user_message.lower() or any(
            len(w) >= 4 and label.startswith(w) for w in msg_words
        )
        if has_facts or mentioned:
            section = _render_node(node, graph)
            if section:
                sections.append(section)
                included_ids.add(node.get("id", ""))

    # Neighbor traversal: for each included hot node, pull 1-level neighbors
    # that have facts. This brings related context without the user needing
    # to name every entity explicitly.
    neighbor_ids = _get_neighbor_ids(included_ids, graph)
    for nid in neighbor_ids:
        if nid in included_ids:
            continue
        neighbor = graph.get_node(nid)
        if neighbor and neighbor.get("facts"):
            section = _render_node(neighbor, graph)
            if section:
                sections.append(section)
                included_ids.add(nid)

    # Warm nodes: only if explicitly mentioned in the message
    for node in warm_nodes:
        if node.get("id", "") in included_ids:
            continue
        if _node_relevant(node, msg_words):
            section = _render_node(node, graph)
            if section:
                sections.append(section)
                included_ids.add(node.get("id", ""))

    if sections:
        parts.extend(sections)

    return "\n\n".join(parts) if parts else ""


def _get_neighbor_ids(node_ids: set[str], graph: TopicGraph) -> list[str]:
    """Get IDs of nodes connected to any of the given nodes (1-level traversal).

    Returns neighbor IDs sorted by edge weight (strongest connections first).
    Caps at 5 neighbors to avoid bloating the context.
    """
    neighbors: dict[str, float] = {}
    for nid in node_ids:
        for neighbor_node, weight in graph.get_neighbors(nid):
            neighbor_id = neighbor_node.get("id", "")
            if neighbor_id and neighbor_id not in node_ids:
                neighbors[neighbor_id] = max(neighbors.get(neighbor_id, 0), weight)
    sorted_ids = sorted(neighbors, key=lambda k: neighbors[k], reverse=True)
    return sorted_ids[:5]


def _find_user_identity(graph: TopicGraph) -> str | None:
    """Search all nodes for facts that reveal the user's name."""
    for node in graph.get_all_nodes():
        for f in node.get("facts", []):
            fact_text = f.get("fact", "")
            match = _IDENTITY_PATTERNS.search(fact_text)
            if match:
                return match.group(1)
            # Also check for persona nodes that were stated by the user
            if (
                node.get("type") in ("Person", "Persona")
                and f.get("source") == "user"
                and ("name" in fact_text.lower() or "nombre" in fact_text.lower())
            ):
                return node.get("label", "")
    return None


def _node_relevant(node: dict, msg_words: set[str]) -> bool:
    """Check if a warm node is relevant to the user's message."""
    label = node.get("label", "").lower()
    # Check if any word from the label appears in the message
    label_words = set(label.split())
    if label_words & msg_words:
        return True
    # Check if any fact content matches a message word
    for f in node.get("facts", []):
        fact_words = set(f.get("fact", "").lower().split())
        if len(fact_words & msg_words) >= 2:  # At least 2 words overlap
            return True
    return False


def _render_node(node: dict, graph: TopicGraph) -> str:
    """Render a single node as a compact text block."""
    label = node.get("label", "?")
    ntype = node.get("type", "?")
    session_count = node.get("session_count", 1)

    # Header
    parts: list[str] = [f"# {label} ({ntype})"]

    # Session history indicator
    if session_count > 1:
        parts.append(f"Mentioned in {session_count} previous sessions.")

    # Facts grouped by source
    by_source: dict[str, list[str]] = {}
    for f in node.get("facts", []):
        fact_text = f.get("fact", "")
        if not fact_text:
            continue
        source = f.get("source", "user")
        by_source.setdefault(source, []).append(fact_text)

    source_labels = {
        "user": "The user mentioned:",
        "web": "From web search:",
        "rag": "From documents:",
    }

    for source, slabel in source_labels.items():
        facts = by_source.get(source, [])
        if facts:
            parts.append(slabel)
            for fact in facts:
                parts.append(f"- {fact}")

    # Related entities (from edges)
    relations = _get_relations(node.get("id", ""), graph)
    if relations:
        parts.append("Relations:")
        for rel_label, rel_type in relations:
            parts.append(f"- {rel_type}: {rel_label}")

    return "\n".join(parts)


def _get_relations(node_id: str, graph: TopicGraph) -> list[tuple[str, str]]:
    """Get labeled relations for a node from the graph edges."""
    if not node_id:
        return []

    relations: list[tuple[str, str]] = []
    for edge in graph.get_edges_for(node_id):
        if edge["source"] == node_id:
            target = graph.get_node(edge["target"])
            if target:
                relations.append((target["label"], edge["relation"]))
        else:
            source = graph.get_node(edge["source"])
            if source:
                relations.append((source["label"], _reverse_relation(edge["relation"])))

    return relations[:5]  # Cap to avoid bloating context


def _reverse_relation(relation: str) -> str:
    """Reverse a relation for display from the target's perspective."""
    reverses = {
        "located_in": "contains",
        "managed_by": "manages",
        "part_of": "includes",
        "belongs_to": "has_member",
        "played_for": "has_player",
        "lives_in": "has_resident",
        # Legacy Spanish relations
        "ubicado_en": "contains",
        "tecnico_de": "manages",
        "parte_de": "includes",
        "hincha_de": "fan",
        "juega_en": "has_player",
        "pertenece_a": "has_member",
    }
    return reverses.get(relation, relation)
