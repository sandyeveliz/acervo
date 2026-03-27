"""CLI implementations for 'acervo graph' subcommands.

Provides inspection and editing of the knowledge graph:
  acervo graph show [entity_id] [--kind KIND] [--json]
  acervo graph search <query> [--kind KIND] [--json]
  acervo graph delete <entity_id> [--yes]
  acervo graph merge <id1> <id2> [--yes]
"""

from __future__ import annotations

import json
import sys
from typing import Any

from acervo.graph import TopicGraph


# -- Show --


def cmd_graph_show(
    graph: TopicGraph,
    entity_id: str | None,
    kind: str | None,
    as_json: bool,
) -> None:
    """List nodes or show detail of a single node."""
    if entity_id:
        _show_detail(graph, entity_id, as_json)
    else:
        _show_list(graph, kind, as_json)


def _show_list(graph: TopicGraph, kind: str | None, as_json: bool) -> None:
    """List all nodes (optionally filtered by kind)."""
    if kind:
        nodes = graph.get_nodes_by_kind(kind)
    else:
        nodes = graph.get_all_nodes()

    if as_json:
        print(json.dumps(nodes, indent=2, default=str))
        return

    if not nodes:
        print("Graph is empty.")
        return

    # Summary line
    kinds: dict[str, int] = {}
    for n in nodes:
        k = n.get("kind", "entity")
        kinds[k] = kinds.get(k, 0) + 1
    summary_parts = [f"{count} {k}" for k, count in sorted(kinds.items())]
    print(f"Nodes ({len(nodes)} total: {', '.join(summary_parts)})")
    print()

    # Table
    print(f"  {'ID':<24} {'Label':<24} {'Type':<14} {'Kind':<8} {'Layer':<10} {'Facts':>5} {'Edges':>5}")
    print(f"  {'-' * 96}")

    for node in sorted(nodes, key=lambda n: n.get("label", n["id"])):
        nid = node["id"]
        label = _trunc(node.get("label", nid), 22)
        ntype = _trunc(node.get("type", ""), 12)
        nkind = node.get("kind", "entity")
        layer = node.get("layer", "")
        facts = len(node.get("facts", []))
        edges = len(graph.get_edges_for(nid))
        print(f"  {nid:<24} {label:<24} {ntype:<14} {nkind:<8} {layer:<10} {facts:>5} {edges:>5}")


def _show_detail(graph: TopicGraph, entity_id: str, as_json: bool) -> None:
    """Show full detail of a single node."""
    node = graph.get_node(entity_id)
    if not node:
        print(f"Node not found: {entity_id}", file=sys.stderr)
        sys.exit(1)

    nid = node["id"]
    edges = graph.get_edges_for(nid)
    files = graph.get_linked_files(nid)

    if as_json:
        data = {**node, "edges": edges, "linked_files": files}
        print(json.dumps(data, indent=2, default=str))
        return

    # Header
    print(f"Node: {nid}")
    print(f"  Label:    {node.get('label', nid)}")
    print(f"  Type:     {node.get('type', '')}")
    print(f"  Kind:     {node.get('kind', 'entity')}")
    print(f"  Layer:    {node.get('layer', '')}")
    source = node.get("source", "")
    if source:
        print(f"  Source:   {source}")
    created = node.get("created_at", "")
    if created:
        print(f"  Created:  {created[:10]}")
    active = node.get("last_active", "")
    if active:
        print(f"  Active:   {active[:10]}")

    # Facts
    facts = node.get("facts", [])
    if facts:
        print(f"\nFacts ({len(facts)}):")
        for i, f in enumerate(facts, 1):
            text = f.get("fact", "")
            src = f.get("source", "")
            date = f.get("date", "")[:10] if f.get("date") else ""
            meta = ", ".join(p for p in [src, date] if p)
            print(f"  {i}. {text} [{meta}]")

    # Edges
    if edges:
        print(f"\nEdges ({len(edges)}):")
        for e in edges:
            src, tgt, rel = e["source"], e["target"], e.get("relation", "")
            if src == nid:
                print(f"  -> {tgt:<24} ({rel})")
            else:
                print(f"  <- {src:<24} ({rel})")

    # Files
    if files:
        print(f"\nLinked files ({len(files)}):")
        for fp in files:
            print(f"  {fp}")

    # Attributes
    attrs = node.get("attributes", {})
    if attrs:
        print(f"\nAttributes:")
        for k, v in attrs.items():
            print(f"  {k}: {v}")


# -- Search --


def cmd_graph_search(
    graph: TopicGraph,
    query: str,
    kind: str | None,
    as_json: bool,
) -> None:
    """Search nodes by label and fact content."""
    query_lower = query.lower()
    results: list[dict[str, Any]] = []

    nodes = graph.get_nodes_by_kind(kind) if kind else graph.get_all_nodes()

    for node in nodes:
        nid = node["id"]
        label = node.get("label", nid)

        # Match on label/ID
        if query_lower in label.lower() or query_lower in nid.lower():
            results.append({**node, "_match": "label"})
            continue

        # Match on facts
        for f in node.get("facts", []):
            if query_lower in f.get("fact", "").lower():
                results.append({**node, "_match": f"fact: {_trunc(f['fact'], 40)}"})
                break

    if as_json:
        # Strip internal _match field for JSON output
        clean = [{k: v for k, v in r.items() if k != "_match"} for r in results]
        print(json.dumps(clean, indent=2, default=str))
        return

    if not results:
        print(f"No nodes matching '{query}'.")
        return

    print(f"Search results for '{query}' ({len(results)} matches):")
    print()
    print(f"  {'ID':<24} {'Label':<24} {'Type':<14} {'Kind':<8} Match")
    print(f"  {'-' * 80}")

    for r in results:
        nid = r["id"]
        label = _trunc(r.get("label", nid), 22)
        ntype = _trunc(r.get("type", ""), 12)
        nkind = r.get("kind", "entity")
        match = r.get("_match", "")
        print(f"  {nid:<24} {label:<24} {ntype:<14} {nkind:<8} {match}")


# -- Delete --


def cmd_graph_delete(
    graph: TopicGraph,
    entity_id: str,
    yes: bool,
) -> None:
    """Delete a node and its edges."""
    node = graph.get_node(entity_id)
    if not node:
        print(f"Node not found: {entity_id}", file=sys.stderr)
        sys.exit(1)

    nid = node["id"]
    edges = graph.get_edges_for(nid)
    facts = node.get("facts", [])

    if not yes:
        print(f"Will delete node: {nid} ({node.get('label', nid)})")
        print(f"  Type:  {node.get('type', '')}")
        print(f"  Facts: {len(facts)}")
        print(f"  Edges: {len(edges)}")
        try:
            answer = input("\nConfirm? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        if answer != "y":
            print("Aborted.")
            return

    removed = graph.remove_node(nid)
    if removed:
        graph.save()
        print(f"Deleted: {nid}")
    else:
        print(f"Failed to delete: {nid}", file=sys.stderr)
        sys.exit(1)


# -- Merge --


def cmd_graph_merge(
    graph: TopicGraph,
    keep_id: str,
    absorb_id: str,
    yes: bool,
) -> None:
    """Merge two nodes (keep first, absorb second)."""
    keep = graph.get_node(keep_id)
    absorb = graph.get_node(absorb_id)

    if not keep:
        print(f"Node not found: {keep_id}", file=sys.stderr)
        sys.exit(1)
    if not absorb:
        print(f"Node not found: {absorb_id}", file=sys.stderr)
        sys.exit(1)

    if not yes:
        print(f"Will merge:")
        print(f"  Keep:   {keep['id']} ({keep.get('label', keep['id'])})")
        print(f"    Facts: {len(keep.get('facts', []))}, Edges: {len(graph.get_edges_for(keep['id']))}")
        print(f"  Absorb: {absorb['id']} ({absorb.get('label', absorb['id'])})")
        print(f"    Facts: {len(absorb.get('facts', []))}, Edges: {len(graph.get_edges_for(absorb['id']))}")
        try:
            answer = input("\nConfirm? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        if answer != "y":
            print("Aborted.")
            return

    merged = graph.merge_nodes(keep["id"], absorb["id"])
    if merged:
        graph.save()
        print(f"Merged: {absorb['id']} -> {keep['id']}")
    else:
        print("Merge failed.", file=sys.stderr)
        sys.exit(1)


# -- Helpers --


def _trunc(text: str, maxlen: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 2] + ".."
