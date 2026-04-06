"""CLI implementations for 'acervo chunks' subcommands.

Provides read-only inspection of ChromaDB vector store contents:
  acervo chunks stats [--file PATH] [--json]
  acervo chunks list [--file PATH] [--node NODE_ID] [--json]
  acervo chunks show <chunk_id> [--json]
  acervo chunks search <query> [--n N] [--json]
"""

from __future__ import annotations

import json
import sys
from typing import Any

from acervo.graph import TopicGraph
from acervo.vector_store import ChromaVectorStore


# -- Stats --


def cmd_chunks_stats(
    store: ChromaVectorStore,
    graph: TopicGraph,
    file_path: str | None,
    as_json: bool,
) -> None:
    """Show chunk statistics across the vector store."""
    chunks = store.get_all_file_chunks(file_path)
    stats = store.get_collection_stats()

    if not chunks:
        if file_path:
            print(f"No chunks found for file: {file_path}")
        else:
            print("No chunks in vector store.")
        return

    # Group by file
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        fp = c["file_path"]
        by_file.setdefault(fp, []).append(c)

    sizes = [len(c["content"]) for c in chunks]
    total_chars = sum(sizes)

    if as_json:
        data = {
            "total_chunks": len(chunks),
            "total_files": len(by_file),
            "facts_count": stats["facts_count"],
            "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "files": {
                fp: {"chunks": len(cs), "total_chars": sum(len(c["content"]) for c in cs)}
                for fp, cs in sorted(by_file.items())
            },
        }
        print(json.dumps(data, indent=2))
        return

    # Header
    print("Chunk Statistics")
    print(f"  Total chunks:  {len(chunks)}")
    print(f"  Total files:   {len(by_file)}")
    print(f"  Facts indexed: {stats['facts_count']}")
    print(f"  Avg size:      {total_chars // len(chunks)} chars")
    print(f"  Min size:      {min(sizes)} chars")
    print(f"  Max size:      {max(sizes)} chars")

    # Distribution
    buckets = {"<100": 0, "100-500": 0, "500-1k": 0, "1k-2k": 0, ">2k": 0}
    for s in sizes:
        if s < 100:
            buckets["<100"] += 1
        elif s < 500:
            buckets["100-500"] += 1
        elif s < 1000:
            buckets["500-1k"] += 1
        elif s < 2000:
            buckets["1k-2k"] += 1
        else:
            buckets[">2k"] += 1

    print("\n  Size distribution:")
    for label, count in buckets.items():
        bar = "#" * min(count, 40)
        print(f"    {label:>7}: {count:>4}  {bar}")

    # Per-file breakdown (sorted by chars/chunk descending to flag under-chunking)
    file_stats = []
    for fp, cs in by_file.items():
        total = sum(len(c["content"]) for c in cs)
        file_stats.append((fp, len(cs), total))
    file_stats.sort(key=lambda x: x[2] / max(x[1], 1), reverse=True)

    print(f"\n  Per-file breakdown ({len(file_stats)} files):")
    print(f"    {'File':<60} {'Chunks':>6} {'Total':>7} {'Avg':>7}")
    print(f"    {'-' * 84}")
    for fp, count, total in file_stats:
        avg = total // count if count else 0
        display_fp = _trunc(fp, 58)
        flag = " !" if avg > 3000 else ""
        print(f"    {display_fp:<60} {count:>6} {total:>6}c {avg:>6}c{flag}")

    # Flag potential issues
    under_chunked = [(fp, count, total) for fp, count, total in file_stats if total > 2000 and count <= 2]
    if under_chunked:
        print(f"\n  Potential under-chunking ({len(under_chunked)} files):")
        for fp, count, total in under_chunked:
            print(f"    {_trunc(fp, 60)}: {count} chunk(s) for {total} chars")


# -- List --


def cmd_chunks_list(
    store: ChromaVectorStore,
    graph: TopicGraph,
    file_path: str | None,
    node_id: str | None,
    as_json: bool,
) -> None:
    """List chunks filtered by file or node."""
    if node_id:
        node = graph.get_node(node_id)
        if not node:
            print(f"Node not found: {node_id}", file=sys.stderr)
            sys.exit(1)
        chunk_ids = node.get("chunk_ids", [])
        if not chunk_ids:
            print(f"Node '{node_id}' has no linked chunks.")
            return
        chunks = store.get_chunks_by_ids(chunk_ids)
        header = f"Chunks linked to node '{node_id}' ({node.get('label', node_id)})"
    elif file_path:
        chunks = store.get_all_file_chunks(file_path)
        header = f"Chunks for file: {file_path}"
    else:
        chunks = store.get_all_file_chunks()
        header = "All chunks"

    if not chunks:
        print("No chunks found.")
        return

    if as_json:
        print(json.dumps(chunks, indent=2))
        return

    print(f"{header} ({len(chunks)} chunks)")
    print()
    print(f"  {'ID':<16} {'File':<40} {'#':>3} {'Size':>6}  Preview")
    print(f"  {'-' * 100}")

    for c in chunks:
        cid = _trunc(c["chunk_id"], 14)
        fp = _trunc(c["file_path"], 38)
        idx = c["chunk_index"]
        size = len(c["content"])
        preview = c["content"][:60].replace("\n", " ").strip()
        print(f"  {cid:<16} {fp:<40} {idx:>3} {size:>5}c  {preview}")


# -- Show --


def cmd_chunks_show(
    store: ChromaVectorStore,
    chunk_id: str,
    as_json: bool,
) -> None:
    """Show full content of a specific chunk."""
    chunks = store.get_chunks_by_ids([chunk_id])
    if not chunks:
        print(f"Chunk not found: {chunk_id}", file=sys.stderr)
        sys.exit(1)

    c = chunks[0]

    if as_json:
        print(json.dumps(c, indent=2))
        return

    print(f"Chunk: {c['chunk_id']}")
    print(f"  File:  {c['file_path']}")
    print(f"  Index: {c['chunk_index']}")
    print(f"  Size:  {len(c['content'])} chars")
    print()
    print("Content:")
    print("-" * 60)
    print(c["content"])
    print("-" * 60)


# -- Search --


async def cmd_chunks_search(
    store: ChromaVectorStore,
    query: str,
    n_results: int,
    as_json: bool,
) -> None:
    """Semantic search against the vector store."""
    results = await store.search(query, n_results=n_results)

    if not results:
        print(f"No results for: {query}")
        return

    if as_json:
        print(json.dumps(results, indent=2))
        return

    print(f"Vector search: \"{query}\" (top {n_results})")
    print()
    print(f"  {'#':>3} {'Score':>6}  {'Source':<6} {'File/Node':<40}  Preview")
    print(f"  {'-' * 100}")

    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        source = r.get("source", "")
        if source == "fact":
            location = _trunc(r.get("label", r.get("node_id", "")), 38)
        else:
            location = _trunc(r.get("file_path", ""), 38)
        preview = r.get("text", "")[:50].replace("\n", " ").strip()
        print(f"  {i:>3} {score:>6.3f}  {source:<6} {location:<40}  {preview}")


# -- Helpers --


def _trunc(text: str, maxlen: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 2] + ".."
