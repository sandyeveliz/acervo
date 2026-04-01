"""acervo CLI — workspace initialization and management.

Usage:
    acervo init [path]               Create .acervo/ (instant, no indexing)
    acervo index [path]              Index workspace files (requires model)
    acervo index [path] --dry-run    Show what would be indexed
    acervo reindex                   Re-index stale files
    acervo status                    Show graph status
    acervo config get <key>          Read a config value
    acervo config set <key> <value>  Write a config value

The .acervo/ directory (like .git/) holds all Acervo state for a project:
    .acervo/
    ├── config.toml      # workspace root, model, indexing, context settings
    ├── data/
    │   └── graph/
    └── .gitignore
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from acervo.config import AcervoConfig
from acervo.project import find_project, init_project, AcervoProject

log = logging.getLogger(__name__)


def _load_env(env_path: str) -> None:
    """Load .env file if it exists (minimal, no deps)."""
    import os

    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _require_project(start: Path | None = None) -> AcervoProject:
    """Find .acervo/ or exit with helpful error."""
    project = find_project(start)
    if not project:
        print("No .acervo/ found. Run 'acervo init' first.", file=sys.stderr)
        sys.exit(1)
    return project


# ── Commands ──


def cmd_init(args: argparse.Namespace) -> None:
    """Create .acervo/ directory structure. Instant — no indexing, no model needed."""
    root = Path(args.path).resolve()

    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    acervo_dir = root / ".acervo"
    config_path = acervo_dir / "config.toml"

    if config_path.exists():
        print(f"Acervo already initialized in {root}")
        print(f"Config: {config_path}")
        return

    project = init_project(root)

    print(f"Initialized Acervo in {root}")
    print(f"  Config: .acervo/config.toml")
    print(f"  Data:   .acervo/data/")
    print()
    print("Next steps:")
    print("  1. Configure your model:")
    print("     acervo config set model.url http://localhost:1234/v1")
    print("     acervo config set model.name qwen2.5-3b-instruct")
    print()
    print("  2. Index your project:")
    print("     acervo index .")
    print()
    print("  3. Check status:")
    print("     acervo status")


def cmd_index(args: argparse.Namespace) -> None:
    """Full index of workspace using the new Indexer pipeline."""
    _load_env(args.env)
    start = Path(args.path).resolve() if args.path else None
    project = _require_project(start)

    config = project.config
    model_cfg = config.resolve_model()
    embed_cfg = config.embeddings.resolve()

    # CLI overrides for embedding config
    embed_model = args.embedding_model or embed_cfg.model
    embed_endpoint = args.embedding_endpoint or embed_cfg.url
    llm_model = args.llm_model or model_cfg.name
    llm_endpoint = args.llm_endpoint or model_cfg.url

    # Merge extensions from config + CLI
    extensions = project.extensions
    if args.extensions:
        extensions = {e.strip() for e in args.extensions.split(",")}

    # Merge excludes from config + CLI
    excludes = set(config.indexing.ignore)
    if args.exclude:
        excludes = {e.strip() for e in args.exclude.split(",")}

    print(f"Project:    {project.acervo_dir}")
    print(f"Workspace:  {project.workspace_root}")
    print(f"LLM:        {llm_model} @ {llm_endpoint}")
    if embed_model and embed_endpoint:
        print(f"Embeddings: {embed_model} @ {embed_endpoint}")
    else:
        print(f"Embeddings: disabled (no model/endpoint configured)")
    print(f"Extensions: {', '.join(sorted(extensions))}")
    print()

    if args.dry_run:
        files = _scan_workspace(project)
        print(f"Would index {len(files)} files:")
        for f in files[:20]:
            print(f"  {f}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return

    from acervo.graph import TopicGraph
    from acervo.indexer import Indexer, IndexResult
    from acervo.indexer import (
        IndexingStarted, FileAnalyzed, FileEnriched,
        FileIndexed, IndexingComplete, IndexingError,
    )

    # Set up LLM client
    llm = None
    if llm_endpoint:
        from acervo.openai_client import OpenAIClient
        llm = OpenAIClient(
            base_url=llm_endpoint,
            model=llm_model,
            api_key=model_cfg.api_key,
        )

    # Set up embedder
    embedder = None
    if embed_model and embed_endpoint:
        from acervo.openai_client import OllamaEmbedder
        embedder = OllamaEmbedder(
            base_url=embed_endpoint,
            model=embed_model,
        )

    graph = TopicGraph(project.graph_path)

    # Set up vector store (for chunk storage + linkage)
    vector_store = None
    if embedder:
        try:
            from acervo.vector_store import ChromaVectorStore
            vectordb_path = project.acervo_dir / "data" / "vectordb"
            vectordb_path.mkdir(parents=True, exist_ok=True)
            embed_batch_fn = getattr(embedder, "embed_batch", None)
            vector_store = ChromaVectorStore(
                persist_path=str(vectordb_path),
                embed_fn=embedder.embed,
                embed_batch_fn=embed_batch_fn,
            )
            print("Vector store: enabled")
        except ImportError:
            print("Vector store: disabled (chromadb not installed)")
        except Exception as e:
            print(f"Vector store: disabled ({e})")

    # Progress callback
    def on_event(event: object) -> None:
        if isinstance(event, IndexingStarted):
            print(f"Scanning {event.total_files} files...")
        elif isinstance(event, FileAnalyzed):
            if args.verbose:
                print(f"  [structural] {event.file_path}: {event.entities} entities")
        elif isinstance(event, FileEnriched):
            if args.verbose:
                print(f"  [semantic]   {event.file_path}: {event.chunks} chunks")
        elif isinstance(event, IndexingError):
            print(f"  [error]      {event.file_path}: {event.error}", file=sys.stderr)
        elif isinstance(event, IndexingComplete):
            pass  # Summary printed below

    indexer = Indexer(
        graph=graph,
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        on_event=on_event,
    )

    result: IndexResult = asyncio.run(
        indexer.index(
            project.workspace_root,
            extensions=extensions,
            excludes=excludes,
        )
    )

    # Summary
    print()
    print(f"Indexed {result.files_analyzed} files "
          f"({result.files_skipped} skipped/unchanged) "
          f"in {result.duration_seconds:.1f}s")
    print(f"  Graph:       {result.total_nodes} nodes, {result.total_edges} edges")
    print(f"  Dependencies: {result.dependency_edges} import edges")
    if result.total_chunks:
        nodes_with_chunks = len(graph.get_nodes_with_chunks())
        print(f"  Chunks:      {result.total_chunks} chunks → {nodes_with_chunks} nodes linked")
    if result.total_summaries:
        print(f"  Summaries:   {result.total_summaries} semantic summaries")
    if result.errors:
        print(f"  Errors:      {len(result.errors)}")

    # NOTE: Legacy FileIngestor block removed in v0.4.0 — the Indexer's
    # Phase 2 (semantic enrichment) already handles entity/fact extraction
    # for all file types including markdown. The old block re-ran extraction
    # on .md files, creating duplicate graph nodes.


def cmd_reindex(args: argparse.Namespace) -> None:
    """Re-index stale files."""
    from acervo.graph import TopicGraph
    from acervo.structural_parser import StructuralParser
    from acervo.reindexer import Reindexer

    project = _require_project()

    graph = TopicGraph(project.graph_path)
    parser = StructuralParser()
    reindexer = Reindexer(graph, parser, project.workspace_root)

    stale = graph.get_stale_files()
    if not stale:
        print("No stale files to reindex.")
        return

    print(f"Found {len(stale)} stale file(s). Reindexing...")
    reindexed = asyncio.run(reindexer.reindex_stale())

    for path in reindexed:
        print(f"  Reindexed: {path}")
    print(f"\nDone: {len(reindexed)} file(s) reindexed.")


def cmd_synthesize(args: argparse.Namespace) -> None:
    """Generate project understanding from indexed content."""
    _load_env(args.env)
    start = Path(args.path).resolve() if args.path else None
    project = _require_project(start)

    config = project.config
    model_cfg = config.resolve_model()

    print(f"Project:   {project.acervo_dir}")
    print(f"Workspace: {project.workspace_root}")
    print(f"LLM:       {model_cfg.name} @ {model_cfg.url}")
    print()

    from acervo.graph import TopicGraph
    from acervo.graph_synthesizer import synthesize_graph

    # Set up LLM client
    from acervo.openai_client import OpenAIClient
    llm = OpenAIClient(
        base_url=model_cfg.url,
        model=model_cfg.name,
        api_key=model_cfg.api_key,
    )

    graph = TopicGraph(project.graph_path)

    def on_progress(event: str, data: dict) -> None:
        if event == "synthesis_started":
            print(f"Synthesizing from {data['file_count']} files (type: {data['content_type']})...")
        elif event == "overview_generated":
            print(f"  Project overview generated ({data['length']} chars)")
        elif event == "modules_generated":
            print(f"  Module summaries generated ({data['count']} modules)")
        elif event == "synthesis_complete":
            print(f"\nDone: {data['nodes_created']} synthesis nodes in {data['duration_seconds']:.1f}s")

    result = asyncio.run(synthesize_graph(
        graph, llm,
        project_description="",
        content_type=args.content_type,
        on_progress=on_progress,
    ))

    if result.errors:
        print(f"\nWarnings: {len(result.errors)}")
        for err in result.errors:
            print(f"  - {err}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show graph status — node counts by kind, stale files."""
    from acervo.graph import TopicGraph

    project = _require_project()
    config = project.config
    model_cfg = config.resolve_model()

    graph = TopicGraph(project.graph_path)

    # Count by kind
    kinds: dict[str, int] = {}
    for node in graph.get_all_nodes():
        kind = node.get("kind", "entity")
        kinds[kind] = kinds.get(kind, 0) + 1

    print(f"Project:   {project.acervo_dir}")
    print(f"Workspace: {project.workspace_root}")
    print(f"Model:     {model_cfg.name} @ {model_cfg.url}")
    print(f"  Total nodes: {graph.node_count}")
    print(f"  Total edges: {graph.edge_count}")
    print()
    for kind, count in sorted(kinds.items()):
        print(f"  {kind}: {count}")

    # Stale files
    stale = graph.get_stale_files()
    if stale:
        print(f"\nStale files ({len(stale)}):")
        for node in stale:
            path = node.get("attributes", {}).get("path", node["id"])
            since = node.get("stale_since", "?")
            print(f"  {path} (since {since})")
    else:
        print("\nNo stale files.")

    # Dependency status (quick health check)
    if config.has_services_config():
        from acervo.services import check_dependencies, format_dep_check
        deps = check_dependencies(config)
        print()
        print(format_dep_check(deps))

def cmd_serve(args: argparse.Namespace) -> None:
    """Start the transparent LLM proxy server."""
    _load_env(args.env)
    project = _require_project()

    config = project.config
    port = args.port or config.proxy.port
    target = args.target or config.proxy.target

    # Override config with CLI args
    config.proxy.port = port
    config.proxy.target = target

    from acervo.proxy import AcervoProxy

    proxy = AcervoProxy(config, project.config_path)
    try:
        asyncio.run(proxy.start(host=args.host, port=port))
    except KeyboardInterrupt:
        pass


def cmd_up(args: argparse.Namespace) -> None:
    """Start Acervo proxy (default) or full dev stack (--dev)."""
    _load_env(args.env)
    project = _require_project()
    config = project.config

    if args.dev:
        # Dev mode: all services with multiplexed logs
        from acervo.services import DevRunner
        runner = DevRunner(config, project.acervo_dir)
        try:
            asyncio.run(runner.run())
        except KeyboardInterrupt:
            pass
        return

    # Default mode: proxy only, with dependency check
    from acervo.services import check_dependencies, format_dep_check, _banner
    from acervo.proxy import AcervoProxy

    port = args.port or config.proxy.port
    target = args.target or config.proxy.target
    config.proxy.port = port
    config.proxy.target = target

    print(_banner(dev=False))
    deps = check_dependencies(config)
    print(format_dep_check(deps))
    print()

    proxy = AcervoProxy(config, project.config_path)
    try:
        asyncio.run(proxy.start(host=args.host, port=port))
    except KeyboardInterrupt:
        pass


def cmd_graph(args: argparse.Namespace) -> None:
    """Graph inspection and editing commands."""
    from acervo.graph import TopicGraph
    from acervo.graph_cli import (
        cmd_graph_show, cmd_graph_search, cmd_graph_delete, cmd_graph_merge,
    )

    project = _require_project()
    graph = TopicGraph(project.graph_path)

    action = args.graph_action
    if action == "show":
        cmd_graph_show(graph, args.entity_id, args.kind, args.json)
    elif action == "search":
        cmd_graph_search(graph, args.query, args.kind, args.json)
    elif action == "delete":
        cmd_graph_delete(graph, args.entity_id, args.yes)
    elif action == "merge":
        cmd_graph_merge(graph, args.keep_id, args.absorb_id, args.yes)
    elif action == "repair":
        report = graph.repair()
        total = sum(report.values())
        if total == 0:
            print("Graph is healthy — no repairs needed.")
        else:
            print(f"Repaired graph: {report['removed_nodes']} bad nodes removed, "
                  f"{report['removed_edges']} orphan edges removed, "
                  f"{report['deduped_edges']} duplicate edges removed, "
                  f"{report['fixed_fields']} missing fields fixed.")
    else:
        args._graph_parser.print_help()


def cmd_chunks(args: argparse.Namespace) -> None:
    """Chunk inspection commands (read-only access to ChromaDB)."""
    import asyncio
    from acervo.graph import TopicGraph
    from acervo.vector_store import ChromaVectorStore

    project = _require_project()
    graph = TopicGraph(project.graph_path)

    vectordb_path = project.acervo_dir / "data" / "vectordb"
    if not vectordb_path.exists():
        print("No vector store found. Run 'acervo index' first.", file=sys.stderr)
        sys.exit(1)

    action = args.chunks_action

    if action == "search":
        # Search needs a real embedder
        config = project.config
        embed_cfg = config.embeddings.resolve()
        if not embed_cfg.model or not embed_cfg.url:
            print("Embeddings not configured. Set [acervo.embeddings] in config.toml.", file=sys.stderr)
            sys.exit(1)
        from acervo.openai_client import OllamaEmbedder
        embedder = OllamaEmbedder(base_url=embed_cfg.url, model=embed_cfg.model)
        embed_batch_fn = getattr(embedder, "embed_batch", None)
        store = ChromaVectorStore(
            persist_path=str(vectordb_path),
            embed_fn=embedder.embed,
            embed_batch_fn=embed_batch_fn,
        )
        from acervo.chunks_cli import cmd_chunks_search
        asyncio.run(cmd_chunks_search(store, args.query, args.n, args.json))
    else:
        # Non-search commands don't need embeddings
        async def _noop_embed(text: str) -> list[float]:
            raise RuntimeError("Embedding not available")

        store = ChromaVectorStore(persist_path=str(vectordb_path), embed_fn=_noop_embed)

        from acervo.chunks_cli import cmd_chunks_stats, cmd_chunks_list, cmd_chunks_show

        if action == "stats":
            cmd_chunks_stats(store, graph, getattr(args, "file", None), args.json)
        elif action == "list":
            cmd_chunks_list(store, graph, getattr(args, "file", None), getattr(args, "node", None), args.json)
        elif action == "show":
            cmd_chunks_show(store, args.chunk_id, args.json)
        else:
            args._chunks_parser.print_help()


def cmd_trace(args: argparse.Namespace) -> None:
    """Show per-turn trace data from conversation sessions."""
    import json

    project = _require_project()
    traces_dir = project.acervo_dir / "data" / "traces"

    if not traces_dir.exists() or not any(traces_dir.glob("*.jsonl")):
        print("No traces found. Start a conversation via 'acervo up' to generate traces.")
        return

    # Find session file
    files = sorted(traces_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    if args.session:
        match = [f for f in files if f.stem == args.session]
        if not match:
            print(f"Session '{args.session}' not found. Available:", file=sys.stderr)
            for f in files[:10]:
                print(f"  {f.stem}", file=sys.stderr)
            sys.exit(1)
        trace_file = match[0]
    else:
        trace_file = files[0]

    # Parse JSONL turns
    turns = []
    for line in trace_file.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            turns.append(json.loads(line))

    if not turns:
        print(f"Session {trace_file.stem}: no turns recorded.")
        return

    print(f"Session: {trace_file.stem}  ({len(turns)} turns)")
    print()

    # Header
    header = f"{'#':>3}  {'Topic':<20}  {'Warm':>5}  {'Hot':>5}  {'Total':>6}  {'Nodes':>5}  {'Ent':>4}  {'Facts':>5}  {'Hit':>3}"
    print(header)
    print("─" * len(header))

    for t in turns:
        turn_num = t.get("turn_number", "?")
        topic = (t.get("topic", "") or "")[:20]
        warm = t.get("warm_tokens", 0)
        hot = t.get("hot_tokens", 0)
        total = t.get("total_context_tokens", 0)
        nodes = t.get("nodes_activated", 0)
        entities = t.get("entities_extracted", 0)
        facts = t.get("facts_added", 0)
        hit = "yes" if t.get("context_hit") else "no"

        print(f"{turn_num:>3}  {topic:<20}  {warm:>5}  {hot:>5}  {total:>6}  {nodes:>5}  {entities:>4}  {facts:>5}  {hit:>3}")

    # Summary
    if len(turns) > 1:
        avg_total = sum(t.get("total_context_tokens", 0) for t in turns) / len(turns)
        hits = sum(1 for t in turns if t.get("context_hit"))
        print()
        print(f"Avg tokens: {avg_total:.0f} | Context hits: {hits}/{len(turns)} ({hits/len(turns):.0%})")


def cmd_config(args: argparse.Namespace) -> None:
    """Get or set config values."""
    project = _require_project()
    config = project.config

    if args.action == "get":
        try:
            value = config.get_value(args.key)
            print(value)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "set":
        if not args.value:
            print("Error: value required for 'config set'", file=sys.stderr)
            sys.exit(1)
        try:
            config.set_value(args.key, args.value)
            config.save(project.config_path)
            print(f"{args.key} = {args.value}")
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


# ── Helpers ──


def _scan_workspace(project: AcervoProject) -> list[Path]:
    """Scan workspace for indexable files."""
    extensions = project.extensions
    ignore = set(project.config.indexing.ignore)
    files: list[Path] = []

    for path in project.workspace_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in extensions:
            continue
        # Check if any parent directory matches ignore patterns
        parts = path.relative_to(project.workspace_root).parts
        if any(p in ignore for p in parts):
            continue
        files.append(path)

    return sorted(files)


# ── Entry point ──


def main() -> None:
    """CLI entry point."""
    from acervo.log_config import setup_logging

    parser = argparse.ArgumentParser(
        prog="acervo",
        description="Acervo — context proxy for AI agents",
    )
    parser.add_argument(
        "--log-level",
        choices=["trace", "debug", "info", "warning", "error"],
        default="warning",
        help="Log verbosity (default: warning)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Shorthand for --log-level debug")
    parser.add_argument("--no-color", action="store_true", help="Disable colored log output")
    sub = parser.add_subparsers(dest="command")

    # init
    init_p = sub.add_parser("init", help="Create .acervo/ (instant, no indexing)")
    init_p.add_argument("path", nargs="?", default=".", help="Project root (default: current dir)")
    init_p.set_defaults(func=cmd_init)

    # index
    index_p = sub.add_parser("index", help="Index workspace files (requires model)")
    index_p.add_argument("path", nargs="?", default=None, help="Project root (default: auto-discover)")
    index_p.add_argument("--env", default=".env", help="Path to .env file for LLM config")
    index_p.add_argument("--dry-run", action="store_true", help="Show what would be indexed")
    index_p.add_argument("--exclude", default=None,
                         help="Comma-separated dirs to exclude (overrides config)")
    index_p.add_argument("--extensions", default=None,
                         help="Comma-separated extensions to index (overrides config)")
    index_p.add_argument("--embedding-model", default=None,
                         help="Embedding model name (e.g., nomic-embed-text)")
    index_p.add_argument("--embedding-endpoint", default=None,
                         help="Embedding endpoint URL (e.g., http://localhost:11434)")
    index_p.add_argument("--llm-model", default=None,
                         help="LLM model name for semantic summaries")
    index_p.add_argument("--llm-endpoint", default=None,
                         help="LLM endpoint URL for semantic summaries")
    index_p.add_argument("--verbose", action="store_true",
                         help="Show per-file progress")
    index_p.set_defaults(func=cmd_index)

    # status
    status_p = sub.add_parser("status", help="Show graph status")
    status_p.set_defaults(func=cmd_status)

    # reindex
    reindex_p = sub.add_parser("reindex", help="Re-index stale files")
    reindex_p.set_defaults(func=cmd_reindex)

    # synthesize
    synth_p = sub.add_parser("synthesize", help="Generate project understanding from indexed content")
    synth_p.add_argument("path", nargs="?", default=None, help="Project root (default: auto-discover)")
    synth_p.add_argument("--env", default=".env", help="Path to .env file for LLM config")
    synth_p.add_argument("--content-type", default="auto", choices=["auto", "code", "prose"],
                         help="Content type hint (default: auto-detect)")
    synth_p.set_defaults(func=cmd_synthesize)

    # serve
    serve_p = sub.add_parser("serve", help="Start the transparent LLM proxy server")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=None, help="Proxy port (default: from config, 9470)")
    serve_p.add_argument("--target", default=None, help="Upstream API URL (default: from config)")
    serve_p.add_argument("--env", default=".env", help="Path to .env file")
    serve_p.set_defaults(func=cmd_serve)

    # trace
    trace_p = sub.add_parser("trace", help="Show per-turn trace data")
    trace_p.add_argument("action", nargs="?", default="show", choices=["show"], help="Action (default: show)")
    trace_p.add_argument("--session", default=None, help="Session ID (default: latest)")
    trace_p.set_defaults(func=cmd_trace)

    # up
    up_p = sub.add_parser("up", help="Start Acervo proxy (or full stack with --dev)")
    up_p.add_argument("--dev", action="store_true", help="Dev mode: start all services with tagged logs")
    up_p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    up_p.add_argument("--port", type=int, default=None, help="Proxy port (default: from config, 9470)")
    up_p.add_argument("--target", default=None, help="Upstream API URL (default: from config)")
    up_p.add_argument("--env", default=".env", help="Path to .env file")
    up_p.set_defaults(func=cmd_up)

    # graph
    graph_p = sub.add_parser("graph", help="Inspect and edit the knowledge graph")
    graph_p.set_defaults(func=cmd_graph, _graph_parser=graph_p)
    graph_sub = graph_p.add_subparsers(dest="graph_action")

    # graph show [entity_id]
    graph_show_p = graph_sub.add_parser("show", help="List nodes or show node detail")
    graph_show_p.add_argument("entity_id", nargs="?", default=None, help="Node ID for detail view")
    graph_show_p.add_argument("--kind", default=None, help="Filter by kind (entity, file, symbol, section)")
    graph_show_p.add_argument("--json", action="store_true", help="JSON output")

    # graph search <query>
    graph_search_p = graph_sub.add_parser("search", help="Search nodes by label and facts")
    graph_search_p.add_argument("query", help="Search text")
    graph_search_p.add_argument("--kind", default=None, help="Filter by kind")
    graph_search_p.add_argument("--json", action="store_true", help="JSON output")

    # graph delete <entity_id>
    graph_delete_p = graph_sub.add_parser("delete", help="Delete a node and its edges")
    graph_delete_p.add_argument("entity_id", help="Node ID to delete")
    graph_delete_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # graph merge <keep_id> <absorb_id>
    graph_merge_p = graph_sub.add_parser("merge", help="Merge two nodes (keep first, absorb second)")
    graph_merge_p.add_argument("keep_id", help="Node ID to keep")
    graph_merge_p.add_argument("absorb_id", help="Node ID to absorb (will be deleted)")
    graph_merge_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # graph repair
    graph_sub.add_parser("repair", help="Detect and fix graph corruption")

    # chunks
    chunks_p = sub.add_parser("chunks", help="Inspect vector store chunks")
    chunks_p.set_defaults(func=cmd_chunks, _chunks_parser=chunks_p)
    chunks_sub = chunks_p.add_subparsers(dest="chunks_action")

    # chunks stats
    chunks_stats_p = chunks_sub.add_parser("stats", help="Show chunk statistics")
    chunks_stats_p.add_argument("--file", default=None, help="Filter by file path")
    chunks_stats_p.add_argument("--json", action="store_true", help="JSON output")

    # chunks list
    chunks_list_p = chunks_sub.add_parser("list", help="List chunks with previews")
    chunks_list_p.add_argument("--file", default=None, help="Filter by file path")
    chunks_list_p.add_argument("--node", default=None, help="Filter by graph node ID")
    chunks_list_p.add_argument("--json", action="store_true", help="JSON output")

    # chunks show <chunk_id>
    chunks_show_p = chunks_sub.add_parser("show", help="Show full content of a chunk")
    chunks_show_p.add_argument("chunk_id", help="Chunk ID to display")
    chunks_show_p.add_argument("--json", action="store_true", help="JSON output")

    # chunks search <query>
    chunks_search_p = chunks_sub.add_parser("search", help="Semantic vector search")
    chunks_search_p.add_argument("query", help="Search query")
    chunks_search_p.add_argument("--n", type=int, default=10, help="Number of results (default: 10)")
    chunks_search_p.add_argument("--json", action="store_true", help="JSON output")

    # config
    config_p = sub.add_parser("config", help="Get or set config values")
    config_p.add_argument("action", choices=["get", "set"], help="Action: get or set")
    config_p.add_argument("key", help="Config key (e.g. model.url, indexing.extensions)")
    config_p.add_argument("value", nargs="?", default=None, help="Value to set")
    config_p.set_defaults(func=cmd_config)

    args = parser.parse_args()

    log_level = args.log_level
    if args.verbose and log_level == "warning":
        log_level = "debug"
    color = not args.no_color
    setup_logging(level=log_level, color=color)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
