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
        print(f"  Embeddings:  {result.total_chunks} chunks")
    if result.total_summaries:
        print(f"  Summaries:   {result.total_summaries} semantic summaries")
    if result.errors:
        print(f"  Errors:      {len(result.errors)}")

    # Legacy compatibility: also run FileIngestor for markdown LLM extraction
    if llm and any(ext == ".md" for ext in extensions):
        from acervo.structural_parser import StructuralParser
        from acervo.file_ingestor import FileIngestor

        parser = StructuralParser()
        ingestor = FileIngestor(llm=llm, graph=graph, structural_parser=parser)
        md_results = asyncio.run(
            ingestor.ingest_all(project.workspace_root, extensions={".md"})
        )
        md_entities = sum(r.entities for r in md_results if not r.skipped)
        md_facts = sum(r.facts for r in md_results if not r.skipped)
        if md_entities or md_facts:
            print(f"  Markdown extraction: {md_entities} entities, {md_facts} facts")


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
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="acervo",
        description="Acervo — context proxy for AI agents",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
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

    # serve
    serve_p = sub.add_parser("serve", help="Start the transparent LLM proxy server")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=None, help="Proxy port (default: from config, 9470)")
    serve_p.add_argument("--target", default=None, help="Upstream API URL (default: from config)")
    serve_p.add_argument("--env", default=".env", help="Path to .env file")
    serve_p.set_defaults(func=cmd_serve)

    # config
    config_p = sub.add_parser("config", help="Get or set config values")
    config_p.add_argument("action", choices=["get", "set"], help="Action: get or set")
    config_p.add_argument("key", help="Config key (e.g. model.url, indexing.extensions)")
    config_p.add_argument("value", nargs="?", default=None, help="Value to set")
    config_p.set_defaults(func=cmd_config)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
