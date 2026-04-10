"""Shared fixtures for E2E integration tests — require a running LLM server.

Run with: pytest tests/integration/ -m integration -v -s

Fixture projects (tests/fixtures/p1, p2, p3) are auto-indexed on first run.
Subsequent runs reuse the existing .acervo/ state (gitignored).
To force re-index, delete the .acervo/ dir inside the fixture.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from acervo import Acervo, OpenAIClient

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


def _test_backend() -> str:
    """Get the graph backend for integration tests.

    Set ACERVO_TEST_BACKEND=ladybug to run against LadybugDB.
    Default: "json" (TopicGraph).
    """
    return os.getenv("ACERVO_TEST_BACKEND", "json")


@pytest.fixture
def llm_client():
    """Create an OpenAIClient from environment variables."""
    return OpenAIClient(
        base_url=os.getenv("ACERVO_LIGHT_MODEL_URL", "http://localhost:11434/v1"),
        model=os.getenv("ACERVO_LIGHT_MODEL", "qwen2.5:7b"),
        api_key=os.getenv("ACERVO_LIGHT_API_KEY", "ollama"),
    )


@pytest.fixture
def e2e_memory(llm_client):
    """Acervo instance with proper .acervo/ directory structure.

    Mirrors the real layout so trace_path resolves correctly:
    tmp/.acervo/data/graph  → persist_path
    tmp/.acervo/traces/     → trace JSONL output

    Respects ACERVO_TEST_BACKEND env var for backend selection.
    """
    tmp = Path(tempfile.mkdtemp()) / ".acervo"
    graph_path = tmp / "data" / "graph"
    graph_path.mkdir(parents=True, exist_ok=True)
    memory = Acervo(
        llm=llm_client, owner="Sandy", persist_path=graph_path,
        graph_backend=_test_backend(),
    )
    yield memory
    shutil.rmtree(tmp.parent, ignore_errors=True)


# ── Fixture project auto-initialization ──


def _ensure_indexed(project_path: Path) -> None:
    """Initialize, index, curate, and synthesize a fixture project if needed.

    Runs the full pipeline: init → index → curate → synthesize.
    Skipped if .acervo/data/graph/nodes.json already exists with data.
    Requires LM Studio + Ollama running locally.
    """
    acervo_dir = project_path / ".acervo"
    backend = _test_backend()

    # Already indexed? Check backend-specific data files
    if backend == "ladybug":
        db_file = acervo_dir / "data" / "graphdb" / "acervo.db"
        if db_file.exists():
            return  # Already has LadybugDB data, skip
    else:
        nodes_file = acervo_dir / "data" / "graph" / "nodes.json"
        if nodes_file.exists():
            content = nodes_file.read_text(encoding="utf-8").strip()
            if content and content != "[]":
                return  # Already has JSON data, skip

    print(f"\n{'='*60}")
    print(f"  AUTO-INDEXING: {project_path.name}")
    print(f"{'='*60}")

    from acervo.project import init_project, load_project
    from acervo.graph import TopicGraph
    from acervo.indexer import Indexer
    from acervo.openai_client import OpenAIClient as AcervoClient, OllamaEmbedder

    # Init
    project = init_project(project_path)

    # Setup dependencies using default config
    config = project.config
    model_cfg = config.resolve_model()
    embed_cfg = config.embeddings.resolve()

    llm = AcervoClient(
        base_url=model_cfg.url,
        model=model_cfg.name,
        api_key=model_cfg.api_key or "lm-studio",
    )

    embedder = None
    vector_store = None
    if embed_cfg.model and embed_cfg.url:
        embedder = OllamaEmbedder(base_url=embed_cfg.url, model=embed_cfg.model)

    backend = _test_backend()
    if backend == "ladybug":
        from acervo.adapters.ladybug_store import LadybugGraphStore
        db_path = project.graph_path.parent / "graphdb" / "acervo.db"
        graph = LadybugGraphStore(db_path)
        print(f"  BACKEND: LadybugDB at {db_path}")
    else:
        graph = TopicGraph(project.graph_path)

    if embedder:
        try:
            from acervo.vector_store import ChromaVectorStore
            project.vectordb_path.mkdir(parents=True, exist_ok=True)
            vector_store = ChromaVectorStore(
                persist_path=str(project.vectordb_path),
                embed_fn=embedder.embed,
                embed_batch_fn=getattr(embedder, "embed_batch", None),
            )
        except ImportError:
            pass

    # Index
    indexer = Indexer(
        graph=graph, llm=llm, embedder=embedder, vector_store=vector_store,
    )
    result = asyncio.get_event_loop().run_until_complete(
        indexer.index(project.workspace_root, extensions=project.extensions)
    )
    print(f"  INDEX: {result.files_analyzed} files, {result.total_nodes} nodes, "
          f"{result.total_edges} edges ({result.duration_seconds:.1f}s)")

    # Curate
    try:
        from acervo.curator import curate_graph
        asyncio.get_event_loop().run_until_complete(curate_graph(graph, llm))
        graph.save()
        entities = sum(1 for n in graph.get_all_nodes() if n.get("kind") == "entity")
        print(f"  CURATE: {graph.node_count} nodes, {entities} entities")
    except Exception as e:
        print(f"  CURATE: FAILED — {e}")

    # Synthesize
    try:
        from acervo.graph_synthesizer import synthesize_graph
        asyncio.get_event_loop().run_until_complete(synthesize_graph(graph, llm))
        graph.save()
        synth = sum(1 for n in graph.get_all_nodes() if n.get("kind") == "synthesis")
        print(f"  SYNTHESIZE: {graph.node_count} nodes, {synth} synthesis nodes")
    except Exception as e:
        print(f"  SYNTHESIZE: FAILED — {e}")


# ── Acervo instance fixtures for Layer 2/3 tests ──


def _is_project_indexed(path: Path) -> bool:
    """Check if a fixture project has been indexed (either backend)."""
    backend = _test_backend()
    if backend == "ladybug":
        return (path / ".acervo" / "data" / "graphdb" / "acervo.db").exists()
    return (path / ".acervo" / "data" / "graph" / "nodes.json").exists()


@pytest.fixture
def p1_acervo():
    """Acervo instance loaded from indexed P1 fixture."""
    path = _FIXTURES / "p1-todo-app"
    if not _is_project_indexed(path):
        pytest.skip("P1 not indexed")
    return Acervo.from_project(path, auto_init=False, graph_backend=_test_backend())


@pytest.fixture
def p2_acervo():
    """Acervo instance loaded from indexed P2 fixture."""
    path = _FIXTURES / "p2-literature"
    if not _is_project_indexed(path):
        pytest.skip("P2 not indexed")
    return Acervo.from_project(path, auto_init=False, graph_backend=_test_backend())


@pytest.fixture
def p3_acervo():
    """Acervo instance loaded from indexed P3 fixture."""
    path = _FIXTURES / "p3-project-docs"
    if not _is_project_indexed(path):
        pytest.skip("P3 not indexed")
    return Acervo.from_project(path, auto_init=False, graph_backend=_test_backend())


def pytest_configure(config):
    """Auto-index fixture projects before any tests run.

    Set ACERVO_SKIP_INDEX=1 to skip auto-indexing (useful when running
    only conversation scenarios that don't need fixture projects).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s %(message)s",
        force=True,
    )

    if os.getenv("ACERVO_SKIP_INDEX", ""):
        return

    if not _FIXTURES.exists():
        return

    for project_dir in sorted(_FIXTURES.iterdir()):
        if not project_dir.is_dir() or project_dir.name.startswith("."):
            continue
        if not project_dir.name.startswith("p"):
            continue
        try:
            _ensure_indexed(project_dir)
        except Exception as e:
            log.warning("Auto-index failed for %s: %s", project_dir.name, e)
