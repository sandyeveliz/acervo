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


@pytest.fixture
def llm_client():
    """Create an OpenAIClient from environment variables."""
    return OpenAIClient(
        base_url=os.getenv("ACERVO_LIGHT_MODEL_URL", "http://localhost:1234/v1"),
        model=os.getenv("ACERVO_LIGHT_MODEL", "acervo-extractor-qwen3.5-9b"),
        api_key=os.getenv("ACERVO_LIGHT_API_KEY", "lm-studio"),
    )


@pytest.fixture
def e2e_memory(llm_client):
    """Acervo instance with proper .acervo/ directory structure.

    Mirrors the real layout so trace_path resolves correctly:
    tmp/.acervo/data/graph  → persist_path
    tmp/.acervo/traces/     → trace JSONL output
    """
    tmp = Path(tempfile.mkdtemp()) / ".acervo"
    graph_path = tmp / "data" / "graph"
    graph_path.mkdir(parents=True, exist_ok=True)
    memory = Acervo(llm=llm_client, owner="Sandy", persist_path=graph_path)
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
    nodes_file = acervo_dir / "data" / "graph" / "nodes.json"

    # Already indexed? Check if nodes.json has content
    if nodes_file.exists():
        content = nodes_file.read_text(encoding="utf-8").strip()
        if content and content != "[]":
            return  # Already has data, skip

    log.info("Auto-indexing fixture: %s", project_path.name)

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
    log.info(
        "  Indexed: %d files, %d nodes, %d edges in %.1fs",
        result.files_analyzed, result.total_nodes, result.total_edges,
        result.duration_seconds,
    )

    # Curate
    try:
        from acervo.curator import curate_graph
        asyncio.get_event_loop().run_until_complete(curate_graph(graph, llm))
        graph.save()
        log.info("  Curated: %d nodes", graph.node_count)
    except Exception as e:
        log.warning("  Curate failed: %s", e)

    # Synthesize
    try:
        from acervo.graph_synthesizer import synthesize_graph
        asyncio.get_event_loop().run_until_complete(synthesize_graph(graph, llm))
        graph.save()
        log.info("  Synthesized: %d nodes", graph.node_count)
    except Exception as e:
        log.warning("  Synthesize failed: %s", e)


def pytest_configure(config):
    """Auto-index fixture projects before any tests run."""
    if not _FIXTURES.exists():
        return

    for project_dir in sorted(_FIXTURES.iterdir()):
        if not project_dir.is_dir() or project_dir.name.startswith("."):
            continue
        # Only process p1, p2, p3 directories
        if not project_dir.name.startswith("p"):
            continue
        try:
            _ensure_indexed(project_dir)
        except Exception as e:
            log.warning("Auto-index failed for %s: %s", project_dir.name, e)
