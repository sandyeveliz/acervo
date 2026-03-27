"""Shared fixtures for E2E integration tests — require a running LLM server.

Run with: pytest tests/integration/ -m integration -v -s
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from acervo import Acervo, OpenAIClient


@pytest.fixture
def llm_client():
    """Create an OpenAIClient from environment variables."""
    return OpenAIClient(
        base_url=os.getenv("ACERVO_LIGHT_MODEL_URL", "http://localhost:1234/v1"),
        model=os.getenv("ACERVO_LIGHT_MODEL", "qwen2.5-3b-instruct"),
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
