"""Shared fixtures for integration tests — require a running LLM server."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from acervo import Acervo, OpenAIClient


@pytest.fixture
def clean_graph():
    """Provide a clean temp directory for graph data, cleaned up after test."""
    tmp = Path(tempfile.mkdtemp()) / "test_graph"
    tmp.mkdir(parents=True, exist_ok=True)
    yield tmp
    shutil.rmtree(tmp.parent, ignore_errors=True)


@pytest.fixture
def llm_client():
    """Create an OpenAIClient from environment variables."""
    return OpenAIClient(
        base_url=os.getenv("ACERVO_LIGHT_MODEL_URL", "http://localhost:1234/v1"),
        model=os.getenv("ACERVO_LIGHT_MODEL", "qwen2.5-3b-instruct"),
        api_key=os.getenv("ACERVO_LIGHT_API_KEY", "lm-studio"),
    )


@pytest.fixture
def memory(llm_client, clean_graph):
    """Create an Acervo instance with real LLM and temp graph."""
    return Acervo(llm=llm_client, owner="Sandy", persist_path=clean_graph)
