"""Shared fixtures for acervo tests."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from acervo.graph import TopicGraph


@pytest.fixture
def tmp_graph_path():
    """Provide a temporary directory for graph persistence, cleaned up after test."""
    tmp = Path(tempfile.mkdtemp()) / "graph"
    yield tmp
    shutil.rmtree(tmp.parent, ignore_errors=True)


@pytest.fixture
def fresh_graph(tmp_graph_path):
    """Provide a fresh TopicGraph in a temp directory."""
    return TopicGraph(tmp_graph_path)


def make_mock_llm(response: str = "{}") -> AsyncMock:
    """Create a mock LLMClient that returns a fixed response.

    Usage:
        llm = make_mock_llm('{"entities": [...]}')
        result = await llm.chat([...])  # returns the response string
    """
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value=response)
    return mock


@pytest.fixture
def mock_llm():
    """Provide a mock LLMClient that returns empty JSON by default."""
    return make_mock_llm("{}")
