"""Tests for acervo.extraction.edge_resolution — mocked LLM + fake graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from acervo.extraction.edge_resolution import (
    EdgeResolution,
    _exact_duplicate,
    _parse_edge_duplicate,
    resolve_extracted_edges,
)


@dataclass
class _FakeFact:
    entity: str
    fact: str
    source: str = "user"
    speaker: str = "user"
    valid_at: str | None = None
    invalid_at: str | None = None
    reference_time: str | None = None


class _FakeGraph:
    """In-memory stub exposing get_node + fact_fulltext_search."""

    def __init__(self, nodes: dict[str, dict] | None = None, fts_hits: list[dict] | None = None):
        self._nodes = nodes or {}
        self._fts_hits = fts_hits or []

    def get_node(self, name_or_id: str) -> dict | None:
        key = (name_or_id or "").lower()
        return self._nodes.get(key) or self._nodes.get(name_or_id)

    def fact_fulltext_search(self, query: str, *, limit: int = 15) -> list[dict]:
        return self._fts_hits[:limit]


class _FakeLLM:
    """Returns canned JSON responses."""

    def __init__(self, response: dict[str, list[int]]):
        self._response = response
        self.calls: list[list[dict]] = []

    async def chat(self, messages: list[dict], **_: Any) -> str:
        self.calls.append(messages)
        return json.dumps(self._response)


# ── _exact_duplicate ────────────────────────────────────────────────────────


def test_exact_duplicate_matches_on_normalized_text():
    existing = [{"fact": "Sandy Vive En Cipolletti"}]
    match = _exact_duplicate("sandy   vive en cipolletti", existing)
    assert match is not None


def test_exact_duplicate_returns_none_when_different():
    existing = [{"fact": "Sandy vive en Cipolletti"}]
    assert _exact_duplicate("Sandy vive en Neuquén", existing) is None


# ── _parse_edge_duplicate ───────────────────────────────────────────────────


def test_parse_edge_duplicate_plain_json():
    parsed = _parse_edge_duplicate('{"duplicate_facts": [0], "contradicted_facts": [1]}')
    assert parsed is not None
    assert parsed.duplicate_facts == [0]
    assert parsed.contradicted_facts == [1]


def test_parse_edge_duplicate_code_fenced_json():
    raw = '```json\n{"duplicate_facts": [], "contradicted_facts": [2]}\n```'
    parsed = _parse_edge_duplicate(raw)
    assert parsed is not None
    assert parsed.contradicted_facts == [2]


def test_parse_edge_duplicate_trailing_text():
    raw = 'Here is the answer: {"duplicate_facts": [], "contradicted_facts": []}  — ok'
    parsed = _parse_edge_duplicate(raw)
    assert parsed is not None


def test_parse_edge_duplicate_malformed_returns_none():
    assert _parse_edge_duplicate("") is None
    assert _parse_edge_duplicate("not json at all") is None
    assert _parse_edge_duplicate('{"duplicate_facts": "nope"}') is None


# ── resolve_extracted_edges ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_facts_returns_empty_resolution():
    res = await resolve_extracted_edges([], graph=_FakeGraph(), llm=None)
    assert isinstance(res, EdgeResolution)
    assert res.new_facts == []
    assert res.duplicates_dropped == []
    assert res.invalidations == []


@pytest.mark.asyncio
async def test_fast_path_exact_duplicate_dropped_without_llm():
    fact = _FakeFact(entity="Sandy", fact="Sandy vive en Cipolletti")
    graph = _FakeGraph(nodes={
        "Sandy": {"facts": [{"fact": "Sandy vive en Cipolletti", "fact_id": "f0"}]},
    })
    llm = _FakeLLM({"duplicate_facts": [], "contradicted_facts": []})

    res = await resolve_extracted_edges([fact], graph=graph, llm=llm)

    assert llm.calls == []  # fast path skipped LLM
    assert res.duplicates_dropped == [("Sandy vive en Cipolletti", "Sandy vive en Cipolletti")]
    assert res.new_facts == []


@pytest.mark.asyncio
async def test_new_fact_with_no_existing_context_persists():
    fact = _FakeFact(entity="Sandy", fact="Sandy compró un auto")
    graph = _FakeGraph()  # empty graph
    llm = _FakeLLM({"duplicate_facts": [], "contradicted_facts": []})

    res = await resolve_extracted_edges([fact], graph=graph, llm=llm)

    assert llm.calls == []  # no candidates → LLM not called
    assert len(res.new_facts) == 1
    assert res.new_facts[0].fact_text == "Sandy compró un auto"


@pytest.mark.asyncio
async def test_llm_marks_duplicate_drops_new_fact():
    fact = _FakeFact(entity="Sandy", fact="Sandy vive en la ciudad de Cipolletti")
    graph = _FakeGraph(nodes={
        "Sandy": {"facts": [
            {"fact": "Sandy vive en Cipolletti", "fact_id": "f0"},
        ]},
    })
    llm = _FakeLLM({"duplicate_facts": [0], "contradicted_facts": []})

    res = await resolve_extracted_edges([fact], graph=graph, llm=llm)

    assert len(llm.calls) == 1
    assert len(res.duplicates_dropped) == 1
    assert res.new_facts == []


@pytest.mark.asyncio
async def test_llm_marks_contradiction_invalidates_old_fact():
    fact = _FakeFact(
        entity="Sandy",
        fact="Sandy se mudó a Neuquén en marzo 2026",
        valid_at="2026-03-01T00:00:00Z",
    )
    graph = _FakeGraph(nodes={
        "Sandy": {"facts": [
            {
                "fact": "Sandy vive en Cipolletti",
                "fact_id": "f0",
                "valid_at": "2020-01-01T00:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            },
        ]},
    })
    llm = _FakeLLM({"duplicate_facts": [], "contradicted_facts": [0]})

    res = await resolve_extracted_edges([fact], graph=graph, llm=llm)

    assert len(res.new_facts) == 1
    assert len(res.invalidations) == 1
    inv = res.invalidations[0]
    assert inv.fact_id == "f0"
    assert inv.invalid_at == "2026-03-01T00:00:00Z"


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_conservative_persistence():
    class _BrokenLLM:
        async def chat(self, *_, **__):
            raise RuntimeError("offline")

    fact = _FakeFact(entity="Sandy", fact="Sandy renunció")
    graph = _FakeGraph(nodes={
        "Sandy": {"facts": [{"fact": "Sandy trabaja en Acme", "fact_id": "f0"}]},
    })

    res = await resolve_extracted_edges([fact], graph=graph, llm=_BrokenLLM())

    # LLM failed → no dedup/contradictions detected → fact persists as new,
    # old fact stays untouched. Better to keep both than to lose info.
    assert len(res.new_facts) == 1
    assert res.invalidations == []
    assert res.duplicates_dropped == []


@pytest.mark.asyncio
async def test_llm_returns_garbage_falls_back_conservatively():
    class _GarbageLLM:
        async def chat(self, *_, **__):
            return "completely unparseable text"

    fact = _FakeFact(entity="Sandy", fact="Sandy renunció")
    graph = _FakeGraph(nodes={
        "Sandy": {"facts": [{"fact": "Sandy trabaja en Acme", "fact_id": "f0"}]},
    })

    res = await resolve_extracted_edges([fact], graph=graph, llm=_GarbageLLM())

    assert len(res.new_facts) == 1
    assert res.invalidations == []
