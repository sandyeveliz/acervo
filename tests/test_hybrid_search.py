"""Tests for acervo.search.hybrid — BFS + vector + fulltext fusion."""

from __future__ import annotations

from typing import Any

import pytest

from acervo.search.hybrid import hybrid_search


class _FakeGraph:
    """Flexible stub that can simulate any combination of methods."""

    def __init__(
        self,
        *,
        bfs_layers: dict[int, list[dict]] | None = None,
        similarity_hits: list[tuple[dict, float]] | None = None,
        fulltext_hits: list[dict] | None = None,
    ):
        self._bfs = bfs_layers
        self._sim = similarity_hits
        self._fts = fulltext_hits
        self.nodes_by_id: dict[str, dict] = {}
        # Build a lookup table from everything we've been configured with.
        if bfs_layers:
            for nodes in bfs_layers.values():
                for n in nodes:
                    self.nodes_by_id[n["id"]] = n
        if similarity_hits:
            for n, _ in similarity_hits:
                self.nodes_by_id[n["id"]] = n
        if fulltext_hits:
            for n in fulltext_hits:
                nid = n.get("_node_id")
                if nid:
                    self.nodes_by_id.setdefault(nid, {"id": nid, "label": nid})

    def traverse_bfs(self, seed_ids: list[str], max_depth: int = 2) -> dict[int, list[dict]]:
        if self._bfs is None:
            raise AttributeError("BFS disabled")
        return self._bfs

    def entity_similarity_search(self, embedding, *, limit=15, min_score=0.6):
        return list(self._sim or [])

    def fact_fulltext_search(self, query: str, *, limit: int = 15) -> list[dict]:
        return list(self._fts or [])

    def get_node(self, node_id: str) -> dict | None:
        return self.nodes_by_id.get(node_id)


def test_hybrid_search_no_active_methods_returns_empty():
    graph = _FakeGraph()
    res = hybrid_search(graph=graph, query="hola", methods=set())
    assert res.fused_nodes == []
    assert res.fused_scores == []


def test_hybrid_search_bfs_only():
    graph = _FakeGraph(bfs_layers={
        0: [{"id": "a", "label": "A"}],
        1: [{"id": "b", "label": "B"}],
        2: [],
    })
    res = hybrid_search(graph=graph, query="ignore", seed_ids=["a"])
    assert [n["id"] for n in res.fused_nodes] == ["a", "b"]
    assert res.per_method_ids.get("bfs") == ["a", "b"]


def test_hybrid_search_vector_only():
    graph = _FakeGraph(similarity_hits=[
        ({"id": "a", "label": "A"}, 0.9),
        ({"id": "b", "label": "B"}, 0.8),
    ])
    res = hybrid_search(
        graph=graph, query="", query_embedding=[0.1, 0.2, 0.3],
    )
    assert [n["id"] for n in res.fused_nodes] == ["a", "b"]
    assert "vector" in res.per_method_ids
    assert "bfs" not in res.per_method_ids  # no seeds


def test_hybrid_search_fulltext_resolves_via_node_id():
    graph = _FakeGraph(fulltext_hits=[
        {"fact_id": "f1", "fact": "Sandy vive en Cipolletti", "_node_id": "sandy"},
    ])
    graph.nodes_by_id["sandy"] = {"id": "sandy", "label": "Sandy"}

    res = hybrid_search(graph=graph, query="Cipolletti")
    assert [n["id"] for n in res.fused_nodes] == ["sandy"]
    assert res.per_method_ids.get("bm25") == ["sandy"]


def test_hybrid_search_fusion_boosts_shared_nodes():
    # Same node "shared" appears in BFS rank 1 AND vector rank 0 → should
    # dominate the RRF fusion.
    graph = _FakeGraph(
        bfs_layers={0: [{"id": "bfs_only"}], 1: [{"id": "shared"}]},
        similarity_hits=[
            ({"id": "shared"}, 0.9),
            ({"id": "vec_only"}, 0.8),
        ],
    )
    graph.nodes_by_id.update({
        "bfs_only": {"id": "bfs_only"},
        "shared": {"id": "shared"},
        "vec_only": {"id": "vec_only"},
    })

    res = hybrid_search(
        graph=graph,
        query="",
        seed_ids=["bfs_only"],
        query_embedding=[0.1, 0.2],
    )
    fused_ids = [n["id"] for n in res.fused_nodes]
    # "shared" beats both solo winners
    assert fused_ids[0] == "shared"
    assert "bfs_only" in fused_ids
    assert "vec_only" in fused_ids


def test_hybrid_search_limit_trims_output():
    layers = {
        0: [{"id": f"n{i}"} for i in range(10)],
        1: [],
        2: [],
    }
    graph = _FakeGraph(bfs_layers=layers)
    for i in range(10):
        graph.nodes_by_id[f"n{i}"] = {"id": f"n{i}"}

    res = hybrid_search(graph=graph, query="", seed_ids=["n0"], limit=3)
    assert len(res.fused_nodes) == 3


def test_hybrid_search_bfs_error_does_not_crash():
    class BrokenGraph(_FakeGraph):
        def traverse_bfs(self, seed_ids, max_depth=2):
            raise RuntimeError("boom")

    graph = BrokenGraph(similarity_hits=[({"id": "a"}, 0.9)])
    graph.nodes_by_id["a"] = {"id": "a"}
    res = hybrid_search(
        graph=graph, query="", seed_ids=["x"], query_embedding=[1.0],
    )
    assert [n["id"] for n in res.fused_nodes] == ["a"]
