"""Tests for acervo.search.fusion — RRF + MMR primitives."""

from __future__ import annotations

import math

import pytest

from acervo.search.fusion import (
    DEFAULT_MMR_LAMBDA,
    maximal_marginal_relevance,
    rrf,
)


# ── rrf ─────────────────────────────────────────────────────────────────────


def test_rrf_empty_input():
    assert rrf([]) == ([], [])


def test_rrf_single_list_preserves_order():
    ids, scores = rrf([["a", "b", "c"]])
    assert ids == ["a", "b", "c"]
    # Scores strictly decreasing
    assert scores[0] > scores[1] > scores[2]


def test_rrf_fuses_two_lists_boosts_shared_items():
    # "b" appears at rank 0 in list 1 AND rank 1 in list 2.
    # Expected scores:
    #   b = 1/1 + 1/2 = 1.5 (winner)
    #   a = 1/1       = 1.0 (rank 0 in list 2 only)
    #   c = 1/2       = 0.5
    #   d = 1/3       = 0.333…
    #   e = 1/3       = 0.333…
    ids, scores = rrf([["b", "c", "d"], ["a", "b", "e"]])
    assert ids[0] == "b"
    assert scores[0] == pytest.approx(1.5)
    # Second place is the rank-0-in-one-list item "a".
    assert ids[1] == "a"
    assert scores[1] == pytest.approx(1.0)


def test_rrf_min_score_filters_low_items():
    ids, scores = rrf([["a", "b"], ["c", "d"]], min_score=0.6)
    # With rank_const=1 + 1/(1+1)=0.5, min_score=0.6 drops rank-2 items
    for s in scores:
        assert s >= 0.6


def test_rrf_preserves_score_ordering_strictly_desc():
    ids, scores = rrf([["a", "b", "c", "d"], ["c", "a", "e", "f"]])
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


# ── maximal_marginal_relevance ──────────────────────────────────────────────


def test_mmr_empty_input():
    assert maximal_marginal_relevance([0.1, 0.2], {}) == ([], [])


def test_mmr_single_candidate_scores_it():
    ids, scores = maximal_marginal_relevance(
        [1.0, 0.0], {"a": [1.0, 0.0]},
    )
    assert ids == ["a"]
    assert len(scores) == 1


def test_mmr_prefers_relevant_at_lambda_1():
    # λ=1 → pure relevance (cosine to query)
    query = [1.0, 0.0]
    candidates = {
        "close": [0.9, 0.1],
        "mid":   [0.7, 0.7],
        "far":   [0.0, 1.0],
    }
    ids, _ = maximal_marginal_relevance(query, candidates, mmr_lambda=1.0)
    assert ids[0] == "close"
    assert ids[-1] == "far"


def test_mmr_prefers_diverse_at_low_lambda():
    # With low λ, after picking the top-relevant item, MMR should favour
    # items orthogonal to it even if they're less relevant.
    query = [1.0, 0.0]
    candidates = {
        "twin_of_close": [0.9, 0.1],   # near-duplicate of close
        "close":         [0.95, 0.05],
        "orthogonal":    [0.0, 1.0],
    }
    ids, _ = maximal_marginal_relevance(query, candidates, mmr_lambda=0.1)
    assert "orthogonal" in ids
    # orthogonal should not be at the very top (low query relevance), but
    # it should rank above the twin.
    assert ids.index("orthogonal") < ids.index("twin_of_close")


def test_mmr_min_score_filters_items_below_floor():
    query = [1.0, 0.0, 0.0]
    candidates = {
        "a": [1.0, 0.0, 0.0],
        "b": [-1.0, 0.0, 0.0],  # opposite direction → negative cosine
    }
    ids, scores = maximal_marginal_relevance(
        query, candidates, mmr_lambda=1.0, min_score=0.0,
    )
    # Only a survives the 0.0 floor (b has negative relevance)
    assert ids == ["a"]
    assert scores[0] >= 0.0


def test_mmr_default_lambda_is_half():
    assert DEFAULT_MMR_LAMBDA == 0.5


def test_mmr_zero_vector_is_safe():
    # Zero vector shouldn't crash the L2 normalizer.
    query = [0.0, 0.0]
    candidates = {"a": [0.0, 0.0]}
    ids, _ = maximal_marginal_relevance(query, candidates)
    assert ids == ["a"]
