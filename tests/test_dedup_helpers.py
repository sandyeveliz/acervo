"""Tests for acervo.extraction.dedup_helpers — MinHash LSH + entropy gate.

Focus on Spanish-language edge cases (accents, typos, short names) that
motivated the adoption of these helpers from Graphiti.
"""

from __future__ import annotations

from acervo.extraction.dedup_helpers import (
    _FUZZY_JACCARD_THRESHOLD,
    DedupResolutionState,
    _build_candidate_indexes,
    _has_high_entropy,
    _jaccard_similarity,
    _name_entropy,
    _normalize_name_for_fuzzy,
    _normalize_string_exact,
    _resolve_with_similarity,
    _shingles,
)
from acervo.extraction.pydantic_schemas import DedupNode


# ── _normalize_string_exact ──────────────────────────────────────────────────


def test_normalize_exact_lowercases_and_trims():
    assert _normalize_string_exact("  Sandy Veliz  ") == "sandy veliz"


def test_normalize_exact_collapses_whitespace():
    assert _normalize_string_exact("Sandy    Veliz") == "sandy veliz"


def test_normalize_exact_preserves_accents():
    # Exact normalization keeps accents so "jose" and "josé" are distinct.
    assert _normalize_string_exact("José") == "josé"
    assert _normalize_string_exact("Jose") == "jose"


# ── _normalize_name_for_fuzzy ────────────────────────────────────────────────


def test_normalize_fuzzy_strips_punctuation():
    assert _normalize_name_for_fuzzy("Dr. Amara Osei") == "dr amara osei"


def test_normalize_fuzzy_keeps_apostrophes():
    assert _normalize_name_for_fuzzy("O'Brien") == "o'brien"


def test_normalize_fuzzy_drops_accents_via_regex():
    # The regex [^a-z0-9' ] strips non-ascii letters. "josé" -> "jos"
    result = _normalize_name_for_fuzzy("José")
    assert "é" not in result
    # Multiple spaces get collapsed
    assert "  " not in result


# ── Entropy gate ────────────────────────────────────────────────────────────


def test_entropy_for_diverse_name():
    # "cipolletti" has 9 distinct chars, high entropy
    assert _name_entropy("cipolletti") > 2.0


def test_entropy_for_repetitive_name():
    # All same char -> entropy 0
    assert _name_entropy("aaaaa") == 0.0


def test_high_entropy_gate_rejects_short_names():
    # "nyc" is 3 chars, 1 token -> too short for fuzzy
    assert _has_high_entropy("nyc") is False


def test_high_entropy_gate_accepts_long_names():
    assert _has_high_entropy("cipolletti") is True


def test_high_entropy_gate_accepts_multi_token_short():
    # "new york" is only 7 chars but 2 tokens and high entropy
    assert _has_high_entropy("new york") is True


def test_high_entropy_gate_rejects_low_entropy():
    # Long but all-same-char string fails entropy check
    assert _has_high_entropy("aaaaaaaa") is False


# ── Shingles + Jaccard ──────────────────────────────────────────────────────


def test_shingles_3grams():
    s = _shingles("cipolletti")
    # cipolletti -> "cipolletti" (no spaces), 3-grams
    assert "cip" in s
    assert "ipo" in s
    assert "tti" in s


def test_jaccard_identical():
    s = _shingles("cipolletti")
    assert _jaccard_similarity(s, s) == 1.0


def test_jaccard_disjoint():
    a = _shingles("cipolletti")
    b = _shingles("cordoba")
    assert _jaccard_similarity(a, b) < 0.3


def test_jaccard_typo_below_threshold_escalates_to_llm():
    # "cipolletti" vs "ciplinetti" — typo case. 3-gram shingles are very
    # position-sensitive: this scores around 0.23, which is well below our
    # 0.85 threshold. The important property is NOT that fuzzy matches them,
    # but that the deterministic path declines and defers to the LLM. This
    # is the correct behaviour — letting the LLM make ambiguous calls.
    a = _shingles(_normalize_name_for_fuzzy("cipolletti"))
    b = _shingles(_normalize_name_for_fuzzy("ciplinetti"))
    sim = _jaccard_similarity(a, b)
    assert sim < _FUZZY_JACCARD_THRESHOLD


def test_jaccard_minor_variation_passes_threshold():
    # Small suffix difference — should pass the 0.85 threshold.
    a = _shingles(_normalize_name_for_fuzzy("cipolletti"))
    b = _shingles(_normalize_name_for_fuzzy("cipollettis"))
    sim = _jaccard_similarity(a, b)
    assert sim >= _FUZZY_JACCARD_THRESHOLD


# ── Full resolution flow ────────────────────────────────────────────────────


def _make_dedup_node(name: str, uuid: str | None = None, type_: str = "person") -> DedupNode:
    return DedupNode(uuid=uuid or f"u_{name}", name=name, type=type_)


def test_resolve_exact_match_single():
    extracted = [_make_dedup_node("Sandy Veliz")]
    existing = [_make_dedup_node("Sandy Veliz", uuid="existing_sandy")]
    indexes = _build_candidate_indexes(existing)
    state = DedupResolutionState(
        resolved_nodes=[None], uuid_map={}, unresolved_indices=[]
    )
    _resolve_with_similarity(extracted, indexes, state)
    assert state.resolved_nodes[0] is not None
    assert state.resolved_nodes[0].uuid == "existing_sandy"
    assert state.uuid_map[extracted[0].uuid] == "existing_sandy"


def test_resolve_exact_match_case_insensitive():
    extracted = [_make_dedup_node("SANDY VELIZ")]
    existing = [_make_dedup_node("sandy veliz", uuid="existing_sandy")]
    indexes = _build_candidate_indexes(existing)
    state = DedupResolutionState(
        resolved_nodes=[None], uuid_map={}, unresolved_indices=[]
    )
    _resolve_with_similarity(extracted, indexes, state)
    assert state.resolved_nodes[0] is not None
    assert state.resolved_nodes[0].uuid == "existing_sandy"


def test_resolve_ambiguous_multiple_matches_escalates():
    # Two existing nodes share the same normalized name -> escalate to LLM.
    extracted = [_make_dedup_node("Sandy")]
    existing = [
        _make_dedup_node("Sandy", uuid="sandy_1", type_="person"),
        _make_dedup_node("Sandy", uuid="sandy_2", type_="person"),
    ]
    indexes = _build_candidate_indexes(existing)
    state = DedupResolutionState(
        resolved_nodes=[None], uuid_map={}, unresolved_indices=[]
    )
    _resolve_with_similarity(extracted, indexes, state)
    assert state.resolved_nodes[0] is None
    assert 0 in state.unresolved_indices


def test_resolve_no_match_escalates():
    extracted = [_make_dedup_node("Cipolletti")]
    existing = [_make_dedup_node("Buenos Aires", uuid="ba")]
    indexes = _build_candidate_indexes(existing)
    state = DedupResolutionState(
        resolved_nodes=[None], uuid_map={}, unresolved_indices=[]
    )
    _resolve_with_similarity(extracted, indexes, state)
    # Not resolved, escalated
    assert state.resolved_nodes[0] is None
    assert 0 in state.unresolved_indices


def test_resolve_short_name_skips_fuzzy_path():
    # "NYC" is short and no exact match → entropy gate kicks in, escalated
    extracted = [_make_dedup_node("NYC")]
    existing = [_make_dedup_node("New York City", uuid="nyc_full")]
    indexes = _build_candidate_indexes(existing)
    state = DedupResolutionState(
        resolved_nodes=[None], uuid_map={}, unresolved_indices=[]
    )
    _resolve_with_similarity(extracted, indexes, state)
    # Entropy gate must block fuzzy → escalated
    assert 0 in state.unresolved_indices


def test_resolve_promotes_generic_type_to_specific():
    # Existing has generic type "entity", extracted has "person"
    extracted = [_make_dedup_node("Sandy Veliz", type_="person")]
    existing = [
        DedupNode(uuid="existing_sandy", name="Sandy Veliz", type="entity")
    ]
    indexes = _build_candidate_indexes(existing)
    state = DedupResolutionState(
        resolved_nodes=[None], uuid_map={}, unresolved_indices=[]
    )
    _resolve_with_similarity(extracted, indexes, state)
    assert state.resolved_nodes[0] is not None
    # _promote_resolved_node should have set the existing node's type to "person"
    assert state.resolved_nodes[0].type == "person"


def test_fuzzy_threshold_constant_is_lowered_for_spanish():
    # We intentionally lowered from Graphiti's 0.9 to 0.85.
    assert _FUZZY_JACCARD_THRESHOLD == 0.85


def test_resolve_empty_extracted_is_noop():
    indexes = _build_candidate_indexes([])
    state = DedupResolutionState(
        resolved_nodes=[], uuid_map={}, unresolved_indices=[]
    )
    _resolve_with_similarity([], indexes, state)
    assert state.resolved_nodes == []
    assert state.unresolved_indices == []
