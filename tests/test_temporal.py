"""Tests for acervo.extraction.temporal — deterministic temporal arbitration."""

from __future__ import annotations

from acervo.extraction.temporal import (
    FactInvalidation,
    _parse_iso,
    resolve_edge_contradictions,
)


def test_parse_iso_z_suffix():
    dt = _parse_iso("2025-01-01T12:00:00Z")
    assert dt is not None
    assert dt.year == 2025
    assert dt.tzinfo is not None


def test_parse_iso_offset_suffix():
    dt = _parse_iso("2025-01-01T12:00:00+00:00")
    assert dt is not None
    assert dt.year == 2025


def test_parse_iso_invalid_returns_none():
    assert _parse_iso("not a date") is None
    assert _parse_iso(None) is None
    assert _parse_iso("") is None


# ── resolve_edge_contradictions ─────────────────────────────────────────────


def test_no_candidates_returns_empty():
    assert resolve_edge_contradictions({"valid_at": "2025-01-01T00:00:00Z"}, []) == []


def test_new_fact_supersedes_older_fact_with_known_valid_at():
    new = {"valid_at": "2026-03-01T00:00:00Z"}
    candidates = [
        {
            "fact_id": "f1",
            "fact": "Alice trabaja en Acme",
            "valid_at": "2024-01-01T00:00:00Z",
            "invalid_at": None,
            "expired_at": None,
        }
    ]
    out = resolve_edge_contradictions(new, candidates)
    assert len(out) == 1
    inv = out[0]
    assert isinstance(inv, FactInvalidation)
    assert inv.fact_id == "f1"
    # invalid_at is stamped to the new fact's valid_at boundary
    assert inv.invalid_at == "2026-03-01T00:00:00Z"
    assert inv.expired_at  # populated with current timestamp


def test_already_expired_candidate_is_skipped():
    new = {"valid_at": "2026-03-01T00:00:00Z"}
    candidates = [
        {
            "fact_id": "f1",
            "fact": "Alice trabaja en Acme",
            "valid_at": "2024-01-01T00:00:00Z",
            "invalid_at": None,
            "expired_at": "2025-12-01T00:00:00Z",  # already marked
        }
    ]
    assert resolve_edge_contradictions(new, candidates) == []


def test_disjoint_candidate_already_ended_is_skipped():
    # Candidate ended in 2022, new fact starts in 2025 — disjoint, leave alone.
    new = {"valid_at": "2025-01-01T00:00:00Z"}
    candidates = [
        {
            "fact_id": "f1",
            "fact": "Alice trabajó en Acme 2020-2022",
            "valid_at": "2020-01-01T00:00:00Z",
            "invalid_at": "2022-01-01T00:00:00Z",
            "expired_at": None,
        }
    ]
    assert resolve_edge_contradictions(new, candidates) == []


def test_disjoint_candidate_starts_after_new_ends_is_skipped():
    # New fact ends in 2020, candidate starts in 2025 — disjoint.
    new = {
        "valid_at": "2018-01-01T00:00:00Z",
        "invalid_at": "2020-01-01T00:00:00Z",
    }
    candidates = [
        {
            "fact_id": "f1",
            "fact": "Alice trabaja en Beta",
            "valid_at": "2025-01-01T00:00:00Z",
            "invalid_at": None,
            "expired_at": None,
        }
    ]
    assert resolve_edge_contradictions(new, candidates) == []


def test_missing_valid_at_on_candidate_invalidates_conservatively():
    # We can't determine the time relationship → conservative invalidation
    # (expired_at stamped, invalid_at None).
    new = {"valid_at": "2026-03-01T00:00:00Z"}
    candidates = [
        {
            "fact_id": "f1",
            "fact": "Alice trabaja en Acme",
            "valid_at": None,
            "invalid_at": None,
            "expired_at": None,
        }
    ]
    out = resolve_edge_contradictions(new, candidates)
    assert len(out) == 1
    assert out[0].expired_at
    assert out[0].invalid_at is None


def test_multiple_candidates_all_invalidated():
    new = {"valid_at": "2026-03-01T00:00:00Z"}
    candidates = [
        {
            "fact_id": "f1",
            "fact": "Sandy vive en Cipolletti",
            "valid_at": "2020-01-01T00:00:00Z",
            "invalid_at": None,
            "expired_at": None,
        },
        {
            "fact_id": "f2",
            "fact": "Sandy vive en General Roca",
            "valid_at": "2022-06-01T00:00:00Z",
            "invalid_at": None,
            "expired_at": None,
        },
    ]
    out = resolve_edge_contradictions(new, candidates)
    assert len(out) == 2
    assert {inv.fact_id for inv in out} == {"f1", "f2"}
