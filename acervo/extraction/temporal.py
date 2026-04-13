"""Temporal arbitration for bi-temporal edge (fact) resolution.

Adapted from Graphiti (Apache-2.0, Zep Software). See acervo/THIRD_PARTY.md.

Original source:
    graphiti_core/utils/maintenance/edge_operations.py::resolve_edge_contradictions

Changes from upstream:
    - Works on Acervo's Fact dict shape (with ``valid_at``, ``invalid_at``,
      ``expired_at``, ``reference_time`` ISO-8601 string fields) rather than
      Graphiti's ``EntityEdge`` Pydantic model.
    - Kept pure Python, zero LLM / network — this is the deterministic
      arbitration step that runs after the LLM has identified contradictions.

The flow is:

    1. The LLM says "new_fact contradicts these existing facts (by idx)".
    2. For each contradicted existing fact, compare time windows:
        - If the existing fact's valid_at is strictly earlier than the new
          fact's valid_at, the new fact supersedes it → stamp the existing
          fact with invalid_at = new_fact.valid_at and expired_at = now.
        - If the existing fact's window is entirely outside the new fact's
          window (e.g. it already ended before the new fact started), leave
          it alone — it was historically true, just no longer.
    3. Return the list of facts that need to be invalidated so the caller
       can persist the changes via ``graph.invalidate_fact``.

---

Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass
class FactInvalidation:
    """Instructions to invalidate a fact found in the graph.

    ``fact_id`` is whatever identifier the graph backend uses (Ladybug
    generates one on insert; TopicGraph uses "node_id::index"). The
    ``expired_at`` timestamp is ISO-8601 UTC and defaults to "now" at the
    moment of arbitration.
    """

    fact_id: str
    fact_text: str
    expired_at: str
    invalid_at: str | None


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string into an aware UTC datetime. Returns None on failure."""
    if not value or not isinstance(value, str):
        return None
    try:
        # Support both "...Z" and "...+00:00" suffixes.
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _now_utc_iso() -> str:
    """Current time as an ISO-8601 UTC string with Z suffix."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_edge_contradictions(
    new_fact: dict,
    invalidation_candidates: list[dict],
) -> list[FactInvalidation]:
    """Deterministic temporal arbitration for contradicted facts.

    Parameters
    ----------
    new_fact:
        Dict with keys ``valid_at`` (str) and optionally ``invalid_at`` (str).
        Represents the fact we're ingesting — the one the LLM claims
        contradicts one or more existing facts.
    invalidation_candidates:
        List of existing fact dicts the LLM flagged as contradicted. Each
        dict should have ``fact_id``, ``fact`` (text), ``valid_at``,
        ``invalid_at``, and ``expired_at`` keys (all strings or None).

    Returns
    -------
    list[FactInvalidation]
        One entry per candidate that should be invalidated by the new fact.
        Candidates whose time window is disjoint from the new fact's (no
        overlap) are returned unchanged and not invalidated — they were
        historically true, just superseded naturally.
    """
    if not invalidation_candidates:
        return []

    now = _now_utc_iso()
    new_valid = _parse_iso(new_fact.get("valid_at"))
    new_invalid = _parse_iso(new_fact.get("invalid_at"))

    out: list[FactInvalidation] = []
    for candidate in invalidation_candidates:
        cand_valid = _parse_iso(candidate.get("valid_at"))
        cand_invalid = _parse_iso(candidate.get("invalid_at"))

        # Already expired or had a known end — nothing to do, the historical
        # fact stays as-is and doesn't need a new invalidation pass.
        if candidate.get("expired_at"):
            continue

        # Disjoint windows where the candidate was already known to be
        # bounded in the past (cand_invalid <= new_valid) → leave alone.
        if (
            cand_invalid is not None
            and new_valid is not None
            and cand_invalid <= new_valid
        ):
            continue
        # Symmetric disjoint case: new fact ends before the candidate starts.
        if (
            new_invalid is not None
            and cand_valid is not None
            and new_invalid <= cand_valid
        ):
            continue

        # The new fact strictly supersedes the candidate: stamp the
        # invalidation window using the new fact's valid_at as the boundary.
        if (
            cand_valid is not None
            and new_valid is not None
            and cand_valid < new_valid
        ):
            out.append(
                FactInvalidation(
                    fact_id=str(candidate.get("fact_id", "")),
                    fact_text=str(candidate.get("fact", "")),
                    expired_at=now,
                    invalid_at=new_fact.get("valid_at"),
                )
            )
            continue

        # When we lack reliable temporal info on one side, conservatively
        # invalidate with expired_at=now but no invalid_at. This preserves
        # the historical fact while marking that we learned something new.
        out.append(
            FactInvalidation(
                fact_id=str(candidate.get("fact_id", "")),
                fact_text=str(candidate.get("fact", "")),
                expired_at=now,
                invalid_at=None,
            )
        )

    return out


__all__ = ["FactInvalidation", "resolve_edge_contradictions", "_parse_iso", "_now_utc_iso"]
