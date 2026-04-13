"""Edge (fact) resolution — LLM-driven dedup + deterministic temporal arbitration.

This module orchestrates Phase 3 of the graph-build pipeline:

    1. **Fast path** — exact fact-text match against an existing fact on the
       same entity. Zero LLM call, just merge episode provenance.
    2. **Candidate gathering** — collect existing facts linked to the same
       entity (duplicate candidates) and facts on semantically-similar
       entities (invalidation candidates), using the graph's native
       ``fact_fulltext_search`` and per-node ``get_node`` helpers.
    3. **LLM call** — single prompt per new fact asking for two lists of idx
       values: ``duplicate_facts`` and ``contradicted_facts``. Output is
       validated with the ``EdgeDuplicate`` Pydantic schema.
    4. **Temporal arbitration** — ``acervo.extraction.temporal.resolve_edge_contradictions``
       decides which contradicted facts actually need to be invalidated
       based on their ``valid_at``/``invalid_at`` windows.

The resulting ``EdgeResolution`` tells S1.5 which facts to persist, which
to drop (pure duplicates), and which existing facts need
``graph.invalidate_fact`` calls.

Adapted conceptually from Graphiti's ``resolve_extracted_edges`` (Apache-2.0,
Zep Software — see acervo/THIRD_PARTY.md). Concrete implementation is
Acervo-specific because our fact-as-node schema is different from
Graphiti's fact-as-edge model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from acervo.extraction.dedup_helpers import _normalize_string_exact
from acervo.extraction.prompts.dedupe_edges import build_resolve_edge_messages
from acervo.extraction.pydantic_schemas import EdgeDuplicate
from acervo.extraction.temporal import (
    FactInvalidation,
    _now_utc_iso,
    resolve_edge_contradictions,
)

log = logging.getLogger(__name__)


# How many semantically-similar fact candidates we feed the LLM per new fact.
DEFAULT_CANDIDATE_LIMIT = 10


@dataclass
class ResolvedFact:
    """A fact after going through the resolution pipeline."""

    fact_text: str
    entity_name: str
    source: str
    speaker: str
    is_duplicate: bool = False
    duplicate_of_text: str | None = None
    # Bi-temporal fields — populated by the S1.5 prompt when extracted from text.
    valid_at: str | None = None
    invalid_at: str | None = None
    reference_time: str | None = None


@dataclass
class EdgeResolution:
    """Aggregate result of resolving a batch of new facts."""

    new_facts: list[ResolvedFact] = field(default_factory=list)
    duplicates_dropped: list[tuple[str, str]] = field(default_factory=list)  # (fact_text, dup_of)
    invalidations: list[FactInvalidation] = field(default_factory=list)


def _gather_existing_facts_for_entity(graph: Any, entity_name: str) -> list[dict]:
    """Return all facts currently linked to ``entity_name`` in the graph.

    Uses only public GraphStorePort methods. When the graph doesn't know
    the entity (new node), returns an empty list.
    """
    try:
        node = graph.get_node(entity_name)
    except Exception:
        node = None
    if not node:
        return []
    return list(node.get("facts", []) or [])


def _gather_invalidation_candidates(
    graph: Any,
    new_fact_text: str,
    existing_facts: list[dict],
    limit: int,
) -> list[dict]:
    """Run graph-wide full-text search and drop anything already in existing_facts."""
    if not hasattr(graph, "fact_fulltext_search"):
        return []
    try:
        hits = graph.fact_fulltext_search(new_fact_text, limit=limit)
    except Exception as exc:
        log.warning("edge_resolution: fact_fulltext_search failed: %s", exc)
        return []

    existing_ids = {f.get("fact_id") for f in existing_facts if f.get("fact_id")}
    deduped: list[dict] = []
    seen: set[str] = set()
    for hit in hits:
        fid = hit.get("fact_id") or ""
        if fid in existing_ids or fid in seen:
            continue
        seen.add(fid)
        deduped.append(hit)
    return deduped


def _exact_duplicate(new_text: str, existing_facts: list[dict]) -> dict | None:
    """Return the existing fact with identical normalized text, if any."""
    target = _normalize_string_exact(new_text)
    for f in existing_facts:
        if _normalize_string_exact(f.get("fact") or "") == target:
            return f
    return None


def _parse_edge_duplicate(raw: str) -> EdgeDuplicate | None:
    """Validate an LLM response as an ``EdgeDuplicate`` object.

    Tolerates code-fenced JSON and trailing text by extracting the first
    balanced ``{...}`` block. Returns None when validation fails so the
    caller can fall back conservatively.
    """
    if not raw:
        return None
    text = raw.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
    # Find the first JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    try:
        return EdgeDuplicate(**obj)
    except Exception:
        return None


async def resolve_extracted_edges(
    facts: list[Any],
    *,
    graph: Any,
    llm: Any,
    reference_time: str | None = None,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> EdgeResolution:
    """Resolve a batch of newly extracted facts against the existing graph.

    Parameters
    ----------
    facts:
        Iterable of fact-like objects. Each item must expose
        ``entity``, ``fact``, ``source`` and ``speaker`` attributes (i.e.
        ``acervo.extractor.ExtractedFact``). Optional bi-temporal fields
        ``valid_at``/``invalid_at``/``reference_time`` are honored if present.
    graph:
        GraphStorePort implementation. Needs ``get_node``,
        ``fact_fulltext_search`` and will be called read-only.
    llm:
        Async LLMClient with a ``chat(messages, ...)`` method returning raw
        string output. When None (tests), the LLM call is skipped and the
        function falls back to the fast path only.
    reference_time:
        ISO-8601 timestamp used as default ``reference_time`` for new facts
        when none is provided. Defaults to "now" UTC.
    candidate_limit:
        Max number of invalidation candidates fetched from full-text search
        per new fact. Keep small to limit prompt size and LLM cost.

    Returns
    -------
    EdgeResolution
        Structured result containing new facts to persist, duplicates that
        were dropped, and invalidation instructions to apply to the graph.
    """
    ref_time = reference_time or _now_utc_iso()
    resolution = EdgeResolution()

    for fact in facts:
        entity_name = getattr(fact, "entity", "") or ""
        fact_text = (getattr(fact, "fact", "") or "").strip()
        if not entity_name or not fact_text:
            continue

        source = getattr(fact, "source", "user") or "user"
        speaker = getattr(fact, "speaker", "user") or "user"
        fact_valid_at = getattr(fact, "valid_at", None)
        fact_invalid_at = getattr(fact, "invalid_at", None)
        fact_ref = getattr(fact, "reference_time", None) or ref_time

        existing_facts = _gather_existing_facts_for_entity(graph, entity_name)

        # 1. Fast path — exact normalized match.
        exact = _exact_duplicate(fact_text, existing_facts)
        if exact is not None:
            resolution.duplicates_dropped.append((fact_text, exact.get("fact", "")))
            continue

        # 2. Gather invalidation candidates via full-text search.
        invalidation_candidates = _gather_invalidation_candidates(
            graph, fact_text, existing_facts, limit=candidate_limit,
        )

        # 3. LLM call — only when we actually have candidates.
        duplicate_fact_text: str | None = None
        contradicted: list[dict] = []
        if llm is not None and (existing_facts or invalidation_candidates):
            existing_ctx = [
                {"idx": i, "fact": f.get("fact", "")}
                for i, f in enumerate(existing_facts)
            ]
            invalidation_offset = len(existing_facts)
            invalidation_ctx = [
                {"idx": invalidation_offset + i, "fact": f.get("fact", "")}
                for i, f in enumerate(invalidation_candidates)
            ]
            messages = build_resolve_edge_messages({
                "existing_edges": existing_ctx,
                "edge_invalidation_candidates": invalidation_ctx,
                "new_edge": fact_text,
            })
            try:
                raw = await llm.chat(messages, temperature=0.0, max_tokens=512)
            except Exception as exc:
                log.warning("edge_resolution: LLM call failed: %s", exc)
                raw = ""
            parsed = _parse_edge_duplicate(raw)
            if parsed is not None:
                # Duplicates: only idx within EXISTING FACTS range.
                for idx in parsed.duplicate_facts:
                    if 0 <= idx < len(existing_facts):
                        duplicate_fact_text = existing_facts[idx].get("fact", "")
                        break
                # Contradictions: idx across both lists.
                for idx in parsed.contradicted_facts:
                    if 0 <= idx < len(existing_facts):
                        contradicted.append(existing_facts[idx])
                    else:
                        shifted = idx - invalidation_offset
                        if 0 <= shifted < len(invalidation_candidates):
                            contradicted.append(invalidation_candidates[shifted])

        if duplicate_fact_text is not None:
            resolution.duplicates_dropped.append((fact_text, duplicate_fact_text))
            continue

        # 4. Temporal arbitration for contradictions.
        if contradicted:
            new_fact_dict = {
                "valid_at": fact_valid_at,
                "invalid_at": fact_invalid_at,
            }
            invalidations = resolve_edge_contradictions(new_fact_dict, contradicted)
            resolution.invalidations.extend(invalidations)

        resolution.new_facts.append(
            ResolvedFact(
                fact_text=fact_text,
                entity_name=entity_name,
                source=source,
                speaker=speaker,
                valid_at=fact_valid_at,
                invalid_at=fact_invalid_at,
                reference_time=fact_ref,
            )
        )

    # Phase 3 telemetry at INFO level so each batch of assistant facts shows
    # a clear trace in end-to-end logs: new vs duplicate vs invalidated.
    if facts:
        log.info(
            "edge_resolution: %d input, %d new, %d exact-duplicates dropped, "
            "%d invalidations",
            len(list(facts)) if hasattr(facts, "__len__") else len(resolution.new_facts) + len(resolution.duplicates_dropped),
            len(resolution.new_facts),
            len(resolution.duplicates_dropped),
            len(resolution.invalidations),
        )
        for old_text, dup_of in resolution.duplicates_dropped:
            log.info(
                "edge_resolution: dropped duplicate fact %r (matches existing %r)",
                old_text[:80],
                dup_of[:80],
            )
        for inv in resolution.invalidations:
            log.info(
                "edge_resolution: invalidated fact_id=%s (%s) — invalid_at=%s",
                inv.fact_id,
                inv.fact_text[:80],
                inv.invalid_at,
            )
    return resolution


__all__ = [
    "ResolvedFact",
    "EdgeResolution",
    "resolve_extracted_edges",
    "DEFAULT_CANDIDATE_LIMIT",
]
