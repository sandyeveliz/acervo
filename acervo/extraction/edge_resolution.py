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


# ── v0.6.1 Change 3 — fact dedup by embedding ───────────────────────────────

# Default thresholds tuned for qwen3-embedding. Cosine sim over normalised
# vectors, so range is [-1, 1] with 1.0 = identical.
#
# drop_threshold is deliberately high (0.92) because embeddings of short
# structural facts like "Precio: 5.000 USD" vs "Valor: 5.500 USD" on the
# same entity can easily score 0.85+ — the template dominates the
# numbers. We start conservative and can tighten to 0.85 once the
# benchmark confirms it's safe. flag_threshold stays at 0.60 so we still
# see the full distribution of possible duplicates in the audit log.
_DEFAULT_DROP_THRESHOLD = 0.92
_DEFAULT_FLAG_THRESHOLD = 0.60


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense float vectors.

    Returns 0.0 when either vector is empty or a zero vector to keep the
    dedup pass robust against malformed data.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


@dataclass
class FactDedupAudit:
    """Summary returned from ``dedupe_facts_by_embedding``."""

    checked: int = 0
    dropped: int = 0
    flagged: int = 0
    by_node: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "dropped": self.dropped,
            "flagged": self.flagged,
            "by_node": dict(self.by_node),
        }


async def dedupe_facts_by_embedding(
    graph: Any,
    embedder: Any,
    node_facts_map: dict[str, list[Any]],
    *,
    drop_threshold: float = _DEFAULT_DROP_THRESHOLD,
    flag_threshold: float = _DEFAULT_FLAG_THRESHOLD,
) -> tuple[dict[str, list[Any]], FactDedupAudit]:
    """Embedding-based deduplication for a batch of new facts.

    For each ``node_id -> [new_facts]`` entry in ``node_facts_map`` this
    function:

    1. Pulls existing facts for the node from ``graph._get_facts_for``
       (or ``graph.get_node(...).facts`` as a fallback for TopicGraph).
    2. Lazy-embeds any existing facts that don't have ``fact_embedding``
       set yet, persisting the embedding back via
       ``graph.set_fact_embedding`` so future turns skip that work.
    3. Embeds the new facts in a single batch call.
    4. Compares each new fact's embedding to every existing fact's
       embedding with cosine similarity and:
         - Drops the new fact when max sim ``>= drop_threshold``.
         - Keeps but marks the new fact as ``_dedup_flag="possible_duplicate"``
           when max sim ``>= flag_threshold`` (but below drop).
         - Keeps the new fact with ``fact_embedding`` attached so
           ``apply_s1_5_result`` can persist it via ``_add_fact(fact_embedding=...)``.

    Returns
    -------
    (kept_by_node, audit)
        - ``kept_by_node`` is a dict parallel to ``node_facts_map`` with
          only the new facts that survived dedup. Kept facts have
          ``fact_embedding`` populated in-place when the input objects
          support assignment.
        - ``audit`` is a ``FactDedupAudit`` summary.

    Zero LLM calls. Embedder is called exactly twice per node in the worst
    case (once for any unembedded existing facts, once for the new batch).
    When the embedder is None this function is a no-op that returns
    ``node_facts_map`` unchanged.
    """
    audit = FactDedupAudit()
    kept_by_node: dict[str, list[Any]] = {}

    if embedder is None:
        return node_facts_map, audit

    for node_id, new_facts in node_facts_map.items():
        if not new_facts:
            kept_by_node[node_id] = []
            continue

        # 1. Get existing facts for this node.
        existing = _load_existing_facts_for_dedup(graph, node_id)

        # 2. Lazy-embed historical facts that lack an embedding.
        missing_emb = [f for f in existing if not f.get("fact_embedding")]
        if missing_emb:
            texts = [f.get("fact") or "" for f in missing_emb]
            try:
                new_embs = await embedder.embed_batch(texts)
            except Exception as exc:
                log.warning(
                    "dedupe_facts_by_embedding: lazy embed_batch failed: %s", exc,
                )
                new_embs = []
            for f, emb in zip(missing_emb, new_embs):
                if not emb:
                    continue
                f["fact_embedding"] = list(emb)
                fid = f.get("fact_id")
                if fid and hasattr(graph, "set_fact_embedding"):
                    try:
                        graph.set_fact_embedding(fid, list(emb))
                    except Exception as exc:
                        log.debug("set_fact_embedding(%s) failed: %s", fid, exc)

        # 3. Embed the new facts.
        new_texts = [(getattr(f, "fact", None) or "").strip() for f in new_facts]
        try:
            new_embs = await embedder.embed_batch(new_texts)
        except Exception as exc:
            log.warning("dedupe_facts_by_embedding: embed_batch failed: %s", exc)
            kept_by_node[node_id] = list(new_facts)
            continue

        kept: list[Any] = []
        node_stats = {"checked": 0, "dropped": 0, "flagged": 0}
        for new_fact, new_emb in zip(new_facts, new_embs):
            node_stats["checked"] += 1
            audit.checked += 1
            if not new_emb:
                kept.append(new_fact)
                continue

            best_sim = 0.0
            best_match_text: str | None = None
            for ex in existing:
                ex_emb = ex.get("fact_embedding")
                if not ex_emb:
                    continue
                sim = _cosine_sim(list(new_emb), list(ex_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_match_text = ex.get("fact") or ""

            if best_sim >= drop_threshold:
                node_stats["dropped"] += 1
                audit.dropped += 1
                log.info(
                    "dedupe: dropped fact (sim=%.3f vs %r): %r",
                    best_sim,
                    (best_match_text or "")[:60],
                    (getattr(new_fact, "fact", "") or "")[:60],
                )
                continue

            # Attach embedding in place so apply_s1_5_result can persist it.
            try:
                new_fact.fact_embedding = list(new_emb)
            except Exception:
                # Fallback for fact types that don't allow setattr
                pass

            if best_sim >= flag_threshold:
                node_stats["flagged"] += 1
                audit.flagged += 1
                log.info(
                    "dedupe: flagged possible duplicate (sim=%.3f): %r",
                    best_sim,
                    (getattr(new_fact, "fact", "") or "")[:60],
                )

            kept.append(new_fact)

        kept_by_node[node_id] = kept
        if node_stats["checked"] > 0:
            audit.by_node[node_id] = node_stats

    return kept_by_node, audit


def _load_existing_facts_for_dedup(graph: Any, node_id: str) -> list[dict]:
    """Best-effort loader for a node's existing facts with embeddings.

    Prefers the Ladybug ``_get_facts_for`` private helper (because it
    surfaces ``fact_id`` and ``fact_embedding`` directly from the SELECT)
    and falls back to ``graph.get_node(node_id)["facts"]`` for TopicGraph.
    """
    # Try the Ladybug path first.
    fn = getattr(graph, "_get_facts_for", None)
    if callable(fn):
        try:
            # Determine the node table — Ladybug needs it, TopicGraph ignores it.
            table = None
            table_resolver = getattr(graph, "_get_node_table_by_id", None)
            if callable(table_resolver):
                table = table_resolver(node_id)
            if table is None:
                table = "EntityNode"
            return list(fn(node_id, table) or [])
        except Exception as exc:
            log.debug("_get_facts_for(%s) failed: %s", node_id, exc)

    # Fallback: read the node dict and take its .facts key.
    try:
        node = graph.get_node(node_id)
    except Exception:
        node = None
    if not node:
        return []
    return list(node.get("facts") or [])


__all__ = [
    "ResolvedFact",
    "EdgeResolution",
    "resolve_extracted_edges",
    "DEFAULT_CANDIDATE_LIMIT",
    "FactDedupAudit",
    "dedupe_facts_by_embedding",
]
