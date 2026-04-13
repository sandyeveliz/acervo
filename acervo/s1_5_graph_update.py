"""S1.5 Graph Update — async post-response graph curation + assistant extraction.

Runs AFTER the LLM responds. Two jobs:
1. Graph curation: merge duplicates, fix types, discard garbage, add relations
2. Extract entities/facts/relations from the assistant's response

This is best-effort/async — race conditions with the next turn are accepted.
Jaccard dedup remains the first line of defense.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from acervo.extractor import (
    Entity,
    ExtractionResult,
    ExtractedFact,
    Relation,
    _clean_response,
    _parse_first_json,
)
from acervo.graph.ids import _make_id
from acervo.layers import Layer
from acervo.llm import LLMClient
from acervo.ontology import map_extractor_type

log = logging.getLogger(__name__)


from acervo.prompts import load_prompt

_DEFAULT_PROMPT = load_prompt("s1_5_graph_update")


# ── Data types ──

@dataclass
class MergeAction:
    from_id: str
    into_id: str
    reason: str


@dataclass
class TypeCorrection:
    node_id: str
    old_type: str
    new_type: str
    reason: str


@dataclass
class DiscardAction:
    node_id: str
    reason: str


@dataclass
class S1_5Result:
    merges: list[MergeAction] = field(default_factory=list)
    new_relations: list[Relation] = field(default_factory=list)
    type_corrections: list[TypeCorrection] = field(default_factory=list)
    discards: list[DiscardAction] = field(default_factory=list)
    assistant_extraction: ExtractionResult = field(default_factory=ExtractionResult)
    # Debug: prompt and raw response for telemetry/annotation
    prompt_sent: str = ""
    raw_response: str = ""


# ── S1.5 Invoker ──

class S1_5GraphUpdate:
    """Async post-response graph curation + assistant extraction."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str | None = None,
    ) -> None:
        self._llm = llm
        self._prompt = prompt_template or _DEFAULT_PROMPT

    async def run(
        self,
        new_entities_json: str,
        existing_nodes_json: str,
        current_assistant_msg: str,
    ) -> S1_5Result:
        """Execute S1.5: graph curation + assistant response extraction.

        Uses ``str.replace`` instead of ``str.format`` to inject the three
        input placeholders, so the prompt can freely contain literal
        ``{``/``}`` characters in JSON examples without Python trying to
        interpret them as format-spec placeholders. The Phase 3 prompt
        rewrite adds several JSON few-shot examples — before this change
        ``format()`` raised ``KeyError('\\n  "merges"')`` on every turn
        because it parsed the first ``{`` in the body as a placeholder.
        """
        prompt = (
            self._prompt
            .replace("{new_entities}", new_entities_json[:1500])
            .replace("{existing_nodes}", existing_nodes_json[:2000])
            .replace("{current_assistant_msg}", current_assistant_msg[:1500])
        )

        try:
            raw_response = await self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            log.error("S1.5 Graph Update LLM call failed: %s", e)
            return S1_5Result()

        raw = _clean_response(raw_response)
        messages_sent = [{"role": "user", "content": prompt}]
        result = _parse_s1_5_response(raw)
        result.prompt_sent = json.dumps(messages_sent, ensure_ascii=False)
        result.raw_response = raw_response
        return result


# ── Parser ──

def _parse_s1_5_response(raw: str) -> S1_5Result:
    """Parse the JSON output from S1.5 Graph Update."""
    from acervo.graph.ids import _make_id

    obj = _parse_first_json(raw, "object")
    if not isinstance(obj, dict):
        log.warning("S1.5: failed to parse JSON response")
        return S1_5Result()

    # Parse merges. The LLM often emits display names in one field and
    # snake_case ids in the other (e.g. {"from": "Cipolletti", "into":
    # "cipolletti"}); comparing the raw strings lets self-merges slip
    # through and causes graph.merge_nodes to delete the surviving node.
    # We normalize via _make_id before the equality check to catch those.
    merges: list[MergeAction] = []
    for m in obj.get("merges", []):
        if not isinstance(m, dict):
            continue
        from_id = str(m.get("from", "")).strip()
        into_id = str(m.get("into", "")).strip()
        reason = str(m.get("reason", "")).strip()
        if not (from_id and into_id):
            continue
        if _make_id(from_id) == _make_id(into_id):
            log.info(
                "S1.5: dropping self-merge %s → %s (same canonical id)",
                from_id, into_id,
            )
            continue
        merges.append(MergeAction(from_id=from_id, into_id=into_id, reason=reason))

    # Parse new relations
    new_relations: list[Relation] = []
    for r in obj.get("new_relations", []):
        if not isinstance(r, dict):
            continue
        src = str(r.get("source", "")).strip()
        tgt = str(r.get("target", "")).strip()
        rel = str(r.get("relation", "")).strip()
        _rel_norm = rel.lower().replace(" ", "_")
        if src and tgt and rel and _rel_norm not in ("related_to", "co_mentioned", "is_related_to", "mentioned_with"):
            new_relations.append(Relation(source=src, target=tgt, relation=rel))

    # Parse type corrections
    type_corrections: list[TypeCorrection] = []
    for tc in obj.get("type_corrections", []):
        if not isinstance(tc, dict):
            continue
        nid = str(tc.get("id", "")).strip()
        old_type = str(tc.get("old_type", "")).strip()
        new_type = str(tc.get("new_type", "")).strip()
        reason = str(tc.get("reason", "")).strip()
        if nid and new_type:
            type_corrections.append(TypeCorrection(
                node_id=nid, old_type=old_type, new_type=new_type, reason=reason,
            ))

    # Parse discards
    discards: list[DiscardAction] = []
    for d in obj.get("discards", []):
        if not isinstance(d, dict):
            continue
        nid = str(d.get("id", "")).strip()
        reason = str(d.get("reason", "")).strip()
        if nid:
            discards.append(DiscardAction(node_id=nid, reason=reason))

    # Parse assistant entities
    entities: list[Entity] = []
    for e_raw in obj.get("assistant_entities", []):
        if not isinstance(e_raw, dict):
            continue
        name = str(e_raw.get("name", "")).strip()
        raw_type = str(e_raw.get("type", "")).strip()
        if not name or not raw_type or len(raw_type) < 2:
            continue
        mapped_type = map_extractor_type(raw_type)
        layer = str(e_raw.get("layer", "")).upper()
        if layer not in ("PERSONAL", "UNIVERSAL"):
            layer = "UNIVERSAL"  # default for assistant-mentioned entities
        attrs = e_raw.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        existing_id = e_raw.get("existing_id")
        if existing_id and isinstance(existing_id, str) and existing_id.lower() not in ("null", "none"):
            attrs["_existing_id"] = existing_id
        entities.append(Entity(name=name, type=mapped_type, layer=layer, attributes=attrs))

    # Parse assistant facts. Phase 3 added optional ``valid_at``/
    # ``invalid_at`` fields the LLM can populate when the assistant text
    # contains explicit temporal bounds. Both stay None when not present —
    # edge_resolution's temporal arbitrator handles the None case safely.
    facts: list[ExtractedFact] = []
    for f_raw in obj.get("assistant_facts", []):
        if not isinstance(f_raw, dict):
            continue
        entity = str(f_raw.get("entity", "")).strip()
        fact = str(f_raw.get("fact", "")).strip()
        if entity and fact and len(fact) >= 10:
            valid_at = f_raw.get("valid_at")
            invalid_at = f_raw.get("invalid_at")
            # Coerce "null"/"none" strings that some local models emit.
            if isinstance(valid_at, str) and valid_at.strip().lower() in ("", "null", "none"):
                valid_at = None
            if isinstance(invalid_at, str) and invalid_at.strip().lower() in ("", "null", "none"):
                invalid_at = None
            facts.append(ExtractedFact(
                entity=entity,
                fact=fact,
                source="assistant",
                speaker="assistant",
                valid_at=valid_at if isinstance(valid_at, str) else None,
                invalid_at=invalid_at if isinstance(invalid_at, str) else None,
            ))

    # Parse assistant relations
    assistant_relations: list[Relation] = []
    for r in obj.get("assistant_relations", []):
        if not isinstance(r, dict):
            continue
        src = str(r.get("source", "")).strip()
        tgt = str(r.get("target", "")).strip()
        rel = str(r.get("relation", "")).strip()
        _rel_norm = rel.lower().replace(" ", "_")
        if src and tgt and rel and _rel_norm not in ("related_to", "co_mentioned", "is_related_to", "mentioned_with"):
            assistant_relations.append(Relation(source=src, target=tgt, relation=rel))

    # Combine new_relations + assistant_relations
    all_relations = new_relations + assistant_relations

    assistant_extraction = ExtractionResult(
        entities=entities, relations=all_relations, facts=facts,
    )

    return S1_5Result(
        merges=merges,
        new_relations=new_relations,
        type_corrections=type_corrections,
        discards=discards,
        assistant_extraction=assistant_extraction,
    )


# ── Graph application ──

def _auto_promote_pending_entities(
    graph: object,
    *,
    session_count_threshold: int = 3,
) -> int:
    """v0.6.1 Change 2: auto-promote ``pending_review`` entities.

    Promotion rules (any triggers):
      1. ``session_count >= session_count_threshold`` — the node keeps
         appearing across distinct sessions, so it has accumulated enough
         evidence to graduate.
      2. ``updated_by == "user"`` — a human explicitly edited it via the
         REST API, which is stronger signal than any LLM heuristic.

    Promoted nodes are updated with ``status="confirmed"`` and their
    confidence bumped to at least 0.8 (1.0 when promoted by user edit).

    Returns the number of promotions applied.
    """
    promoted = 0
    try:
        nodes = graph.get_all_nodes()
    except Exception as exc:
        log.debug("auto_promote: get_all_nodes failed: %s", exc)
        return 0
    for node in nodes:
        if (node.get("status") or "") != "pending_review":
            continue
        session_count = int(node.get("session_count") or 0)
        updated_by = (node.get("updated_by") or "") or ""
        promote_reason: str | None = None
        new_confidence: float | None = None
        if updated_by == "user":
            promote_reason = "user edit"
            new_confidence = 1.0
        elif session_count >= session_count_threshold:
            promote_reason = f"session_count={session_count}"
            new_confidence = max(float(node.get("confidence_for_owner") or 0.0), 0.8)
        if promote_reason is None:
            continue
        try:
            ok = graph.update_node(
                node["id"],
                status="confirmed",
                confidence_for_owner=new_confidence,
                updated_by="system",
            )
        except Exception as exc:
            log.debug("auto_promote: update_node failed for %s: %s", node["id"], exc)
            ok = False
        if ok:
            promoted += 1
            log.info(
                "auto_promote: %s → confirmed (reason=%s, confidence=%.2f)",
                node["id"], promote_reason, new_confidence,
            )
    return promoted


def _persist_fact_embeddings_for_entity(
    graph: object,
    entity_name: str,
    fact_objs: list,
) -> int:
    """After upsert_entities has written the facts, look up each one by its
    text, grab the auto-generated fact_id, and call
    ``graph.set_fact_embedding`` when the in-memory ExtractedFact carries a
    cached embedding from the v0.6.1 dedup pre-pass.

    Returns the number of embeddings persisted.
    """
    if not fact_objs:
        return 0
    has_setter = hasattr(graph, "set_fact_embedding")
    if not has_setter:
        return 0
    # Filter to facts that actually have a cached embedding.
    targets = [f for f in fact_objs if getattr(f, "fact_embedding", None)]
    if not targets:
        return 0
    try:
        node = graph.get_node(entity_name)
    except Exception:
        node = None
    if not node:
        return 0
    existing_facts = node.get("facts") or []
    # Index by normalized text so multiple facts written in the same
    # upsert don't collide.
    index: dict[str, str] = {}
    for f in existing_facts:
        text = (f.get("fact") or "").strip().lower()
        fid = f.get("fact_id")
        if text and fid and text not in index:
            index[text] = fid
    written = 0
    for fobj in targets:
        key = (getattr(fobj, "fact", "") or "").strip().lower()
        fid = index.get(key)
        if not fid:
            continue
        try:
            if graph.set_fact_embedding(fid, list(fobj.fact_embedding)):
                written += 1
        except Exception as exc:
            log.debug("set_fact_embedding(%s) failed: %s", fid, exc)
    return written


async def dedupe_s1_5_facts_by_embedding(
    result: S1_5Result,
    last_s1_extraction: object | None,
    graph: object,
    embedder: object | None,
) -> dict:
    """Run the v0.6.1 embedding-based fact dedup on the assistant facts.

    Mutates ``result.assistant_extraction.facts`` in place to drop facts
    whose cosine similarity against an existing fact on the same entity
    exceeds the drop threshold. Kept facts receive their computed
    embedding via the ``fact_embedding`` attribute so downstream persistence
    can write it in one call.

    The ``last_s1_extraction`` argument is accepted for future use (so we
    can restrict dedup to nodes actually touched this turn) but currently
    we dedup against every assistant fact's entity — the cost is bounded
    because there's usually only 1-5 new facts per turn.

    Returns an audit dict compatible with the existing S1.5 telemetry log.
    """
    audit = {
        "dedup_checked": 0,
        "dedup_dropped": 0,
        "dedup_flagged": 0,
    }

    if embedder is None:
        return audit

    assistant_ext = getattr(result, "assistant_extraction", None)
    if assistant_ext is None:
        return audit
    facts = list(getattr(assistant_ext, "facts", []) or [])
    if not facts:
        return audit

    try:
        from acervo.extraction.edge_resolution import dedupe_facts_by_embedding
    except Exception as exc:  # pragma: no cover — import guard
        log.warning("dedupe_s1_5_facts_by_embedding: import failed: %s", exc)
        return audit

    # Build {node_id: [facts]}. We key on _make_id(entity_name) so it
    # matches Ladybug's id scheme.
    from acervo.graph.ids import _make_id
    facts_by_node: dict[str, list] = {}
    # Side index so we can rebuild the list in original order after dedup.
    order_index: list[tuple[str, int]] = []  # (node_id, idx_in_that_bucket)
    for f in facts:
        ent = getattr(f, "entity", "") or ""
        if not ent:
            continue
        nid = _make_id(ent)
        bucket = facts_by_node.setdefault(nid, [])
        order_index.append((nid, len(bucket)))
        bucket.append(f)

    if not facts_by_node:
        return audit

    try:
        kept_by_node, dedup_audit = await dedupe_facts_by_embedding(
            graph, embedder, facts_by_node,
        )
    except Exception as exc:
        log.warning(
            "dedupe_s1_5_facts_by_embedding: dedup pass failed: %s", exc,
        )
        return audit

    # Reconstruct the mutable facts list, preserving the set of kept facts
    # while dropping the dedupe'd ones. We build a set of (node_id, idx)
    # keys for what survived.
    survivors: set[int] = set()
    kept_buckets_seen: dict[str, int] = {node_id: 0 for node_id in kept_by_node}
    # kept_by_node buckets are in the same order as the input bucket, so
    # we can walk them in lockstep.
    for (node_id, orig_bucket_idx), fact in zip(order_index, facts):
        kept_bucket = kept_by_node.get(node_id) or []
        # The kept list preserves original order — just check if the
        # pointer equality holds at the current "walk position".
        pos = kept_buckets_seen[node_id]
        if pos < len(kept_bucket) and kept_bucket[pos] is fact:
            survivors.add(id(fact))
            kept_buckets_seen[node_id] = pos + 1

    new_facts = [f for f in facts if id(f) in survivors]
    assistant_ext.facts = new_facts

    audit["dedup_checked"] = dedup_audit.checked
    audit["dedup_dropped"] = dedup_audit.dropped
    audit["dedup_flagged"] = dedup_audit.flagged
    if dedup_audit.dropped or dedup_audit.flagged:
        log.info(
            "S1.5 embedding dedup: checked=%d dropped=%d flagged=%d",
            dedup_audit.checked, dedup_audit.dropped, dedup_audit.flagged,
        )
    return audit


async def resolve_s1_5_facts(
    result: S1_5Result,
    graph: object,
    llm: object | None,
) -> dict:
    """Phase 3 hook: run edge_resolution on the assistant-side facts before
    ``apply_s1_5_result`` persists them.

    This is a pre-processing step: the S1.5 LLM call produces candidate
    facts from the assistant response, and here we run them through
    ``acervo.extraction.edge_resolution.resolve_extracted_edges`` to:

        1. Drop exact duplicates already present on the same entity.
        2. Ask the LLM for semantic duplicates and contradictions against
           related facts retrieved via ``graph.fact_fulltext_search``.
        3. Compute ``FactInvalidation`` instructions for contradicted
           facts via deterministic temporal arbitration.

    The function mutates ``result.assistant_extraction.facts`` in place to
    keep only the facts that should be persisted as new. The returned
    dict includes the list of invalidations the caller must apply via
    ``graph.invalidate_fact`` after ``apply_s1_5_result`` finishes.

    When ``llm`` is None or ``edge_resolution`` isn't available the hook
    is a no-op: all assistant facts are persisted as-is (Phase 1 behaviour).

    Returns a small dict with audit counters that the caller can log.
    """
    audit = {
        "resolved_new": 0,
        "resolved_duplicates_dropped": 0,
        "resolved_invalidations": 0,
    }

    assistant_ext = getattr(result, "assistant_extraction", None)
    if assistant_ext is None or not getattr(assistant_ext, "facts", None):
        return audit

    if llm is None:
        # Phase 1 fallback — no LLM available, keep all facts as-is.
        return audit

    try:
        from acervo.extraction.edge_resolution import resolve_extracted_edges
    except Exception as exc:
        log.warning("resolve_s1_5_facts: edge_resolution unavailable: %s", exc)
        return audit

    try:
        resolution = await resolve_extracted_edges(
            list(assistant_ext.facts),
            graph=graph,
            llm=llm,
        )
    except Exception as exc:
        log.warning("resolve_s1_5_facts: edge resolution failed: %s", exc)
        return audit

    # Rebuild the assistant facts list from the resolved set, preserving
    # the legacy ExtractedFact shape so apply_s1_5_result can persist them
    # unchanged.
    kept_facts: list[ExtractedFact] = []
    for new_fact in resolution.new_facts:
        kept_facts.append(
            ExtractedFact(
                entity=new_fact.entity_name,
                fact=new_fact.fact_text,
                source=new_fact.source,
                speaker=new_fact.speaker,
            )
        )
    assistant_ext.facts = kept_facts

    audit["resolved_new"] = len(kept_facts)
    audit["resolved_duplicates_dropped"] = len(resolution.duplicates_dropped)
    audit["resolved_invalidations"] = len(resolution.invalidations)

    # Persist invalidations right away — this keeps apply_s1_5_result
    # unchanged and avoids threading the invalidation list through another
    # return channel.
    if resolution.invalidations and hasattr(graph, "invalidate_fact"):
        for inv in resolution.invalidations:
            try:
                graph.invalidate_fact(
                    inv.fact_id,
                    expired_at=inv.expired_at,
                    invalid_at=inv.invalid_at,
                )
            except Exception as exc:
                log.warning(
                    "resolve_s1_5_facts: invalidate_fact(%s) failed: %s",
                    inv.fact_id,
                    exc,
                )

    if audit["resolved_duplicates_dropped"] or audit["resolved_invalidations"]:
        log.info(
            "S1.5 edge resolution: kept=%d dropped=%d invalidated=%d",
            audit["resolved_new"],
            audit["resolved_duplicates_dropped"],
            audit["resolved_invalidations"],
        )

    return audit


def apply_s1_5_result(
    graph: object,  # GraphStorePort — works with TopicGraph or LadybugGraphStore
    result: S1_5Result,
    owner: str = "",
) -> dict:
    """Apply S1.5 curation actions to the graph. Returns audit log.

    Uses only GraphStorePort methods (no dict mutation) so it works
    with both TopicGraph (JSON) and LadybugGraphStore (LadybugDB).
    """
    audit: dict = {
        "merges_applied": 0,
        "type_corrections": 0,
        "discards": 0,
        "relations_added": 0,
        "entities_added": 0,
        "facts_added": 0,
    }

    # 1. Apply merges — use graph.merge_nodes() instead of dict mutation.
    # Defense in depth: the parser already rejects self-merges by canonical
    # id, but we double-check here after resolving the nodes in case either
    # side references a node whose current id differs from the LLM's input.
    from acervo.graph.ids import _make_id as _make_id_apply
    for merge in result.merges:
        from_node = graph.get_node(merge.from_id)
        into_node = graph.get_node(merge.into_id)
        if not from_node or not into_node:
            log.info("S1.5 merge skipped (node not found): %s → %s", merge.from_id, merge.into_id)
            continue

        from_resolved = from_node.get("id") or _make_id_apply(merge.from_id)
        into_resolved = into_node.get("id") or _make_id_apply(merge.into_id)
        if from_resolved == into_resolved:
            log.info(
                "S1.5 merge skipped (both sides resolve to %s): %s → %s",
                from_resolved, merge.from_id, merge.into_id,
            )
            continue

        ok = graph.merge_nodes(merge.into_id, merge.from_id, updated_by="llm")
        if ok:
            audit["merges_applied"] += 1
            log.info("S1.5 merged: %s → %s (%s)", merge.from_id, merge.into_id, merge.reason)

    # ── OntologyValidator for S1.5 ──
    from acervo.graph.ontology_validator import OntologyValidator
    validator = OntologyValidator(
        source_stage="s1_5",
        session_id=getattr(graph, "session_id", ""),
    )

    # 2. Apply type corrections — validate new_type before applying
    for tc in result.type_corrections:
        node = graph.get_node(tc.node_id)
        if not node:
            continue
        old = node.get("type", "")
        vt = validator.validate_entity_type(tc.new_type, entity_name=tc.node_id)
        graph.update_node(tc.node_id, type=vt.resolved, updated_by="llm")
        audit["type_corrections"] += 1
        log.info("S1.5 type fix: %s %s → %s (%s)", tc.node_id, old, vt.resolved, tc.reason)

    # 3. Apply discards
    for discard in result.discards:
        node = graph.get_node(discard.node_id)
        if not node:
            continue
        facts = node.get("facts", [])
        if len(facts) > 2:
            log.info("S1.5 discard skipped (has %d facts): %s", len(facts), discard.node_id)
            continue
        graph.remove_node(discard.node_id)
        audit["discards"] += 1
        log.info("S1.5 discarded: %s (%s)", discard.node_id, discard.reason)

    # 4. Add new relations — validate relation types
    if result.new_relations:
        validated = []
        for r in result.new_relations:
            vr = validator.validate_relation(r.relation, entity_name=r.source)
            if vr.resolved is not None:
                validated.append((r.source, r.target, vr.resolved))
        if validated:
            graph.upsert_entities(
                [], validated,
                layer=Layer.PERSONAL,
                source="llm",
                owner=owner or None,
                updated_by="llm",
            )
        audit["relations_added"] += len(validated)

    # 5. Persist assistant entities + facts — validate types and relations
    ext = result.assistant_extraction
    if ext.entities:
        for entity in ext.entities:
            vt = validator.validate_entity_type(entity.type, entity_name=entity.name)
            layer = Layer.UNIVERSAL if entity.layer == "UNIVERSAL" else Layer.PERSONAL
            entity_fact_objs = [f for f in ext.facts if f.entity == entity.name]
            entity_facts = [
                (f.entity, f.fact, f.source) for f in entity_fact_objs
            ]
            graph.upsert_entities(
                [(entity.name, vt.resolved)],
                None,
                entity_facts if entity_facts else None,
                layer=layer,
                source="llm",
                owner=owner or None,
                updated_by="llm",
            )
            audit["entities_added"] += 1
            audit["facts_added"] += len(entity_facts)

            # v0.6.1 Change 3: persist fact_embedding on any fact that the
            # dedup pre-pass cached on the ExtractedFact. Look up the node
            # after upsert, find the freshly written fact by text match,
            # and call set_fact_embedding with its fact_id.
            _persist_fact_embeddings_for_entity(graph, entity.name, entity_fact_objs)

        if ext.relations:
            validated_rels = []
            for r in ext.relations:
                vr = validator.validate_relation(r.relation, entity_name=r.source)
                if vr.resolved is not None:
                    validated_rels.append((r.source, r.target, vr.resolved))
            if validated_rels:
                graph.upsert_entities(
                    [], validated_rels,
                    layer=Layer.PERSONAL,
                    source="llm",
                    owner=owner or None,
                    updated_by="llm",
                )
            audit["relations_added"] += len(validated_rels)

    # Persist validation decisions
    log_entries = validator.drain_log()
    if log_entries:
        graph.persist_validation_log(log_entries)
        mapped = sum(1 for e in log_entries if e.action == "mapped")
        rejected = sum(1 for e in log_entries if e.action == "rejected")
        if mapped or rejected:
            log.info("S1.5 validation: %d mapped, %d rejected", mapped, rejected)

    # v0.6.1 Change 2: auto-promote pending_review entities that have
    # accumulated enough evidence (session_count >= 3) or were edited by a
    # user. This runs once per S1.5 cycle to catch anything the current
    # turn just nudged over the threshold.
    try:
        promoted = _auto_promote_pending_entities(graph)
        audit["promoted_pending"] = promoted
    except Exception as exc:
        log.debug("auto_promote: pass failed: %s", exc)
        audit["promoted_pending"] = 0

    graph.save()
    return audit
