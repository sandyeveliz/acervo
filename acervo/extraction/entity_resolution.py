"""Entity resolution — deterministic dedup with optional LLM escalation.

This module is the Acervo-specific glue between the Graphiti-derived
`dedup_helpers` (MinHash LSH + entropy gate) and the rest of the extraction
pipeline. It converts between raw graph-store dicts / domain `Entity` dataclasses
and the `DedupNode` working type, runs the deterministic resolution pass, and
(when Phase 2 lands) delegates to an LLM for the remaining ambiguous cases.

Phase 1 scope: no semantic candidate search. Operates on the full set of
existing graph nodes passed in by the caller. Phase 2 will add a pre-filter
using `graph.entity_similarity_search(...)`.

Returns a tuple `(resolved_entities, uuid_map, duplicate_pairs)` where:
    - `resolved_entities` is the input list with each entity pointing to its
      canonical form (either itself if new, or the existing node it duplicates)
    - `uuid_map` maps extracted-entity UUID → canonical UUID so callers can
      rewrite any relations/facts that reference them
    - `duplicate_pairs` lists each (extracted, canonical) pair where a merge
      happened, for audit logging
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from acervo.extraction.dedup_helpers import (
    DedupResolutionState,
    _build_candidate_indexes,
    _resolve_with_similarity,
)
from acervo.extraction.pydantic_schemas import DedupNode

log = logging.getLogger(__name__)


def _dict_to_dedup_node(node: dict[str, Any]) -> DedupNode:
    """Convert a graph-store dict into a DedupNode for the helpers."""
    uuid = node.get('id') or node.get('uuid') or str(uuid4())
    name = node.get('label') or node.get('name', '')
    type_ = node.get('type', 'entity')
    return DedupNode(
        uuid=str(uuid),
        name=str(name),
        type=str(type_),
        attributes=dict(node.get('attributes') or {}),
    )


def _entity_to_dedup_node(entity: Any) -> DedupNode:
    """Convert an acervo.domain.models.Entity-like dataclass to a DedupNode.

    We accept any object with `name` and `type` attributes and optional
    `attributes` dict, so this also works for the extractor's Entity type from
    `acervo.extractor`.
    """
    name = getattr(entity, 'name', '') or ''
    type_ = getattr(entity, 'type', 'entity') or 'entity'
    attrs = dict(getattr(entity, 'attributes', {}) or {})
    # Reuse an existing UUID if the entity was hydrated from the graph; otherwise
    # generate a fresh one (it's only used for mapping, not persistence).
    uuid = attrs.get('_existing_id') or str(uuid4())
    return DedupNode(
        uuid=str(uuid),
        name=str(name),
        type=str(type_),
        attributes=attrs,
    )


def resolve_extracted_nodes(
    extracted: list[Any],
    existing: list[dict[str, Any]],
    *,
    graph: Any = None,
    semantic_candidate_limit: int = 15,
    semantic_min_score: float = 0.6,
) -> tuple[list[DedupNode], dict[str, str], list[tuple[DedupNode, DedupNode]]]:
    """Resolve extracted entities against the existing graph using deterministic dedup.

    Parameters
    ----------
    extracted:
        List of Entity-like dataclasses from S1 extraction (must have `.name`
        and `.type` attributes).
    existing:
        List of graph-store dicts representing current graph nodes. Each dict
        should have at minimum `id`/`uuid`, `label`/`name`, and `type`.
    graph:
        Optional graph store that implements ``entity_similarity_search``.
        When provided, Phase 2 semantic pre-filter kicks in: for each
        extracted entity that carries a ``name_embedding`` in its attributes,
        the candidate universe is narrowed from the full ``existing`` list
        to the top ``semantic_candidate_limit`` most similar graph nodes.
        Entities without an embedding fall back to matching against the full
        ``existing`` list.
    semantic_candidate_limit:
        How many top candidates to pull from semantic search per entity.
    semantic_min_score:
        Cosine similarity floor for semantic candidates.

    Returns
    -------
    (resolved, uuid_map, duplicate_pairs)
        - `resolved`: one DedupNode per input in the same order; either a new
          node or an existing one if a duplicate was found.
        - `uuid_map`: dict mapping extracted.uuid -> canonical.uuid.
        - `duplicate_pairs`: list of (extracted, canonical) pairs that were merged.
    """
    if not extracted:
        return [], {}, []

    extracted_nodes = [_entity_to_dedup_node(e) for e in extracted]
    existing_nodes = [_dict_to_dedup_node(n) for n in existing]

    # Phase 2 semantic pre-filter: when the graph supports similarity search
    # and we have an embedding for the entity, swap its candidate list for
    # the top-K most similar graph nodes. This is essential for large graphs
    # where we don't want to MinHash-LSH-index every single node per turn.
    per_entity_candidates: list[list[DedupNode]] | None = None
    if graph is not None and hasattr(graph, "entity_similarity_search"):
        per_entity_candidates = []
        for orig_entity, extracted_node in zip(extracted, extracted_nodes, strict=True):
            emb = getattr(orig_entity, "attributes", {}).get("name_embedding")
            if not emb:
                per_entity_candidates.append(existing_nodes)
                continue
            try:
                hits = graph.entity_similarity_search(
                    emb,
                    limit=semantic_candidate_limit,
                    min_score=semantic_min_score,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "entity_resolution: semantic search failed for %r: %s",
                    extracted_node.name,
                    exc,
                )
                per_entity_candidates.append(existing_nodes)
                continue
            per_entity_candidates.append([_dict_to_dedup_node(n) for n, _score in hits])

    if not existing_nodes and per_entity_candidates is None:
        # No graph yet — every extracted entity is novel.
        uuid_map = {n.uuid: n.uuid for n in extracted_nodes}
        return extracted_nodes, uuid_map, []

    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted_nodes),
        uuid_map={},
        unresolved_indices=[],
    )

    if per_entity_candidates is not None:
        # Phase 2 path: each entity has its own narrowed candidate list from
        # the semantic pre-filter. We run _resolve_with_similarity per entity
        # against a freshly-built MinHash index, which is cheap because each
        # candidate list is at most ``semantic_candidate_limit`` items.
        for idx, (node, candidates) in enumerate(
            zip(extracted_nodes, per_entity_candidates, strict=True)
        ):
            if not candidates:
                continue
            local_indexes = _build_candidate_indexes(candidates)
            local_state = DedupResolutionState(
                resolved_nodes=[None],
                uuid_map={},
                unresolved_indices=[],
            )
            _resolve_with_similarity([node], local_indexes, local_state)
            if local_state.resolved_nodes[0] is not None:
                state.resolved_nodes[idx] = local_state.resolved_nodes[0]
                state.uuid_map.update(local_state.uuid_map)
                state.duplicate_pairs.extend(local_state.duplicate_pairs)
            else:
                state.unresolved_indices.append(idx)
    else:
        # Phase 1 path: no semantic pre-filter, use the full existing graph.
        indexes = _build_candidate_indexes(existing_nodes)
        _resolve_with_similarity(extracted_nodes, indexes, state)

    # Phase 1: anything unresolved stays as a new node. Phase 3 will replace
    # this with an LLM batched dedup call.
    for idx, node in enumerate(extracted_nodes):
        if state.resolved_nodes[idx] is None:
            state.resolved_nodes[idx] = node
            state.uuid_map[node.uuid] = node.uuid

    # Phase 1/2 telemetry: one compact INFO line per resolve call so the
    # pipeline's dedup behaviour shows up in end-to-end logs. This is the
    # number that proves "MinHash LSH is actually firing" without needing
    # to crank log level to DEBUG.
    semantic_active = per_entity_candidates is not None
    merged_count = len(state.duplicate_pairs)
    if extracted_nodes:
        log.info(
            "entity_resolution: %d extracted, %d merged against graph, %d new "
            "(semantic_prefilter=%s)",
            len(extracted_nodes),
            merged_count,
            len(extracted_nodes) - merged_count,
            "on" if semantic_active else "off",
        )
        if merged_count:
            for extracted_node, canonical in state.duplicate_pairs:
                log.info(
                    "entity_resolution: merged %r → %r (canonical=%s)",
                    extracted_node.name,
                    canonical.name,
                    canonical.uuid,
                )

    resolved: list[DedupNode] = [n for n in state.resolved_nodes if n is not None]
    return resolved, state.uuid_map, state.duplicate_pairs


__all__ = ['resolve_extracted_nodes']
