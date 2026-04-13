"""S1 Unified — combined topic classifier + knowledge extractor.

Runs ALWAYS before S2 Gather. L1/L2 results are passed as hints, not gates.
Uses the extraction LLM (qwen3.5:9b by default) with a structured prompt.

Entity deduplication during validation is delegated to
``acervo.extraction.dedup_helpers`` (MinHash LSH + entropy gate, adapted from
Graphiti — see ``acervo/THIRD_PARTY.md``). The legacy difflib-based path was
replaced in Phase 1 of the graph build quality work.

Output: S1Result with topic classification + entities/relations/facts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from acervo._text import strip_think_blocks
from acervo.graph.ids import _make_id
from acervo.extractor import (
    Entity,
    ExtractionResult,
    ExtractedFact,
    Relation,
    _clean_response,
    _parse_first_json,
)
from acervo.llm import LLMClient
from acervo.ontology import map_extractor_type
from acervo.topic_detector import TopicVerdict

log = logging.getLogger(__name__)


# ── System prompt for the extractor model ──

from acervo.prompts import load_prompt

_SYSTEM_PROMPT = load_prompt("s1_unified")

# Valid relation types the extractor model produces
_VALID_RELATIONS: frozenset[str] = frozenset({
    "part_of", "created_by", "maintains", "works_at", "member_of",
    "uses_technology", "depends_on", "alternative_to",
    "located_in", "deployed_on", "produces", "serves", "documented_in",
    "participated_in", "triggered_by", "resulted_in",
})

# Valid entity types the extractor model produces
_VALID_ENTITY_TYPES: frozenset[str] = frozenset({
    "person", "organization", "project", "technology",
    "place", "event", "document", "concept",
})


# ── Data types ──

@dataclass
class TopicResult:
    action: str  # "same" | "subtopic" | "changed"
    label: str | None  # topic label (only if subtopic/changed)


@dataclass
class S1Result:
    topic: TopicResult
    extraction: ExtractionResult
    intent: str = "specific"  # "overview" | "specific" | "followup" | "chat"
    retrieval: str | None = None  # "summary_only" | "with_chunks" | None (v1 fallback)
    # Debug: prompt and raw response for telemetry/annotation
    prompt_sent: str = ""       # JSON of messages array sent to extractor
    raw_response: str = ""      # Raw model output before JSON parsing
    # Validation diagnostics (populated by _validate_s1)
    raw_entity_count: int = 0
    raw_relation_count: int = 0
    raw_fact_count: int = 0
    dropped_facts: list[dict] = field(default_factory=list)


# ── Graph summary builder ──

def build_graph_summary(
    nodes: list[dict],
    query_text: str,
    max_nodes: int = 20,
) -> str:
    """Build a compact JSON array of the most relevant existing nodes.

    Returns a JSON string for direct insertion into the user message.
    Uses keyword matching on labels + status boost. Embeddings are not required.
    """
    if not nodes:
        return "[]"

    query_tokens = set(query_text.lower().split())
    scored: list[tuple[float, dict]] = []

    for node in nodes:
        label = node.get("label", "")
        label_tokens = set(label.lower().replace("_", " ").split())

        # Jaccard-like overlap score
        intersection = query_tokens & label_tokens
        union = query_tokens | label_tokens
        score = len(intersection) / len(union) if union else 0.0

        # Boost PERSONAL nodes (more likely relevant to user)
        if node.get("layer") == "PERSONAL":
            score += 0.1

        # Boost warm/hot nodes (recently active in conversation)
        status = node.get("status", "")
        if status == "hot":
            score += 0.2
        elif status == "warm":
            score += 0.15

        # Boost by session_count (frequently referenced)
        sc = node.get("session_count", 0)
        if sc > 0:
            score += 0.05 * min(sc, 5)

        if score > 0:
            scored.append((score, node))

    # Sort by score, take top-k
    scored.sort(key=lambda x: x[0], reverse=True)
    top_nodes = [n for _, n in scored[:max_nodes]]

    # Fallback: if no matches, use most recently active nodes
    if not top_nodes:
        recent = sorted(
            [n for n in nodes if n.get("last_active")],
            key=lambda n: n.get("last_active", ""),
            reverse=True,
        )
        top_nodes = recent[:max_nodes]

    if not top_nodes:
        return "[]"

    # Build compact entries matching the extractor model's expected format
    items = []
    for n in top_nodes:
        entry: dict = {
            "id": n.get("id", ""),
            "label": n.get("label", ""),
            "type": n.get("type", ""),
            "layer": n.get("layer", ""),
        }
        attrs = n.get("attributes", {})
        if attrs:
            key_attrs = {
                k: v for k, v in attrs.items()
                if k in ("purpose", "description", "url", "location", "platforms",
                         "tech_stack", "role", "stack", "status", "domain")
            }
            if key_attrs:
                entry["attributes"] = key_attrs
        # Include relations preview (from edges)
        relations = n.get("_relations", [])
        if relations:
            entry["relations"] = relations[:5] if isinstance(relations, list) else []
        items.append(entry)

    return json.dumps(items, ensure_ascii=False)


# ── Topic hint generator ──

def generate_topic_hint(
    l1_keyword: str | None,
    l2_similarity: float | None,
    l2_verdict: TopicVerdict | None,
    current_topic: str,
) -> str:
    """Generate a topic hint string from L1/L2 results for the extractor model."""
    if l1_keyword:
        return "same (high confidence from keyword match)"

    if l2_verdict is not None and l2_similarity is not None:
        if l2_similarity >= 0.80:
            return "same (high confidence from embedding similarity)"
        if l2_similarity < 0.65:
            return "changed (medium confidence — verify)"
        # Ambiguous range
        if l2_verdict == TopicVerdict.SUBTOPIC:
            return f"subtopic of {current_topic} (medium confidence — verify)"
        if l2_verdict == TopicVerdict.CHANGED:
            return "changed (medium confidence — verify)"
        return "same (high confidence from embedding similarity)"

    return "unresolved — classify the topic yourself"


# ── S1 Unified invoker ──

class S1Unified:
    """Combined topic classifier + knowledge extractor.

    Uses the utility model with a structured system+user message format.
    Always runs on every turn. L1/L2 provide hints, not gates.
    """

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str | None = None,
        embedder: Any = None,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        # Optional embedder — when set, S1 batch-embeds entity names after
        # validation so downstream entity resolution and S2 retrieval can
        # use semantic similarity. Without it, the pipeline falls back to
        # the Phase-1 deterministic path (MinHash LSH only).
        self._embedder = embedder

    async def run(
        self,
        user_msg: str,
        prev_assistant_msg: str,
        current_topic: str,
        topic_hint: str,
        existing_nodes_summary: str,
        existing_node_names: set[str] | None = None,
        existing_nodes: list[dict] | None = None,
        graph: Any = None,
    ) -> S1Result:
        """Execute S1 Unified: topic classification + extraction in one LLM call.

        Builds the system+user message format the extractor model expects.

        When ``existing_nodes`` is provided (list of full graph-store dicts),
        the validated entities are additionally passed through
        ``entity_resolution.resolve_extracted_nodes`` so duplicates against
        the existing graph are merged deterministically (exact normalization
        + MinHash LSH via dedup_helpers). Relations and facts referencing the
        merged entities are rewritten to use the canonical name.
        """
        # Limit previous assistant to first 150 chars — enough for followup detection
        # but too short to dominate topic classification with hallucinated topics
        prev_summary = prev_assistant_msg[:150] if prev_assistant_msg else "null"
        user_content = (
            f"EXISTING NODES:\n{existing_nodes_summary}\n\n"
            f"TOPIC HINT: {topic_hint}\n"
            f"CURRENT TOPIC: {current_topic if current_topic != 'none' else 'null'}\n\n"
            f"PREVIOUS ASSISTANT: {prev_summary}\n"
            f"USER: {user_msg[:800]}"
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            raw_response = await self._llm.chat(
                messages,
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            log.error("S1 Unified LLM call failed: %s", e)
            return _fallback_result(current_topic)

        raw = strip_think_blocks(_clean_response(raw_response))
        log.debug("S1 raw response:\n%s", raw[:2000])
        result = _parse_s1_response(raw)

        # Retry once with lower temperature if JSON parse failed
        if result.extraction == ExtractionResult() and result.topic.action == "same" and result.topic.label is None:
            obj = _parse_first_json(raw, "object")
            if not isinstance(obj, dict):
                log.info("S1 Unified: retrying with temperature=0.0 after JSON parse failure")
                try:
                    raw_response = await self._llm.chat(
                        messages,
                        temperature=0.0,
                        max_tokens=2048,
                    )
                    raw = strip_think_blocks(_clean_response(raw_response))
                    result = _parse_s1_response(raw)
                except Exception as e:
                    log.warning("S1 Unified retry failed: %s", e)

        # Validate extraction against conversation text
        combined_text = f"{user_msg} {prev_assistant_msg}"
        result = _validate_s1(result, combined_text, existing_node_names or set())

        # Phase 2: batch-embed NEW entity names BEFORE resolving against the
        # graph, so that resolve_extracted_nodes can use those embeddings
        # for semantic candidate pre-filter against the graph.
        if self._embedder is not None and result.extraction.entities:
            await _embed_new_entities(result.extraction.entities, self._embedder)

        # Resolve entities against the existing graph via deterministic dedup.
        # Runs even when ``existing_nodes`` is empty so the per-turn log line
        # is always emitted (makes the Phase 1 path visible in diagnostics)
        # and the graph semantic-search hook gets exercised. The function
        # short-circuits internally when there's nothing to compare against.
        if result.extraction.entities:
            result = _resolve_against_graph(
                result, existing_nodes or [], graph=graph,
            )

        # Attach debug data for telemetry
        result.prompt_sent = json.dumps(messages, ensure_ascii=False)
        result.raw_response = raw_response

        return result


# ── Parser ──

def _parse_s1_response(raw: str) -> S1Result:
    """Parse the JSON output from the extractor model.

    The model outputs fields: id, label (for entities), text (for facts).
    Falls back to name/fact fields for backwards compatibility.
    """
    obj = _parse_first_json(raw, "object")
    if not isinstance(obj, dict):
        log.warning("S1 Unified: failed to parse JSON response")
        return S1Result(
            topic=TopicResult(action="same", label=None),
            extraction=ExtractionResult(),
        )

    # Parse topic — model returns {"action": ..., "label": ...},
    # or a plain string (e.g. "Proyectos")
    topic_raw = obj.get("topic", {})
    if isinstance(topic_raw, str):
        # Generic model returned topic as string — treat as new topic
        label = topic_raw.strip() if topic_raw.strip().lower() not in ("null", "none", "") else None
        action = "changed" if label else "same"
    elif isinstance(topic_raw, dict):
        action = topic_raw.get("action", "same")
        label = topic_raw.get("label")
        if label and isinstance(label, str):
            label = label.strip()
            if label.lower() in ("null", "none", ""):
                label = None
        else:
            label = None
    else:
        action = "same"
        label = None
    if action not in ("same", "subtopic", "changed"):
        action = "same"

    topic = TopicResult(action=action, label=label)

    # Parse intent — supports both flat string and nested {"type": ..., "retrieval": ...}
    intent_raw = obj.get("intent", "specific")
    if isinstance(intent_raw, dict):
        intent = str(intent_raw.get("type", "specific")).strip()
        retrieval = intent_raw.get("retrieval")
        if retrieval is not None:
            retrieval = str(retrieval).strip()
            if retrieval not in ("summary_only", "with_chunks"):
                retrieval = None
    elif isinstance(intent_raw, str):
        intent = intent_raw.strip()
        retrieval = obj.get("retrieval")
        if retrieval is not None:
            if not isinstance(retrieval, str) or retrieval not in ("summary_only", "with_chunks"):
                retrieval = None
    else:
        intent = "specific"
        retrieval = None
    if intent not in ("overview", "specific", "followup", "chat"):
        intent = "specific"

    # Parse entities — model uses "label" and "id", with nested "facts".
    # May also return plain strings instead of objects — handle both.
    entities: list[Entity] = []
    entity_facts: list[ExtractedFact] = []  # collected from nested entity facts
    for e_raw in obj.get("entities", []):
        # Generic models may return entities as plain strings (e.g. "Butaco")
        if isinstance(e_raw, str):
            name = e_raw.strip()
            if name and len(name) >= 2:
                entities.append(Entity(name=name, type="concept", layer="", attributes={}))
            continue
        if not isinstance(e_raw, dict):
            continue
        # Primary: "label" field. Fallback: "name" for compat.
        name = str(e_raw.get("label", "") or e_raw.get("name", "")).strip()
        raw_type = str(e_raw.get("type", "")).strip()
        if not name:
            continue
        if not raw_type or len(raw_type) < 2:
            raw_type = "concept"  # default type for generic models
        mapped_type = map_extractor_type(raw_type)
        layer = str(e_raw.get("layer", "")).upper()
        if layer not in ("PERSONAL", "UNIVERSAL"):
            layer = ""
        attrs = e_raw.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        # Preserve existing_id for downstream resolution
        existing_id = e_raw.get("existing_id")
        if existing_id and isinstance(existing_id, str) and existing_id.lower() not in ("null", "none"):
            attrs["_existing_id"] = existing_id
        # Preserve description for graph summary
        description = e_raw.get("description")
        if description and isinstance(description, str) and description.strip():
            attrs["description"] = description.strip()
        # v0.6.1 Change 2 — confidence scoring. Tolerate missing/malformed.
        conf_raw = e_raw.get("confidence")
        try:
            confidence = float(conf_raw) if conf_raw is not None else 1.0
        except (TypeError, ValueError):
            confidence = 1.0
        if confidence < 0.0 or confidence > 1.0:
            confidence = 1.0
        entities.append(Entity(
            name=name, type=mapped_type, layer=layer,
            attributes=attrs, confidence=confidence,
        ))

        # Collect nested facts from entity
        for f_raw in e_raw.get("facts", []):
            if not isinstance(f_raw, dict):
                continue
            fact_text = str(f_raw.get("text", "") or f_raw.get("fact", "")).strip()
            speaker = str(f_raw.get("speaker", "user")).strip().lower()
            if speaker not in ("user", "assistant"):
                speaker = "user"
            if fact_text:
                entity_facts.append(ExtractedFact(
                    entity=name, fact=fact_text, source="user", speaker=speaker,
                ))

    # Build map: model-generated ID → entity label (for relation resolution)
    # The model may generate "id": "supabase_db" but label: "Supabase".
    # Relations reference IDs, but graph nodes use _make_id(label).
    _id_to_label: dict[str, str] = {}
    for e_raw in obj.get("entities", []):
        if not isinstance(e_raw, dict):
            continue
        eid = str(e_raw.get("id", "")).strip().lower()
        label = str(e_raw.get("label", "") or e_raw.get("name", "")).strip()
        if eid and label:
            _id_to_label[eid] = label
            _id_to_label[label.lower()] = label

    def _resolve_name(name: str) -> str:
        """Resolve a model ID or label to the canonical entity label."""
        return _id_to_label.get(name.lower(), name)

    # Parse relations — both top-level AND nested inside entities
    relations: list[Relation] = []
    seen_relations: set[tuple[str, str, str]] = set()

    def _add_relation(src: str, tgt: str, rel: str, confidence: float = 1.0) -> None:
        src = _resolve_name(src)
        tgt = _resolve_name(tgt)
        if not src or not tgt or not rel or src.lower() == tgt.lower():
            return  # skip empty or self-referencing relations
        key = (src.lower(), tgt.lower(), rel.lower())
        if key not in seen_relations:
            seen_relations.add(key)
            relations.append(Relation(
                source=src, target=tgt, relation=rel, confidence=confidence,
            ))

    # Top-level relations array
    for r_raw in obj.get("relations", []):
        if not isinstance(r_raw, dict):
            continue
        r_conf_raw = r_raw.get("confidence")
        try:
            r_conf = float(r_conf_raw) if r_conf_raw is not None else 1.0
        except (TypeError, ValueError):
            r_conf = 1.0
        if r_conf < 0.0 or r_conf > 1.0:
            r_conf = 1.0
        _add_relation(
            str(r_raw.get("source", "")).strip(),
            str(r_raw.get("target", "")).strip(),
            str(r_raw.get("relation", "")).strip(),
            confidence=r_conf,
        )

    # Nested relations inside entities
    # Use entity LABEL (not model-generated ID) for relation source,
    # so _make_id(label) in graph matches the node ID.
    for e_raw in obj.get("entities", []):
        if not isinstance(e_raw, dict):
            continue
        entity_label = str(e_raw.get("label", "") or e_raw.get("name", "") or e_raw.get("id", "")).strip()
        for r_raw in e_raw.get("relations", []):
            if not isinstance(r_raw, dict):
                continue
            # Use label for source; target may reference another entity by ID,
            # so try to resolve via label_by_id map
            target_raw = str(r_raw.get("target", "")).strip()
            _add_relation(
                str(r_raw.get("source", entity_label)).strip(),
                target_raw,
                str(r_raw.get("relation", "")).strip(),
            )

    # Parse top-level facts — supports both "entity" and "entity_id" keys
    facts: list[ExtractedFact] = []
    for f_raw in obj.get("facts", []):
        if not isinstance(f_raw, dict):
            continue
        entity = str(f_raw.get("entity", "") or f_raw.get("entity_id", "")).strip()
        # Resolve entity_id to label if possible
        entity = _resolve_name(entity) if entity else ""
        fact_text = str(f_raw.get("text", "") or f_raw.get("fact", "")).strip()
        speaker = str(f_raw.get("speaker", "user")).strip().lower()
        if speaker not in ("user", "assistant"):
            speaker = "user"
        if entity and fact_text:
            facts.append(ExtractedFact(
                entity=entity, fact=fact_text, source="user", speaker=speaker,
            ))

    # Merge nested entity facts with top-level facts
    all_facts = entity_facts + facts

    extraction = ExtractionResult(entities=entities, relations=relations, facts=all_facts)
    return S1Result(topic=topic, extraction=extraction, intent=intent, retrieval=retrieval)


# ── Validator ──

_VAGUE_FACTS = frozenset({
    "was mentioned", "is mentioned", "appeared in conversation",
    "fue mencionado", "se mencionó", "was discussed",
})

# Entity names that are technical jargon, not knowledge graph entities.
# These are implementation details that leak from technical conversations.
_GARBAGE_ENTITY_PATTERNS: frozenset[str] = frozenset({
    "index", "query", "table", "column", "row", "field",
    "file", "folder", "directory", "path", "extension",
    "endpoint", "request", "response", "header", "body",
    "variable", "function", "method", "class", "module",
    "parameter", "argument", "flag", "option", "config",
    "error", "exception", "warning", "log", "debug",
    "migration", "schema", "constraint", "trigger",
})

# Exact names that should never be entities
_GARBAGE_ENTITY_EXACT: frozenset[str] = frozenset({
    "xlsx", "csv", "json", "yaml", "toml", "xml", "html", "css",
    "gin index", "brin index", "btree index", "hash index",
    "primary key", "foreign key", "unique constraint",
    "null", "true", "false", "none", "undefined",
    "select", "insert", "update", "delete", "join",
})


def _is_garbage_entity(name: str) -> bool:
    """Return True if the entity name is technical jargon, not a real entity."""
    lower = name.lower().strip()
    if lower in _GARBAGE_ENTITY_EXACT:
        return True
    # Single-word matches against pattern set (e.g., "index", "query")
    if lower in _GARBAGE_ENTITY_PATTERNS:
        return True
    # Multi-word ending with a pattern word (e.g., "GIN index", "BRIN index")
    words = lower.split()
    if len(words) >= 2 and words[-1] in _GARBAGE_ENTITY_PATTERNS:
        return True
    # Too short to be meaningful
    if len(lower) < 2:
        return True
    return False


def _fuzzy_match(name: str, candidates: set[str], threshold: float = 0.85) -> str | None:
    """Find the best fuzzy match for a name among candidate strings.

    Uses the same primitives as ``acervo.extraction.dedup_helpers`` so the
    whole S1 pipeline shares a single normalization + similarity strategy.

    The algorithm runs in three passes, from cheapest to costliest:

    1. **Exact normalized match.** Lowercase + whitespace collapse (same rule
       as MinHash dedup) — catches "Sandy Veliz" vs "sandy veliz".
    2. **ID-level match.** ``_make_id`` strips accents and punctuation so
       "josé pérez" and "Jose Perez" map to the same canonical ID.
    3. **Jaccard over 3-gram character shingles.** Only runs if the name has
       enough entropy (skips short/repetitive strings) so we don't get false
       positives on generic tokens like "app" or "data". Matches the
       0.85 default, tunable by the caller.

    The ``threshold`` argument is the minimum Jaccard similarity required for
    the fuzzy pass; it defaults to 0.85 (matching
    ``dedup_helpers._FUZZY_JACCARD_THRESHOLD``). Callers that want looser
    matching can pass a lower value (e.g. 0.70 for fact-entity resolution).

    Adapted from Graphiti's approach (see acervo/THIRD_PARTY.md).
    """
    if not name or not candidates:
        return None

    from acervo.extraction.dedup_helpers import (
        _cached_shingles,
        _has_high_entropy,
        _jaccard_similarity,
        _normalize_name_for_fuzzy,
        _normalize_string_exact,
    )

    normalized_name_exact = _normalize_string_exact(name)
    normalized_name_fuzzy = _normalize_name_for_fuzzy(name)
    name_id = _make_id(name)

    # 1. Exact normalized match — lowercase + whitespace collapsed.
    for cand in candidates:
        if _normalize_string_exact(cand) == normalized_name_exact:
            return cand

    # 2. ID-level match — strips accents and punctuation.
    for cand in candidates:
        if _make_id(cand) == name_id:
            return cand

    # 3. Jaccard fuzzy over 3-gram shingles. Entropy gate blocks unreliable
    #    short / low-information names from reaching this path.
    if not _has_high_entropy(normalized_name_fuzzy):
        return None

    name_shingles = _cached_shingles(normalized_name_fuzzy)
    if not name_shingles:
        return None

    best_cand: str | None = None
    best_score = 0.0
    for cand in candidates:
        cand_fuzzy = _normalize_name_for_fuzzy(cand)
        if not _has_high_entropy(cand_fuzzy):
            continue
        cand_shingles = _cached_shingles(cand_fuzzy)
        score = _jaccard_similarity(name_shingles, cand_shingles)
        if score > best_score:
            best_score = score
            best_cand = cand

    if best_cand is not None and best_score >= threshold:
        return best_cand

    return None


def _validate_s1(
    result: S1Result, conversation_text: str,
    existing_node_names: set[str] | None = None,
) -> S1Result:
    """Post-parse validation: reject hallucinated entities, vague relations/facts."""
    conv_lower = conversation_text.lower()
    ext = result.extraction
    _existing_lower = {n.lower() for n in (existing_node_names or set())}

    # Filter entities: must appear in conversation text, not garbage
    valid_entities: list[Entity] = []
    valid_names_lower: set[str] = set()
    for e in ext.entities:
        name_lower = e.name.lower()
        # v0.6.1 Change 2: bypass the garbage filter when the LLM itself
        # flagged the entity as low confidence (< 0.7). It's already marked
        # uncertain — we don't need a second filter telling it off. Tech
        # jargon like "Prisma schema" hits the pattern list but is a valid
        # entity when the LLM explicitly downgraded its own confidence.
        bypass_garbage = e.confidence < 0.7
        if not bypass_garbage and _is_garbage_entity(e.name):
            log.info("S1: rejected garbage entity: %s", e.name)
            continue
        if name_lower not in conv_lower:
            # Try fuzzy match against conversation text words.
            # Threshold 0.85 matches the dedup_helpers default; short names
            # are filtered out by the entropy gate inside _fuzzy_match.
            conv_words = set(conv_lower.split())
            fuzzy = _fuzzy_match(name_lower, conv_words, threshold=0.85)
            if fuzzy:
                log.info("S1: fuzzy matched entity '%s' → '%s' in conversation", e.name, fuzzy)
            elif e.attributes.get("_existing_id"):
                pass  # LLM matched to graph node, allow
            elif _fuzzy_match(name_lower, _existing_lower, threshold=0.85):
                log.info("S1: fuzzy matched entity '%s' to existing graph node", e.name)
            else:
                log.info("S1: rejected hallucinated entity: %s", e.name)
                continue
        valid_entities.append(e)
        valid_names_lower.add(name_lower)

    # Facts can reference entities from this turn OR existing graph nodes
    # Build a combined set + fuzzy resolution map
    fact_valid_names = set(valid_names_lower) | _existing_lower

    def _resolve_fact_entity(entity_name: str) -> str | None:
        """Resolve a fact's entity reference, with fuzzy fallback.

        Uses a slightly looser threshold (0.80) than the 0.85 default
        because fact entities are often referred to with variations the
        LLM introduces (e.g. "Sandy" vs "Sandy Veliz" inside the same turn),
        and we prefer to attach the fact rather than drop it.
        """
        lower = entity_name.lower()
        if lower in fact_valid_names:
            return entity_name
        # Fuzzy match against all known entities
        match = _fuzzy_match(lower, fact_valid_names, threshold=0.80)
        if match:
            log.info("S1: fuzzy resolved fact entity '%s' → '%s'", entity_name, match)
            return match
        return None

    # Filter relations: must reference valid entities, no vague relation names.
    # Threshold 0.85 matches the dedup_helpers default.
    valid_relations = [
        r for r in ext.relations
        if (r.source.lower() in valid_names_lower or r.target.lower() in valid_names_lower
            or _fuzzy_match(r.source.lower(), valid_names_lower, 0.85) is not None
            or _fuzzy_match(r.target.lower(), valid_names_lower, 0.85) is not None)
        and r.relation.lower().replace(" ", "_") not in (
            "related_to", "co_mentioned", "is_related_to", "mentioned_with",
        )
    ]

    # Cap edges at 2x entity count
    max_edges = max(len(valid_entities) * 2, 1)
    if len(valid_relations) > max_edges:
        log.info("S1: capped relations from %d to %d", len(valid_relations), max_edges)
        valid_relations = valid_relations[:max_edges]

    # Filter facts: reject vague/short, must reference valid entity or existing node
    valid_facts: list[ExtractedFact] = []
    dropped_facts: list[dict] = []
    for f in ext.facts:
        resolved = _resolve_fact_entity(f.entity)
        if resolved is None:
            dropped_facts.append({"entity": f.entity, "text": f.fact, "reason": f"entity_not_found: {f.entity}"})
            log.info("S1: dropped fact (entity not found): %s → %s", f.entity, f.fact[:60])
            continue
        # Use the resolved entity name
        if resolved != f.entity:
            f = ExtractedFact(entity=resolved, fact=f.fact, source=f.source, speaker=f.speaker)
        fact_lower = f.fact.lower().strip()
        if len(fact_lower) < 10:
            dropped_facts.append({"entity": f.entity, "text": f.fact, "reason": "too_short"})
            log.info("S1: rejected short fact for %s: %s", f.entity, f.fact)
            continue
        if fact_lower in _VAGUE_FACTS:
            dropped_facts.append({"entity": f.entity, "text": f.fact, "reason": "vague"})
            log.info("S1: rejected vague fact for %s: %s", f.entity, f.fact)
            continue
        valid_facts.append(f)

    return S1Result(
        topic=result.topic,
        extraction=ExtractionResult(
            entities=valid_entities,
            relations=valid_relations,
            facts=valid_facts,
        ),
        intent=result.intent,
        retrieval=result.retrieval,
        raw_entity_count=len(ext.entities),
        raw_relation_count=len(ext.relations),
        raw_fact_count=len(ext.facts),
        dropped_facts=dropped_facts,
    )


def _resolve_against_graph(
    result: S1Result,
    existing_nodes: list[dict],
    graph: Any = None,
) -> S1Result:
    """Merge extracted entities against the existing graph via dedup_helpers.

    For each extracted entity, runs the deterministic resolution pipeline
    (exact normalize → MinHash LSH fuzzy → entropy gate). When a match
    against an existing graph node is found, mutates the Entity in-place to:
      - adopt the canonical name from the existing node
      - stamp ``_existing_id`` in attributes so downstream upserts target
        the correct row

    Also rewrites Relation.source/target and ExtractedFact.entity references
    whose names changed during canonicalization, keeping the S1Result
    self-consistent.

    When ``graph`` is provided and exposes ``entity_similarity_search``,
    Phase 2 semantic pre-filter is active: the candidate universe per
    entity is narrowed from ``existing_nodes`` to the top-K most similar
    graph nodes using the entity's name embedding. Otherwise we fall back
    to the Phase 1 path (MinHash LSH over the full list).
    """
    from acervo.extraction.entity_resolution import resolve_extracted_nodes

    entities = result.extraction.entities
    if not entities:
        return result

    existing_ids: set[str] = {
        str(n.get("id") or n.get("uuid") or "") for n in existing_nodes
    }
    existing_ids.discard("")

    resolved_nodes, _uuid_map, _duplicate_pairs = resolve_extracted_nodes(
        entities, existing_nodes, graph=graph
    )

    name_rewrites: dict[str, str] = {}
    merged_count = 0
    for orig_entity, canonical in zip(entities, resolved_nodes, strict=False):
        # An entity is "merged" when its canonical resolution points at a
        # real graph node id. Fresh entities get a synthetic uuid that is not
        # in existing_ids.
        if canonical.uuid in existing_ids:
            old_name = orig_entity.name
            if canonical.name and canonical.name != old_name:
                name_rewrites[old_name.lower()] = canonical.name
                orig_entity.name = canonical.name
                log.info(
                    "S1._resolve_against_graph: merged %r -> canonical %r (existing_id=%s)",
                    old_name, canonical.name, canonical.uuid,
                )
            else:
                log.info(
                    "S1._resolve_against_graph: stamped _existing_id=%s on %r",
                    canonical.uuid, old_name,
                )
            orig_entity.attributes["_existing_id"] = canonical.uuid
            merged_count += 1

    if merged_count:
        log.info(
            "S1: merged %d/%d extracted entities against existing graph nodes",
            merged_count,
            len(entities),
        )

    if name_rewrites:
        for rel in result.extraction.relations:
            new_source = name_rewrites.get(rel.source.lower())
            if new_source is not None:
                rel.source = new_source
            new_target = name_rewrites.get(rel.target.lower())
            if new_target is not None:
                rel.target = new_target
        for fact in result.extraction.facts:
            new_entity = name_rewrites.get(fact.entity.lower())
            if new_entity is not None:
                fact.entity = new_entity

    return result


async def _embed_new_entities(entities: list[Entity], embedder: Any) -> None:
    """Batch-embed names of newly extracted entities in place.

    Puts the embedding on ``entity.attributes['name_embedding']`` so
    downstream upsert paths can persist it (LadybugGraphStore reads
    ``attributes['name_embedding']`` when writing EntityNode rows; the
    Phase 2 DDL gives the column a dedicated slot).

    Entities that were already merged against a graph node (``_existing_id``
    is stamped) are skipped — their embedding already lives on the canonical
    row.
    """
    targets: list[Entity] = [
        e for e in entities
        if not e.attributes.get("_existing_id")
        and not e.attributes.get("name_embedding")
        and e.name.strip()
    ]
    if not targets:
        return

    names = [e.name.replace("\n", " ") for e in targets]
    try:
        vectors = await embedder.embed_batch(names)
    except Exception as exc:
        log.warning("S1: entity batch embedding failed (%d names): %s", len(names), exc)
        return

    if len(vectors) != len(targets):
        log.warning(
            "S1: embedder returned %d vectors for %d entities; skipping attach",
            len(vectors),
            len(targets),
        )
        return

    for entity, vec in zip(targets, vectors, strict=True):
        entity.attributes["name_embedding"] = list(vec)


def _fallback_result(current_topic: str) -> S1Result:
    """Return a safe fallback when S1 fails (graceful degradation)."""
    return S1Result(
        topic=TopicResult(action="same", label=None),
        extraction=ExtractionResult(),
    )
