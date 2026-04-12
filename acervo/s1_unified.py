"""S1 Unified — combined topic classifier + knowledge extractor.

Runs ALWAYS before S2 Gather. L1/L2 results are passed as hints, not gates.
Uses the utility model (qwen2.5:3b by default) with a structured prompt.

Output: S1Result with topic classification + entities/relations/facts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

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

    def __init__(self, llm: LLMClient, system_prompt: str | None = None) -> None:
        self._llm = llm
        self._system_prompt = system_prompt or _SYSTEM_PROMPT

    async def run(
        self,
        user_msg: str,
        prev_assistant_msg: str,
        current_topic: str,
        topic_hint: str,
        existing_nodes_summary: str,
        existing_node_names: set[str] | None = None,
    ) -> S1Result:
        """Execute S1 Unified: topic classification + extraction in one LLM call.

        Builds the system+user message format the extractor model expects.
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
        entities.append(Entity(name=name, type=mapped_type, layer=layer, attributes=attrs))

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

    def _add_relation(src: str, tgt: str, rel: str) -> None:
        src = _resolve_name(src)
        tgt = _resolve_name(tgt)
        if not src or not tgt or not rel or src.lower() == tgt.lower():
            return  # skip empty or self-referencing relations
        key = (src.lower(), tgt.lower(), rel.lower())
        if key not in seen_relations:
            seen_relations.add(key)
            relations.append(Relation(source=src, target=tgt, relation=rel))

    # Top-level relations array
    for r_raw in obj.get("relations", []):
        if not isinstance(r_raw, dict):
            continue
        _add_relation(
            str(r_raw.get("source", "")).strip(),
            str(r_raw.get("target", "")).strip(),
            str(r_raw.get("relation", "")).strip(),
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


def _fuzzy_match(name: str, candidates: set[str], threshold: float = 0.60) -> str | None:
    """Find the best fuzzy match for a name among candidates.

    Uses difflib.SequenceMatcher (Ratcliff/Obershelp) which handles
    typos like ciplinetti→cipolletti, plus substring containment
    and _make_id normalization for ID-level matches.
    """
    if not name or not candidates:
        return None

    from difflib import get_close_matches

    name_lower = name.lower().strip()
    name_id = _make_id(name)

    # 1. Exact _make_id match (accents, case, punctuation)
    for cand in candidates:
        if _make_id(cand) == name_id:
            return cand

    # 2. Substring containment (min 4 chars to avoid matching "en", "la", etc.)
    if len(name_lower) >= 4:
        for cand in candidates:
            cand_lower = cand.lower().strip()
            if len(cand_lower) >= 4 and (name_lower in cand_lower or cand_lower in name_lower):
                return cand

    # 3. difflib fuzzy match (handles typos)
    matches = get_close_matches(name_lower, [c.lower() for c in candidates], n=1, cutoff=threshold)
    if matches:
        # Return the original-case candidate
        for cand in candidates:
            if cand.lower() == matches[0]:
                return cand

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
        if _is_garbage_entity(e.name):
            log.info("S1: rejected garbage entity: %s", e.name)
            continue
        if name_lower not in conv_lower:
            # Try fuzzy match against conversation text words
            conv_words = set(conv_lower.split())
            fuzzy = _fuzzy_match(name_lower, conv_words, threshold=0.75)
            if fuzzy:
                log.info("S1: fuzzy matched entity '%s' → '%s' in conversation", e.name, fuzzy)
            elif e.attributes.get("_existing_id"):
                pass  # LLM matched to graph node, allow
            elif _fuzzy_match(name_lower, _existing_lower, threshold=0.75):
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
        """Resolve a fact's entity reference, with fuzzy fallback."""
        lower = entity_name.lower()
        if lower in fact_valid_names:
            return entity_name
        # Fuzzy match against all known entities
        match = _fuzzy_match(lower, fact_valid_names, threshold=0.70)
        if match:
            log.info("S1: fuzzy resolved fact entity '%s' → '%s'", entity_name, match)
            return match
        return None

    # Filter relations: must reference valid entities, no vague relation names
    valid_relations = [
        r for r in ext.relations
        if (r.source.lower() in valid_names_lower or r.target.lower() in valid_names_lower
            or _fuzzy_match(r.source.lower(), valid_names_lower, 0.75) is not None
            or _fuzzy_match(r.target.lower(), valid_names_lower, 0.75) is not None)
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


def _fallback_result(current_topic: str) -> S1Result:
    """Return a safe fallback when S1 fails (graceful degradation)."""
    return S1Result(
        topic=TopicResult(action="same", label=None),
        extraction=ExtractionResult(),
    )
