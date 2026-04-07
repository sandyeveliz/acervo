"""Pipeline orchestrator — thin wiring between S1, S2, S3, S1.5.

No business logic here. Just calls stages in order and builds results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acervo.domain.models import ExtractionResult
from acervo.domain.s1_extractor import S1Unified, build_graph_summary, generate_topic_hint
from acervo.domain.s2_activator import S2Activator
from acervo.domain.s3_assembler import S3Assembler
from acervo.domain.s15_updater import S1_5GraphUpdate, apply_s1_5_result
from acervo.token_counter import count_tokens

log = logging.getLogger(__name__)


# ── PrepareResult (unchanged contract) ──

@dataclass
class PrepareResult:
    """Result of prepare() — everything the client needs to call the LLM."""
    context_stack: list[dict]
    topic: str
    hot_tokens: int = 0
    warm_tokens: int = 0
    total_tokens: int = 0
    warm_content: str = ""
    has_context: bool = False
    needs_tool: bool = False
    stages: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)

    def add_web_results(self, web_content: str) -> None:
        """Inject web search results into the context stack."""
        if not web_content:
            return
        new_stack: list[dict] = [self.context_stack[0]]
        new_stack.append({
            "role": "user",
            "content": (
                "[WEB SEARCH RESULTS — real and up-to-date data]\n"
                f"{web_content}\n"
                "[END RESULTS]\n"
                "Use these results to answer the user. You may cite the sources."
            ),
        })
        new_stack.append({"role": "assistant", "content": "Understood."})
        for msg in self.context_stack[1:]:
            if msg.get("content", "").startswith(("[VERIFIED CONTEXT]", "[CONTEXTO VERIFICADO]")):
                continue
            if msg.get("content") in ("Understood.", "Entendido.") and msg.get("role") == "assistant":
                continue
            new_stack.append(msg)
        self.context_stack = new_stack
        self.warm_content = web_content


class Pipeline:
    """Thin orchestrator — calls S1 → S2 → S3 for prepare, S1.5 for process.

    All business logic lives in the stage modules. This class only wires
    them together and collects debug data for telemetry.
    """

    def __init__(
        self,
        *,
        s1: S1Unified,
        s2: S2Activator,
        s3: S3Assembler,
        s15_llm: Any,  # LLMPort — for S1.5 (can be same as S1's LLM)
        s15_prompt: str | None = None,
        graph: Any,  # TopicGraph
        topic_detector: Any,  # TopicDetector
        context_index: Any,  # ContextIndex
        embedder: Any | None = None,  # EmbedderPort
        vector_store: Any | None = None,  # VectorStorePort
        metrics: Any | None = None,  # SessionMetrics
        workspace_path: Path | None = None,
        owner: str = "",
        project_description: str = "",
        warm_token_budget: int = 400,
    ) -> None:
        self._s1 = s1
        self._s2 = s2
        self._s3 = s3
        self._s15_llm = s15_llm
        self._s15_prompt = s15_prompt
        self._graph = graph
        self._topic_detector = topic_detector
        self._context_index = context_index
        self._embedder = embedder
        self._vector_store = vector_store
        self._metrics = metrics
        self._workspace_path = workspace_path
        self._owner = owner
        self._project_description = project_description
        self._warm_token_budget = warm_token_budget
        # Internal state
        self._last_s1_extraction: ExtractionResult | None = None
        self._last_active_node_ids: set[str] = set()
        self._last_s15_debug: dict | None = None

    # ── Public properties ──

    @property
    def graph(self) -> Any:
        return self._graph

    @property
    def topic_detector(self) -> Any:
        return self._topic_detector

    @property
    def context_index(self) -> Any:
        return self._context_index

    @property
    def metrics(self) -> Any:
        return self._metrics

    # ── prepare() ──

    async def prepare(self, user_text: str, history: list[dict]) -> PrepareResult:
        """Pre-LLM pipeline: S1 → S2 → S3."""
        _t0 = time.monotonic()
        _stages: list[str] = []

        # ── S1: Topic detection + Entity extraction ──
        _s1_t0 = time.perf_counter()

        # Embedding (optional)
        user_embedding = None
        if self._embedder:
            try:
                user_embedding = await self._embedder.embed(user_text)
            except Exception as e:
                log.warning("User embedding failed: %s", e)

        # Topic hints
        current_topic = self._topic_detector.current_topic
        detection = await self._topic_detector.detect_hints(user_text, pre_embedding=user_embedding)
        topic_hint = generate_topic_hint(
            detection.keyword, detection.similarity,
            detection.verdict if hasattr(detection, 'verdict') else None,
            current_topic,
        )

        # Graph summary for S1 context
        all_nodes = self._graph.get_all_nodes()
        existing_nodes_summary = build_graph_summary([dict(n) for n in all_nodes], user_text)

        # Previous assistant message
        prev_assistant_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                prev_assistant_msg = msg.get("content", "")
                break

        # Run S1
        s1_result = await self._s1.run(
            user_msg=user_text,
            prev_assistant_msg=prev_assistant_msg,
            current_topic=current_topic,
            topic_hint=topic_hint,
            existing_nodes_summary=existing_nodes_summary,
        )
        _s1_ms = round((time.perf_counter() - _s1_t0) * 1000)

        s1_topic = s1_result.topic
        s1_intent = s1_result.intent
        s1_extraction = s1_result.extraction
        s1_retrieval = s1_result.retrieval

        # Intent override (keyword fallback)
        if s1_intent != "overview":
            _overview = user_text.lower()
            if any(p in _overview for p in (
                "cuantos", "cuántos", "how many", "what do we have",
                "que tenemos", "qué tenemos", "list all", "show all",
                "resumen", "summary", "overview", "inventory",
                "estructura", "structure",
            )):
                s1_intent = "overview"

        # Update topic
        if s1_topic.action in ("changed", "subtopic") and s1_topic.label:
            current_topic = s1_topic.label
            self._topic_detector.current_topic = current_topic
        elif s1_topic.action == "changed" and not s1_topic.label:
            if detection.keyword:
                current_topic = detection.keyword
                self._topic_detector.current_topic = current_topic

        _stages.append(
            f"S1: intent={s1_intent}, topic={current_topic}, "
            f"{len(s1_extraction.entities)}E {len(s1_extraction.relations)}R {len(s1_extraction.facts)}F"
        )
        log.info("[acervo] %s", _stages[-1])

        # Persist S1 entities to graph
        self._last_s1_extraction = s1_extraction
        if s1_extraction.entities:
            self._persist_s1_entities(s1_extraction, user_text, current_topic)

        # ── S2: BFS Seed Selection + Traversal ──
        s2_result = self._s2.run(
            user_text, s1_result, self._graph,
            intent=s1_intent,
            vector_store=self._vector_store,
            user_embedding=user_embedding,
        )
        self._last_active_node_ids = s2_result.active_node_ids

        _stages.append(
            f"S2: seeds={len(s2_result.layered.seeds_used)}, "
            f"hot={len(s2_result.layered.hot)}, "
            f"warm={len(s2_result.layered.warm)}, "
            f"cold={len(s2_result.layered.cold)}"
        )

        # ── S3: Layer-aware Context Assembly ──
        project_overview = self._build_project_overview()
        s3_result = self._s3.run(
            layered=s2_result.layered,
            intent=s1_intent,
            graph=self._graph,
            project_overview=project_overview,
            context_index=self._context_index,
            history=history,
            current_topic=current_topic,
            warm_budget_override=self._warm_token_budget,
        )

        _stages.append(
            f"S3: warm={s3_result.warm_tokens}tk hot={s3_result.hot_tokens}tk "
            f"total={s3_result.total_tokens}tk"
        )

        # ── Build debug dict for telemetry ──
        _debug = self._build_debug(
            s1_result, s1_intent, s1_retrieval, s1_extraction, _s1_ms,
            detection, topic_hint, current_topic,
            s2_result, s3_result,
        )

        _elapsed_ms = int((time.monotonic() - _t0) * 1000)
        log.info(
            "prepare done: topic=%s tokens=%d (warm=%d hot=%d) %dms",
            current_topic, s3_result.total_tokens, s3_result.warm_tokens,
            s3_result.hot_tokens, _elapsed_ms,
        )

        return PrepareResult(
            context_stack=s3_result.context_stack,
            topic=current_topic,
            hot_tokens=s3_result.hot_tokens,
            warm_tokens=s3_result.warm_tokens,
            total_tokens=s3_result.total_tokens,
            warm_content=s3_result.warm_content,
            has_context=s3_result.has_context,
            needs_tool=s3_result.needs_tool,
            stages=_stages,
            debug=_debug,
        )

    # ── process() ──

    async def process(
        self, user_text: str, assistant_text: str, web_results: str = "",
    ) -> ExtractionResult:
        """Post-LLM pipeline: S1.5 graph curation."""
        from acervo.s1_5_graph_update import S1_5GraphUpdate, apply_s1_5_result
        _s15_t0 = time.perf_counter()

        # Skip no-data responses
        _SKIP = ("i don't have information", "i don't have data", "i can search",
                 "would you like me to search", "shall i look",
                 "no tengo información", "no tengo datos", "puedo buscar",
                 "busque en internet", "querés que busque", "quieres que busque")
        if any(p in assistant_text.lower() for p in _SKIP) and not web_results:
            return ExtractionResult()

        # S1 entities for S1.5 context
        s1_entities = self._last_s1_extraction
        new_entities_json = "[]"
        if s1_entities and s1_entities.entities:
            new_entities_json = json.dumps([
                {"name": e.name, "type": e.type, "layer": e.layer}
                for e in s1_entities.entities
            ], ensure_ascii=False)

        existing_nodes_json = build_graph_summary(
            [dict(n) for n in self._graph.get_all_nodes()], user_text,
        )

        # Run S1.5
        s15 = S1_5GraphUpdate(self._s15_llm, prompt_template=self._s15_prompt)
        s15_result = await s15.run(
            new_entities_json=new_entities_json,
            existing_nodes_json=existing_nodes_json,
            current_assistant_msg=assistant_text,
        )
        _s15_ms = round((time.perf_counter() - _s15_t0) * 1000)

        # Apply curation
        audit = apply_s1_5_result(self._graph, s15_result, owner=self._owner)
        log.info(
            "[acervo] S1.5 — merges=%d, fixes=%d, discards=%d, entities=%d, facts=%d (%dms)",
            audit["merges_applied"], audit["type_corrections"],
            audit["discards"], audit["entities_added"], audit["facts_added"], _s15_ms,
        )

        # Always save after mutations
        self._graph.save()

        # Store debug for proxy telemetry
        self._last_s15_debug = {
            "s15_prompt": s15_result.prompt_sent,
            "s15_raw_response": s15_result.raw_response,
            "s15_latency_ms": _s15_ms,
            "s15_actions": {
                "merges_applied": audit["merges_applied"],
                "type_corrections": audit["type_corrections"],
                "discards": audit["discards"],
                "entities_added": audit["entities_added"],
                "facts_added": audit["facts_added"],
            },
            "assistant_msg": assistant_text,
        }

        return s15_result.assistant_extraction

    # ── Internal helpers ──

    def _persist_s1_entities(
        self, extraction: ExtractionResult, user_text: str, current_topic: str,
    ) -> None:
        """Persist S1 entities, relations, and facts to the graph."""
        from acervo.graph import _make_id
        from acervo.layers import Layer
        from acervo.ontology import is_likely_universal

        for entity in extraction.entities:
            layer = Layer.UNIVERSAL if is_likely_universal(entity.type) else Layer.PERSONAL
            if entity.layer == "UNIVERSAL":
                layer = Layer.UNIVERSAL
            elif entity.layer == "PERSONAL":
                layer = Layer.PERSONAL

            entity_facts = [
                (f.entity, f.fact, f.source) for f in extraction.facts
                if f.entity.lower() == entity.name.lower()
            ]

            self._graph.upsert_entities(
                [(entity.name, entity.type)],
                None,
                entity_facts if entity_facts else None,
                layer=layer,
                source="user_assertion",
                owner=self._owner or None,
            )

        if extraction.relations:
            # Build a map from model IDs to entity labels for normalization.
            # The model may use IDs like "supabase_db" in relations but the
            # entity was created with label "Supabase" → node ID "supabase".
            entity_label_map: dict[str, str] = {}
            for e in extraction.entities:
                entity_label_map[e.name.lower()] = e.name
                # Also map the model's generated ID if it has one
                eid = e.attributes.get("_model_id") or e.name.lower().replace(" ", "_")
                entity_label_map[eid.lower()] = e.name

            def _resolve(name: str) -> str:
                """Resolve a relation endpoint to an entity label."""
                return entity_label_map.get(name.lower(), name)

            relations = [
                (_resolve(r.source), _resolve(r.target), r.relation)
                for r in extraction.relations
            ]
            self._graph.upsert_entities(
                [], relations,
                layer=Layer.PERSONAL,
                source="user_assertion",
                owner=self._owner or None,
            )

        # Topic membership
        if current_topic != "none":
            topic_id = _make_id(current_topic)
            for entity in extraction.entities:
                entity_id = _make_id(entity.name)
                if entity_id != topic_id:
                    self._graph.upsert_entities(
                        [], [(entity.name, current_topic, "part_of")],
                        layer=Layer.PERSONAL, source="user_assertion",
                    )

        self._graph.save()

    def _build_project_overview(self) -> str:
        """Build project overview text from description and synthesis nodes."""
        parts: list[str] = []
        if self._project_description:
            parts.append(f"Project: {self._project_description}")

        synthesis_nodes = self._graph.get_nodes_by_kind("synthesis")
        for node in synthesis_nodes:
            if node.get("type") == "project_overview":
                summary = node.get("attributes", {}).get("summary", "")
                if summary:
                    parts.append(summary)
                break

        return "\n\n".join(parts) if parts else ""

    def _build_debug(
        self, s1_result, s1_intent, s1_retrieval, s1_extraction, s1_ms,
        detection, topic_hint, current_topic,
        s2_result, s3_result,
    ) -> dict:
        """Build the debug dict for telemetry/trace."""
        return {
            "s1_detection": {
                "topic_action": s1_result.topic.action,
                "topic_label": s1_result.topic.label,
                "current_topic": current_topic,
                "topic_hint": topic_hint,
                "hint_level": detection.level,
                "hint_verdict": detection.verdict.name if hasattr(detection.verdict, 'name') else str(detection.verdict),
                "hint_keyword": detection.keyword,
                "hint_similarity": detection.similarity,
                "intent": s1_intent,
                "retrieval": s1_retrieval or "none",
                "entities_extracted": len(s1_extraction.entities),
                "relations_extracted": len(s1_extraction.relations),
                "facts_extracted": len(s1_extraction.facts),
                "entities": [
                    {"name": e.name, "type": e.type, "layer": e.layer}
                    for e in s1_extraction.entities[:10]
                ],
                "relations": [
                    {"source": r.source, "target": r.target, "relation": r.relation}
                    for r in s1_extraction.relations[:10]
                ],
                "facts": [
                    {"entity": f.entity, "fact": f.fact, "speaker": f.speaker}
                    for f in s1_extraction.facts[:10]
                ],
            },
            "s1_prompt": s1_result.prompt_sent,
            "s1_raw_response": s1_result.raw_response,
            "s1_latency_ms": s1_ms,
            "s2_gathered": {
                "seeds": s2_result.layered.seeds_used,
                "hot_count": len(s2_result.layered.hot),
                "warm_count": len(s2_result.layered.warm),
                "cold_count": len(s2_result.layered.cold),
                "nodes_total": len(s2_result.layered.hot) + len(s2_result.layered.warm) + len(s2_result.layered.cold),
                "nodes": [
                    {
                        "id": n.get("id", ""),
                        "label": n.get("label", ""),
                        "type": n.get("type", ""),
                        "layer": "hot" if n in s2_result.layered.hot else "warm" if n in s2_result.layered.warm else "cold",
                    }
                    for n in (s2_result.layered.hot + s2_result.layered.warm + s2_result.layered.cold)[:20]
                ],
                "vector_hits": [
                    {"node_id": v.get("node_id", ""), "score": round(v.get("score", 0), 3)}
                    for v in s2_result.vector_hits[:10]
                ],
            },
            "s3_context": {
                "warm_tokens": s3_result.warm_tokens,
                "hot_tokens": s3_result.hot_tokens,
                "total_tokens": s3_result.total_tokens,
                "warm_budget": self._warm_token_budget,
                "warm_source": "graph" if s3_result.warm_content else "none",
                "warm_content_preview": s3_result.warm_content[:500] if s3_result.warm_content else "",
                "has_context": s3_result.has_context,
            },
        }
