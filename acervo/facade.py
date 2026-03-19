"""Acervo — context proxy for AI agents.

Sits between the user and the LLM. Enriches context before the LLM call
(prepare), and extracts knowledge after the LLM responds (process).

Usage:
    from acervo import Acervo

    memory = Acervo(llm=my_utility_client, owner="Sandy")

    # Before LLM call
    prep = await memory.prepare(user_text, history)
    # Client calls LLM with prep.context_stack, handles MCP tools, etc.

    # After LLM call
    await memory.process(user_text, assistant_response, web_results="...")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from acervo.graph import TopicGraph, _make_id
from acervo.extractor import (
    ConversationExtractor,
    SearchExtractor,
    ExtractionResult,
)
from acervo.layers import Layer
from acervo.llm import LLMClient, Embedder
from acervo.ontology import is_likely_universal
from acervo.synthesizer import synthesize
from acervo.query_planner import QueryPlanner, PlanResult
from acervo.topic_detector import TopicDetector, TopicVerdict
from acervo.context_index import ContextIndex

log = logging.getLogger(__name__)


@dataclass
class PrepareResult:
    """Result of Acervo.prepare() — everything the client needs to call the LLM."""

    context_stack: list[dict]  # messages ready for LLM (system + warm + hot + user)
    plan: PlanResult           # what tool the planner recommends (GRAPH_ALL, WEB_SEARCH, etc.)
    topic: str                 # current topic label
    hot_tokens: int = 0
    warm_tokens: int = 0
    total_tokens: int = 0
    warm_content: str = ""     # the synthesized warm layer text (for display)
    has_context: bool = False  # True if graph had relevant data for this query
    needs_tool: bool = False   # True if planner wants a tool (WEB_SEARCH) but no data yet

    def add_web_results(self, web_content: str) -> None:
        """Inject web search results into the context stack.

        Call this after the client executes a WEB_SEARCH and before calling the LLM.
        Replaces the warm layer with web results.
        """
        if not web_content:
            return

        # Find the warm layer injection point (after system msg)
        # Remove existing warm content if present
        new_stack: list[dict] = [self.context_stack[0]]  # system msg

        # Add web results as warm layer
        new_stack.append({
            "role": "user",
            "content": (
                "[RESULTADOS DE BÚSQUEDA WEB — datos reales y actualizados]\n"
                f"{web_content}\n"
                "[FIN RESULTADOS]\n"
                "Usá estos resultados para responder al usuario. Podés citar las fuentes."
            ),
        })
        new_stack.append({"role": "assistant", "content": "Entendido."})

        # Add remaining messages (hot layer + user message), skip old warm if present
        for msg in self.context_stack[1:]:
            # Skip old warm layer (identified by the context markers)
            if msg.get("content", "").startswith("[CONTEXTO VERIFICADO]"):
                continue
            if msg.get("content") == "Entendido." and msg.get("role") == "assistant":
                continue
            new_stack.append(msg)

        self.context_stack = new_stack
        self.warm_content = web_content


class Acervo:
    """Context proxy for AI agents.

    - prepare(): pre-LLM — topic detection, planning, context building
    - process(): post-LLM — entity extraction, fact persistence
    - commit()/materialize(): lower-level API (still available)
    """

    def __init__(
        self,
        llm: LLMClient,
        owner: str = "",
        persist_path: str | Path = "data/graph",
        embedder: Embedder | None = None,
        embed_threshold: float = 0.65,
        hot_layer_max_messages: int = 2,
        hot_layer_max_tokens: int = 500,
        compaction_trigger_tokens: int = 2000,
    ) -> None:
        self._graph = TopicGraph(Path(persist_path))
        self._llm = llm
        self._extractor = ConversationExtractor(llm)
        self._search_extractor = SearchExtractor(llm)
        self._owner = owner

        # Pipeline components
        self._topic_detector = TopicDetector(
            llm, embedder=embedder, embed_threshold=embed_threshold,
        )
        self._planner = QueryPlanner(llm)
        self._context_index = ContextIndex(
            self._graph, llm,
            hot_layer_max_messages=hot_layer_max_messages,
            hot_layer_max_tokens=hot_layer_max_tokens,
            compaction_trigger_tokens=compaction_trigger_tokens,
        )

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = ".env",
        owner: str = "",
        persist_path: str | Path = "data/graph",
    ) -> Acervo:
        """Create an Acervo instance from environment variables."""
        from acervo.openai_client import OpenAIClient

        if env_file:
            _load_dotenv(Path(env_file))

        base_url = os.environ.get("ACERVO_LIGHT_MODEL_URL", "http://localhost:1234/v1")
        model = os.environ.get("ACERVO_LIGHT_MODEL", "qwen2.5-3b-instruct")
        api_key = os.environ.get("ACERVO_LIGHT_API_KEY", "lm-studio")

        llm = OpenAIClient(base_url=base_url, model=model, api_key=api_key)
        return cls(llm=llm, owner=owner, persist_path=persist_path)

    @property
    def graph(self) -> TopicGraph:
        return self._graph

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def topic_detector(self) -> TopicDetector:
        return self._topic_detector

    @property
    def context_index(self) -> ContextIndex:
        return self._context_index

    # ── prepare/process API ──

    async def prepare(
        self,
        user_text: str,
        history: list[dict],
    ) -> PrepareResult:
        """Pre-LLM processing: detect topic, plan retrieval, build context.

        Args:
            user_text: the user's current message
            history: full conversation history as list of {"role", "content"} dicts

        Returns:
            PrepareResult with context_stack ready for LLM, plus the plan
        """
        # Cycle graph status: hot → warm → cold
        self._graph.cycle_status()

        # Topic detection
        detection = await self._topic_detector.detect(user_text)
        if detection.verdict in (TopicVerdict.CHANGED, TopicVerdict.SUBTOPIC):
            new_label = await self._topic_detector.extract_topic_label(user_text)
            self._topic_detector.current_topic = new_label

        current_topic = self._topic_detector.current_topic

        # Activate mentioned nodes
        self._activate_mentioned_nodes(user_text)

        # Keep current topic node hot
        if current_topic != "none":
            topic_id = _make_id(current_topic)
            topic_node = self._graph.get_node(topic_id)
            if topic_node:
                self._graph.set_node_status(topic_id, "hot")

        # Find entity node for the planner
        entity_node = self._graph.get_node(_make_id(current_topic)) if current_topic != "none" else None
        if not entity_node:
            msg_lower = user_text.lower()
            for node in self._graph.get_all_nodes():
                label = node.get("label", "").lower()
                if label and len(label) >= 3 and label in msg_lower:
                    entity_node = node
                    break

        # Build facts summary for planner (facts + relations + existence)
        if entity_node:
            parts = []
            facts = entity_node.get("facts", [])
            if facts:
                parts.append("Hechos: " + ", ".join(f.get("fact", "") for f in facts))
            edges = self._graph.get_edges_for(entity_node.get("id", ""))
            if edges:
                rel_labels = [e.get("relation", "") for e in edges[:5]]
                parts.append(f"Relaciones: {', '.join(rel_labels)}")
            if not parts:
                parts.append(
                    f"Nodo existe (mencionado {entity_node.get('session_count', 0)} veces) "
                    "pero sin hechos verificados"
                )
            facts_summary = ". ".join(parts)[:500]
        else:
            facts_summary = ""

        # Query planner
        plan = await self._planner.plan(
            user_text,
            current_topic if current_topic != "none" else "",
            entity_node.get("type", "") if entity_node else "",
            facts_summary,
        )

        # Build context stack
        warm_override = ""
        warm_source = ""
        if plan.tool in ("GRAPH_ALL", "GRAPH_SEARCH"):
            # Synthesize from graph
            entity_hint = plan.entity or current_topic
            if entity_node:
                self._graph.set_node_status(entity_node.get("id", ""), "hot")
            warm_override = synthesize(self._graph, entity_hint)
            warm_source = "graph"

        context_stack, hot_tk, warm_tk, total_tk = self._context_index.build_context_stack(
            history, current_topic,
            warm_override=warm_override,
            warm_source=warm_source,
        )

        # has_context is True only when warm content has actual facts, not just node headers
        # Node headers are "# Name (Type)" — facts start with "- "
        has_context = bool(warm_override) and "- " in warm_override
        needs_tool = not has_context

        return PrepareResult(
            context_stack=context_stack,
            plan=plan,
            topic=current_topic,
            hot_tokens=hot_tk,
            warm_tokens=warm_tk,
            total_tokens=total_tk,
            warm_content=warm_override,
            has_context=has_context,
            needs_tool=needs_tool,
        )

    async def process(
        self,
        user_text: str,
        assistant_text: str,
        web_results: str = "",
    ) -> ExtractionResult:
        """Post-LLM processing: extract entities/facts and persist to graph.

        Skips extraction when the LLM response is a "no data" instruction-following
        response (e.g., "no tengo datos, queres que busque?") to avoid storing
        hallucinated entities from non-informative responses.

        Args:
            user_text: what the user said
            assistant_text: what the LLM responded
            web_results: raw web search content (if any) to extract facts from

        Returns:
            ExtractionResult with extracted entities, relations, facts
        """
        # Check if LLM response is a "no data" instruction-following response
        _SKIP_PHRASES = ("no tengo información", "no tengo datos", "puedo buscar",
                         "busque en internet", "querés que busque", "quieres que busque")
        assistant_lower = assistant_text.lower()
        is_no_data_response = any(p in assistant_lower for p in _SKIP_PHRASES)

        if is_no_data_response and not web_results:
            # Only create the primary entity the user asked about (not hallucinated ones)
            # The extractor tends to hallucinate related entities from training data
            result = await self._extractor.extract(user_text, "")
            if result.entities:
                # Keep only the first entity (the one the user actually mentioned)
                primary = result.entities[0]
                self._graph.upsert_entities(
                    [(primary.name, primary.type)],
                    owner=self._owner or None,
                )
                node = self._graph.get_node(_make_id(primary.name))
                if node and not node.get("facts"):
                    node["status"] = "pending_verification"
                result.entities = [primary]
                result.relations = []
                result.facts = []
            return result

        # Extract from conversation (only user facts stored)
        result = await self.commit(user_text, assistant_text)

        # If web results were used, extract and persist web facts separately
        if web_results:
            await self._persist_web_facts(user_text, web_results)

        return result

    # ── Lower-level API (still available) ──

    async def commit(
        self,
        user_msg: str,
        assistant_msg: str = "",
    ) -> ExtractionResult:
        """Extract entities/facts from text and persist to graph.

        Only user-spoken facts are stored. Assistant facts are filtered out.
        """
        result = await self._extractor.extract(user_msg, assistant_msg)

        if result.entities:
            pairs = [(e.name, e.type) for e in result.entities]
            relations = (
                [(r.source, r.target, r.relation) for r in result.relations]
                if result.relations else None
            )
            user_facts = [
                (f.entity, f.fact, f.source)
                for f in result.facts
                if f.speaker == "user"
            ]

            has_universal = any(
                is_likely_universal(e.type) for e in result.entities
            )
            layer = Layer.UNIVERSAL if has_universal else Layer.PERSONAL
            source = "world" if has_universal else "user_assertion"

            self._graph.upsert_entities(
                pairs,
                relations,
                user_facts if user_facts else None,
                layer=layer,
                source=source,
                owner=self._owner or None,
            )

            # Mark entities without user facts as pending_verification
            # (e.g., extracted from a question, not a statement)
            if not user_facts:
                for name, _ in pairs:
                    nid = _make_id(name)
                    node = self._graph.get_node(nid)
                    if node and not node.get("facts"):
                        node["status"] = "pending_verification"

        return result

    def materialize(self, query: str, token_budget: int = 800) -> str:
        """Build a context string from relevant graph nodes."""
        return synthesize(self._graph, query)

    def cycle_status(self) -> None:
        """Demote node statuses: hot -> warm -> cold."""
        self._graph.cycle_status()

    # ── Internal helpers ──

    def _activate_mentioned_nodes(self, user_text: str) -> None:
        """Set graph nodes to 'hot' if the user message mentions them."""
        msg_lower = user_text.lower()
        msg_words = set(msg_lower.split())
        for node in self._graph.get_all_nodes():
            label = node.get("label", "").lower()
            nid = node.get("id", "")
            if not label or len(label) < 3:
                continue
            if label in msg_lower:
                self._graph.set_node_status(nid, "hot")
                continue
            for word in msg_words:
                if len(word) >= 4 and label.startswith(word):
                    self._graph.set_node_status(nid, "hot")
                    break
            label_words = set(label.split())
            if len(label_words) > 1 and label_words.issubset(msg_words):
                self._graph.set_node_status(nid, "hot")

    async def _persist_web_facts(self, query: str, web_content: str) -> None:
        """Extract and persist entities, relations, and facts from web search results.

        Filters out entities unrelated to the search query to prevent
        garbage nodes from irrelevant search results.
        """
        try:
            result = await self._search_extractor.extract(query, web_content)

            # Filter: only keep entities related to the query or existing graph nodes
            query_lower = query.lower()
            def _is_relevant(name: str) -> bool:
                name_lower = name.lower()
                # Entity name appears in query or query appears in entity name
                if name_lower in query_lower or query_lower in name_lower:
                    return True
                # Entity already exists in graph
                if self._graph.get_node(_make_id(name)):
                    return True
                # Entity is mentioned in a relation with a relevant entity
                for r in result.relations:
                    if r.source.lower() == name_lower or r.target.lower() == name_lower:
                        other = r.target if r.source.lower() == name_lower else r.source
                        if other.lower() in query_lower or self._graph.get_node(_make_id(other)):
                            return True
                return False

            # Build entity pairs from extracted entities + fact entities
            entity_pairs = []
            seen = set()
            for e in result.entities:
                if e.name not in seen and _is_relevant(e.name):
                    entity_pairs.append((e.name, e.type))
                    seen.add(e.name)
            for f in result.facts:
                if f.entity not in seen and _is_relevant(f.entity):
                    existing = self._graph.get_node(_make_id(f.entity))
                    etype = existing["type"] if existing else "Unknown"
                    entity_pairs.append((f.entity, etype))
                    seen.add(f.entity)

            # Filter relations to only include relevant entities
            if result.relations:
                result.relations = [
                    r for r in result.relations
                    if r.source in seen or r.target in seen
                ]

            if not entity_pairs and not result.facts:
                return

            # Build relation tuples
            relations = (
                [(r.source, r.target, r.relation) for r in result.relations]
                if result.relations else None
            )

            # Build fact tuples
            fact_tuples = (
                [(f.entity, f.fact, "web") for f in result.facts]
                if result.facts else None
            )

            self._graph.upsert_entities(
                entity_pairs,
                relations=relations,
                facts=fact_tuples,
                layer=Layer.UNIVERSAL,
                source="web",
            )
        except Exception as e:
            log.warning("Web fact persistence failed: %s", e)


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — no dependencies."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
