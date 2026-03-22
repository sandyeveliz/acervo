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

from acervo.context_builder import (
    ContextBuilder,
    GatheredInfo,
    RankedChunk,
    select_chunks_by_budget,
    format_chunks_as_context,
)
from acervo.graph import TopicGraph, _make_id
from acervo.extractor import (
    ConversationExtractor,
    SearchExtractor,
    ExtractionResult,
)
from acervo.layers import Layer
from acervo.llm import LLMClient, Embedder, VectorStore
from acervo.metrics import SessionMetrics
from acervo.ontology import is_likely_universal
from acervo.reindexer import Reindexer, hash_file
from acervo.synthesizer import synthesize
from acervo.query_planner import QueryPlanner, PlanResult
from acervo.topic_detector import TopicDetector, TopicVerdict
from acervo.context_index import ContextIndex

log = logging.getLogger(__name__)


@dataclass
class PrepareResult:
    """Result of Acervo.prepare() — everything the client needs to call the LLM."""

    context_stack: list[dict]  # messages ready for LLM (system + warm + hot + user)
    topic: str                 # current topic label
    hot_tokens: int = 0
    warm_tokens: int = 0
    total_tokens: int = 0
    warm_content: str = ""     # the synthesized warm layer text (for display)
    has_context: bool = False  # True if graph had relevant data for this query
    needs_tool: bool = False   # True if no context found (client may want to web search)
    stages: list[str] = field(default_factory=list)  # human-readable stage logs
    debug: dict = field(default_factory=dict)  # structured per-stage data for trace UI

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
                "[WEB SEARCH RESULTS — real and up-to-date data]\n"
                f"{web_content}\n"
                "[END RESULTS]\n"
                "Use these results to answer the user. You may cite the sources."
            ),
        })
        new_stack.append({"role": "assistant", "content": "Understood."})

        # Add remaining messages (hot layer + user message), skip old warm if present
        for msg in self.context_stack[1:]:
            # Skip old warm layer (identified by the context markers)
            if msg.get("content", "").startswith(("[VERIFIED CONTEXT]", "[CONTEXTO VERIFICADO]")):
                continue
            if msg.get("content") in ("Understood.", "Entendido.") and msg.get("role") == "assistant":
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
        vector_store: VectorStore | None = None,
        workspace_path: str | Path | None = None,
        prompts: dict[str, str] | None = None,
        structural_parser: object | None = None,
    ) -> None:
        self._graph = TopicGraph(Path(persist_path))
        self._llm = llm
        p = prompts or {}
        self._extractor = ConversationExtractor(llm, prompt_template=p.get("extractor_conversation"))
        self._search_extractor = SearchExtractor(llm, prompt_template=p.get("extractor_search"))
        self._owner = owner
        self._embedder = embedder
        self._vector_store = vector_store
        self._workspace_path = Path(workspace_path) if workspace_path else None

        # Reindexer (optional — requires structural_parser + workspace_path)
        self._reindexer: Reindexer | None = None
        if structural_parser and self._workspace_path:
            self._reindexer = Reindexer(self._graph, structural_parser, self._workspace_path)

        # Metrics
        self._metrics = SessionMetrics(session_id=self._graph.session_id)

        # Pipeline components
        self._topic_detector = TopicDetector(
            llm, embedder=embedder, embed_threshold=embed_threshold,
            classify_prompt=p.get("topic_classifier"),
        )
        self._planner = QueryPlanner(llm, prompt_template=p.get("planner"))
        self._context_builder = ContextBuilder(llm, prompt_template=p.get("context_builder"))
        self._context_index = ContextIndex(
            self._graph, llm,
            hot_layer_max_messages=hot_layer_max_messages,
            hot_layer_max_tokens=hot_layer_max_tokens,
            compaction_trigger_tokens=compaction_trigger_tokens,
            summarize_prompt=p.get("summarizer"),
        )

    @classmethod
    def from_project(
        cls,
        path: str | Path | None = None,
        llm: LLMClient | None = None,
        auto_init: bool = True,
        **overrides,
    ) -> Acervo:
        """Create Acervo from a project directory.

        This is the recommended way to use Acervo as a library. Discovers or
        creates a .acervo/ directory, loads config, and returns a ready instance.

        Args:
            path: Project root (default: cwd). Looks for .acervo/ here.
            llm: Client-provided LLM. If None, creates one from config/env.
            auto_init: If True, creates .acervo/ if not found (default True).
            **overrides: Additional params forwarded to Acervo.__init__.
        """
        from acervo.project import find_project, init_project

        root = Path(path or ".").resolve()

        # Find or create .acervo/
        project = find_project(root)
        if project is None:
            if auto_init:
                project = init_project(root)
            else:
                raise FileNotFoundError(
                    f"No .acervo/ found in {root}. Run 'acervo init' or pass auto_init=True."
                )

        return cls._from_project(project, llm=llm, **overrides)

    @classmethod
    def _from_project(
        cls,
        project,
        llm: LLMClient | None = None,
        **overrides,
    ) -> Acervo:
        """Internal factory: create Acervo from an AcervoProject instance."""
        from acervo.structural_parser import StructuralParser

        # LLM: client-provided > config.toml > env vars > defaults
        if llm is None:
            from acervo.openai_client import OpenAIClient

            llm_cfg = project.llm_config()
            llm = OpenAIClient(
                base_url=llm_cfg["base_url"],
                model=llm_cfg["model"],
                api_key=llm_cfg["api_key"],
            )

        owner = overrides.pop("owner", project.owner)

        return cls(
            llm=llm,
            persist_path=project.graph_path,
            workspace_path=project.workspace_root,
            owner=owner,
            structural_parser=StructuralParser(),
            **overrides,
        )

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = ".env",
        owner: str = "",
        persist_path: str | Path = "data/graph",
    ) -> Acervo:
        """Create an Acervo instance from environment variables.

        Tries .acervo/ discovery first. Falls back to env vars if not found.
        """
        from acervo.project import find_project

        if env_file:
            _load_dotenv(Path(env_file))

        # Try .acervo/ discovery first
        project = find_project()
        if project:
            return cls.from_project(project.acervo_dir.parent, owner=owner)

        # Fallback to env vars (legacy behavior)
        from acervo.openai_client import OpenAIClient

        base_url = os.environ.get("ACERVO_LIGHT_MODEL_URL", "http://localhost:1234/v1")
        model = os.environ.get("ACERVO_LIGHT_MODEL", "qwen2.5-3b-instruct")
        api_key = os.environ.get("ACERVO_LIGHT_API_KEY", "lm-studio")

        llm = OpenAIClient(base_url=base_url, model=model, api_key=api_key)
        return cls(llm=llm, owner=owner, persist_path=persist_path)

    # ── Public properties ──

    @property
    def graph(self) -> TopicGraph:
        return self._graph

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def llm(self) -> LLMClient:
        return self._llm

    @property
    def vector_store(self) -> VectorStore | None:
        return self._vector_store

    @property
    def workspace_path(self) -> Path | None:
        return self._workspace_path

    @property
    def topic_detector(self) -> TopicDetector:
        return self._topic_detector

    @property
    def context_index(self) -> ContextIndex:
        return self._context_index

    @property
    def metrics(self) -> SessionMetrics:
        return self._metrics

    @property
    def reindexer(self) -> Reindexer | None:
        return self._reindexer

    # ── Public query methods (for plugin consumers) ──

    def lookup_node(self, name: str) -> dict | None:
        """Find a node by name/label. Returns the node dict or None."""
        node_id = _make_id(name)
        node = self._graph.get_node(node_id)
        if node:
            return node
        # Fuzzy match: find a node whose label contains the name
        name_lower = name.lower()
        for n in self._graph.get_all_nodes():
            if name_lower in n.get("label", "").lower():
                return n
        return None

    def get_related_nodes(self, name: str, max_depth: int = 1) -> list[dict]:
        """Get nodes related to the given node name.

        Returns list of neighbor node dicts with their relations.
        """
        node = self.lookup_node(name)
        if not node:
            return []
        nid = node.get("id", "")
        neighbors = self._graph.get_neighbors(nid, max_count=5 * max_depth)
        return [nbr for nbr, _weight in neighbors]

    def get_node_context(self, name: str) -> str:
        """Get rendered text representation of a node and its context.

        Returns the same format as the synthesizer's _render_node.
        """
        from acervo.synthesizer import _render_node

        node = self.lookup_node(name)
        if not node:
            return ""
        return _render_node(node, self._graph)

    def search_nodes(self, query: str, kinds: list[str] | None = None) -> list[dict]:
        """Search nodes by label text, optionally filtered by kind.

        Args:
            query: Text to match against node labels.
            kinds: Filter by node kinds (e.g. ["entity", "file"]).
        """
        query_lower = query.lower()
        results: list[dict] = []
        for node in self._graph.get_all_nodes():
            if kinds and node.get("kind", "entity") not in kinds:
                continue
            label = node.get("label", "").lower()
            if query_lower in label or label in query_lower:
                results.append(node)
        return results

    def get_graph_stats(self) -> dict:
        """Return graph statistics for status displays."""
        graph = self._graph
        kinds: dict[str, int] = {}
        for node in graph.get_all_nodes():
            kind = node.get("kind", "entity")
            kinds[kind] = kinds.get(kind, 0) + 1

        stats = {
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "nodes_by_kind": kinds,
            "hot_nodes": len(graph.get_nodes_by_status("hot")),
            "warm_nodes": len(graph.get_nodes_by_status("warm")),
            "stale_files": len(graph.get_stale_files()),
        }
        if self._metrics:
            stats["metrics"] = {
                "avg_total_tokens": round(self._metrics.avg_total_tokens, 1),
                "context_hit_rate": round(self._metrics.context_hit_rate, 3),
                "graph_growth_rate": round(self._metrics.graph_growth_rate, 2),
                "fact_density": round(self._metrics.fact_density, 2),
                "total_facts_added": self._metrics.total_facts_added,
                "total_facts_deduped": self._metrics.total_facts_deduped,
            }
        return stats

    def find_user_identity(self) -> str | None:
        """Search graph for facts that reveal the user's name."""
        from acervo.synthesizer import _find_user_identity
        return _find_user_identity(self._graph)

    async def ingest_file(self, file_path: str | Path) -> dict:
        """Ingest a single file through the facade.

        Returns dict with: file, entities, facts, relations, symbols, skipped.
        """
        from acervo.file_ingestor import FileIngestor
        from acervo.structural_parser import StructuralParser

        ingestor = FileIngestor(
            llm=self._llm,
            graph=self._graph,
            vector_store=self._vector_store,
            structural_parser=StructuralParser(),
        )
        file_path = Path(file_path)
        workspace = self._workspace_path or file_path.parent
        result = await ingestor.ingest(file_path, workspace)
        return {
            "file": result.file,
            "entities": result.entities,
            "facts": result.facts,
            "relations": result.relations,
            "symbols": result.symbols,
            "skipped": result.skipped,
        }

    async def ingest_all(self, extensions: set[str] | None = None) -> list[dict]:
        """Ingest all supported files in workspace.

        Returns list of dicts, each with: file, entities, facts, relations, symbols, skipped.
        """
        from acervo.file_ingestor import FileIngestor
        from acervo.structural_parser import StructuralParser

        if not self._workspace_path:
            return []

        ingestor = FileIngestor(
            llm=self._llm,
            graph=self._graph,
            vector_store=self._vector_store,
            structural_parser=StructuralParser(),
        )
        results = await ingestor.ingest_all(self._workspace_path, extensions=extensions)
        return [
            {
                "file": r.file,
                "entities": r.entities,
                "facts": r.facts,
                "relations": r.relations,
                "symbols": r.symbols,
                "skipped": r.skipped,
            }
            for r in results
        ]

    # ── prepare/process API ──

    async def prepare(
        self,
        user_text: str,
        history: list[dict],
    ) -> PrepareResult:
        """Pre-LLM processing: 3-stage pipeline.

        S1 — Topic Detection: detect topic via 3B LLM, activate mentioned nodes
        S2 — Gather: collect graph nodes, file contents, vector search → ranked chunks
        S3 — Context Assembly: budget-aware chunk selection (deterministic, no LLM)

        Args:
            user_text: the user's current message
            history: full conversation history as list of {"role", "content"} dicts

        Returns:
            PrepareResult with context_stack ready for LLM
        """
        from acervo.token_counter import count_tokens

        # ── S1: Topic Detection ──
        self._graph.cycle_status()

        detection = await self._topic_detector.detect(user_text)
        if detection.verdict in (TopicVerdict.CHANGED, TopicVerdict.SUBTOPIC):
            new_label = await self._topic_detector.extract_topic_label(user_text)
            self._topic_detector.current_topic = new_label

        current_topic = self._topic_detector.current_topic
        _stages: list[str] = []

        # Activate mentioned nodes
        self._activate_mentioned_nodes(user_text)

        # Keep current topic node hot — or create placeholder if new
        placeholder_created = False
        if current_topic != "none":
            topic_id = _make_id(current_topic)
            existing = self._graph.get_node(topic_id)
            if existing:
                self._graph.set_node_status(topic_id, "hot")
            else:
                # New topic — create placeholder node so next turn can find it
                self._graph.upsert_entities(
                    [(current_topic, "topic")],
                    owner=self._owner or None,
                    source="conversation",
                    confidence=0.5,
                )
                node = self._graph.get_node(topic_id)
                if node:
                    node["status"] = "placeholder"
                    self._graph.set_node_status(topic_id, "hot")
                    placeholder_created = True
                    log.info("[acervo] S1 — Created placeholder node: %s", current_topic)

                    # Find related nodes via vector search
                    if self._vector_store:
                        try:
                            related = await self._vector_store.search(
                                current_topic, n_results=5,
                            )
                            for hit in related:
                                score = hit.get("score", 0)
                                if score >= 0.7:
                                    target_id = hit.get("node_id", "")
                                    if target_id and target_id != topic_id:
                                        self._graph.upsert_entities(
                                            [], relations=[(current_topic, target_id, "related_to")],
                                        )
                        except Exception as e:
                            log.warning("Vector search for related nodes failed: %s", e)

        # S1 stage log (after placeholder creation so we can report it)
        s1_msg = f"Topic: {current_topic} (verdict={detection.verdict.name}, L{detection.level}, conf={detection.confidence:.2f})"
        if placeholder_created:
            s1_msg += " [NEW — placeholder created]"
        _stages.append(s1_msg)
        log.info("[acervo] S1 — %s", _stages[-1])

        # ── S2: Gather (always, no planner) ──
        gathered = GatheredInfo()
        gathered.nodes = self._gather_graph_nodes(current_topic)

        # Gather linked file contents (line-precise when symbols available)
        if self._workspace_path:
            for node in gathered.nodes:
                for file_path in node.get("files", []):
                    if file_path in gathered.file_contents:
                        continue
                    full_path = self._workspace_path / file_path
                    if not full_path.is_file():
                        continue

                    symbols = self._graph.get_file_symbols(file_path)
                    if symbols:
                        file_node = self._graph.get_node(_make_id(file_path))
                        is_fresh = True
                        if file_node:
                            stored_hash = file_node.get("attributes", {}).get("content_hash", "")
                            try:
                                current_hash = hash_file(full_path)
                                if current_hash != stored_hash:
                                    is_fresh = False
                                    self._graph.mark_file_stale(file_path)
                                    log.info("Stale file detected: %s", file_path)
                            except Exception:
                                is_fresh = False

                        if is_fresh:
                            for sym in symbols:
                                content = self._graph.get_symbol_content(sym["id"], self._workspace_path)
                                if content:
                                    attrs = sym.get("attributes", {})
                                    key = f"{file_path}:{sym.get('label', '')} (L{attrs.get('start_line')}-{attrs.get('end_line')})"
                                    gathered.file_contents[key] = content
                        else:
                            try:
                                gathered.file_contents[file_path] = full_path.read_text(encoding="utf-8")
                            except Exception as e:
                                log.warning("Failed to read stale file %s: %s", file_path, e)
                    else:
                        try:
                            gathered.file_contents[file_path] = full_path.read_text(encoding="utf-8")
                        except Exception as e:
                            log.warning("Failed to read linked file %s: %s", file_path, e)

        # Vector search for semantic matches
        if self._vector_store:
            try:
                vector_hits = await self._vector_store.search(user_text, n_results=5)
                gathered.vector_results = vector_hits
                for hit in vector_hits:
                    node_id = hit.get("node_id", "")
                    if node_id and self._graph.get_node(node_id):
                        self._graph.set_node_status(node_id, "hot")
            except Exception as e:
                log.warning("Vector search failed: %s", e)

        # Convert gathered info → ranked chunks, tracking verification status
        # Verified: source=world/indexed/web, or has verified=True attribute
        # Unverified: source=conversation/user_assertion, placeholder/enriched nodes
        chunks: list[RankedChunk] = []
        verified_labels: set[str] = set()  # node labels with verified content
        for node in gathered.nodes:
            # Skip placeholder nodes — nothing useful to inject
            if node.get("status") == "placeholder":
                continue
            is_hot = node.get("_hot", True)
            base_score = 1.0 if is_hot else 0.8
            is_verified = (
                node.get("source") in ("world", "web")
                or node.get("attributes", {}).get("verified", False)
                or node.get("kind") in ("file", "symbol")
            )
            if is_verified:
                verified_labels.add(node.get("label", ""))
            for fact in node.get("facts", []):
                fact_text = fact.get("fact", "") if isinstance(fact, dict) else str(fact)
                text = f"**{node.get('label', '')}**: {fact_text}"
                # Use source prefix to distinguish verified vs unverified
                src = "verified_fact" if is_verified else "conversation_fact"
                chunks.append(RankedChunk(
                    text=text, score=base_score, source=src,
                    label=node.get("label", ""), tokens=count_tokens(text),
                ))
            for rel in node.get("_relations", []):
                text = f"**{node.get('label', '')}** → {rel}"
                src = "verified_relation" if is_verified else "conversation_relation"
                chunks.append(RankedChunk(
                    text=text, score=base_score * 0.9, source=src,
                    label=node.get("label", ""), tokens=count_tokens(text),
                ))
        for hit in gathered.vector_results:
            text = hit.get("text", "")
            chunks.append(RankedChunk(
                text=text, score=hit.get("score", 0.5), source="vector",
                label=hit.get("node_id", ""), tokens=count_tokens(text),
            ))
        for path, content in gathered.file_contents.items():
            preview = content[:2000]
            text = f"**{path}:**\n{preview}"
            chunks.append(RankedChunk(
                text=text, score=0.7, source="verified_file",
                label=path, tokens=count_tokens(text),
            ))

        _stages.append(
            f"Gathered: {len(gathered.nodes)} nodes, "
            f"{len(gathered.vector_results)} vector, "
            f"{len(gathered.file_contents)} files → {len(chunks)} chunks"
        )
        log.info("[acervo] S2 — %s", _stages[-1])

        # ── S3: Context Assembly (deterministic, no LLM) ──
        # Split chunks into verified and conversation (unverified)
        warm_budget = 1500
        selected, warm_used = select_chunks_by_budget(chunks, warm_budget)

        verified_chunks = [c for c in selected if c.source.startswith("verified")]
        conversation_chunks = [c for c in selected if not c.source.startswith("verified")]

        # Build warm override with verification markers
        parts: list[str] = []
        if verified_chunks:
            parts.append(format_chunks_as_context(verified_chunks))
        if conversation_chunks:
            if parts:
                parts.append("")  # blank line separator
            parts.append("[CONVERSATION CONTEXT (unverified)]\n"
                         + format_chunks_as_context(conversation_chunks)
                         + "\n(This information came from previous conversations "
                         "and has not been verified against primary sources.)")

        warm_override = "\n".join(parts) if parts else ""
        warm_source = "graph" if warm_override else ""

        context_stack, hot_tk, warm_tk, total_tk = self._context_index.build_context_stack(
            history, current_topic,
            warm_override=warm_override,
            warm_source=warm_source,
        )

        has_context = warm_tk > 0
        needs_tool = not has_context
        _stages.append(
            f"Context: {len(selected)}/{len(chunks)} chunks, "
            f"warm={warm_tk}tk, hot={hot_tk}tk, total={total_tk}tk"
        )
        log.info("[acervo] S3 — %s", _stages[-1])

        # Record prepare-phase metrics
        nodes_activated = len(self._graph.get_nodes_by_status("hot"))
        self._pending_metric = dict(
            warm_tokens=warm_tk,
            hot_tokens=hot_tk,
            total_context_tokens=total_tk,
            node_count=self._graph.node_count,
            edge_count=self._graph.edge_count,
            nodes_activated=nodes_activated,
            topic=current_topic,
            plan_tool="GATHER",
            context_hit=has_context,
        )

        _debug = {
            "s1_detection": {
                "verdict": detection.verdict.name,
                "level": detection.level,
                "confidence": detection.confidence,
                "current_topic": current_topic,
                "detected_topic": detection.detected_topic,
                "keyword": detection.keyword,
                "similarity": detection.similarity,
                "detail": detection.detail,
                "placeholder_created": placeholder_created,
            },
            "s2_gathered": {
                "nodes": [
                    {
                        "id": n.get("id", ""),
                        "label": n.get("label", ""),
                        "type": n.get("type", ""),
                        "facts": [
                            f.get("fact", "") if isinstance(f, dict) else str(f)
                            for f in n.get("facts", [])
                        ],
                    }
                    for n in gathered.nodes[:20]
                ],
                "files": list(gathered.file_contents.keys())[:20],
                "vector_hits": [
                    {"node_id": v.get("node_id", ""), "score": round(v.get("score", 0), 3)}
                    for v in gathered.vector_results[:10]
                ],
                "chunks_total": len(chunks),
                "chunks_selected": len(selected),
                "chunks": [
                    {
                        "text": c.text[:100],
                        "score": round(c.score, 3),
                        "source": c.source,
                        "label": c.label,
                        "tokens": c.tokens,
                    }
                    for c in selected[:20]
                ],
            },
            "s3_context": {
                "warm_tokens": warm_tk,
                "hot_tokens": hot_tk,
                "total_tokens": total_tk,
                "warm_budget": warm_budget,
                "chunks_used": len(selected),
                "verified_chunks": len(verified_chunks),
                "conversation_chunks": len(conversation_chunks),
                "warm_source": warm_source or "none",
                "warm_content_preview": warm_override[:500] if warm_override else "",
                "has_context": has_context,
            },
        }

        return PrepareResult(
            context_stack=context_stack,
            topic=current_topic,
            hot_tokens=hot_tk,
            warm_tokens=warm_tk,
            total_tokens=total_tk,
            warm_content=warm_override,
            has_context=has_context,
            needs_tool=needs_tool,
            stages=_stages,
            debug=_debug,
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
        _SKIP_PHRASES = ("i don't have information", "i don't have data", "i can search",
                         "would you like me to search", "shall i look",
                         # Legacy Spanish phrases
                         "no tengo información", "no tengo datos", "puedo buscar",
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
            self._record_turn_metric(result)
            return result

        # Extract from conversation (only user facts stored)
        result = await self.commit(user_text, assistant_text)

        # If web results were used, extract and persist web facts separately
        has_tool_results = bool(web_results)
        if web_results:
            await self._persist_web_facts(user_text, web_results)

        # ── Conversational indexing: enrich nodes post-extraction ──
        await self._enrich_nodes_post_llm(result, has_tool_results)

        # Finalize turn metrics
        self._record_turn_metric(result)

        # Deferred reindex — re-parse any files marked stale during Gather
        if self._reindexer:
            try:
                reindexed = await self._reindexer.reindex_stale()
                if reindexed:
                    log.info("Post-turn reindex: %d files updated", len(reindexed))
            except Exception as e:
                log.warning("Post-turn reindex failed: %s", e)

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

    async def _enrich_nodes_post_llm(
        self,
        result: ExtractionResult,
        has_tool_results: bool,
    ) -> None:
        """Conversational indexing: update nodes after extraction.

        - Promote placeholder → enriched for entities that gained facts
        - Set source/verified flags based on how the knowledge was obtained
        - Generate embeddings for new/updated entities
        """
        if not result.entities:
            return

        source = "tool_result" if has_tool_results else "conversation"
        verified = has_tool_results  # tool results are considered verified

        for entity in result.entities:
            node_id = _make_id(entity.name)
            node = self._graph.get_node(node_id)
            if not node:
                continue

            # Update source and verified status
            node["source"] = source if node.get("source") != "world" else "world"
            node.setdefault("attributes", {})["verified"] = verified or node.get("source") == "world"

            # Promote placeholder → enriched if we have facts
            entity_facts = [f for f in result.facts if f.entity == entity.name]
            if node.get("status") == "placeholder" and entity_facts:
                node["status"] = "enriched"
                log.info("[acervo] Node enriched: %s (%d facts)", entity.name, len(entity_facts))
            elif node.get("status") == "placeholder" and not entity_facts:
                # Keep placeholder but mark as enriched if it has any content
                if node.get("facts"):
                    node["status"] = "enriched"

            # Generate embedding for the entity
            if self._embedder and self._vector_store:
                try:
                    # Build a summary for embedding
                    facts_text = " ".join(
                        f.get("fact", "") if isinstance(f, dict) else str(f)
                        for f in node.get("facts", [])
                    )
                    embed_text = f"{entity.name}: {facts_text}" if facts_text else entity.name
                    embedding = await self._embedder.embed(embed_text)
                    await self._vector_store.index_facts(
                        node_id, entity.name,
                        [embed_text],
                    )
                    log.info("[acervo] Embedded: %s", entity.name)
                except Exception as e:
                    log.warning("Embedding failed for %s: %s", entity.name, e)

        self._graph.save()

    def materialize(self, query: str, token_budget: int = 800) -> str:
        """Build a context string from relevant graph nodes."""
        return synthesize(self._graph, query)

    def cycle_status(self) -> None:
        """Demote node statuses: hot -> warm -> cold."""
        self._graph.cycle_status()

    # ── Internal helpers ──

    def _build_facts_summary(self, entity_node: dict | None) -> str:
        """Build a facts summary string for the planner."""
        if not entity_node:
            return ""
        parts = []
        facts = entity_node.get("facts", [])
        if facts:
            parts.append("Facts: " + ", ".join(f.get("fact", "") for f in facts))
        edges = self._graph.get_edges_for(entity_node.get("id", ""))
        if edges:
            rel_labels = [e.get("relation", "") for e in edges[:5]]
            parts.append(f"Relations: {', '.join(rel_labels)}")
        if not parts:
            parts.append(
                f"Node exists (mentioned {entity_node.get('session_count', 0)} times) "
                "but no verified facts"
            )
        return ". ".join(parts)[:500]

    def _gather_graph_nodes(self, entity_hint: str) -> list[dict]:
        """Gather hot nodes and their neighbors with relations for context building.

        Each node gets a ``_hot`` flag: True for directly-activated nodes,
        False for neighbor-expanded nodes (used for scoring in ranked chunks).
        """
        nodes: list[dict] = []
        seen_ids: set[str] = set()

        # Collect hot nodes
        for node in self._graph.get_nodes_by_status("hot"):
            nid = node.get("id", "")
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            edges = self._graph.get_edges_for(nid)
            relations = []
            for e in edges[:5]:
                other_id = e["target"] if e["source"] == nid else e["source"]
                other = self._graph.get_node(other_id)
                if other:
                    rel = e.get("relation", "related_to")
                    relations.append(f"{rel}: {other.get('label', other_id)}")
            node_copy = dict(node)
            node_copy["_relations"] = relations
            node_copy["_hot"] = True
            nodes.append(node_copy)

        # Add neighbors of hot nodes (1-level)
        for node in list(nodes):
            nid = node.get("id", "")
            neighbors = self._graph.get_neighbors(nid, max_count=3)
            for nbr, weight in neighbors:
                nbr_id = nbr.get("id", "")
                if nbr_id in seen_ids:
                    continue
                seen_ids.add(nbr_id)
                if nbr.get("facts"):
                    nbr_copy = dict(nbr)
                    nbr_copy["_relations"] = []
                    nbr_copy["_hot"] = False
                    nodes.append(nbr_copy)

        return nodes

    def _record_turn_metric(self, result: ExtractionResult) -> None:
        """Combine pending prepare-phase metrics with extraction results."""
        pending = getattr(self, "_pending_metric", {})
        dedup_log = self._graph.dedup_log
        user_facts = [f for f in result.facts if f.speaker == "user"]
        # Override graph counts with post-extraction values
        pending["node_count"] = self._graph.node_count
        pending["edge_count"] = self._graph.edge_count
        self._metrics.record_turn(
            **pending,
            entities_extracted=len(result.entities),
            facts_added=len(user_facts),
            facts_deduped=len(dedup_log),
        )
        self._pending_metric = {}

    def _activate_mentioned_nodes(self, user_text: str) -> None:
        """Set graph nodes to 'hot' if the user message mentions them."""
        msg_lower = user_text.lower()
        msg_words = set(msg_lower.split())
        for node in self._graph.get_all_nodes():
            # Skip symbol/file/section nodes — only activate entity nodes
            if node.get("kind", "entity") != "entity":
                continue
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
