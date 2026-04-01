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

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from acervo.context_builder import (
    GatheredInfo,
    RankedChunk,
    select_chunks_by_budget,
    format_chunks_compact,
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
from acervo.specificity import classify_specificity
from acervo.reindexer import Reindexer, hash_file
from acervo.s1_unified import S1Unified, build_graph_summary, generate_topic_hint
from acervo.synthesizer import synthesize
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
        warm_token_budget: int = 400,
        compaction_trigger_tokens: int = 2000,
        vector_store: VectorStore | None = None,
        workspace_path: str | Path | None = None,
        prompts: dict[str, str] | None = None,
        structural_parser: object | None = None,
        role_llms: dict[str, LLMClient] | None = None,
        description: str = "",
    ) -> None:
        self._project_description = description
        self._graph = TopicGraph(Path(persist_path))
        self._llm = llm
        p = prompts or {}
        roles = role_llms or {}

        # Per-role LLM: use override if provided, else fall back to default llm
        extractor_llm = roles.get("extractor", llm)
        summarizer_llm = roles.get("summarizer", llm)

        self._extractor_llm = extractor_llm
        self._s1_unified = S1Unified(extractor_llm, system_prompt=p.get("s1_unified"))
        self._s1_5_prompt = p.get("s1_5_graph_update")
        self._extractor = ConversationExtractor(extractor_llm, prompt_template=p.get("extractor_conversation"))
        self._search_extractor = SearchExtractor(llm, prompt_template=p.get("extractor_search"))
        self._owner = owner
        self._warm_token_budget = warm_token_budget
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
            embedder=embedder, embed_threshold=embed_threshold,
        )
        self._context_index = ContextIndex(
            self._graph, summarizer_llm,
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

        from acervo.openai_client import OpenAIClient

        # Default LLM: client-provided > config.toml > env vars > defaults
        if llm is None:
            llm_cfg = project.llm_config()
            llm = OpenAIClient(
                base_url=llm_cfg["base_url"],
                model=llm_cfg["model"],
                api_key=llm_cfg["api_key"],
            )

        # Per-role LLM overrides from [acervo.models.*]
        default_model = project.config.model
        models_cfg = project.config.models
        role_llms: dict[str, LLMClient] = {}
        for role in ("extractor", "summarizer"):
            role_model = models_cfg.resolve_for_role(role, default_model)
            if role_model.name != default_model.name or role_model.url != default_model.url:
                role_llms[role] = OpenAIClient(
                    base_url=role_model.url,
                    model=role_model.name,
                    api_key=role_model.api_key or default_model.api_key,
                )
                log.info("Model override for %s: %s @ %s", role, role_model.name, role_model.url)

        owner = overrides.pop("owner", project.owner)

        # Load prompts from .acervo/prompts/*.txt, merge with caller overrides
        config_prompts = project.config.prompts.load_prompts(project.workspace_root)
        caller_prompts = overrides.pop("prompts", None) or {}
        merged_prompts = {**config_prompts, **caller_prompts} if (config_prompts or caller_prompts) else None

        return cls(
            llm=llm,
            persist_path=project.graph_path,
            workspace_path=project.workspace_root,
            owner=owner,
            structural_parser=StructuralParser(),
            prompts=merged_prompts,
            role_llms=role_llms if role_llms else None,
            description=project.config.description,
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
        model = os.environ.get("ACERVO_LIGHT_MODEL", "qwen3.5-9b")
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

    async def sync_vector_store(self) -> int:
        """Index existing graph facts into vector store if missing.

        Call after init when enabling vector store on a graph that already has data.
        Returns number of nodes indexed.
        """
        if not self._vector_store:
            return 0
        indexed = 0
        for node in self._graph.get_all_nodes():
            facts = node.get("facts", [])
            if not facts:
                continue
            fact_texts = [
                f.get("fact", "") if isinstance(f, dict) else str(f)
                for f in facts
            ]
            fact_texts = [t for t in fact_texts if t.strip()]
            if fact_texts:
                nid = node.get("id", "")
                label = node.get("label", "")
                try:
                    await self._vector_store.index_facts(nid, label, fact_texts)
                    indexed += 1
                except Exception as e:
                    log.warning("Sync failed for %s: %s", label, e)
        if indexed:
            log.info("[acervo] Synced %d nodes to vector store", indexed)
        return indexed

    async def index_document(self, file_path: Path) -> dict:
        """Index a document through the full pipeline (parse + enrich + chunks + graph).

        Args:
            file_path: Absolute path to the .md file.

        Returns:
            {"document_id": str, "chunk_count": int, "chunk_ids": list, "node_id": str}
        """
        from acervo.structural_parser import StructuralParser
        from acervo.semantic_enricher import SemanticEnricher
        from acervo.graph import make_symbol_id

        workspace = file_path.parent
        parser = StructuralParser()
        structure = parser.parse(file_path, workspace)

        # Phase 1: structural nodes
        self._graph.upsert_file_structure(
            structure.file_path, structure.language,
            structure.units, structure.content_hash,
        )
        file_id = _make_id(structure.file_path)

        # Phase 2: semantic enrichment (chunks + embeddings)
        all_chunk_ids: list[str] = []
        chunk_count = 0

        if self._embedder:
            enricher = SemanticEnricher(llm=self._llm, embedder=self._embedder)
            content = file_path.read_text(encoding="utf-8")
            enrichment = await enricher.enrich_file(structure, content)

            if enrichment.chunks and self._vector_store:
                await self._vector_store.index_file_chunks(
                    file_path=enrichment.chunks[0].file_path,
                    chunks=[c.content for c in enrichment.chunks],
                    chunk_ids=[c.chunk_id for c in enrichment.chunks],
                    embeddings=[c.embedding for c in enrichment.chunks],
                    extra_metadata={"file_id": file_id},
                )
                all_chunk_ids = [c.chunk_id for c in enrichment.chunks]
                chunk_count = len(enrichment.chunks)

                # Link to file node
                self._graph.link_chunks(file_id, all_chunk_ids)

                # Link to section nodes
                for unit in structure.units:
                    node_id = make_symbol_id(structure.file_path, unit.name, unit.parent)
                    matching = [
                        c.chunk_id for c in enrichment.chunks
                        if c.line_start >= unit.start_line and c.line_end <= unit.end_line
                    ]
                    if matching:
                        self._graph.link_chunks(node_id, matching)

        self._graph.save()
        return {
            "document_id": file_id,
            "chunk_count": chunk_count,
            "chunk_ids": all_chunk_ids,
            "node_id": file_id,
        }

    async def delete_document(self, document_id: str) -> dict:
        """Delete a document, its chunks from ChromaDB, and its graph nodes.

        Returns:
            {"deleted": str, "chunks_removed": int}
        """
        node = self._graph.get_node(document_id)
        if not node:
            return {"deleted": document_id, "chunks_removed": 0}

        # Collect all chunk_ids (file node + children)
        chunk_ids = list(node.get("chunk_ids", []))
        prefix = document_id + "__"
        child_ids = [nid for nid in self._graph._nodes if nid.startswith(prefix)]
        for cid in child_ids:
            child = self._graph.get_node(cid)
            if child:
                chunk_ids.extend(child.get("chunk_ids", []))

        # Remove from ChromaDB
        if chunk_ids and self._vector_store and hasattr(self._vector_store, "remove_by_chunk_ids"):
            self._vector_store.remove_by_chunk_ids(chunk_ids)

        # Remove graph nodes (children + file node)
        self._graph._remove_file_children(document_id)
        self._graph.remove_node(document_id)
        self._graph.save()

        return {"deleted": document_id, "chunks_removed": len(chunk_ids)}

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
            "entity_nodes": len([n for n in graph.get_all_nodes() if n.get("kind", "entity") == "entity"]),
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

        S1 — S1 Unified: L1/L2 hints + LLM topic classification + entity extraction (9b)
        S2 — Gather: collect graph nodes, file contents, vector search → ranked chunks
        S3 — Context Assembly: budget-aware chunk selection (deterministic, no LLM)

        Args:
            user_text: the user's current message
            history: full conversation history as list of {"role", "content"} dicts

        Returns:
            PrepareResult with context_stack ready for LLM
        """
        from acervo.token_counter import count_tokens

        _t0 = time.monotonic()

        # ── S1: Topic Detection + Extraction (S1 Unified) ──
        _stages: list[str] = []

        # Pre-compute user text embedding once (reused by topic L2 + vector search)
        user_embedding: list[float] | None = None
        if self._embedder:
            try:
                user_embedding = await self._embedder.embed(user_text)
            except Exception as e:
                log.warning("User text embedding failed: %s", e)

        log.info("prepare: user_text=%d chars", len(user_text))

        # L1/L2 as hints (no L3 LLM call)
        detection = await self._topic_detector.detect_hints(user_text, pre_embedding=user_embedding)

        # Build graph summary for S1 prompt
        all_nodes = self._graph.get_all_nodes()
        existing_nodes_summary = build_graph_summary(
            [dict(n) for n in all_nodes], user_text,
        )

        # Generate topic hint from L1/L2 results
        current_topic = self._topic_detector.current_topic
        topic_hint = generate_topic_hint(
            l1_keyword=detection.keyword,
            l2_similarity=detection.similarity,
            l2_verdict=detection.verdict if detection.level == 2 else None,
            current_topic=current_topic,
        )

        # Get previous assistant message for S1 context
        prev_assistant_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                prev_assistant_msg = msg.get("content", "")
                break

        # S1 Unified: topic classification + extraction in one LLM call (ALWAYS runs)
        s1_result = await self._s1_unified.run(
            user_msg=user_text,
            prev_assistant_msg=prev_assistant_msg,
            current_topic=current_topic,
            topic_hint=topic_hint,
            existing_nodes_summary=existing_nodes_summary,
        )

        # Apply S1's topic + intent decision
        s1_topic = s1_result.topic
        s1_intent = s1_result.intent

        # Keyword fallback for intent: detect overview patterns the model may miss
        if s1_intent != "overview":
            _overview = user_text.lower()
            if any(p in _overview for p in (
                "cuantos", "cuántos", "how many", "what do we have",
                "que tenemos", "qué tenemos", "list all", "show all",
                "resumen", "summary", "overview", "inventory",
                "estructura", "structure",
            )):
                s1_intent = "overview"

        log.info(
            "[acervo] S1 detail: action=%s label=%s, intent=%s, %dE %dR %dF",
            s1_topic.action, s1_topic.label, s1_intent,
            len(s1_result.extraction.entities),
            len(s1_result.extraction.relations),
            len(s1_result.extraction.facts),
        )
        if s1_topic.action in ("changed", "subtopic") and s1_topic.label:
            self._topic_detector.current_topic = s1_topic.label
        elif s1_topic.action == "changed" and not s1_topic.label:
            # S1 said changed but didn't provide a label — use keyword fallback
            if detection.keyword:
                self._topic_detector.current_topic = detection.keyword
        current_topic = self._topic_detector.current_topic

        # Build the set of active node IDs for this turn (ephemeral, not persisted)
        active_node_ids = self._find_active_node_ids(user_text, current_topic, intent=s1_intent)
        self._last_active_node_ids = active_node_ids  # used by process() → _mark_touched_nodes()

        # Persist S1 extraction results SYNC (so S2 Gather sees the new nodes)
        s1_extraction = s1_result.extraction
        self._last_s1_extraction = s1_extraction  # saved for S1.5 in process()
        if s1_extraction.entities:
            for entity in s1_extraction.entities:
                entity_layer = self._resolve_entity_layer(entity, user_text)
                entity_source = "world" if entity_layer == Layer.UNIVERSAL else "user_assertion"

                entity_facts = [
                    (f.entity, f.fact, f.source)
                    for f in s1_extraction.facts
                    if f.speaker == "user" and f.entity.lower() == entity.name.lower()
                ]

                self._graph.upsert_entities(
                    [(entity.name, entity.type)],
                    None,
                    entity_facts if entity_facts else None,
                    layer=entity_layer,
                    source=entity_source,
                    owner=self._owner or None,
                )

                # Merge extracted attributes into graph node
                nid = _make_id(entity.name)
                node = self._graph.get_node(nid)
                if node:
                    if entity.attributes:
                        # Filter out internal attributes
                        attrs = {k: v for k, v in entity.attributes.items() if not k.startswith("_")}
                        if attrs:
                            node.setdefault("attributes", {}).update(attrs)
                    # Stamp topic membership for topic-based layer management
                    if current_topic != "none":
                        node["_topic_id"] = _make_id(current_topic)

            # Add relations
            if s1_extraction.relations:
                relations = [(r.source, r.target, r.relation) for r in s1_extraction.relations]
                self._graph.upsert_entities(
                    [],
                    relations,
                    layer=Layer.PERSONAL,
                    source="user_assertion",
                    owner=self._owner or None,
                )

            # Enrich nodes post-extraction (embeddings)
            await self._enrich_nodes_post_llm(s1_extraction, has_tool_results=False)

        # Add current topic node to active set
        if current_topic != "none":
            topic_id = _make_id(current_topic)
            if self._graph.get_node(topic_id):
                active_node_ids.add(topic_id)

        s1_msg = (
            f"S1 Unified: topic={current_topic} (action={s1_topic.action}), "
            f"hint=L{detection.level}, "
            f"extracted: {len(s1_extraction.entities)}E {len(s1_extraction.relations)}R {len(s1_extraction.facts)}F"
        )
        _stages.append(s1_msg)
        log.info("[acervo] S1 — %s", s1_msg)

        # ── S2: Gather (always, no planner) ──
        gathered = GatheredInfo()
        gathered.nodes = self._gather_graph_nodes(active_node_ids)

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

        # Node-scoped chunk retrieval: search within chunks linked to activated nodes
        node_chunk_ids: list[str] = []
        for node in gathered.nodes:
            node_chunk_ids.extend(node.get("chunk_ids", []))

        # Specificity classifier: only fetch chunks for detail-oriented questions
        query_specificity = classify_specificity(user_text)
        log.info(
            "[acervo] S2 gather: %d active nodes, %d chunk_ids, specificity=%s",
            len(gathered.nodes), len(node_chunk_ids), query_specificity,
        )

        if node_chunk_ids and self._vector_store and user_embedding and query_specificity == "specific":
            try:
                if hasattr(self._vector_store, "search_by_chunk_ids"):
                    scoped_hits = await self._vector_store.search_by_chunk_ids(
                        node_chunk_ids, user_embedding, n_results=3,
                    )
                    for hit in scoped_hits:
                        hit["score"] = min(hit.get("score", 0.5) + 0.15, 1.0)
                    gathered.vector_results.extend(scoped_hits)
            except Exception as e:
                log.warning("Node-scoped chunk search failed: %s", e)

        # Vector search for semantic matches (topic-scoped boost)
        # Skip for overview intent — vector hits are section-level noise
        if self._vector_store and s1_intent != "overview":
            try:
                if user_embedding:
                    vector_hits = await self._vector_store.search_with_embedding(user_embedding, n_results=5)
                else:
                    vector_hits = await self._vector_store.search(user_text, n_results=5)
                # Boost on-topic results
                current_topic_id = _make_id(current_topic) if current_topic != "none" else None
                for hit in vector_hits:
                    node_id = hit.get("node_id", "")
                    node = self._graph.get_node(node_id) if node_id else None
                    if node:
                        active_node_ids.add(node_id)
                        if current_topic_id and node_id in active_node_ids:
                            hit["score"] = min(hit.get("score", 0.5) + 0.2, 1.0)
                gathered.vector_results = vector_hits
            except Exception as e:
                log.warning("Vector search failed: %s", e)

        # Convert gathered info → ranked chunks, tracking verification status
        # Verified: source=world/indexed/web, or has verified=True attribute
        # Unverified: source=conversation/user_assertion, placeholder/enriched nodes
        chunks: list[RankedChunk] = []
        verified_labels: set[str] = set()  # node labels with verified content
        user_lower = user_text.lower()
        for node in gathered.nodes:
            kind = node.get("kind", "entity")
            has_facts = bool(node.get("facts"))
            has_relations = bool(node.get("_relations"))
            summary = node.get("attributes", {}).get("summary", "")

            # Skip nodes with no useful content at all
            if not has_facts and not has_relations and not summary:
                continue

            is_hot = node.get("_hot", True)
            base_score = 1.0 if is_hot else 0.7
            # Query-relevance boost: entities mentioned in user message rank higher
            label_lower = node.get("label", "").lower()
            if label_lower and label_lower in user_lower:
                base_score = min(base_score + 0.3, 1.0)
            is_verified = (
                node.get("source") in ("world", "web")
                or node.get("attributes", {}).get("verified", False)
                or kind in ("file", "section", "symbol", "folder")
            )
            if is_verified:
                verified_labels.add(node.get("label", ""))

            # Indexed nodes: use summary as a verified chunk
            if kind in ("file", "section", "symbol", "folder") and summary and not has_facts:
                text = f"**{node.get('label', '')}**: {summary}"
                chunks.append(RankedChunk(
                    text=text, score=base_score, source="verified_summary",
                    label=node.get("label", ""), tokens=count_tokens(text),
                ))

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
            source = hit.get("source", "vector")
            chunks.append(RankedChunk(
                text=text, score=hit.get("score", 0.5), source=source,
                label=hit.get("node_id", hit.get("chunk_id", "")), tokens=count_tokens(text),
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
        # Log top chunks for terminal debugging
        for ci, ch in enumerate(sorted(chunks, key=lambda c: c.score, reverse=True)[:5]):
            log.info(
                "[acervo] S2   chunk[%d] score=%.2f src=%s: %s",
                ci, ch.score, ch.source, ch.text[:120],
            )

        # ── S3: Context Assembly (deterministic, no LLM) ──

        # TIER 1: Project overview — always injected if graph has data.
        # This is structural metadata, NOT subject to budget selection.
        project_overview = self._build_project_overview()

        # Initialize chunk lists used across all branches + debug output
        verified_chunks: list = []
        conversation_chunks: list = []

        if s1_intent == "overview":
            # Overview intent: project overview + file/folder summaries only.
            # Skip vector chunks (section-level noise) — they match semantically
            # but are irrelevant for "how many books?" type questions.
            overview_chunks = [
                c for c in chunks
                if c.source.startswith("verified")
            ]
            selected, warm_used = select_chunks_by_budget(overview_chunks, self._warm_token_budget)
            parts: list[str] = []
            if project_overview:
                parts.append(project_overview)
            verified_chunks = selected  # all are verified for overview
            if verified_chunks:
                parts.append(format_chunks_compact(verified_chunks))
            warm_override = "\n".join(parts) if parts else ""
        elif s1_intent == "chat":
            # Chat intent: provide overview for awareness, skip specific chunks
            selected = []
            warm_override = project_overview or ""
        else:
            # Specific intent: normal budget-gated assembly
            selected, warm_used = select_chunks_by_budget(chunks, self._warm_token_budget)

            verified_chunks = [c for c in selected if c.source.startswith("verified")]
            conversation_chunks = [c for c in selected if not c.source.startswith("verified")]

            parts = []
            if project_overview:
                parts.append(project_overview)
            if verified_chunks:
                parts.append(format_chunks_compact(verified_chunks))
            if conversation_chunks:
                if parts:
                    parts.append("")  # blank line separator
                parts.append("[UNVERIFIED]\n"
                             + format_chunks_compact(conversation_chunks))

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
        if warm_override:
            log.info("[acervo] S3 warm preview: %s", warm_override[:200])

        # Record prepare-phase metrics
        self._pending_metric = dict(
            warm_tokens=warm_tk,
            hot_tokens=hot_tk,
            total_context_tokens=total_tk,
            node_count=self._graph.node_count,
            edge_count=self._graph.edge_count,
            nodes_activated=len(active_node_ids),
            topic=current_topic,
            plan_tool="GATHER",
            context_hit=has_context,
        )

        _debug = {
            "s1_detection": {
                "topic_action": s1_topic.action,
                "topic_label": s1_topic.label,
                "current_topic": current_topic,
                "topic_hint": topic_hint,
                "hint_level": detection.level,
                "hint_verdict": detection.verdict.name,
                "hint_keyword": detection.keyword,
                "hint_similarity": detection.similarity,
                "intent": s1_intent,
                "entities_extracted": len(s1_extraction.entities),
                "relations_extracted": len(s1_extraction.relations),
                "facts_extracted": len(s1_extraction.facts),
                "entities": [
                    {"name": e.name, "type": e.type, "layer": e.layer}
                    for e in s1_extraction.entities[:10]
                ],
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
                        "text": c.text[:200],
                        "score": round(c.score, 3),
                        "source": c.source,
                        "label": c.label,
                        "tokens": c.tokens,
                    }
                    for c in selected[:20]
                ],
                "all_chunks": [
                    {
                        "text": c.text[:150],
                        "score": round(c.score, 3),
                        "source": c.source,
                        "label": c.label,
                        "tokens": c.tokens,
                    }
                    for c in sorted(chunks, key=lambda x: x.score, reverse=True)[:30]
                ],
            },
            "s3_context": {
                "warm_tokens": warm_tk,
                "hot_tokens": hot_tk,
                "total_tokens": total_tk,
                "warm_budget": self._warm_token_budget,
                "chunks_used": len(selected),
                "verified_chunks": len(verified_chunks),
                "conversation_chunks": len(conversation_chunks),
                "warm_source": warm_source or "none",
                "warm_content_preview": warm_override[:500] if warm_override else "",
                "has_context": has_context,
            },
            "chunks": {
                "documents_with_chunks_activated": len([n for n in gathered.nodes if n.get("chunk_ids")]),
                "chunks_retrieved": len([h for h in gathered.vector_results if h.get("source") == "node_scoped_chunk"]),
                "chunks_total_on_activated_nodes": len(node_chunk_ids),
                "retrieval_scope": "node_scoped" if node_chunk_ids and query_specificity == "specific" else "global",
                "query_specificity": query_specificity,
            },
        }

        _elapsed_ms = int((time.monotonic() - _t0) * 1000)
        log.info(
            "prepare done: topic=%s tokens=%d (warm=%d hot=%d) %dms",
            current_topic, total_tk, warm_tk, hot_tk, _elapsed_ms,
        )

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
        """Post-LLM processing: S1.5 graph curation + assistant extraction.

        Runs S1.5 Graph Update asynchronously to:
        1. Curate graph (merge duplicates, fix types, discard garbage)
        2. Extract entities/facts from the assistant's response

        Skips when the LLM response is a "no data" response.

        Args:
            user_text: what the user said
            assistant_text: what the LLM responded
            web_results: raw web search content (if any) to extract facts from

        Returns:
            ExtractionResult with extracted entities, relations, facts
        """
        from acervo.s1_5_graph_update import S1_5GraphUpdate, apply_s1_5_result

        # Check if LLM response is a "no data" instruction-following response
        _SKIP_PHRASES = ("i don't have information", "i don't have data", "i can search",
                         "would you like me to search", "shall i look",
                         "no tengo información", "no tengo datos", "puedo buscar",
                         "busque en internet", "querés que busque", "quieres que busque")
        assistant_lower = assistant_text.lower()
        is_no_data_response = any(p in assistant_lower for p in _SKIP_PHRASES)

        if is_no_data_response and not web_results:
            self._record_turn_metric(ExtractionResult())
            return ExtractionResult()

        # If web results were used, extract and persist web facts
        has_tool_results = bool(web_results)
        if web_results:
            await self._persist_web_facts(user_text, web_results)

        # Build S1.5 input: new entities from S1 (stored as last extraction)
        s1_entities = getattr(self, "_last_s1_extraction", None)
        new_entities_json = "[]"
        if s1_entities and s1_entities.entities:
            new_entities_json = json.dumps([
                {"name": e.name, "type": e.type, "layer": e.layer}
                for e in s1_entities.entities
            ], ensure_ascii=False)

        # Build existing nodes summary for S1.5
        all_nodes = self._graph.get_all_nodes()
        existing_nodes_json = build_graph_summary(
            [dict(n) for n in all_nodes], user_text,
        )

        # Run S1.5 Graph Update
        s1_5 = S1_5GraphUpdate(self._extractor_llm, prompt_template=self._s1_5_prompt)
        s1_5_result = await s1_5.run(
            new_entities_json=new_entities_json,
            existing_nodes_json=existing_nodes_json,
            current_assistant_msg=assistant_text,
        )

        # Apply curation actions + persist assistant entities
        audit = apply_s1_5_result(self._graph, s1_5_result, owner=self._owner)
        log.info(
            "[acervo] S1.5 — merges=%d, fixes=%d, discards=%d, entities=%d, facts=%d",
            audit["merges_applied"], audit["type_corrections"],
            audit["discards"], audit["entities_added"], audit["facts_added"],
        )

        # Enrich newly added assistant entities
        if s1_5_result.assistant_extraction.entities:
            await self._enrich_nodes_post_llm(s1_5_result.assistant_extraction, has_tool_results)

        # Build result for metrics
        result = s1_5_result.assistant_extraction
        self._record_turn_metric(result)

        # Mark touched nodes (last_active + session_count)
        self._mark_touched_nodes()

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
        Layer is determined per-entity: LLM layer hint > possessive detection > type heuristic.
        """
        result = await self._extractor.extract(user_msg, assistant_msg)

        if result.entities:
            relations = (
                [(r.source, r.target, r.relation) for r in result.relations]
                if result.relations else None
            )
            user_facts = [
                (f.entity, f.fact, f.source)
                for f in result.facts
                if f.speaker == "user"
            ]

            # Per-entity layer assignment
            for entity in result.entities:
                entity_layer = self._resolve_entity_layer(entity, user_msg)
                entity_source = "world" if entity_layer == Layer.UNIVERSAL else "user_assertion"

                # Collect facts for this entity only
                entity_facts = [
                    (f.entity, f.fact, f.source)
                    for f in result.facts
                    if f.speaker == "user" and f.entity == entity.name
                ]

                self._graph.upsert_entities(
                    [(entity.name, entity.type)],
                    None,  # relations added separately below
                    entity_facts if entity_facts else None,
                    layer=entity_layer,
                    source=entity_source,
                    owner=self._owner or None,
                )

                # Merge extracted attributes into graph node
                if entity.attributes:
                    nid = _make_id(entity.name)
                    node = self._graph.get_node(nid)
                    if node:
                        node.setdefault("attributes", {}).update(entity.attributes)

            # Add relations in a single pass (uses first entity's layer for edge metadata)
            if relations:
                self._graph.upsert_entities(
                    [],  # no new entities
                    relations,
                    layer=Layer.PERSONAL,  # default for relation edges
                    source="user_assertion",
                    owner=self._owner or None,
                )

            # Mark entities without user facts as pending_verification
            if not user_facts:
                for entity in result.entities:
                    nid = _make_id(entity.name)
                    node = self._graph.get_node(nid)
                    if node and not node.get("facts"):
                        node["status"] = "pending_verification"

        return result

    @staticmethod
    def _resolve_entity_layer(entity, user_msg: str) -> Layer:
        """Determine layer for a single entity.

        Priority: LLM layer hint > possessive detection > type heuristic.
        """
        # 1. LLM explicitly said PERSONAL or UNIVERSAL
        if entity.layer == "PERSONAL":
            return Layer.PERSONAL
        if entity.layer == "UNIVERSAL":
            return Layer.UNIVERSAL

        # 2. Possessive detection: "my X", "mi X" → PERSONAL
        import re
        name_lower = entity.name.lower()
        msg_lower = user_msg.lower()
        possessive_patterns = [
            rf"\bmy\s+{re.escape(name_lower)}\b",
            rf"\bmi\s+{re.escape(name_lower)}\b",
            rf"\bnuestro\s+{re.escape(name_lower)}\b",
            rf"\bnuestra\s+{re.escape(name_lower)}\b",
        ]
        for pattern in possessive_patterns:
            if re.search(pattern, msg_lower):
                return Layer.PERSONAL

        # 3. Type heuristic
        if is_likely_universal(entity.type):
            return Layer.UNIVERSAL

        return Layer.PERSONAL

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

            # Index entity facts in vector store (background — don't block response)
            if self._vector_store:
                fact_texts = [
                    f.get("fact", "") if isinstance(f, dict) else str(f)
                    for f in node.get("facts", [])
                ]
                fact_texts = [t for t in fact_texts if t.strip()]
                if fact_texts:
                    asyncio.create_task(
                        self._bg_index_facts(node_id, entity.name, fact_texts)
                    )

        self._graph.save()

    async def _bg_index_facts(
        self, node_id: str, label: str, fact_texts: list[str],
    ) -> None:
        """Background task: index facts in vector store without blocking response."""
        try:
            await self._vector_store.index_facts(node_id, label, fact_texts)
            log.info("[acervo] Indexed %d facts for: %s", len(fact_texts), label)
        except Exception as e:
            log.warning("Vector indexing failed for %s: %s", label, e)

    def materialize(self, query: str, token_budget: int = 800) -> str:
        """Build a context string from relevant graph nodes."""
        active = self._find_active_node_ids(query)
        return synthesize(self._graph, query, active_node_ids=active)

    def _mark_touched_nodes(self) -> None:
        """Update last_active on active nodes, increment session_count once per session."""
        from datetime import datetime

        now = datetime.now().isoformat(timespec="seconds")
        active_ids = getattr(self, "_last_active_node_ids", set())
        already_counted = getattr(self, "_session_counted_nodes", set())
        for node in self._graph.get_nodes_by_ids(active_ids):
            nid = node.get("id", "")
            node["last_active"] = now
            if nid not in already_counted:
                node["session_count"] = node.get("session_count", 0) + 1
                already_counted.add(nid)
        self._session_counted_nodes = already_counted
        self._graph.save()

    # ── Internal helpers ──

    def _gather_graph_nodes(self, active_node_ids: set[str]) -> list[dict]:
        """Gather active nodes and their neighbors with relations for context.

        Each node gets a ``_hot`` flag: True for directly-activated nodes,
        False for neighbor-expanded nodes (used for scoring in ranked chunks).
        """
        nodes: list[dict] = []
        seen_ids: set[str] = set()

        # Collect active nodes
        for node in self._graph.get_nodes_by_ids(active_node_ids):
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

        # Add neighbors of active nodes (1-level expansion)
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

        # For indexed nodes (section/symbol): inject summary or content snippet
        # as a pseudo-fact so it flows through the ranked chunk pipeline
        if self._workspace_path:
            for node in nodes:
                kind = node.get("kind")
                if kind not in ("section", "symbol"):
                    continue
                if node.get("facts"):
                    continue  # already has facts, skip
                summary = node.get("attributes", {}).get("summary", "")
                if summary:
                    node.setdefault("facts", []).append({"fact": summary, "source": "rag"})
                else:
                    content = self._graph.get_symbol_content(node["id"], self._workspace_path)
                    if content:
                        snippet = content[:500].strip()
                        if snippet:
                            node.setdefault("facts", []).append({"fact": snippet, "source": "rag"})

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

    def _build_project_overview(self) -> str:
        """Build a compact overview of the project from graph data.

        Prefers synthesis-generated overview if available (from graph_synthesizer).
        Falls back to structural file/folder listing otherwise.
        """
        all_nodes = self._graph.get_all_nodes()
        file_nodes = [n for n in all_nodes if n.get("kind") == "file"]

        if not file_nodes and not self._project_description:
            return ""

        lines: list[str] = []

        # Optional client-provided description
        if self._project_description:
            lines.append(f"Project: {self._project_description}")

        # Check for synthesis-generated overview (richer than structural listing)
        synthesis_nodes = self._graph.get_nodes_by_kind("synthesis")
        overview_node = next(
            (n for n in synthesis_nodes if n.get("type") == "project_overview"),
            None,
        )
        if overview_node:
            summary = overview_node.get("attributes", {}).get("summary", "")
            if summary:
                lines.append(summary)
                # Still include file list for structural awareness
                lines.append(self._build_file_list(file_nodes, all_nodes))
                return "\n".join(lines)

        # Fallback: structural listing (no synthesis available)
        if not file_nodes:
            return "\n".join(lines)

        lines.append(self._build_file_list(file_nodes, all_nodes))
        return "\n".join(lines)

    def _build_file_list(self, file_nodes: list[dict], all_nodes: list[dict]) -> str:
        """Build a structural file/folder listing."""
        folder_nodes = [n for n in all_nodes if n.get("kind") == "folder"]
        lines: list[str] = []

        if folder_nodes:
            lines.append(f"This project contains {len(file_nodes)} indexed files:")
            folder_files: dict[str, list[dict]] = {}
            orphan_files: list[dict] = []
            for fnode in file_nodes:
                path = fnode.get("attributes", {}).get("path", "")
                parts = path.rsplit("/", 1)
                folder_path = parts[0] if len(parts) > 1 else ""
                if folder_path:
                    folder_files.setdefault(folder_path, []).append(fnode)
                else:
                    orphan_files.append(fnode)

            for folder in sorted(folder_nodes, key=lambda n: n.get("attributes", {}).get("path", "")):
                fpath = folder.get("attributes", {}).get("path", "")
                flabel = folder.get("label", fpath)
                files = folder_files.get(fpath, [])
                if files:
                    lines.append(f"- {flabel}/ ({len(files)} files)")
                    for fnode in files[:10]:
                        label = fnode.get("label", "")
                        lines.append(f"  - {label}")
                    if len(files) > 10:
                        lines.append(f"  - ... and {len(files) - 10} more")

            for fnode in orphan_files[:10]:
                lines.append(f"- {fnode.get('label', '')}")
        else:
            lines.append(f"This project contains {len(file_nodes)} indexed files:")
            for node in file_nodes[:30]:
                lines.append(f"- {node.get('label', '')}")

        section_count = sum(1 for n in all_nodes if n.get("kind") == "section")
        if section_count:
            lines.append(f"\nTotal sections across all files: {section_count}")

        return "\n".join(lines)

    @staticmethod
    def _text_matches(label: str, msg_lower: str, msg_words: set[str]) -> bool:
        """Check if a node label is mentioned in the user message."""
        if label in msg_lower:
            return True
        for word in msg_words:
            if len(word) >= 4 and label.startswith(word):
                return True
        label_words = set(label.split())
        if len(label_words) > 1 and label_words.issubset(msg_words):
            return True
        return False

    def _find_active_node_ids(self, user_text: str, current_topic: str = "none", intent: str = "specific") -> set[str]:
        """Find graph node IDs relevant to the user message.

        Matches ALL node kinds with kind-appropriate logic:
        - entity: label text matching (prefix, substring, multi-word)
        - file: filename stem or path matching
        - folder: folder name matching
        - section/symbol: label text matching (same as entity)

        For "overview" intent, activates all file and folder nodes so the
        gathering step can build a complete picture of the project.

        Returns a set of node IDs. Does NOT mutate the graph — activation
        is ephemeral and lives in the conversation context.
        """
        active: set[str] = set()
        msg_lower = user_text.lower()
        msg_words = set(msg_lower.split())

        for node in self._graph.get_all_nodes():
            kind = node.get("kind", "entity")
            label = node.get("label", "").lower()
            nid = node.get("id", "")
            if not label or len(label) < 3:
                continue

            # Overview intent: activate file, folder, and synthesis nodes only.
            # Skip section/symbol nodes — they add chapter-level noise.
            if intent == "overview":
                if kind in ("file", "folder", "synthesis"):
                    active.add(nid)
                continue  # skip all other kinds for overview

            # Synthesis nodes: always activate on keyword match against summary
            if kind == "synthesis":
                summary = node.get("attributes", {}).get("summary", "").lower()
                if summary:
                    match_count = sum(1 for w in msg_words if len(w) >= 4 and w in summary)
                    if match_count >= 1:
                        active.add(nid)
                continue

            if kind == "file":
                # Match filename stem (without extension) or full path
                from pathlib import PurePosixPath
                stem = PurePosixPath(label).stem.lower() if "." in label else label
                path_str = node.get("attributes", {}).get("path", "").lower()
                if (len(stem) >= 3 and stem in msg_lower) or (path_str and path_str in msg_lower):
                    active.add(nid)
                    continue
            elif kind == "folder":
                # Match folder name or path
                path_str = node.get("attributes", {}).get("path", "").lower()
                if self._text_matches(label, msg_lower, msg_words):
                    active.add(nid)
                    continue
                if path_str and path_str in msg_lower:
                    active.add(nid)
                    continue
            else:
                # entity, section, symbol: text matching on label
                if self._text_matches(label, msg_lower, msg_words):
                    active.add(nid)
                    continue

            # Also match against node summary and topics (semantic relevance)
            summary = node.get("attributes", {}).get("summary", "").lower()
            topics_raw = node.get("attributes", {}).get("topics", [])
            topics_str = " ".join(t.lower() for t in topics_raw) if isinstance(topics_raw, list) else str(topics_raw).lower()
            searchable = f"{summary} {topics_str}"
            if searchable.strip():
                for word in msg_words:
                    if len(word) >= 4 and word in searchable:
                        active.add(nid)
                        break

        # Expand: if a folder is active, activate all contained files
        folder_ids = [
            nid for nid in list(active)
            if (n := self._graph.get_node(nid)) and n.get("kind") == "folder"
        ]
        for folder_id in folder_ids:
            for edge in self._graph.get_edges_for(folder_id):
                if edge.get("relation") == "contains":
                    child_id = edge["target"] if edge["source"] == folder_id else edge["source"]
                    active.add(child_id)

        # Expand: if a file is active, also activate its child sections/symbols
        file_ids = [
            nid for nid in list(active)
            if (n := self._graph.get_node(nid)) and n.get("kind") == "file"
        ]
        for file_id in file_ids:
            for edge in self._graph.get_edges_for(file_id):
                if edge.get("relation") == "contains":
                    child_id = edge["target"] if edge["source"] == file_id else edge["source"]
                    active.add(child_id)

        return active

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
