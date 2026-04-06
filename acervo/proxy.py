"""acervo serve — transparent LLM proxy with context enrichment.

Sits between any LLM client (Claude Code, Cursor, etc.) and the real LLM API.
Enriches the first user turn with context from the 3-stage pipeline, passes
through mid-loop tool-use turns without modification, and watches responses
for file-modifying tool calls to update a changelog.

Supports both Anthropic Messages API and OpenAI Chat Completions API,
streaming and non-streaming.

Usage:
    proxy = AcervoProxy(config, config_path)
    await proxy.start()   # binds to 0.0.0.0:{config.proxy.port}
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import aiohttp
from aiohttp import web

from acervo.config import AcervoConfig
from acervo.infra_prompt import build_system_message

log = logging.getLogger(__name__)


class AcervoProxy:
    """Transparent LLM proxy with Acervo context enrichment."""

    def __init__(
        self,
        config: AcervoConfig,
        config_path: Path,
    ) -> None:
        self._config = config
        self._config_path = config_path
        self._acervo = None  # Lazy init
        self._session: aiohttp.ClientSession | None = None
        self._changelog: list[dict] = []
        self._turn_count = 0
        self._last_enrichment: dict = {}
        self._last_forwarded_body: dict | None = None
        self._pending_context_msgs: list[dict] | None = None  # context for tool continuations
        self._app = web.Application()
        self._setup_routes()

    def _setup_routes(self) -> None:
        self._app.router.add_route("POST", "/v1/messages", self._handle_anthropic)
        self._app.router.add_route("POST", "/v1/chat/completions", self._handle_openai)
        self._app.router.add_get("/acervo/status", self._handle_status)
        self._app.router.add_get("/acervo/changelog", self._handle_changelog)
        self._app.router.add_post("/acervo/reset", self._handle_reset)
        self._app.router.add_post("/acervo/reload-graph", self._handle_reload_graph)
        self._app.router.add_post("/acervo/clear", self._handle_clear)
        self._app.router.add_post("/acervo/switch-project", self._handle_switch_project)
        self._app.router.add_get("/acervo/last-turn", self._handle_last_turn)
        self._app.router.add_get("/acervo/last-request", self._handle_last_request)
        self._app.router.add_get("/acervo/traces", self._handle_traces_list)
        self._app.router.add_get("/acervo/traces/{session_id}", self._handle_trace)
        self._app.router.add_get("/acervo/traces/{session_id}/summary", self._handle_trace_summary)
        # Graph CRUD
        self._app.router.add_get("/acervo/graph/nodes", self._handle_graph_list)
        self._app.router.add_get("/acervo/graph/search", self._handle_graph_search)
        self._app.router.add_get("/acervo/graph/nodes/{node_id}", self._handle_graph_get)
        self._app.router.add_delete("/acervo/graph/nodes/{node_id}", self._handle_graph_delete)
        self._app.router.add_post("/acervo/graph/merge", self._handle_graph_merge)
        # Document management
        self._app.router.add_post("/acervo/documents", self._handle_document_upload)
        self._app.router.add_get("/acervo/documents", self._handle_documents_list)
        self._app.router.add_get("/acervo/documents/{doc_id}", self._handle_document_get)
        self._app.router.add_delete("/acervo/documents/{doc_id}", self._handle_document_delete)

    def _reload_config(self) -> None:
        """Reload config from disk (cheap — single TOML parse)."""
        try:
            self._config = AcervoConfig.load(self._config_path)
        except Exception:
            pass  # Keep existing config on error

    async def start(self, host: str = "0.0.0.0", port: int | None = None) -> None:
        """Start the proxy server."""
        port = port or self._config.proxy.port
        self._session = aiohttp.ClientSession()

        await self._init_acervo()

        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        target = self._config.proxy.target or "(from X-Forward-To header)"
        log.info("Acervo proxy listening on %s:%d -> %s", host, port, target)
        print(f"Acervo proxy listening on http://{host}:{port}")
        print(f"  Target: {target}")
        print(f"  Anthropic: POST /v1/messages")
        print(f"  OpenAI:    POST /v1/chat/completions")
        print(f"  Status:    GET  /acervo/status")
        print()

        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            pass
        finally:
            print("\nShutting down Acervo proxy...")
            if self._session:
                await self._session.close()
            await runner.cleanup()
            print("Done.")

    async def _init_acervo(self) -> None:
        """Initialize Acervo from the project config."""
        try:
            from acervo import Acervo

            # Embeddings + vector store (for document chunk indexing)
            embedder = None
            vector_store = None
            embed_cfg = self._config.embeddings
            if embed_cfg.url and embed_cfg.model:
                from acervo.openai_client import OllamaEmbedder
                resolved = embed_cfg.resolve()
                _embedder = OllamaEmbedder(
                    base_url=resolved.url,
                    model=resolved.model,
                )
                try:
                    await _embedder.embed("health")
                    embedder = _embedder
                    log.info("Embedder configured: %s @ %s", resolved.model, resolved.url)
                except Exception as e:
                    log.warning("Ollama unavailable (%s) — running without embeddings", e)
                    print(f"  WARNING: Ollama unavailable — embeddings disabled")

            project_root = self._config_path.parent.parent

            if embedder:
                try:
                    from acervo.vector_store import ChromaVectorStore
                    vectordb_path = project_root / ".acervo" / "data" / "vectordb"
                    vectordb_path.mkdir(parents=True, exist_ok=True)
                    embed_batch_fn = getattr(embedder, "embed_batch", None)
                    vector_store = ChromaVectorStore(
                        persist_path=str(vectordb_path),
                        embed_fn=embedder.embed,
                        embed_batch_fn=embed_batch_fn,
                    )
                    log.info("Vector store initialized at %s", vectordb_path)
                except ImportError:
                    log.info("chromadb not installed — vector search disabled")
                except Exception as e:
                    log.warning("Vector store init failed: %s", e)

            self._acervo = Acervo.from_project(
                project_root, auto_init=False,
                embedder=embedder, vector_store=vector_store,
            )
            stats = self._acervo.get_graph_stats()
            print(f"  Graph: {stats['node_count']} nodes, {stats['edge_count']} edges")
            embed_status = f"enabled ({embed_cfg.model})" if embedder else "disabled"
            vs_status = "enabled" if vector_store else "disabled"
            print(f"  Embeddings: {embed_status}, Vector store: {vs_status}")
        except Exception as e:
            log.warning("Acervo init failed (proxy will pass-through): %s", e)
            print(f"  WARNING: Acervo init failed: {e}")
            self._acervo = None

    # ── Route handlers ──

    async def _handle_anthropic(self, request: web.Request) -> web.StreamResponse:
        """Handle Anthropic Messages API requests."""
        body = await request.json()
        is_stream = body.get("stream", False)

        # Always compose the infra block in the system message.
        self._reload_config()
        body = self._compose_system_message_anthropic(body)

        if self._is_new_user_turn_anthropic(body):
            body = await self._enrich_anthropic(body)
            body = self._window_history_anthropic(body)
            self._turn_count += 1

        # Capture what we're actually sending to the LLM (for trace/debug)
        self._last_forwarded_body = body

        target = self._resolve_target(request)
        target_url = f"{target}/v1/messages"
        headers = self._forward_headers(request)

        try:
            if is_stream:
                return await self._stream_and_forward(
                    request, target_url, headers, body, "anthropic",
                )

            async with self._session.post(target_url, headers=headers, json=body) as resp:
                response_body = await resp.json()
                has_tool_calls = self._watch_tool_calls_anthropic(response_body)
                if not has_tool_calls:
                    await self._process_response(body, response_body)
                return web.json_response(response_body, status=resp.status)
        except aiohttp.ClientConnectorError:
            name = self._config.proxy.provider_name or target
            log.error("Cannot connect to provider '%s' at %s — is it running?", name, target)
            return web.json_response(
                {
                    "error": {
                        "message": f"Cannot connect to '{name}' at {target}. Make sure the provider is running.",
                        "type": "connection_error",
                        "code": "upstream_unreachable",
                    }
                },
                status=502,
            )

    async def _handle_openai(self, request: web.Request) -> web.StreamResponse:
        """Handle OpenAI Chat Completions API requests."""
        body = await request.json()
        is_stream = body.get("stream", False)

        # Always compose the infra block in the system message —
        # even on tool-continuation requests that skip full enrichment.
        self._reload_config()
        body = self._compose_system_message_openai(body)

        if self._is_new_user_turn_openai(body):
            msg_count = len(body.get("messages", []))
            log.info("New user turn detected (%d messages)", msg_count)
            self._last_enrichment = {
                "enriched": False, "topic": "", "warm_tokens": 0,
                "warm_content_preview": "", "entities_extracted": 0,
                "facts_extracted": 0,
            }
            body = await self._enrich_openai(body)
            body = self._window_history_openai(body)
            self._turn_count += 1
        elif self._pending_context_msgs:
            # Tool continuation — re-inject context from the initial enrichment
            # so the LLM has graph context for the streaming response too.
            body = self._reinject_context_openai(body)

        # Capture what we're actually sending to the LLM (for trace/debug)
        self._last_forwarded_body = body

        target = self._resolve_target(request)
        target_url = f"{target}/v1/chat/completions"
        headers = self._forward_headers(request)

        try:
            if is_stream:
                return await self._stream_and_forward(
                    request, target_url, headers, body, "openai",
                )

            async with self._session.post(target_url, headers=headers, json=body) as resp:
                response_body = await resp.json()
                has_tool_calls = self._watch_tool_calls_openai(response_body)
                # Only extract on final text responses, not tool-use rounds
                if not has_tool_calls:
                    await self._process_response(body, response_body)
                return web.json_response(response_body, status=resp.status)
        except aiohttp.ClientConnectorError:
            name = self._config.proxy.provider_name or target
            log.error("Cannot connect to provider '%s' at %s — is it running?", name, target)
            return web.json_response(
                {
                    "error": {
                        "message": f"Cannot connect to '{name}' at {target}. Make sure the provider is running.",
                        "type": "connection_error",
                        "code": "upstream_unreachable",
                    }
                },
                status=502,
            )

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Return Acervo proxy status."""
        stats = self._acervo.get_graph_stats() if self._acervo else {}
        project_root = str(self._config_path.parent.parent)
        return web.json_response({
            "status": "active" if self._acervo else "pass-through",
            "turns": self._turn_count,
            "changelog_entries": len(self._changelog),
            "target": self._config.proxy.target or "(from X-Forward-To header)",
            "graph": stats,
            "project_path": project_root,
        })

    async def _handle_changelog(self, request: web.Request) -> web.Response:
        """Return the changelog of file modifications."""
        return web.json_response({"changelog": self._changelog})

    async def _handle_reset(self, request: web.Request) -> web.Response:
        """Reset proxy state and reinitialize Acervo facade from disk."""
        self._changelog.clear()
        self._turn_count = 0
        self._pending_context_msgs = None
        self._last_forwarded_body = None
        self._last_enrichment = {}
        # Full reinit — rebuilds facade, topic detector, metrics, graph from disk
        await self._init_acervo()
        return web.json_response({"status": "reset"})

    async def _handle_reload_graph(self, request: web.Request) -> web.Response:
        """Reload graph from disk without resetting conversation state.

        Called after external indexing writes new data to nodes.json/edges.json.
        The proxy's in-memory graph would otherwise be stale (and overwrite the
        new data on the next save()).
        """
        if self._acervo:
            self._acervo.graph.reload()
            stats = self._acervo.get_graph_stats()
            log.info("Graph reloaded: %s", stats)
            return web.json_response({"status": "reloaded", "graph": stats})
        return web.json_response({"status": "no_acervo"}, status=503)

    async def _handle_clear(self, request: web.Request) -> web.Response:
        """Clear ALL Acervo data (graph + vectordb) and reinitialize clean.

        Unlike /reset (which reloads from disk), this deletes all persistent
        data first. Handles ChromaDB file locks by dropping the client before
        deleting the vectordb directory.
        """
        import shutil

        # Drop the current facade (releases ChromaDB file locks)
        self._acervo = None
        self._changelog.clear()
        self._turn_count = 0
        self._pending_context_msgs = None
        self._last_forwarded_body = None
        self._last_enrichment = {}

        # Delete all data
        project_root = self._config_path.parent.parent
        data_dir = project_root / ".acervo" / "data"
        deleted: list[str] = []
        skipped: list[str] = []
        if data_dir.exists():
            for child in data_dir.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                    deleted.append(child.name)
                except PermissionError:
                    skipped.append(child.name)
                    log.warning("Could not delete %s (file locked)", child)

        # Reinitialize with empty state
        await self._init_acervo()
        return web.json_response({
            "status": "cleared",
            "deleted": deleted,
            "skipped": skipped,
        })

    async def _handle_switch_project(self, request: web.Request) -> web.Response:
        """Switch the proxy to a different project's graph and config.

        Body: {"project_path": "/absolute/path/to/project"}
        Re-initializes the Acervo facade with the new project's .acervo/ directory.
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        project_path = body.get("project_path", "")
        if not project_path:
            return web.json_response({"error": "project_path required"}, status=400)

        new_root = Path(project_path)
        config_path = new_root / ".acervo" / "config.toml"
        if not config_path.exists():
            return web.json_response(
                {"error": f"No .acervo/config.toml at {new_root}"}, status=400,
            )

        # Update config path and reload
        self._config_path = config_path
        self._reload_config()

        # Reset conversation state
        self._changelog.clear()
        self._turn_count = 0
        self._pending_context_msgs = None
        self._last_forwarded_body = None
        self._last_enrichment = {}

        # Re-initialize Acervo with new project
        await self._init_acervo()

        stats = self._acervo.get_graph_stats() if self._acervo else {}
        log.info("Switched to project: %s (%s)", new_root, stats)
        return web.json_response({
            "status": "switched",
            "project_path": str(new_root),
            "graph": stats,
        })

    async def _handle_last_turn(self, request: web.Request) -> web.Response:
        """Return enrichment details from the last turn."""
        return web.json_response(self._last_enrichment or {
            "enriched": False, "topic": "", "warm_tokens": 0,
            "warm_content_preview": "", "entities_extracted": 0,
            "facts_extracted": 0,
        })

    async def _handle_last_request(self, request: web.Request) -> web.Response:
        """Return the last request body forwarded to the LLM.

        This shows what the LLM actually received, including the infra block
        and enriched context — as opposed to the raw request from the client.
        """
        if not self._last_forwarded_body:
            return web.json_response({"messages": [], "message_count": 0})
        messages = self._last_forwarded_body.get("messages", [])
        return web.json_response({
            "messages": [
                {
                    "role": m.get("role", ""),
                    "content": m.get("content", ""),
                    "content_length": len(m.get("content", "")),
                }
                for m in messages
            ],
            "message_count": len(messages),
            "model": self._last_forwarded_body.get("model", ""),
            "temperature": self._last_forwarded_body.get("temperature"),
            "stream": self._last_forwarded_body.get("stream"),
            "has_tools": bool(self._last_forwarded_body.get("tools")),
        })

    async def _handle_traces_list(self, request: web.Request) -> web.Response:
        """List available trace files."""
        if not self._acervo:
            return web.json_response({"traces": []})
        traces_dir = self._acervo.metrics.trace_path.parent if self._acervo.metrics.trace_path else None
        if not traces_dir or not traces_dir.exists():
            return web.json_response({"traces": []})
        files = sorted(traces_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        return web.json_response({
            "traces": [{"session_id": f.stem, "size_kb": round(f.stat().st_size / 1024, 1)} for f in files],
        })

    async def _handle_trace(self, request: web.Request) -> web.Response:
        """Return all turns from a trace session as a JSON array."""
        import json
        session_id = request.match_info["session_id"]
        if not self._acervo or not self._acervo.metrics.trace_path:
            return web.json_response({"error": "traces not configured"}, status=404)
        trace_file = self._acervo.metrics.trace_path.parent / f"{session_id}.jsonl"
        if not trace_file.exists():
            return web.json_response({"error": f"trace {session_id} not found"}, status=404)
        turns = []
        for line in trace_file.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                turns.append(json.loads(line))
        return web.json_response({"session_id": session_id, "turns": turns})

    async def _handle_trace_summary(self, request: web.Request) -> web.Response:
        """Return aggregated metrics for a trace session."""
        import json
        from acervo.metrics import SessionMetrics, TurnMetric
        session_id = request.match_info["session_id"]
        if not self._acervo or not self._acervo.metrics.trace_path:
            return web.json_response({"error": "traces not configured"}, status=404)
        trace_file = self._acervo.metrics.trace_path.parent / f"{session_id}.jsonl"
        if not trace_file.exists():
            return web.json_response({"error": f"trace {session_id} not found"}, status=404)
        # Reconstruct SessionMetrics from JSONL
        metrics = SessionMetrics(session_id=session_id)
        for line in trace_file.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                data = json.loads(line)
                turn = TurnMetric(**{k: v for k, v in data.items() if k in TurnMetric.__dataclass_fields__})
                metrics.turns.append(turn)
        return web.json_response(metrics.export_json())

    # ── Graph CRUD handlers ──

    def _require_graph(self):
        """Get the TopicGraph or raise 503."""
        if not self._acervo:
            raise web.HTTPServiceUnavailable(text="Acervo not initialized")
        return self._acervo.graph

    async def _handle_graph_list(self, request: web.Request) -> web.Response:
        """GET /acervo/graph/nodes — list nodes."""
        graph = self._require_graph()
        kind = request.query.get("kind")
        nodes = graph.get_nodes_by_kind(kind) if kind else graph.get_all_nodes()
        limit = int(request.query.get("limit", 0))
        if limit > 0:
            nodes = nodes[:limit]
        return web.json_response(nodes)

    async def _handle_graph_get(self, request: web.Request) -> web.Response:
        """GET /acervo/graph/nodes/{node_id} — node detail with edges."""
        graph = self._require_graph()
        node_id = request.match_info["node_id"]
        node = graph.get_node(node_id)
        if not node:
            return web.json_response({"error": f"node not found: {node_id}"}, status=404)
        edges = graph.get_edges_for(node["id"])
        files = graph.get_linked_files(node["id"])
        return web.json_response({**node, "edges": edges, "linked_files": files})

    async def _handle_graph_search(self, request: web.Request) -> web.Response:
        """GET /acervo/graph/search?q=query&kind=entity — search nodes."""
        graph = self._require_graph()
        query = request.query.get("q", "").lower()
        if not query:
            return web.json_response({"error": "missing q parameter"}, status=400)
        kind = request.query.get("kind")
        nodes = graph.get_nodes_by_kind(kind) if kind else graph.get_all_nodes()
        results = []
        for node in nodes:
            label = node.get("label", node["id"])
            if query in label.lower() or query in node["id"].lower():
                results.append(node)
                continue
            for f in node.get("facts", []):
                if query in f.get("fact", "").lower():
                    results.append(node)
                    break
        return web.json_response(results)

    async def _handle_graph_delete(self, request: web.Request) -> web.Response:
        """DELETE /acervo/graph/nodes/{node_id} — delete node and edges."""
        graph = self._require_graph()
        node_id = request.match_info["node_id"]
        node = graph.get_node(node_id)
        if not node:
            return web.json_response({"error": f"node not found: {node_id}"}, status=404)
        nid = node["id"]
        graph.remove_node(nid)
        graph.save()
        return web.json_response({"deleted": nid})

    async def _handle_graph_merge(self, request: web.Request) -> web.Response:
        """POST /acervo/graph/merge — body: {"keep": "id1", "absorb": "id2"}."""
        graph = self._require_graph()
        body = await request.json()
        keep_id = body.get("keep", "")
        absorb_id = body.get("absorb", "")
        if not keep_id or not absorb_id:
            return web.json_response({"error": "missing keep or absorb"}, status=400)
        ok = graph.merge_nodes(keep_id, absorb_id)
        if not ok:
            return web.json_response({"error": "one or both nodes not found"}, status=404)
        graph.save()
        return web.json_response({"merged": {"keep": keep_id, "absorbed": absorb_id}})

    # ── Document management handlers ──

    async def _handle_document_upload(self, request: web.Request) -> web.Response:
        """POST /acervo/documents — upload and index a .md file."""
        if not self._acervo:
            raise web.HTTPServiceUnavailable(text="Acervo not initialized")

        reader = await request.multipart()
        if not reader:
            return web.json_response({"error": "multipart form data required"}, status=400)

        field = await reader.next()
        if not field or field.name != "file":
            return web.json_response({"error": "missing 'file' field"}, status=400)

        filename = field.filename or "document.md"
        if not filename.endswith(".md"):
            return web.json_response(
                {"error": "only .md files supported in v0.4.0"}, status=400,
            )

        content = await field.read(decode=True)

        # Save to .acervo/data/documents/
        project_root = self._config_path.parent.parent
        docs_dir = project_root / ".acervo" / "data" / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        dest = docs_dir / filename
        dest.write_bytes(content)

        result = await self._acervo.index_document(dest)
        return web.json_response(result)

    async def _handle_documents_list(self, request: web.Request) -> web.Response:
        """GET /acervo/documents — list indexed documents."""
        graph = self._require_graph()
        nodes = graph.get_nodes_with_chunks()
        docs = []
        for n in nodes:
            if n.get("kind") != "file":
                continue
            docs.append({
                "id": n["id"],
                "label": n.get("label", ""),
                "file_path": n.get("attributes", {}).get("path", ""),
                "chunk_count": len(n.get("chunk_ids", [])),
                "indexed_at": n.get("indexed_at", ""),
            })
        return web.json_response(docs)

    async def _handle_document_get(self, request: web.Request) -> web.Response:
        """GET /acervo/documents/{doc_id} — document detail with chunk_ids."""
        graph = self._require_graph()
        doc_id = request.match_info["doc_id"]
        node = graph.get_node(doc_id)
        if not node or node.get("kind") != "file" or not node.get("chunk_ids"):
            return web.json_response({"error": f"document not found: {doc_id}"}, status=404)

        # Include section children
        sections = []
        prefix = doc_id + "__"
        for nid, n in graph._nodes.items():
            if nid.startswith(prefix) and n.get("kind") == "section":
                sections.append({
                    "id": nid,
                    "label": n.get("label", ""),
                    "chunk_ids": n.get("chunk_ids", []),
                    "start_line": n.get("attributes", {}).get("start_line"),
                    "end_line": n.get("attributes", {}).get("end_line"),
                })

        return web.json_response({
            **node,
            "sections": sections,
        })

    async def _handle_document_delete(self, request: web.Request) -> web.Response:
        """DELETE /acervo/documents/{doc_id} — delete doc + chunks + nodes."""
        if not self._acervo:
            raise web.HTTPServiceUnavailable(text="Acervo not initialized")

        doc_id = request.match_info["doc_id"]
        result = await self._acervo.delete_document(doc_id)
        if result["chunks_removed"] == 0:
            node = self._acervo.graph.get_node(doc_id)
            if not node:
                return web.json_response({"error": f"document not found: {doc_id}"}, status=404)
        return web.json_response(result)

    def _resolve_target(self, request: web.Request) -> str:
        """Resolve the upstream LLM target URL.

        Priority: X-Forward-To header > config.proxy.target.
        The returned URL is the base (no /v1 suffix) — callers append the
        full path (e.g. /v1/chat/completions) themselves.
        """
        header = request.headers.get("x-forward-to", "").strip().rstrip("/")
        if header:
            return header.removesuffix("/v1")
        target = self._config.proxy.target.strip().rstrip("/")
        if target:
            return target.removesuffix("/v1")
        raise web.HTTPBadRequest(
            text="No target URL: set proxy.target in config or send X-Forward-To header",
        )

    # ── Turn detection ──

    def _is_new_user_turn_anthropic(self, body: dict) -> bool:
        """Detect new user turn vs tool-use continuation (Anthropic format).

        A tool-use continuation has the last user message containing
        tool_result content blocks.
        """
        messages = body.get("messages", [])
        if not messages:
            return False

        last = messages[-1]
        if last.get("role") != "user":
            return False

        content = last.get("content", "")
        if isinstance(content, list):
            return not any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            )
        return True  # String content = new user turn

    def _is_new_user_turn_openai(self, body: dict) -> bool:
        """Detect new user turn vs tool-use continuation (OpenAI format)."""
        messages = body.get("messages", [])
        if not messages:
            return False

        last = messages[-1]
        if last.get("role") == "tool":
            return False
        if last.get("role") == "assistant" and last.get("tool_calls"):
            return False
        return last.get("role") == "user"

    # ── System message composition (always applied) ──

    def _compose_system_message_openai(self, body: dict) -> dict:
        """Compose infra block + user prompt in system message (OpenAI format).

        Applied to ALL requests so the LLM always has context/tool instructions,
        even on tool-continuation rounds that skip full enrichment.
        """
        messages = body.get("messages", [])
        if not messages or messages[0].get("role") != "system":
            return body

        user_prompt = messages[0]["content"]
        composed = build_system_message(
            user_prompt, plan_mode=self._config.context.plan_mode,
        )
        # Skip if already composed (idempotent)
        if "ACERVO INFRA" in user_prompt:
            return body

        messages = [dict(m) for m in messages]
        messages[0] = dict(messages[0])
        messages[0]["content"] = composed
        body = dict(body)
        body["messages"] = messages
        return body

    def _compose_system_message_anthropic(self, body: dict) -> dict:
        """Compose infra block + user prompt in system message (Anthropic format)."""
        existing_system = body.get("system", "")
        if isinstance(existing_system, list):
            existing_text = " ".join(
                b.get("text", "") for b in existing_system
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            existing_text = existing_system

        if not existing_text or "ACERVO INFRA" in existing_text:
            return body

        body = dict(body)
        body["system"] = build_system_message(
            existing_text, plan_mode=self._config.context.plan_mode,
        )
        return body

    # ── Context enrichment ──

    async def _enrich_anthropic(self, body: dict) -> dict:
        """Enrich Anthropic request by appending context to the system message."""
        if not self._acervo:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        user_text = self._extract_text_anthropic(messages[-1])
        if not user_text:
            return body

        try:
            history = self._build_history_anthropic(body)
            prep = await self._acervo.prepare(user_text, history)

            if not prep.warm_content:
                return body

            context_block = (
                "\n\n[ACERVO CONTEXT — verified facts from knowledge graph]\n"
                f"{prep.warm_content}\n"
                "[END CONTEXT]"
            )
            body = dict(body)
            body["system"] = body.get("system", "") + context_block

            log.info(
                "Enriched Anthropic request: +%d warm tokens, topic=%s",
                prep.warm_tokens, prep.topic,
            )
        except Exception as e:
            log.warning("Enrichment failed (passing through): %s", e)

        return body

    async def _enrich_openai(self, body: dict) -> dict:
        """Enrich OpenAI request by inserting context from S1/S2/S3 pipeline.

        System message composition (infra block) is handled separately by
        _compose_system_message_openai() which runs on ALL requests.
        """
        if not self._acervo:
            log.info("Pass-through (Acervo not initialized)")
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        last = messages[-1]
        user_text = last.get("content", "") if last.get("role") == "user" else ""
        if not user_text:
            return body

        try:
            history = [
                {"role": m["role"], "content": m.get("content", "")}
                for m in messages
            ]
            log.info("Running prepare() with %d messages, user=%s...",
                     len(history), user_text[:60])
            prep = await self._acervo.prepare(user_text, history)

            self._last_enrichment.update({
                "topic": prep.topic,
                "stages": prep.stages,
                "debug": prep.debug,
            })

            log.info(
                "prepare() returned: warm_content=%d chars, warm_tokens=%d, topic=%s, stages=%s",
                len(prep.warm_content), prep.warm_tokens, prep.topic, prep.stages,
            )
            if not prep.warm_content:
                log.info("No context to inject (topic=%s)", prep.topic)
                return body

            # warm_content may contain [CONVERSATION CONTEXT] sections
            # If it starts with that marker, there's no verified content
            if "[CONVERSATION CONTEXT" in prep.warm_content:
                # Mixed or conversation-only context — already has markers
                ctx_text = (
                    "[VERIFIED CONTEXT]\n"
                    f"{prep.warm_content}\n"
                    "[END CONTEXT]"
                )
            else:
                # All verified content
                ctx_text = (
                    "[VERIFIED CONTEXT]\n"
                    f"{prep.warm_content}\n"
                    "[END CONTEXT]"
                )
            context_msg = {"role": "user", "content": ctx_text}
            ack_msg = {"role": "assistant", "content": "Understood."}

            # Store for tool-continuation re-injection
            self._pending_context_msgs = [context_msg, ack_msg]

            body = dict(body)
            new_messages = list(messages)
            insert_idx = 1 if new_messages[0].get("role") == "system" else 0
            new_messages.insert(insert_idx, context_msg)
            new_messages.insert(insert_idx + 1, ack_msg)
            body["messages"] = new_messages

            self._last_enrichment.update({
                "enriched": True,
                "warm_tokens": prep.warm_tokens,
                "warm_content_preview": prep.warm_content[:300],
            })
            log.info("Enriched: +%dtk, topic=%s", prep.warm_tokens, prep.topic)
        except Exception as e:
            log.error("Enrichment CRASHED: %s", e, exc_info=True)

        return body

    def _window_history_openai(self, body: dict) -> dict:
        """Trim conversation history, keeping only the last N messages.

        Windowing applies ALWAYS — with graph context, older turns are covered
        by the enriched context blocks. Without graph context, older turns
        aren't useful anyway; the last N messages provide conversational flow.

        Message layout after enrichment:
          [system, context_user, context_ack, ...old_conv..., current_user]
        After windowing:
          [system, context_user, context_ack, last_assistant, current_user]
        """
        window = self._config.context.history_window
        if window <= 0:
            return body  # 0 = disabled

        messages = body.get("messages", [])

        # Count prefix: system message + optional context injection pair
        prefix_len = 1  # system
        if len(messages) > 1 and "[VERIFIED CONTEXT]" in messages[1].get("content", ""):
            prefix_len += 2  # context_msg + ack_msg

        conversation = messages[prefix_len:]
        if len(conversation) <= window:
            return body  # Already within window

        trimmed_count = len(conversation) - window
        body = dict(body)
        body["messages"] = messages[:prefix_len] + conversation[-window:]
        log.info(
            "History windowed: %d → %d messages (trimmed %d, window=%d)",
            len(messages), len(body["messages"]), trimmed_count, window,
        )
        return body

    def _window_history_anthropic(self, body: dict) -> dict:
        """Trim Anthropic conversation history, keeping only the last N messages.

        Anthropic format: system is a top-level string, messages is pure
        conversation (no system message in the array). Just keep the last
        `window` messages.
        """
        window = self._config.context.history_window
        if window <= 0:
            return body  # 0 = disabled

        messages = body.get("messages", [])
        if len(messages) <= window:
            return body  # Already within window

        trimmed_count = len(messages) - window
        body = dict(body)
        body["messages"] = messages[-window:]
        log.info(
            "History windowed (Anthropic): %d → %d messages (trimmed %d, window=%d)",
            len(messages), len(body["messages"]), trimmed_count, window,
        )
        return body

    def _reinject_context_openai(self, body: dict) -> dict:
        """Re-inject stored context messages into a tool-continuation request.

        When the LLM uses tools, subsequent requests (tool results, streaming)
        don't trigger enrichment. This method re-inserts the context from the
        initial enrichment so the LLM always has graph context.
        """
        messages = body.get("messages", [])
        if not messages:
            return body

        # Check if context is already present (idempotent)
        for m in messages:
            if m.get("role") == "user" and "[VERIFIED CONTEXT]" in m.get("content", ""):
                return body

        body = dict(body)
        new_messages = list(messages)
        insert_idx = 1 if new_messages[0].get("role") == "system" else 0
        for i, ctx_msg in enumerate(self._pending_context_msgs):
            new_messages.insert(insert_idx + i, ctx_msg)
        body["messages"] = new_messages
        log.info("Re-injected context for tool continuation (%d msgs)", len(self._pending_context_msgs))
        return body

    # ── Forwarding ──

    def _forward_headers(self, request: web.Request) -> dict[str, str]:
        """Build headers for the upstream request (pass auth through)."""
        headers: dict[str, str] = {}
        for key in (
            "authorization", "x-api-key", "anthropic-version",
            "anthropic-beta", "content-type",
        ):
            if key in request.headers:
                headers[key] = request.headers[key]
        if "content-type" not in headers:
            headers["content-type"] = "application/json"
        return headers

    async def _stream_and_forward(
        self,
        request: web.Request,
        target_url: str,
        headers: dict,
        body: dict,
        api_format: str,
    ) -> web.StreamResponse:
        """Stream SSE from upstream, forwarding chunks immediately.

        Parses events in the background to detect tool calls and accumulate
        assistant text for post-response processing.
        """
        accumulated_text = ""
        accumulated_tool_calls: list[dict] = []
        current_tool: dict = {}

        async with self._session.post(target_url, headers=headers, json=body) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                headers={
                    "Content-Type": upstream.headers.get(
                        "Content-Type", "text/event-stream",
                    ),
                    "Cache-Control": "no-cache",
                },
            )
            try:
                await response.prepare(request)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
                        aiohttp.ClientConnectionResetError):
                log.warning("Client disconnected before stream started")
                return response

            buffer = b""
            async for chunk in upstream.content.iter_any():
                # Forward immediately for real-time token display
                try:
                    await response.write(chunk)
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
                            aiohttp.ClientConnectionResetError):
                    log.warning("Client disconnected during stream")
                    return response

                # Parse SSE events for tool call detection
                buffer += chunk
                while b"\n\n" in buffer:
                    event_bytes, buffer = buffer.split(b"\n\n", 1)
                    event_str = event_bytes.decode("utf-8", errors="replace")

                    for line in event_str.split("\n"):
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        if api_format == "anthropic":
                            accumulated_text, current_tool = self._parse_sse_anthropic(
                                event, accumulated_text, current_tool,
                                accumulated_tool_calls,
                            )
                        else:
                            accumulated_text, current_tool = self._parse_sse_openai(
                                event, accumulated_text, current_tool,
                                accumulated_tool_calls,
                            )

            try:
                await response.write_eof()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
                        aiohttp.ClientConnectionResetError):
                pass  # Client already gone

        # Post-stream: check tool calls for changelog entries
        for tc in accumulated_tool_calls:
            self._check_tool_call(tc.get("name", ""), tc.get("input", ""))

        # Only extract knowledge from FINAL text responses (no tool calls).
        # Tool-use responses trigger more rounds — extraction should wait
        # until the LLM produces the final text answer.
        if accumulated_text and not accumulated_tool_calls:
            await self._process_response_text(body, accumulated_text)

        return response

    # ── SSE event parsers ──

    def _parse_sse_anthropic(
        self,
        event: dict,
        text: str,
        current_tool: dict,
        tool_calls: list[dict],
    ) -> tuple[str, dict]:
        """Parse a single Anthropic SSE event."""
        delta = event.get("delta", {})
        if delta.get("type") == "text_delta":
            text += delta.get("text", "")

        if event.get("type") == "content_block_start":
            cb = event.get("content_block", {})
            if cb.get("type") == "tool_use":
                current_tool = {"name": cb.get("name", ""), "input": ""}

        if event.get("type") == "content_block_delta":
            d = event.get("delta", {})
            if d.get("type") == "input_json_delta" and current_tool:
                current_tool["input"] += d.get("partial_json", "")

        if event.get("type") == "content_block_stop" and current_tool.get("name"):
            tool_calls.append(dict(current_tool))
            current_tool = {}

        return text, current_tool

    def _parse_sse_openai(
        self,
        event: dict,
        text: str,
        current_tool: dict,
        tool_calls: list[dict],
    ) -> tuple[str, dict]:
        """Parse a single OpenAI SSE event."""
        choices = event.get("choices", [])
        if not choices:
            return text, current_tool

        delta = choices[0].get("delta", {})

        if delta.get("content"):
            text += delta["content"]

        for tc_delta in delta.get("tool_calls", []):
            fn = tc_delta.get("function", {})
            if fn.get("name"):
                if current_tool.get("name"):
                    tool_calls.append(dict(current_tool))
                current_tool = {
                    "name": fn["name"],
                    "input": fn.get("arguments", ""),
                }
            elif current_tool:
                current_tool["input"] += fn.get("arguments", "")

        finish = choices[0].get("finish_reason")
        if finish and current_tool.get("name"):
            tool_calls.append(dict(current_tool))
            current_tool = {}

        return text, current_tool

    # ── Tool call watching (changelog) ──

    def _watch_tool_calls_anthropic(self, response: dict) -> bool:
        """Check non-streaming Anthropic response for file-modifying tool calls.

        Returns True if the response contains tool calls.
        """
        found = False
        for block in response.get("content", []):
            if block.get("type") == "tool_use":
                found = True
                self._check_tool_call(
                    block.get("name", ""),
                    json.dumps(block.get("input", {})),
                )
        return found

    def _watch_tool_calls_openai(self, response: dict) -> bool:
        """Check non-streaming OpenAI response for file-modifying tool calls.

        Returns True if the response contains tool calls (not a final text response).
        """
        found = False
        for choice in response.get("choices", []):
            tool_calls = choice.get("message", {}).get("tool_calls", [])
            if tool_calls:
                found = True
            for tc in tool_calls:
                fn = tc.get("function", {})
                self._check_tool_call(fn.get("name", ""), fn.get("arguments", ""))
        return found

    def _check_tool_call(self, name: str, input_json: str) -> None:
        """Record file-modifying tool calls in the changelog."""
        changelog_cfg = self._config.changelog
        action = None

        if name in changelog_cfg.write_tools:
            action = "write"
        elif name in changelog_cfg.delete_tools:
            action = "delete"

        if not action:
            return

        file_path = ""
        try:
            args = json.loads(input_json) if input_json else {}
            for key in ("path", "file_path", "file", "filename", "file_name"):
                if key in args:
                    file_path = str(args[key])
                    break
        except (json.JSONDecodeError, TypeError):
            pass

        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "tool": name,
            "file": file_path,
        }
        self._changelog.append(entry)
        log.info("Changelog: %s %s via %s", action, file_path, name)

    # ── Post-response processing ──

    async def _process_response(self, request_body: dict, response_body: dict) -> None:
        """Extract assistant text from a non-streaming response and process it."""
        if not self._acervo:
            return

        # Anthropic format
        content = response_body.get("content", [])
        if content:
            assistant_text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            # OpenAI format
            choices = response_body.get("choices", [])
            assistant_text = (
                choices[0].get("message", {}).get("content", "") if choices else ""
            )

        if assistant_text:
            await self._process_response_text(request_body, assistant_text)

    async def _process_response_text(
        self, request_body: dict, assistant_text: str,
    ) -> None:
        """Run Acervo process() to extract knowledge from the LLM response."""
        if not self._acervo:
            return

        messages = request_body.get("messages", [])
        if not messages:
            return

        user_text = self._extract_text_anthropic(messages[-1])
        if not user_text:
            user_text = messages[-1].get("content", "")
        if not user_text:
            return

        try:
            # Check if tool results (e.g., web search) were used
            has_tool_results = any(
                m.get("role") == "tool" for m in messages
            )
            web_results = ""
            if has_tool_results:
                # Collect tool result content as web_results
                web_results = "\n".join(
                    m.get("content", "") for m in messages
                    if m.get("role") == "tool"
                )

            result = await self._acervo.process(
                user_text, assistant_text, web_results=web_results,
            )
            if result:
                source = "tool_result" if has_tool_results else "conversation"
                self._last_enrichment["entities_extracted"] = len(result.entities)
                self._last_enrichment["facts_extracted"] = len(result.facts)
                self._last_enrichment["indexing_source"] = source
                self._last_enrichment["indexing_verified"] = has_tool_results
                log.info("Indexed: %d entities, %d facts, source=%s",
                         len(result.entities), len(result.facts), source)
            # Merge S1.5 debug data into enrichment for telemetry
            s15_debug = getattr(self._acervo, "_last_s15_debug", None)
            if s15_debug and isinstance(self._last_enrichment.get("debug"), dict):
                self._last_enrichment["debug"].update(s15_debug)
        except Exception as e:
            log.warning("process() failed: %s", e)

    # ── Helpers ──

    def _extract_text_anthropic(self, message: dict) -> str:
        """Extract text from an Anthropic message (string or content blocks)."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    def _build_history_anthropic(self, body: dict) -> list[dict]:
        """Convert Anthropic request body to simple history format for Acervo."""
        history: list[dict] = []

        system = body.get("system", "")
        if isinstance(system, list):
            system = " ".join(
                b.get("text", "") for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if system:
            history.append({"role": "system", "content": system})

        for msg in body.get("messages", []):
            text = self._extract_text_anthropic(msg)
            if text:
                history.append({"role": msg["role"], "content": text})

        return history
