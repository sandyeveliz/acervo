# v0.5.0 — "Usable from anywhere"

> MCP server as the killer feature. If you use Claude, Cursor, or any IDE
> with MCP, Acervo works without changing anything.

---

## M1 — MCP Server

**The most important integration.** An Acervo MCP server exposing:

```
Tools:
  acervo_prepare    — enrich context before LLM call
  acervo_process    — extract knowledge after LLM response
  acervo_index      — index a file/directory
  acervo_search     — search the knowledge graph
  acervo_status     — graph stats

Resources:
  acervo://graph          — current graph state
  acervo://nodes/{id}     — node detail
  acervo://traces/latest  — last conversation trace
```

- Compatible with Claude Desktop, Cursor, Windsurf, Continue.dev
- User installs Acervo, adds the MCP server to their config,
  and has persistent memory in any tool
- `npx acervo-mcp` or `uvx acervo-mcp` to start

**Why MCP first:**
- Zero friction: doesn't change the user's framework
- Perfect audience: developers using AI IDEs
- Doesn't compete with LangChain/LlamaIndex, complements them
- A well-made MCP server IS the marketing — people share it

---

## M2 — TypeScript SDK

- `npm install acervo-client`
- REST API wrapper for the JS/TS ecosystem
- Types for all endpoints
- Example: Vercel AI SDK + Acervo memory
- Example: Next.js chatbot with persistent memory
- Needed so the MCP server has a native Node client

---

## M3 — LangChain / LlamaIndex integrations

Secondary to MCP, but expands reach:

- `AcervoMemory` — drop-in for LangChain ConversationBufferMemory
- `AcervoRetriever` — drop-in for LlamaIndex retriever
- Competitive advantage: same interface, but with knowledge graph
  compression instead of raw chunks

---

## M4 — Blog post v0.4.0 + indexation benchmarks

- Blog: results from the 4 indexation domains
- Visual comparison: Acervo indexation vs plain RAG
- LinkedIn/X: one hero chart per domain
- GitHub Pages: indexation benchmark reports
