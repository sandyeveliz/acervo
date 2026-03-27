# v0.5.0 Progress Tracker

> Updated: 2026-03-27

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

---

## M1 — MCP Server

- [ ] MCP server package (`acervo-mcp`)
- [ ] Tool: `acervo_prepare` — enrich context before LLM call
- [ ] Tool: `acervo_process` — extract knowledge after LLM response
- [ ] Tool: `acervo_index` — index a file/directory
- [ ] Tool: `acervo_search` — search the knowledge graph
- [ ] Tool: `acervo_status` — graph stats
- [ ] Resource: `acervo://graph` — current graph state
- [ ] Resource: `acervo://nodes/{id}` — node detail
- [ ] Resource: `acervo://traces/latest` — last conversation trace
- [ ] Test with Claude Desktop
- [ ] Test with Cursor
- [ ] `npx acervo-mcp` or `uvx acervo-mcp` launcher
- [ ] Documentation: MCP setup guide

---

## M2 — TypeScript SDK

- [ ] `acervo-client` npm package
- [ ] REST API wrapper with full types
- [ ] Example: Vercel AI SDK + Acervo memory
- [ ] Example: Next.js chatbot with persistent memory
- [ ] README + npm publish

---

## M3 — LangChain / LlamaIndex integrations

- [ ] `AcervoMemory` — LangChain ConversationBufferMemory drop-in
- [ ] `AcervoRetriever` — LlamaIndex retriever drop-in
- [ ] Integration tests
- [ ] Documentation + examples

---

## M4 — Blog post v0.4.0 + benchmarks

- [ ] Blog post with indexation domain results
- [ ] Visual comparison: Acervo vs plain RAG
- [ ] LinkedIn/X posts
- [ ] GitHub Pages benchmark reports

---

## Release Criteria
- [ ] MCP server works with Claude Desktop + Cursor
- [ ] TypeScript SDK published on npm
- [ ] LangChain/LlamaIndex drop-in integrations working
- [ ] Blog post published
