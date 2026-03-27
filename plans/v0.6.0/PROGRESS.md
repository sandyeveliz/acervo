# v0.6.0 Progress Tracker

> Updated: 2026-03-27

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

---

## M1 — Docker Compose

- [ ] `docker-compose.yml` (proxy + Ollama + model + ChromaDB)
- [ ] GPU passthrough (nvidia-docker)
- [ ] Persistent volumes
- [ ] "Try Acervo in 30 seconds" README section
- [ ] Smoke test: `docker compose up` -> conversation works

---

## M2 — Progressive retrieval

- [ ] Insufficient context detection (low confidence, "no info" response)
- [ ] Auto-escalation: hot -> warm -> cold
- [ ] Configurable budget per layer
- [ ] Metrics: escalation_count, escalation_reason
- [ ] Tests: escalation triggers correctly

---

## M3 — Runtime metrics

- [ ] `GET /acervo/metrics` endpoint (Prometheus-compatible)
- [ ] Tokens saved metric
- [ ] Compression ratio metric
- [ ] Latency metric (prepare_ms, process_ms)
- [ ] Hit rate metric
- [ ] Acervo Studio minimal dashboard
- [ ] Configurable alerts

---

## M4 — Advanced error recovery

- [ ] Automatic graph backup before destructive ops
- [ ] `acervo graph rollback` command
- [ ] LLM down -> cache last known graph context
- [ ] Tests: rollback restores previous state

---

## Release Criteria
- [ ] `docker compose up` -> full stack running with GPU
- [ ] Progressive retrieval escalates automatically
- [ ] Prometheus metrics endpoint working
- [ ] Graph rollback working
