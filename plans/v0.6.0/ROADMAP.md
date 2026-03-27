# v0.6.0 — "Production-ready"

> Docker, runtime metrics, progressive retrieval. What someone needs
> to use Acervo in a deployed service, not just on their local machine.

---

## M1 — Docker Compose

- `docker-compose.yml`: Acervo proxy + Ollama + model + ChromaDB
- `docker compose up` -> everything running
- GPU passthrough (nvidia-docker)
- Volumes for persistence
- README: "Try Acervo in 30 seconds"

---

## M2 — Progressive retrieval

- Hot layer insufficient -> automatically escalate to warm
- Insufficient context detection (low topic confidence,
  LLM says "I don't have info")
- Configurable budget per layer
- Metrics: escalation_count, escalation_reason

---

## M3 — Runtime metrics

- `GET /acervo/metrics` — Prometheus-compatible
- Tokens saved, compression ratio, latency, hit rate
- Minimal dashboard in Acervo Studio
- Configurable alerts

---

## M4 — Advanced error recovery

- Automatic graph backup before destructive operations
- `acervo graph rollback` — revert to previous snapshot
- Improved graceful degradation (LLM unresponsive -> cache
  last known graph context)
