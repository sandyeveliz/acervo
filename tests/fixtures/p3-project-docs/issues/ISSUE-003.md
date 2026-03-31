# ISSUE-003: Docker Compose setup blocked

**Type:** Infrastructure
**Priority:** Medium
**Assignee:** Sandy
**Status:** Blocked
**Created:** 2026-03-26

## Description

The Docker Compose configuration for local development cannot be tested because Sandy's machine doesn't have a Docker Desktop license (company policy requires paid licenses for commercial use). Need to either get the license or switch to an alternative.

## Options

1. **Podman** — Drop-in Docker replacement, no license needed. Requires adjusting compose files to use `podman-compose`.
2. **Docker Desktop license** — Request through IT. ETA: 1-2 weeks.
3. **Skip Docker for now** — Developers run services directly. Less consistent environments.

## Decision

Pending — Sandy to evaluate Podman compatibility by 2026-03-31.

## Impact

- INF-01 task blocked in Sprint 1
- No containerized development environment for new team members
- CI/CD pipeline will need Docker regardless (GitHub Actions has Docker)

## Related

- Sprint 1 backlog: INF-01
