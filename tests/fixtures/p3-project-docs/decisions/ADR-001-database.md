# ADR-001: Use SQLite for initial development

**Date:** 2026-03-20
**Status:** Accepted
**Deciders:** Sandy, Lucas

## Context

We need a database for the Todo App. Options considered: PostgreSQL, MySQL, SQLite, MongoDB.

## Decision

Use **SQLite** via `better-sqlite3` for v0.1.0 through v0.3.0. Migrate to PostgreSQL for v1.0.0 production launch.

## Rationale

1. **Zero setup** — No database server to install or configure. The database is a single file.
2. **Synchronous API** — `better-sqlite3` is synchronous, which simplifies the model layer (no async/await for queries).
3. **Performance** — For single-user or small team use, SQLite outperforms client-server databases due to zero network overhead.
4. **Portability** — Database file can be copied, backed up, or shared trivially.
5. **WAL mode** — Write-Ahead Logging allows concurrent reads during writes, sufficient for our scale.

## Consequences

### Positive
- Faster development cycle (no database server management)
- Simpler CI/CD (no database container needed for tests)
- Easy to inspect data (single file, can use DB Browser for SQLite)

### Negative
- **No concurrent writes** — SQLite locks the entire database during writes. This limits us to one write at a time.
- **No built-in replication** — Can't scale horizontally.
- **Migration effort** — Moving to PostgreSQL for v1.0.0 will require rewriting SQL queries (SQLite-specific syntax like `datetime('now')`).

### Migration Plan

For v1.0.0, we'll:
1. Replace `better-sqlite3` with `pg` (node-postgres)
2. Convert SQLite-specific SQL to PostgreSQL syntax
3. Add a migration script to transfer existing data
4. The model layer interface stays the same — only the implementation changes

## Related

- `backend/src/config/database.ts` — Database initialization and migrations
- `backend/src/models/` — All model files use `getDatabase()` for queries
