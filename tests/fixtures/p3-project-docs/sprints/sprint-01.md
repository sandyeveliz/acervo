# Sprint 1 — Foundation

**Period:** 2026-03-24 → 2026-04-07 (2 weeks)
**Goal:** Working auth + todo CRUD with basic frontend

## Backlog

### Backend (Sandy + Lucas)

| ID | Task | Points | Assignee | Status |
|----|------|--------|----------|--------|
| BE-01 | SQLite setup + migrations | 2 | Lucas | Done |
| BE-02 | User model + auth service | 3 | Lucas | Done |
| BE-03 | Todo model + service layer | 3 | Sandy | Done |
| BE-04 | REST routes + controllers | 2 | Sandy | Done |
| BE-05 | Auth middleware (JWT) | 2 | Lucas | Done |
| BE-06 | Input sanitization | 2 | Sandy | In Progress |
| BE-07 | Rate limiting | 1 | Sandy | Not Started |

### Frontend (Martin)

| ID | Task | Points | Assignee | Status |
|----|------|--------|----------|--------|
| FE-01 | Login/register forms | 2 | Martin | Done |
| FE-02 | Todo list + item components | 3 | Martin | Done |
| FE-03 | Create todo form | 2 | Martin | Done |
| FE-04 | useTodos hook + API service | 3 | Martin | Done |
| FE-05 | Error toast notifications | 2 | Martin | Blocked (ISSUE-005) |
| FE-06 | Loading skeleton components | 1 | Martin | Not Started |

### Infrastructure

| ID | Task | Points | Assignee | Status |
|----|------|--------|----------|--------|
| INF-01 | Docker Compose | 2 | Sandy | Blocked (ISSUE-003) |

### Testing

| ID | Task | Points | Assignee | Status |
|----|------|--------|----------|--------|
| TEST-01 | Todo service unit tests | 2 | Lucas | Done |
| TEST-02 | Auth service unit tests | 2 | Lucas | In Progress |

## Retrospective Notes

(To be filled at sprint end)

### What went well
- Backend CRUD was straightforward with better-sqlite3
- Martin's component design was clean first pass

### What could improve
- Docker setup should have been validated before sprint planning
- Need to decide on notification library earlier — blocking FE work

### Action Items
- [ ] Sandy: Set up Podman as Docker alternative by next sprint
- [ ] Martin: Make notification library decision by 2026-03-28
- [ ] Lucas: Finish auth service tests before starting Sprint 2
