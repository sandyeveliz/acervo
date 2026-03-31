# Architecture

## Overview

The Todo App follows a classic fullstack architecture with clear separation of concerns:

- **Backend**: Express.js REST API with layered architecture (Controller → Service → Model)
- **Frontend**: React SPA with hooks-based state management
- **Database**: SQLite via better-sqlite3 (synchronous, embedded, zero-config)

## Backend Layers

### Controllers (`backend/src/controllers/`)

Controllers handle HTTP request/response. They:
- Parse and validate request parameters
- Call the appropriate service method
- Format the response (JSON envelope: `{ data: ... }`)
- Map service errors to HTTP status codes

Controllers never contain business logic or database queries.

### Services (`backend/src/services/`)

Services contain all business logic:
- Input validation (title length, date format, email format)
- Authorization checks (user owns the resource)
- Orchestration (multi-step operations)
- Custom error types (ValidationError, NotFoundError, AuthError)

Services are the only layer that throws domain-specific errors.

### Models (`backend/src/models/`)

Models are the data access layer:
- Direct SQL queries via better-sqlite3
- Return plain TypeScript interfaces (no ORM classes)
- Handle data normalization (e.g., SQLite integer → boolean)
- No business logic — just CRUD operations

### Middleware (`backend/src/middleware/`)

Express middleware for cross-cutting concerns:
- **auth.middleware.ts**: JWT token verification, sets `req.userId`
- **error.middleware.ts**: Global error handler, 404 catch-all

## Frontend Architecture

### Components (`frontend/src/components/`)

Presentational React components:
- **Header**: Navigation bar with user info and logout
- **TodoList**: Container for todo items with filtering
- **TodoItem**: Single todo display with toggle/edit/delete actions
- **TodoForm**: Create new todo form

### Hooks (`frontend/src/hooks/`)

Custom hooks encapsulate all state management:
- **useTodos**: CRUD operations, optimistic updates, filter state
- **useAuth**: Login/register/logout, token management

### Services (`frontend/src/services/`)

API client layer:
- **api.ts**: Centralized fetch wrapper with auth header injection, error handling, type-safe responses

## Key Decisions

### Why SQLite?

- Zero configuration — no separate database server
- Perfect for single-user or small team use
- WAL mode for concurrent reads during writes
- Data is a single file — easy backup and migration

### Why JWT for auth?

- Stateless — no session store needed
- Token contains user ID — no database lookup per request
- Short expiry (24h) limits exposure if compromised

### Why separate controller/service/model?

- Controllers are thin — easy to test services in isolation
- Services can be reused (CLI, background jobs, etc.)
- Models can be swapped (e.g., SQLite → PostgreSQL) without touching business logic

## Data Flow

```
User Action → React Component → Custom Hook → API Service → Express Controller
                                                               ↓
                                                         Service Layer
                                                               ↓
                                                         Model (SQLite)
                                                               ↓
                                                         Response JSON
                                                               ↓
                                               Custom Hook (state update)
                                                               ↓
                                               React Component (re-render)
```
