# Todo App

A fullstack TODO application with Node/Express backend and React frontend.

## Architecture

- **Backend**: Express.js + SQLite (better-sqlite3)
- **Frontend**: React + TypeScript + Vite
- **Database**: SQLite with migration system

## Project Structure

```
backend/
  src/
    controllers/   — Request handlers (parse input, call services, format response)
    services/      — Business logic (validation, orchestration)
    models/        — Database access layer (SQL queries, data mapping)
    middleware/    — Express middleware (auth, error handling, validation)
    routes/        — Route definitions (URL mapping to controllers)
    config/        — App configuration (database, environment)
  tests/           — Unit and integration tests

frontend/
  src/
    components/    — React components (TodoList, TodoItem, Header, etc.)
    hooks/         — Custom React hooks (useTodos, useAuth)
    services/      — API client and data fetching
    types/         — TypeScript type definitions
    styles/        — CSS modules and global styles

docs/
  architecture.md  — Technical architecture decisions
  api.md           — REST API documentation
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/todos | List all todos (with filters) |
| POST | /api/todos | Create a new todo |
| GET | /api/todos/:id | Get a single todo |
| PUT | /api/todos/:id | Update a todo |
| DELETE | /api/todos/:id | Delete a todo |
| POST | /api/auth/login | Authenticate user |
| POST | /api/auth/register | Register new user |

## Getting Started

```bash
# Backend
cd backend && npm install && npm run dev

# Frontend
cd frontend && npm install && npm run dev
```
