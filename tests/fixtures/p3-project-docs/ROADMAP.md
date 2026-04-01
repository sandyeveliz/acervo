# Roadmap

## v0.1.0 — Auth + Basic CRUD (Target: 2026-04-07)

### Backend

- [x] SQLite database setup with migrations (`backend/src/config/database.ts`)
- [x] User model with email/password fields (`backend/src/models/user.model.ts`)
- [x] Todo model with CRUD operations (`backend/src/models/todo.model.ts`)
- [x] JWT authentication service (`backend/src/services/auth.service.ts`)
- [x] Auth middleware for protected routes (`backend/src/middleware/auth.middleware.ts`)
- [x] Auth routes: POST /login, POST /register (`backend/src/routes/auth.routes.ts`)
- [x] Todo routes: GET, POST, PUT, DELETE (`backend/src/routes/todo.routes.ts`)
- [ ] Input sanitization middleware (prevent XSS in title/description)
- [ ] Rate limiting on auth endpoints (5 attempts per minute)

### Frontend

- [x] Login form component
- [x] Todo list with add/toggle/delete (`frontend/src/components/TodoList.tsx`)
- [x] Individual todo item display (`frontend/src/components/TodoItem.tsx`)
- [x] Create todo form (`frontend/src/components/TodoForm.tsx`)
- [ ] Error toast notifications
- [ ] Loading skeleton components

### Infrastructure

- [ ] Docker Compose for local development
- [ ] GitHub Actions CI (lint + test on PR)
- [ ] Seed script for demo data

---

## v0.2.0 — Filters + Stats (Target: 2026-04-21)

### Backend

- [ ] Pagination support (limit/offset on GET /api/todos)
- [ ] Sorting options (by date, priority, title)
- [ ] Bulk operations endpoint (PATCH /api/todos/bulk)
- [ ] Todo categories/tags (new table, many-to-many)

### Frontend

- [ ] Filter bar component (by status, priority, date range)
- [ ] Search input with debounce
- [ ] Stats dashboard (completion rate chart, priority breakdown)
- [ ] Keyboard shortcuts (n = new, j/k = navigate, x = toggle)

### API Changes

- [ ] `GET /api/todos?page=1&limit=20&sort=created_at:desc`
- [ ] `GET /api/todos/stats` — aggregated metrics
- [ ] `PATCH /api/todos/bulk` — `{ ids: [1,2,3], action: "complete" }`

---

## v0.3.0 — Notifications + Mobile (Target: 2026-05-12)

### Backend

- [ ] Due date reminder system (check every hour)
- [ ] WebSocket for real-time updates (when other clients modify)
- [ ] Email notifications (SendGrid integration)
- [ ] Export todos as CSV/JSON

### Frontend

- [ ] Push notifications for due dates (browser Notification API)
- [ ] Responsive design for mobile
- [ ] Offline support with service worker
- [ ] Dark mode toggle

---

## v1.0.0 — Public Launch (Target: 2026-06-01)

### Requirements

- [ ] All v0.1-v0.3 features complete
- [ ] 90%+ test coverage
- [ ] Performance: <200ms API response time
- [ ] Security audit (OWASP top 10 checklist)
- [ ] Landing page with demo
- [ ] Documentation site

### Deployment

- [ ] Production database (PostgreSQL migration)
- [ ] CDN for static assets
- [ ] Monitoring (error tracking, uptime)
- [ ] Automated backups
