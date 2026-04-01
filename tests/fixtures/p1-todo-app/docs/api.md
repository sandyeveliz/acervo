# REST API Reference

## Authentication

All endpoints except `/api/auth/login` and `/api/auth/register` require a Bearer token in the `Authorization` header.

### POST /api/auth/register

Create a new user account.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepassword",
  "name": "John Doe"
}
```

**Response (201):**
```json
{
  "data": {
    "user": { "id": 1, "email": "user@example.com", "name": "John Doe" },
    "token": "eyJhbGciOiJIUzI1NiIs..."
  }
}
```

**Errors:**
- `400` — Email already registered, password too short

### POST /api/auth/login

Authenticate and receive a JWT token.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response (200):**
```json
{
  "data": {
    "user": { "id": 1, "email": "user@example.com", "name": "John Doe" },
    "token": "eyJhbGciOiJIUzI1NiIs..."
  }
}
```

**Errors:**
- `401` — Invalid email or password

### GET /api/auth/me

Get current user profile. Requires authentication.

**Response (200):**
```json
{
  "data": { "id": 1, "email": "user@example.com", "name": "John Doe", "created_at": "2024-01-15T10:30:00" }
}
```

---

## Todos

All todo endpoints require authentication.

### GET /api/todos

List all todos for the authenticated user.

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `completed` | boolean | Filter by completion status |
| `priority` | string | Filter by priority: `low`, `medium`, `high` |
| `search` | string | Search in title and description |

**Response (200):**
```json
{
  "data": [
    {
      "id": 1,
      "user_id": 1,
      "title": "Buy groceries",
      "description": "Milk, eggs, bread",
      "completed": false,
      "priority": "medium",
      "due_date": "2024-01-20",
      "created_at": "2024-01-15T10:30:00",
      "updated_at": "2024-01-15T10:30:00"
    }
  ],
  "count": 1
}
```

### POST /api/todos

Create a new todo.

**Request:**
```json
{
  "title": "Buy groceries",
  "description": "Milk, eggs, bread",
  "priority": "high",
  "due_date": "2024-01-20"
}
```

Only `title` is required. Defaults: `priority = "medium"`, `description = ""`.

**Response (201):** Returns the created todo object.

**Errors:**
- `400` — Empty title, title too long (>200), invalid date

### GET /api/todos/:id

Get a single todo by ID.

**Response (200):** Returns the todo object.

**Errors:**
- `404` — Todo not found or belongs to another user

### PUT /api/todos/:id

Update a todo. Only include fields to change.

**Request:**
```json
{
  "completed": true,
  "priority": "low"
}
```

**Response (200):** Returns the updated todo object.

### DELETE /api/todos/:id

Delete a todo.

**Response:** `204 No Content`

**Errors:**
- `404` — Todo not found

### GET /api/todos/stats

Get aggregated statistics for the authenticated user.

**Response (200):**
```json
{
  "data": {
    "total": 15,
    "completed": 8,
    "pending": 7,
    "completionRate": 53
  }
}
```

---

## Error Format

All errors return a JSON body:

```json
{
  "error": "Human-readable error message"
}
```

## Health Check

### GET /api/health

No authentication required.

```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```
