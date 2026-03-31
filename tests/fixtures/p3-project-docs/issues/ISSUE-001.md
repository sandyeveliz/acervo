# ISSUE-001: Todo title allows HTML injection

**Type:** Bug (Security)
**Priority:** High
**Assignee:** Sandy
**Status:** Open
**Created:** 2026-03-25

## Description

The `createTodo` endpoint accepts raw HTML in the `title` field. When rendered in the frontend, the HTML is interpreted by the browser, allowing potential XSS attacks.

## Steps to Reproduce

1. POST /api/todos with body: `{"title": "<script>alert('xss')</script>"}`
2. The todo is created successfully
3. When the TodoItem component renders the title, the script executes

## Expected Behavior

HTML should be escaped or stripped before storage. The title field should only contain plain text.

## Proposed Fix

Add input sanitization middleware in `backend/src/middleware/` that strips HTML tags from string fields. Apply to all POST/PUT endpoints.

Alternative: Use React's default JSX escaping (which already prevents this in most cases) but also sanitize server-side for API consumers.

## Related Files

- `backend/src/services/todo.service.ts` — createTodo validation
- `backend/src/middleware/` — needs new sanitization middleware
- `frontend/src/components/TodoItem.tsx` — renders title
