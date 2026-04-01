# ISSUE-005: Notification library decision needed

**Type:** Decision
**Priority:** Medium
**Assignee:** Martin
**Status:** Open
**Created:** 2026-03-26
**Deadline:** 2026-03-28

## Context

FE-05 (Error toast notifications) is blocked because we haven't decided on a notification library. This blocks the error handling UX for the entire frontend.

## Options Evaluated

### react-hot-toast
- **Pros:** Lightweight (5kb), simple API, good defaults
- **Cons:** Limited customization, no built-in promise handling for async operations
- **Example:** `toast.success("Todo created!")`

### sonner
- **Pros:** Beautiful default styles, promise API built-in, rich content support
- **Cons:** Slightly larger (8kb), newer library (less battle-tested)
- **Example:** `toast.promise(createTodo(input), { loading: "Creating...", success: "Done!", error: "Failed" })`

### Custom implementation
- **Pros:** Full control, no dependency
- **Cons:** More development time, need to handle animations, stacking, dismissal

## Recommendation

**sonner** — the promise API is exactly what we need for async operations (creating/deleting todos), and the default styles match our design system. The 3kb size difference is negligible.

## Decision

Pending Martin's final evaluation. Must decide by 2026-03-28 to unblock FE-05.
