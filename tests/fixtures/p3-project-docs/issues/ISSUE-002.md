# ISSUE-002: Overdue todos not visually distinguished

**Type:** Enhancement
**Priority:** Medium
**Assignee:** Martin
**Status:** In Progress
**Created:** 2026-03-25

## Description

Todos with a `due_date` in the past should be visually distinct (red border, warning icon) so users can quickly identify overdue items. Currently, the `TodoItem` component has the CSS class `todo-overdue` but the logic only applies to the left border color.

## Acceptance Criteria

- [ ] Overdue todos show a warning icon next to the due date
- [ ] The due date text turns red for overdue items
- [ ] Overdue filter option added to the filter bar
- [ ] Overdue count shown in the stats dashboard

## Related Files

- `frontend/src/components/TodoItem.tsx` — line 32, `isOverdue` logic
- `frontend/src/styles/todo.css` — `.todo-overdue` class
- `frontend/src/hooks/useTodos.ts` — filter state
