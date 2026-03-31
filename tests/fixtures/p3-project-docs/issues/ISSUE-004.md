# ISSUE-004: Token expiry causes silent logout

**Type:** Bug
**Priority:** High
**Assignee:** Martin
**Status:** Open
**Created:** 2026-03-26

## Description

When the JWT token expires (after 24h), API calls fail with 401 but the frontend doesn't handle this gracefully. The user sees generic "Failed to fetch todos" error instead of being redirected to login.

## Steps to Reproduce

1. Login and use the app normally
2. Wait 24 hours (or manually set a short expiry for testing)
3. Try to create or list todos
4. Frontend shows error message, not a login redirect

## Expected Behavior

On 401 response:
1. Clear the stored token
2. Redirect to login page
3. Show a message: "Session expired. Please login again."

## Proposed Fix

Add a response interceptor in `frontend/src/services/api.ts` that catches 401 responses and triggers the auth cleanup:

```typescript
if (response.status === 401) {
  localStorage.removeItem("auth_token");
  window.location.href = "/login";
}
```

## Related Files

- `frontend/src/services/api.ts` — `request()` function
- `frontend/src/hooks/useAuth.ts` — logout logic
- `backend/src/services/auth.service.ts` — token expiry config
