# ADR-002: JWT-based authentication

**Date:** 2026-03-20
**Status:** Accepted
**Deciders:** Sandy, Lucas

## Context

The Todo App needs user authentication. Options: session cookies, JWT tokens, OAuth 2.0.

## Decision

Use **JWT tokens** stored in localStorage, with a 24-hour expiry.

## Rationale

1. **Stateless** — No server-side session store needed. Each request carries its own authentication.
2. **API-first** — Works naturally with REST APIs. The frontend sends `Authorization: Bearer <token>`.
3. **Simple implementation** — `jsonwebtoken` library handles signing and verification.
4. **Decoupled** — Backend doesn't need to track active sessions.

## Security Considerations

- **localStorage vs httpOnly cookie**: We chose localStorage for simplicity. This means the token is accessible to JavaScript (XSS risk). For v1.0.0, we should switch to httpOnly cookies.
- **Token expiry**: 24 hours is a balance between convenience and security. Refresh tokens are not implemented in v0.1.0.
- **Password hashing**: Using bcrypt with 10 salt rounds. This is standard and resistant to brute force.

## Consequences

### Positive
- No session table or Redis needed
- Works with any frontend (web, mobile, CLI)
- Easy to test (just include the header)

### Negative
- Cannot revoke tokens (until they expire)
- Token stored in localStorage is vulnerable to XSS
- No refresh token flow — user must re-login after 24h

## Related

- `backend/src/services/auth.service.ts` — Token generation and verification
- `backend/src/middleware/auth.middleware.ts` — Token extraction from headers
- `backend/src/config/env.ts` — JWT secret and expiry configuration
- ISSUE-004 — Token expiry causes silent logout (UX bug)
