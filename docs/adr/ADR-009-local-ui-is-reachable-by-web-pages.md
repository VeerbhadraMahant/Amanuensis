# ADR-009: The local UI is treated as reachable by hostile web pages

Status: accepted

## Context
The correction UI listens on 127.0.0.1. That does not make it private from the browser: any web page the owner
visits can send requests to `http://127.0.0.1:8765`, and DNS rebinding can make a hostile hostname resolve to it.
The UI can approve a model promotion and approve lexicon entries (which feed prompt biasing), so those endpoints
must not be clickable by an arbitrary page. A code review found them open to simple cross-site POSTs.

## Options considered
1. Rely on binding to localhost.
2. A per-launch secret token in the page and every request.
3. Refuse foreign `Host` headers and require a custom header on every state-changing request.

## Decision
Option 3, as middleware on every route. A browser will not send a custom header on a cross-site request without a
CORS preflight, and the server never grants one. The Host check blocks DNS rebinding. Reads (GET) stay open to the
local page. Tests cover a bare cross-site-style POST, a foreign Host, and the real client path.

## Consequences
- Anything that scripts the API (curl, tests) must send `X-Amanuensis-UI: 1` and a local Host.
- This is not authentication: another local process can still call the API. For a single-user machine that is
  accepted. A token (option 2) is the next step if the UI is ever exposed beyond localhost.
