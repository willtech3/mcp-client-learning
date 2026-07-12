# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

An educational MCP **client**, dual-era: protocol 2026-07-28 ("modern",
implemented from scratch in `mcp_client/modern/`) with fallback to
2025-11-25 and earlier ("legacy", via the official `mcp` Python SDK).
Sibling of `../mcp-learning/virtual-library-mcp` (the server it is
demoed against).

## Critical rules

1. **Use `just` commands** for all workflows (`just test`, `just lint`, `just typecheck`, `just check`).
2. **Never `git add .`** — add files by name.
3. **Never commit secrets.** Tokens live in the OS keychain; the Anthropic key comes from `ANTHROPIC_API_KEY`.
4. **Read the MCP spec before protocol changes** — the modern era targets **2026-07-28** (<https://modelcontextprotocol.io/specification/draft>, wire string `2026-07-28`); the legacy era is 2025-11-25 and earlier. Verify SDK shapes by introspection (`uv run python -c "import mcp.types as t; ..."`), not memory. The `modern/` package hand-implements the wire protocol — do not assume the pinned `mcp` 1.x SDK supports the modern era (it does not; that's `mcp==2.0.0b1`).
5. **Security invariants are non-negotiable**: PKCE S256 only, constant-time state comparison, HTTPS-only discovery (localhost exempt), loopback-only callback listener, tokens never logged. Tests in `tests/test_oauth.py` encode these — they must never be weakened to make a change pass.
6. **All tests pass + lint clean before committing.**
7. **Keep it modular without over-abstraction** — one module per concern, plain functions over frameworks.

## Stack

Python 3.12+, uv, official `mcp` SDK, anthropic, httpx, keyring,
typer + rich, pytest, ruff (with bandit `S` rules), pyright.

## Architecture

- `oauth.py` — OAuth 2.1 flow (discovery, CIMD/DCR, PKCE, RFC 9207 iss,
  issuer-keyed credentials, scope step-up) as `httpx.Auth`
- `callback_server.py` — loopback redirect listener (RFC 8252)
- `token_store.py` — keychain/file persistence (tokens by server URL,
  client registrations by AS issuer per SEP-2352)
- `handlers.py` — sampling (Anthropic; SEP-1577 tools), elicitation,
  logging, plus MRTR wire adapters for the modern era
- `session.py` — legacy transport + capabilities + lifecycle (official SDK)
- `modern/` — protocol 2026-07-28 from scratch:
  - `types.py` — constants, `_meta` keys, error codes, base64 sentinel codec
  - `transport.py` — Streamable HTTP POSTs + required headers + SSE parser
  - `session.py` — ModernSession with the MRTR driver and x-mcp-header duty
  - `compat.py` — era detection (`server/discover` probe) + unified connect
- `cli.py` — typer commands incl. `demo` (the guided protocol tour),
  `discover`, `skills` (SEP-2640 digest verification), `listen`
