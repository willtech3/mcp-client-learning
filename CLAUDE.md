# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

An educational MCP **client** targeting protocol revision 2025-11-25,
built on the official `mcp` Python SDK. Sibling of
`../mcp-learning/virtual-library-mcp` (the server it is demoed against).

## Critical rules

1. **Use `just` commands** for all workflows (`just test`, `just lint`, `just typecheck`, `just check`).
2. **Never `git add .`** — add files by name.
3. **Never commit secrets.** Tokens live in the OS keychain; the Anthropic key comes from `ANTHROPIC_API_KEY`.
4. **Read the MCP spec before protocol changes** — current revision 2025-11-25 at modelcontextprotocol.io; verify SDK shapes by introspection (`uv run python -c "import mcp.types as t; ..."`), not memory.
5. **Security invariants are non-negotiable**: PKCE S256 only, constant-time state comparison, HTTPS-only discovery (localhost exempt), loopback-only callback listener, tokens never logged. Tests in `tests/test_oauth.py` encode these — they must never be weakened to make a change pass.
6. **All tests pass + lint clean before committing.**
7. **Keep it modular without over-abstraction** — one module per concern, plain functions over frameworks.

## Stack

Python 3.12+, uv, official `mcp` SDK, anthropic, httpx, keyring,
typer + rich, pytest, ruff (with bandit `S` rules), pyright.

## Architecture

- `oauth.py` — OAuth 2.1 flow (discovery, DCR, PKCE, refresh) as `httpx.Auth`
- `callback_server.py` — loopback redirect listener (RFC 8252)
- `token_store.py` — keychain/file token persistence
- `handlers.py` — sampling (Anthropic; SEP-1577 tools), elicitation, logging
- `session.py` — transport + capabilities + lifecycle
- `cli.py` — typer commands incl. `demo` (the guided protocol tour)
