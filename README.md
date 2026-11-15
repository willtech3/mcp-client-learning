# mcp-client-learning

A full, modern, **secure** MCP client built from first principles — the
client-side companion to [mcp-learning](https://github.com/willtech3/mcp-learning)'s
Virtual Library server. **Dual-era:** it speaks the current stateless
**[2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28)** protocol
(implemented from scratch in [`mcp_client/modern/`](mcp_client/modern/)) and
falls back to **2025-11-25** and earlier via the official Python SDK. The era
is auto-detected with a `server/discover` probe.

> **Spec:** <https://modelcontextprotocol.io/specification/2026-07-28> ·
> [changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog) ·
> [MRTR](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr) ·
> [transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http) ·
> [authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) ·
> [skills (SEP-2640)](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2640)

Where most examples hide the interesting parts inside a library, this repo
implements them in the open:

| Capability | Where | What you'll learn |
|---|---|---|
| 2026-07-28 protocol | `mcp_client/modern/` | Stateless transport, per-request `_meta`, `server/discover`, the **MRTR** driver, `subscriptions/listen`, `Mcp-Method`/`Mcp-Name`/`x-mcp-header` headers |
| Era detection | `mcp_client/modern/compat.py` | `server/discover` probe → modern; `-32022` → modern with a supported version; otherwise legacy fallback |
| OAuth 2.1 from scratch | `mcp_client/oauth.py` | RFC 9728 discovery → RFC 8414/OIDC metadata (issuer-validated) → **CIMD** / RFC 7591 registration → Authorization Code + **PKCE S256** → RFC 8707 resource → **RFC 9207 `iss`** → issuer-keyed credentials (SEP-2352) → scope step-up → refresh |
| Loopback redirect | `mcp_client/callback_server.py` | RFC 8252 native-app flow: 127.0.0.1, ephemeral port, single-use, `iss` capture |
| Secure token storage | `mcp_client/token_store.py` | OS keychain via `keyring`, 0600-file fallback; client registrations keyed by AS issuer |
| Sampling handler | `mcp_client/handlers.py` | Serving `sampling/createMessage` with the Anthropic API, model-hint mapping, human-in-the-loop approval, **tool-enabled sampling (SEP-1577)** |
| Elicitation handler | `mcp_client/handlers.py` | Form fields, enum selects, URL mode, accept/decline/cancel — routed through MRTR in the modern era |
| Skills verification | `mcp_client/cli.py` (`skills`) | SEP-2640 host duties: `skill://index.json`, **sha256 digest + frontmatter verification**, refuse-on-mismatch |
| Session orchestration | `mcp_client/session.py`, `modern/session.py` | Capability advertising, MRTR loop, stdio + Streamable HTTP |

## Quickstart

```bash
just install

# Against a local dual-era Virtual Library server (no auth). The client probes
# server/discover and picks the 2026-07-28 modern path automatically:
#   in mcp-learning/virtual-library-mcp:
#     VIRTUAL_LIBRARY_TRANSPORT=http VIRTUAL_LIBRARY_ALLOW_INSECURE_HTTP=true uv run python server.py
mcp-client discover --server http://127.0.0.1:8080/mcp --anonymous   # capabilities + extensions
mcp-client tools    --server http://127.0.0.1:8080/mcp --anonymous
mcp-client skills   --server http://127.0.0.1:8080/mcp --anonymous   # SEP-2640 digest-verified
mcp-client call search_catalog --args '{"query": "1984"}' \
  --server http://127.0.0.1:8080/mcp --anonymous

# Against an OAuth-protected server (browser opens for sign-in):
mcp-client demo --server https://<host>/mcp
```

For sampling demos, export `ANTHROPIC_API_KEY` — the *client* pays for
tokens, which is exactly the point of MCP sampling.

## The guided demo

`mcp-client demo` walks the entire protocol against the Virtual Library:
typed tool schemas → structured search output → resource reads →
**elicitation** (the server asks you which membership term to apply) →
**sampling** (the server asks *your* Claude to write book insights, with
the catalog-search tool attached) → progress + `list_changed`
notifications during catalog maintenance.

## Security model

- PKCE S256 only; servers not advertising it are refused (no downgrade)
- `state` validated with constant-time comparison; mismatch aborts
- HTTPS required for every discovered endpoint (localhost exempt)
- Redirect listener binds 127.0.0.1 on an OS-assigned port, accepts one hit
- Tokens: OS keychain (or 0600 file), never logged, never in argv
- Tokens are resource-bound (RFC 8707) — a token for one server is useless at another
- Sampling requests require human approval before tokens are spent
  (set `MCP_CLIENT_AUTO_APPROVE_SAMPLING=true` for unattended demos)

## Development

```bash
just test        # pytest
just lint        # ruff (incl. bandit security rules)
just typecheck   # pyright
just check       # all of the above
```
