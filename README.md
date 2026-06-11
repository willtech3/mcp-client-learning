# mcp-client-learning

A full, modern, **secure** MCP client built from first principles — the
client-side companion to [mcp-learning](https://github.com/willtech3/mcp-learning)'s
Virtual Library server. Targets MCP protocol revision **2025-11-25** on the
official Python SDK.

Where most examples hide the interesting parts inside a library, this repo
implements them in the open:

| Capability | Where | What you'll learn |
|---|---|---|
| OAuth 2.1 from scratch | `mcp_client/oauth.py` | RFC 9728 discovery → RFC 8414/OIDC metadata → RFC 7591 dynamic registration → Authorization Code + **PKCE S256** → RFC 8707 resource binding → refresh |
| Loopback redirect | `mcp_client/callback_server.py` | RFC 8252 native-app flow: 127.0.0.1, ephemeral port, single-use |
| Secure token storage | `mcp_client/token_store.py` | OS keychain via `keyring`, 0600-file fallback |
| Sampling handler | `mcp_client/handlers.py` | Serving `sampling/createMessage` with the Anthropic API, model-hint mapping, human-in-the-loop approval, **tool-enabled sampling (SEP-1577)** |
| Elicitation handler | `mcp_client/handlers.py` | Form fields, enum selects (SEP-1330), URL mode (SEP-1036), accept/decline/cancel |
| Session orchestration | `mcp_client/session.py` | Capability advertising, initialize negotiation, stdio + Streamable HTTP |

## Quickstart

```bash
just install

# Against a local Virtual Library server (no auth):
#   in mcp-learning/virtual-library-mcp:  just dev-http
mcp-client tools  --server http://127.0.0.1:8080/mcp --anonymous
mcp-client call search_catalog --args '{"query": "dune"}' \
  --server http://127.0.0.1:8080/mcp --anonymous

# Against the OAuth-protected Cloud Run deployment (browser opens for sign-in):
mcp-client demo --server https://<service>.run.app/mcp
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
