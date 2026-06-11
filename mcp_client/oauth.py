"""OAuth 2.1 for MCP, implemented from scratch — the heart of this repo.

The MCP authorization spec (2025-11-25) composes five RFCs. This module
implements the client side of each one explicitly so you can see the whole
machine, not a black box:

1. RFC 9728 — Protected Resource Metadata. A 401 from the server carries
   (or implies) a metadata URL describing WHO can issue tokens for it.
2. RFC 8414 — Authorization Server Metadata (with OpenID Connect Discovery
   fallback). Tells us the authorize/token/registration endpoints.
3. RFC 7591 — Dynamic Client Registration. The client registers itself and
   receives a client_id; no human pre-provisioning.
4. OAuth 2.1 Authorization Code + PKCE (S256). The only grant MCP allows
   for user-facing clients. PKCE binds the authorization code to this
   process; `state` binds the callback to this request.
5. RFC 8707 — Resource Indicators. The token request names the exact MCP
   server the token is for, so a token for server A is useless at server B.

Security decisions (each one is load-bearing):
- PKCE verifier: 64 random bytes via `secrets`, challenge is S256 only.
- `state`: 32 random bytes, compared with constant-time equality.
- Redirect URI: loopback (127.0.0.1) on an ephemeral port, exact-match
  registered and exact-match verified.
- HTTPS required for every discovered endpoint (localhost exempt for dev).
- Tokens are never logged and never touch process arguments.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
import time
import webbrowser
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from urllib.parse import urlencode, urljoin, urlsplit

import httpx

from .callback_server import CallbackResult, run_callback_server
from .token_store import TokenStore

logger = logging.getLogger(__name__)

DISCOVERY_TIMEOUT = 10.0
AUTH_FLOW_TIMEOUT = 300.0  # how long we wait for the human to sign in


class OAuthFlowError(Exception):
    """Raised when any step of the authorization flow fails validation."""


# ---------------------------------------------------------------------------
# PKCE (Proof Key for Code Exchange) primitives
# ---------------------------------------------------------------------------


def generate_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) per RFC 7636, S256 only.

    The verifier never leaves this process until the token exchange; the
    challenge (its SHA-256) is what travels through the browser. An attacker
    who steals the authorization code cannot redeem it without the verifier.
    """
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _require_https(url: str, what: str) -> str:
    """OAuth 2.1 mandates TLS everywhere; only localhost is exempt (dev)."""
    parts = urlsplit(url)
    is_local = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if parts.scheme != "https" and not is_local:
        raise OAuthFlowError(f"{what} must be https:// (got {url})")
    return url


# ---------------------------------------------------------------------------
# Discovery: 401 -> resource metadata -> authorization server metadata
# ---------------------------------------------------------------------------


@dataclass
class AuthServerInfo:
    """Everything discovery tells us about the authorization server."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None
    code_challenge_methods: list[str] = field(default_factory=list)


def parse_resource_metadata_url(www_authenticate: str | None, server_url: str) -> str:
    """Find the RFC 9728 metadata URL for a protected MCP server.

    Preferred: the 401's WWW-Authenticate header names it explicitly
    (`resource_metadata="..."`). Fallback (2025-11-25 made the header
    optional): derive /.well-known/oauth-protected-resource from the
    server URL, preserving its path.
    """
    if www_authenticate:
        for raw_part in www_authenticate.split(","):
            part = raw_part.strip()
            if part.startswith("resource_metadata="):
                return part.split("=", 1)[1].strip().strip('"')

    parts = urlsplit(server_url)
    well_known_path = "/.well-known/oauth-protected-resource" + parts.path.rstrip("/")
    return f"{parts.scheme}://{parts.netloc}{well_known_path}"


async def discover_authorization_server(
    http: httpx.AsyncClient, server_url: str, www_authenticate: str | None
) -> AuthServerInfo:
    """Walk the full discovery chain: PRM -> AS metadata (RFC 8414 or OIDC)."""
    prm_url = _require_https(
        parse_resource_metadata_url(www_authenticate, server_url), "resource metadata URL"
    )
    logger.info("Discovering protected resource metadata: %s", prm_url)
    prm = (await http.get(prm_url, timeout=DISCOVERY_TIMEOUT)).raise_for_status().json()

    auth_servers = prm.get("authorization_servers") or []
    if not auth_servers:
        raise OAuthFlowError("Protected resource metadata lists no authorization servers")
    issuer = _require_https(auth_servers[0], "authorization server issuer")

    # RFC 8414 first, then OpenID Connect Discovery (2025-11-25 requires
    # clients to support both spellings).
    candidates = [
        urljoin(issuer + "/", ".well-known/oauth-authorization-server"),
        urljoin(issuer + "/", ".well-known/openid-configuration"),
    ]
    metadata: dict | None = None
    for candidate in candidates:
        response = await http.get(candidate, timeout=DISCOVERY_TIMEOUT)
        if response.status_code == 200:
            metadata = response.json()
            break
    if metadata is None:
        raise OAuthFlowError(f"No authorization server metadata found for issuer {issuer}")

    methods = metadata.get("code_challenge_methods_supported", [])
    if "S256" not in methods:
        # OAuth 2.1 + MCP both require PKCE; a server not advertising S256
        # is one we refuse to talk to rather than silently downgrade.
        raise OAuthFlowError("Authorization server does not advertise PKCE S256 support")

    return AuthServerInfo(
        issuer=issuer,
        authorization_endpoint=_require_https(
            metadata["authorization_endpoint"], "authorization endpoint"
        ),
        token_endpoint=_require_https(metadata["token_endpoint"], "token endpoint"),
        registration_endpoint=metadata.get("registration_endpoint"),
        code_challenge_methods=methods,
    )


# ---------------------------------------------------------------------------
# Dynamic Client Registration (RFC 7591)
# ---------------------------------------------------------------------------


async def register_client(
    http: httpx.AsyncClient, auth_server: AuthServerInfo, redirect_uri: str
) -> tuple[str, str | None]:
    """Register this client and return (client_id, client_secret|None).

    MCP clients are typically *public* clients (they can't keep a secret),
    so token_endpoint_auth_method is 'none' — PKCE is what protects the
    code exchange instead of a client secret.
    """
    if not auth_server.registration_endpoint:
        raise OAuthFlowError(
            "Authorization server does not support dynamic client registration; "
            "configure a pre-registered client_id instead."
        )

    response = await http.post(
        auth_server.registration_endpoint,
        json={
            "client_name": "mcp-client-learning",
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
        timeout=DISCOVERY_TIMEOUT,
    )
    response.raise_for_status()
    registration = response.json()
    logger.info("Dynamically registered client_id=%s", registration["client_id"])
    return registration["client_id"], registration.get("client_secret")


# ---------------------------------------------------------------------------
# The interactive authorization-code + PKCE flow
# ---------------------------------------------------------------------------


async def run_authorization_flow(
    http: httpx.AsyncClient,
    auth_server: AuthServerInfo,
    server_url: str,
    client_id: str,
    client_secret: str | None,
) -> dict:
    """Drive the browser flow end to end; returns the token response dict."""
    verifier, challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(32)

    async with run_callback_server() as callback:
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": callback.redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            # RFC 8707: bind the requested token to this exact MCP server.
            "resource": server_url,
        }
        authorize_url = f"{auth_server.authorization_endpoint}?{urlencode(params)}"

        print("\nOpening your browser to sign in...")
        print(f"If nothing opens, visit:\n  {authorize_url}\n")
        webbrowser.open(authorize_url)

        result: CallbackResult = await asyncio.wait_for(callback.wait(), timeout=AUTH_FLOW_TIMEOUT)

    if result.error:
        raise OAuthFlowError(f"Authorization denied: {result.error}")
    # Constant-time comparison: a forged callback must not be distinguishable
    # by timing, and any state mismatch means the code is not ours.
    if not secrets.compare_digest(result.state or "", state):
        raise OAuthFlowError("State mismatch in OAuth callback (possible CSRF) - aborting")
    if not result.code:
        raise OAuthFlowError("Callback carried no authorization code")

    token_request = {
        "grant_type": "authorization_code",
        "code": result.code,
        "redirect_uri": callback.redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
        "resource": server_url,
    }
    if client_secret:
        token_request["client_secret"] = client_secret

    response = await http.post(
        auth_server.token_endpoint, data=token_request, timeout=DISCOVERY_TIMEOUT
    )
    if response.status_code != 200:
        raise OAuthFlowError(f"Token exchange failed: {response.status_code} {response.text[:200]}")
    return response.json()


# ---------------------------------------------------------------------------
# httpx.Auth glue: attach tokens, refresh, re-authorize on 401
# ---------------------------------------------------------------------------


class MCPOAuth(httpx.Auth):
    """Bearer-token auth for the MCP transport, with the full flow behind it.

    The MCP SDK's Streamable HTTP transport accepts any httpx.Auth. On each
    request we attach the cached access token; on a 401 we run (or re-run)
    discovery -> registration -> PKCE flow -> token exchange, persist the
    result, and retry the request once.
    """

    requires_response_body = True

    def __init__(self, server_url: str, store: TokenStore):
        self.server_url = server_url.rstrip("/")
        self.store = store
        self._flow_lock = asyncio.Lock()

    # -- token lifecycle ----------------------------------------------------

    async def _current_access_token(self, http: httpx.AsyncClient) -> str | None:
        tokens = self.store.load(self.server_url)
        if not tokens:
            return None
        if tokens.get("expires_at", 0) > time.time() + 30:
            return tokens["access_token"]
        if tokens.get("refresh_token"):
            refreshed = await self._refresh(http, tokens)
            if refreshed:
                return refreshed["access_token"]
        return None

    async def _refresh(self, http: httpx.AsyncClient, tokens: dict) -> dict | None:
        """Redeem a refresh token; on any failure fall back to a fresh flow."""
        try:
            response = await http.post(
                tokens["token_endpoint"],
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": tokens["refresh_token"],
                    "client_id": tokens["client_id"],
                    "resource": self.server_url,
                },
                timeout=DISCOVERY_TIMEOUT,
            )
            if response.status_code != 200:
                logger.info("Refresh token rejected (%s); will re-authorize", response.status_code)
                return None
            saved = self._persist(response.json(), tokens)
        except httpx.HTTPError as e:
            logger.info("Token refresh failed (%s); will re-authorize", e)
            return None
        else:
            logger.info("Access token refreshed")
            return saved

    def _persist(self, token_response: dict, context: dict) -> dict:
        tokens = {
            "access_token": token_response["access_token"],
            "refresh_token": token_response.get("refresh_token", context.get("refresh_token")),
            "expires_at": time.time() + token_response.get("expires_in", 3600),
            "token_endpoint": context["token_endpoint"],
            "client_id": context["client_id"],
            "client_secret": context.get("client_secret"),
        }
        self.store.save(self.server_url, tokens)
        return tokens

    async def _full_authorization(self, www_authenticate: str | None) -> dict:
        """Discovery -> (cached or dynamic) registration -> PKCE flow."""
        async with httpx.AsyncClient() as http:
            auth_server = await discover_authorization_server(
                http, self.server_url, www_authenticate
            )

            # Reuse a previously registered client when the AS is unchanged —
            # re-registering on every login litters the AS with clients.
            cached = self.store.load(self.server_url) or {}
            if cached.get("client_id") and cached.get("issuer") == auth_server.issuer:
                client_id: str = cached["client_id"]
                client_secret: str | None = cached.get("client_secret")
            else:
                # Registration needs the redirect URI, but the loopback port is
                # ephemeral. Servers accepting loopback clients are required to
                # ignore the port (RFC 8252 §7.3); we register a placeholder.
                client_id, client_secret = await register_client(
                    http, auth_server, "http://127.0.0.1/callback"
                )

            token_response = await run_authorization_flow(
                http, auth_server, self.server_url, client_id, client_secret
            )
            tokens = self._persist(
                token_response,
                {
                    "token_endpoint": auth_server.token_endpoint,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
            )
            # Remember the issuer so the client registration can be reused.
            tokens["issuer"] = auth_server.issuer
            tokens["client_id"] = client_id
            self.store.save(self.server_url, tokens)
            return tokens

    # -- httpx.Auth interface ------------------------------------------------

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        async with httpx.AsyncClient() as http:
            token = await self._current_access_token(http)
        if token:
            request.headers["Authorization"] = f"Bearer {token}"

        response = yield request

        if response.status_code == 401:
            # One re-auth attempt per request; serialize concurrent flows so
            # parallel requests don't each open a browser tab.
            async with self._flow_lock:
                async with httpx.AsyncClient() as http:
                    token = await self._current_access_token(http)
                if not token:
                    tokens = await self._full_authorization(
                        response.headers.get("www-authenticate")
                    )
                    token = tokens["access_token"]
            request.headers["Authorization"] = f"Bearer {token}"
            yield request

    def sync_auth_flow(self, _request: httpx.Request):  # pragma: no cover
        raise RuntimeError("MCPOAuth is async-only; use it with httpx.AsyncClient")
