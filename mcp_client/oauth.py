"""OAuth 2.1 for MCP, implemented from scratch — the heart of this repo.

The MCP authorization spec (2026-07-28 revision) composes six specs. This
module implements the client side of each one explicitly so you can see the
whole machine, not a black box:

1. RFC 9728 — Protected Resource Metadata. A 401 from the server carries
   (or implies) a metadata URL describing WHO can issue tokens for it.
2. RFC 8414 — Authorization Server Metadata (with OpenID Connect Discovery
   fallback). Tells us the authorize/token/registration endpoints. The
   metadata's `issuer` MUST equal the issuer we asked about (§3.3).
3. Client identity: CIMD (draft-ietf-oauth-client-id-metadata-document —
   the https URL of a hosted metadata document IS the client_id) when the
   AS advertises support, else RFC 7591 Dynamic Client Registration
   (deprecated in 2026-07-28 but kept for compatibility; SEP-837 requires
   `application_type` when used). Registered credentials are keyed by AS
   issuer (SEP-2352) and never replayed at a different AS.
4. OAuth 2.1 Authorization Code + PKCE (S256). The only grant MCP allows
   for user-facing clients. PKCE binds the authorization code to this
   process; `state` binds the callback to this request.
5. RFC 8707 — Resource Indicators. The token request names the exact MCP
   server the token is for, so a token for server A is useless at server B.
6. RFC 9207 — the `iss` authorization-response parameter (SEP-2468). The
   callback names the AS that answered; we compare it — exact string match,
   no normalization — against the issuer recorded before the redirect,
   which is what defeats mix-up attacks that PKCE alone cannot.

Security decisions (each one is load-bearing):
- PKCE verifier: 64 random bytes via `secrets`, challenge is S256 only.
- `state`: 32 random bytes, compared with constant-time equality.
- `iss`: validated before ANYTHING else in the callback is acted on —
  including error fields (RFC 9207 §2.4 applies to error responses too).
- Redirect URI: loopback (127.0.0.1) on an ephemeral port, exact-match
  registered and exact-match verified.
- HTTPS required for every discovered endpoint (localhost exempt for dev),
  including the registration endpoint and CIMD client_id URLs.
- PRM `resource`, when present, must canonically match the server we asked
  about — a mismatched document is someone else's (or an attacker's).
- Scope requests follow the spec priority (401 challenge scope, else PRM
  scopes_supported, else omit) and step-up re-authorization unions the new
  challenge with previously granted scopes so permissions are never lost.
- Tokens are never logged and never touch process arguments.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import secrets
import time
import webbrowser
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlsplit

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


def _canonical_resource(url: str) -> str:
    """Canonical comparison form for RFC 8707/RFC 9728 resource URIs.

    Canonicalization is deliberately minimal: lowercase the scheme and host
    (case-insensitive per RFC 3986 §6.2.2.1), drop the fragment (a resource
    URI must not carry one), and trim a trailing slash (the spec prefers the
    no-trailing-slash form unless the slash is significant). Port and path
    are compared verbatim — `https://a/mcp` and `https://a/other` are
    different resources, and pretending otherwise would let one server's
    metadata authorize tokens for another.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    port = f":{parts.port}" if parts.port is not None else ""
    path = parts.path.rstrip("/")
    return f"{parts.scheme.lower()}://{host}{port}{path}"


# ---------------------------------------------------------------------------
# WWW-Authenticate parsing (RFC 7235 / RFC 9728 §5.1)
# ---------------------------------------------------------------------------

# auth-param = token "=" ( token / quoted-string ). A naive comma-split
# corrupts quoted strings containing commas (error_description="Need scopes
# a, b" would truncate); this regex consumes a full quoted-string (with
# backslash escapes) or a bare token per parameter instead.
_AUTH_PARAM = re.compile(r'([A-Za-z0-9_\-]+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s",]+))')


def parse_www_authenticate(www_authenticate: str | None) -> dict[str, str]:
    """Extract auth-params from a WWW-Authenticate header value.

    Handles quoted strings containing commas and backslash escapes, multiple
    parameters in any order, and is forgiving about the scheme token
    (`Bearer`) and surrounding whitespace. First occurrence of a parameter
    wins. Returns {} for a missing header.
    """
    params: dict[str, str] = {}
    if not www_authenticate:
        return params
    for match in _AUTH_PARAM.finditer(www_authenticate):
        name = match.group(1).lower()
        if match.group(2) is not None:
            value = re.sub(r"\\(.)", r"\1", match.group(2))  # unescape quoted-pair
        else:
            value = match.group(3)
        params.setdefault(name, value)
    return params


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
    # RFC 9207 §2.3: an AS that emits `iss` MUST advertise it. When this is
    # True, a callback WITHOUT `iss` must be rejected (downgrade defense).
    iss_parameter_supported: bool = False
    # CIMD: the AS accepts https metadata-document URLs as client_ids.
    cimd_supported: bool = False
    # RFC 9728 scopes_supported from the PRM document (priority 2 in the
    # spec's scope-selection strategy).
    scopes_supported: list[str] | None = None


def parse_resource_metadata_url(www_authenticate: str | None, server_url: str) -> list[str]:
    """RFC 9728 metadata URL candidates for a protected MCP server, in order.

    Preferred: the 401's WWW-Authenticate header names it explicitly
    (`resource_metadata="..."`) — then that is the only candidate. Otherwise
    (the header is optional) the client MUST try the well-known URIs *in
    order*: the path-inserted form first, then the root form
    (research-auth.md §5). A server with a path (e.g. `/mcp`) may host its
    PRM at either location, so trying only the path-inserted form fails
    against a spec-compliant server that uses the root form.
    """
    from_header = parse_www_authenticate(www_authenticate).get("resource_metadata")
    if from_header:
        return [from_header]

    parts = urlsplit(server_url)
    base = f"{parts.scheme}://{parts.netloc}"
    path_inserted = f"{base}/.well-known/oauth-protected-resource{parts.path.rstrip('/')}"
    root = f"{base}/.well-known/oauth-protected-resource"
    # De-duplicate: the two coincide when the server URL has no path.
    return [path_inserted] if path_inserted == root else [path_inserted, root]


def _metadata_candidates(issuer: str) -> list[str]:
    """Well-known URL candidates for AS metadata, in spec priority order.

    RFC 8414 path-INSERTION comes first (`/.well-known/suffix/tenant`), then
    the OIDC spellings. For a path-less issuer the two path-inserted forms
    collapse to the plain well-known URLs. (research-auth §7 order.)
    """
    parts = urlsplit(issuer)
    base = f"{parts.scheme}://{parts.netloc}"
    path = parts.path.rstrip("/")
    if path:
        return [
            f"{base}/.well-known/oauth-authorization-server{path}",
            f"{base}/.well-known/openid-configuration{path}",
            f"{base}{path}/.well-known/openid-configuration",
        ]
    return [
        f"{base}/.well-known/oauth-authorization-server",
        f"{base}/.well-known/openid-configuration",
    ]


async def discover_authorization_server(
    http: httpx.AsyncClient, server_url: str, www_authenticate: str | None
) -> AuthServerInfo:
    """Walk the full discovery chain: PRM -> AS metadata (RFC 8414 or OIDC)."""
    # Try each PRM candidate in spec order (path-inserted, then root); the
    # first 200 wins. A single fixed URL would fail against a compliant
    # server that hosts its PRM at the other well-known location.
    prm = None
    for candidate in parse_resource_metadata_url(www_authenticate, server_url):
        prm_url = _require_https(candidate, "resource metadata URL")
        logger.info("Discovering protected resource metadata: %s", prm_url)
        response = await http.get(prm_url, timeout=DISCOVERY_TIMEOUT)
        if response.status_code == 200:
            prm = response.json()
            break
    if prm is None:
        raise OAuthFlowError(
            "No protected resource metadata found at any well-known location"
        )

    # RFC 9728: the PRM `resource` field, when present, MUST describe the
    # server we are trying to reach. Accepting a document about a different
    # resource would let a misconfigured (or hostile) endpoint steer this
    # client toward an AS that issues tokens for something else entirely.
    prm_resource = prm.get("resource")
    if prm_resource is not None and _canonical_resource(prm_resource) != _canonical_resource(
        server_url
    ):
        raise OAuthFlowError(
            f"Protected resource metadata is for {prm_resource!r}, not the requested "
            f"server {server_url!r} — refusing to use it"
        )

    auth_servers = prm.get("authorization_servers") or []
    if not auth_servers:
        raise OAuthFlowError("Protected resource metadata lists no authorization servers")
    issuer = _require_https(auth_servers[0], "authorization server issuer")

    metadata: dict | None = None
    for candidate in _metadata_candidates(issuer):
        response = await http.get(candidate, timeout=DISCOVERY_TIMEOUT)
        if response.status_code == 200:
            metadata = response.json()
            break
    if metadata is None:
        raise OAuthFlowError(f"No authorization server metadata found for issuer {issuer}")

    # RFC 8414 §3.3 / OIDC Discovery §4.3 (2026-07-28 makes this a MUST):
    # the `issuer` inside the document must be IDENTICAL to the issuer we
    # built the well-known URL from. Otherwise an attacker-hosted metadata
    # document could claim to be an honest AS while pointing every endpoint
    # at attacker infrastructure.
    if metadata.get("issuer") != issuer:
        raise OAuthFlowError(
            f"AS metadata issuer mismatch: document says {metadata.get('issuer')!r} "
            f"but was fetched for issuer {issuer!r} — rejecting (RFC 8414 §3.3)"
        )

    methods = metadata.get("code_challenge_methods_supported", [])
    if "S256" not in methods:
        # OAuth 2.1 + MCP both require PKCE; a server not advertising S256
        # is one we refuse to talk to rather than silently downgrade.
        raise OAuthFlowError("Authorization server does not advertise PKCE S256 support")

    # The registration endpoint receives our redirect URIs and mints our
    # client identity — it gets the SAME https-only treatment as the token
    # and authorization endpoints (previously a gap).
    registration_endpoint = metadata.get("registration_endpoint")
    if registration_endpoint is not None:
        registration_endpoint = _require_https(registration_endpoint, "registration endpoint")

    return AuthServerInfo(
        issuer=issuer,
        authorization_endpoint=_require_https(
            metadata["authorization_endpoint"], "authorization endpoint"
        ),
        token_endpoint=_require_https(metadata["token_endpoint"], "token endpoint"),
        registration_endpoint=registration_endpoint,
        code_challenge_methods=methods,
        # RFC 9207 §2.3 advertisement — drives the iss validation table.
        iss_parameter_supported=bool(metadata.get("authorization_response_iss_parameter_supported")),
        cimd_supported=bool(metadata.get("client_id_metadata_document_supported")),
        scopes_supported=prm.get("scopes_supported"),
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
            # SEP-837: MCP clients MUST specify an application_type in DCR.
            # Omitting it defaults to "web" under OIDC Dynamic Client
            # Registration, which rejects native-style loopback redirect
            # URIs like ours; a CLI is a native app.
            "application_type": "native",
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
    *,
    scope: str | None = None,
) -> dict:
    """Drive the browser flow end to end; returns the token response dict."""
    verifier, challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(32)
    # RFC 9207 (SEP-2468): record the expected issuer from the VALIDATED AS
    # metadata BEFORE redirecting, associated with this flow's per-request
    # record (verifier + state live in the same stack frame — that IS the
    # record). The callback's `iss` is compared against this value.
    expected_issuer = auth_server.issuer

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
        if scope:
            params["scope"] = scope
        authorize_url = f"{auth_server.authorization_endpoint}?{urlencode(params)}"

        print("\nOpening your browser to sign in...")
        print(f"If nothing opens, visit:\n  {authorize_url}\n")
        webbrowser.open(authorize_url)

        result: CallbackResult = await asyncio.wait_for(callback.wait(), timeout=AUTH_FLOW_TIMEOUT)

    # RFC 9207 §2.4 validation table — applied BEFORE acting on anything
    # else in the response, error fields included (a mix-up attacker can
    # forge `error` just as easily as `code`):
    #   - advertised true,  iss present -> exact string compare vs recorded issuer
    #   - advertised true,  iss absent  -> REJECT the response (advertised-but-absent)
    #   - advertised false, iss present -> exact string compare vs recorded issuer
    #   - advertised false, iss absent  -> proceed
    # "Exact" means RFC 3986 §6.2.1 simple string comparison: no case
    # folding, no default-port elision, no trailing-slash or percent-encoding
    # normalization after the form-urlencoded decode.
    if result.iss is not None:
        if result.iss != expected_issuer:
            raise OAuthFlowError(
                f"Authorization response `iss` mismatch: got {result.iss!r}, expected "
                f"{expected_issuer!r} — possible mix-up attack, aborting (RFC 9207)"
            )
    elif auth_server.iss_parameter_supported:
        raise OAuthFlowError(
            "Authorization server advertises authorization_response_iss_parameter_supported "
            "but the callback carried no `iss` — rejecting the response (RFC 9207 §2.4)"
        )

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

    Both MCP transports in this repo accept any httpx.Auth (the SDK's
    Streamable HTTP client and mcp_client.modern's from-scratch transport).
    On each request we attach the cached access token; on a 401 we run (or
    re-run) discovery -> client identity -> PKCE flow -> token exchange,
    persist the result, and retry the request once. A 403 with
    error="insufficient_scope" triggers the step-up flow: re-authorize with
    the UNION of prior and newly challenged scopes, retry once.
    """

    # We never read response bodies in the auth flow (only status + headers),
    # and requiring the body would force httpx to buffer SSE streams — the
    # modern transport's subscriptions/listen responses never end.
    requires_response_body = False

    def __init__(
        self,
        server_url: str,
        store: TokenStore,
        *,
        client_metadata_url: str | None = None,
        http_factory: Callable[[], httpx.AsyncClient] | None = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.store = store
        # CIMD (SHOULD): the https URL of a metadata document this client
        # hosts somewhere; when the AS supports CIMD, this URL IS the
        # client_id and no registration round-trip happens.
        self.client_metadata_url = client_metadata_url
        # Test seam: the flow opens short-lived clients for discovery and
        # token traffic; tests substitute a MockTransport-backed factory.
        self._http_factory: Callable[[], httpx.AsyncClient] = http_factory or httpx.AsyncClient
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
            # SEP-2352: tokens stay keyed by server URL but RECORD the AS
            # issuer, so an AS change is detectable on the next flow.
            "issuer": context.get("issuer"),
            # The granted scope set (AS's word if it echoed one, else what we
            # asked for) — the base for future step-up unions.
            "scope": token_response.get("scope", context.get("scope")),
        }
        self.store.save(self.server_url, tokens)
        return tokens

    async def _obtain_client(
        self, http: httpx.AsyncClient, auth_server: AuthServerInfo
    ) -> tuple[str, str | None]:
        """Resolve a client identity, in the spec's preference order.

        1. CIMD (SHOULD, replaces DCR): when the AS advertises
           client_id_metadata_document_supported and we have a hosted
           metadata document, its https URL IS the client_id. CIMD ids are
           portable across ASes, so nothing is persisted per issuer.
        2. Issuer-keyed cached DCR registration (SEP-2352): reuse the
           client_id previously minted by THIS issuer, never one from
           another AS.
        3. Fresh DCR (deprecated in 2026-07-28, kept for compatibility),
           persisted keyed by the issuer that minted it.
        """
        if auth_server.cimd_supported and self.client_metadata_url:
            parts = urlsplit(self.client_metadata_url)
            # CIMD requirements: https scheme AND a path component.
            if parts.scheme != "https" or not parts.path or parts.path == "/":
                raise OAuthFlowError(
                    "client_metadata_url must be an https URL with a path component "
                    f"to be used as a CIMD client_id (got {self.client_metadata_url!r})"
                )
            logger.info("Using CIMD client_id %s (no registration)", self.client_metadata_url)
            return self.client_metadata_url, None

        cached = self.store.load_client(auth_server.issuer)
        if cached and cached.get("client_id"):
            return cached["client_id"], cached.get("client_secret")

        # Registration needs the redirect URI, but the loopback port is
        # ephemeral. Servers accepting loopback clients are required to
        # ignore the port (RFC 8252 §7.3); we register a placeholder.
        client_id, client_secret = await register_client(
            http, auth_server, "http://127.0.0.1/callback"
        )
        self.store.save_client(
            auth_server.issuer, {"client_id": client_id, "client_secret": client_secret}
        )
        return client_id, client_secret

    def _select_scope(
        self,
        www_authenticate: str | None,
        auth_server: AuthServerInfo,
        extra_scopes: frozenset[str] | None,
    ) -> str | None:
        """Spec scope-selection priority, plus client-side accumulation.

        Priority: (1) the `scope` parameter of the WWW-Authenticate challenge
        is authoritative for the current operation; (2) else ALL of the PRM's
        scopes_supported (the advertised baseline); (3) else omit `scope`
        entirely and let the AS apply its defaults. Scope ACCUMULATION is a
        client-side duty: on step-up we union previously granted scopes in
        (`extra_scopes`) so a narrow challenge never strips permissions the
        user already granted.
        """
        challenge_scope = parse_www_authenticate(www_authenticate).get("scope")
        if challenge_scope:
            scopes = challenge_scope.split()
        elif auth_server.scopes_supported:
            scopes = list(auth_server.scopes_supported)
        else:
            scopes = []
        if extra_scopes:
            scopes = sorted(set(scopes) | extra_scopes)
        if not scopes:
            return None
        return " ".join(dict.fromkeys(scopes))

    async def _full_authorization(
        self,
        www_authenticate: str | None,
        extra_scopes: frozenset[str] | None = None,
    ) -> dict:
        """Discovery -> client identity -> scope selection -> PKCE flow."""
        async with self._http_factory() as http:
            auth_server = await discover_authorization_server(
                http, self.server_url, www_authenticate
            )

            # SEP-2352: if the AS advertised by the (fresh) PRM differs from
            # the one our stored tokens came from, those tokens and any
            # implied bindings belong to a different AS — drop them and start
            # clean rather than silently mixing credential domains.
            previous = self.store.load(self.server_url) or {}
            if previous.get("issuer") and previous["issuer"] != auth_server.issuer:
                logger.warning(
                    "Authorization server for %s changed (%s -> %s); dropping stored "
                    "tokens and re-registering with the new AS",
                    self.server_url,
                    previous["issuer"],
                    auth_server.issuer,
                )
                self.store.clear(self.server_url)

            client_id, client_secret = await self._obtain_client(http, auth_server)
            scope = self._select_scope(www_authenticate, auth_server, extra_scopes)

            token_response = await run_authorization_flow(
                http, auth_server, self.server_url, client_id, client_secret, scope=scope
            )
            return self._persist(
                token_response,
                {
                    "token_endpoint": auth_server.token_endpoint,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "issuer": auth_server.issuer,
                    "scope": scope,
                },
            )

    # -- httpx.Auth interface ------------------------------------------------

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        async with self._http_factory() as http:
            token = await self._current_access_token(http)
        if token:
            request.headers["Authorization"] = f"Bearer {token}"

        response = yield request

        if response.status_code == 401:
            # One re-auth attempt per request; serialize concurrent flows so
            # parallel requests don't each open a browser tab.
            async with self._flow_lock:
                async with self._http_factory() as http:
                    token = await self._current_access_token(http)
                if not token:
                    tokens = await self._full_authorization(
                        response.headers.get("www-authenticate")
                    )
                    token = tokens["access_token"]
            request.headers["Authorization"] = f"Bearer {token}"
            yield request
        elif response.status_code == 403:
            # Step-up authorization: a 403 challenge with
            # error="insufficient_scope" names the scopes the operation
            # needs. Servers are NOT required to echo previously granted
            # scopes, so the union with what we already hold happens here,
            # client-side. One retry, then the failure is permanent.
            challenge = parse_www_authenticate(response.headers.get("www-authenticate"))
            if challenge.get("error") == "insufficient_scope":
                async with self._flow_lock:
                    prior = (self.store.load(self.server_url) or {}).get("scope") or ""
                    tokens = await self._full_authorization(
                        response.headers.get("www-authenticate"),
                        extra_scopes=frozenset(prior.split()),
                    )
                request.headers["Authorization"] = f"Bearer {tokens['access_token']}"
                yield request

    def sync_auth_flow(self, _request: httpx.Request):  # pragma: no cover
        raise RuntimeError("MCPOAuth is async-only; use it with httpx.AsyncClient")
