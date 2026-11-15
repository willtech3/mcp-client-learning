"""Red-team tests for the 2026-07-28 authorization hardening.

Each test encodes one of the 2026-07-28 client MUSTs added on top of the
original OAuth layer (whose invariants live untouched in test_oauth.py):

- RFC 9207 / SEP-2468 `iss` validation (mix-up attack defense)
- RFC 8414 §3.3 metadata issuer validation
- SEP-2352 issuer-keyed client credentials + re-registration on AS change
- SEP-837 `application_type` in DCR
- HTTPS enforcement on the registration endpoint
- RFC 9728 PRM `resource` validation
- CIMD client_id selection
- RFC 7235-grade WWW-Authenticate parsing (quoted strings with commas)
- Scope selection priority + 403 insufficient_scope step-up with union

No network: httpx.MockTransport everywhere; the "browser" is a monkeypatched
webbrowser.open that parses the authorize URL and steers the fake callback.
"""

import json
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

import mcp_client.token_store as token_store_module
from mcp_client.callback_server import CallbackResult
from mcp_client.oauth import (
    AuthServerInfo,
    MCPOAuth,
    OAuthFlowError,
    discover_authorization_server,
    parse_resource_metadata_url,
    parse_www_authenticate,
    register_client,
    run_authorization_flow,
)
from mcp_client.token_store import TokenStore

SERVER = "https://srv.example/mcp"
ISSUER = "https://auth.example"

AS_METADATA = {
    "issuer": ISSUER,
    "authorization_endpoint": f"{ISSUER}/authorize",
    "token_endpoint": f"{ISSUER}/token",
    "registration_endpoint": f"{ISSUER}/register",
    "code_challenge_methods_supported": ["S256"],
    "authorization_response_iss_parameter_supported": True,
}

PRM = {
    "resource": SERVER,
    "authorization_servers": [ISSUER],
    "scopes_supported": ["library:read"],
}


def auth_info(**overrides) -> AuthServerInfo:
    base: dict = {
        "issuer": ISSUER,
        "authorization_endpoint": f"{ISSUER}/authorize",
        "token_endpoint": f"{ISSUER}/token",
        "registration_endpoint": f"{ISSUER}/register",
        "code_challenge_methods": ["S256"],
    }
    base.update(overrides)
    return AuthServerInfo(**base)


class BrowserSim:
    """Captured state of the simulated authorization redirect."""

    def __init__(self):
        self.iss: str | None = None
        self.error: str | None = None
        self.code: str | None = "code-123"
        self.echo_state = True
        self.authorize_urls: list[str] = []

    def authorize_query(self, index: int = -1) -> dict:
        return {k: v[0] for k, v in parse_qs(urlsplit(self.authorize_urls[index]).query).items()}


@pytest.fixture
def browser(monkeypatch) -> BrowserSim:
    """Simulate browser + AS: capture the authorize URL, echo back a callback
    whose code/state/iss the test controls."""
    sim = BrowserSim()
    state_seen: dict = {}

    def fake_open(url):
        sim.authorize_urls.append(url)
        query = parse_qs(urlsplit(url).query)
        state_seen["state"] = query.get("state", [""])[0]
        return True

    @asynccontextmanager
    async def fake_callback_server():
        class Listener:
            redirect_uri = "http://127.0.0.1:9/callback"

            async def wait(self):
                return CallbackResult(
                    code=sim.code,
                    error=sim.error,
                    state=state_seen.get("state") if sim.echo_state else "WRONG-STATE",
                    iss=sim.iss,
                )

        yield Listener()

    monkeypatch.setattr("webbrowser.open", fake_open)
    monkeypatch.setattr("mcp_client.oauth.run_callback_server", fake_callback_server)
    return sim


def _token_http() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "at-1", "expires_in": 3600})
        return httpx.Response(500)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# 1. RFC 9207 `iss` validation
# ---------------------------------------------------------------------------


class TestIssValidation:
    async def test_iss_mismatch_aborts_before_token_exchange(self, browser):
        """A code delivered under the wrong issuer is a mix-up attack; the
        verifier must never reach the (attacker's) token endpoint."""
        browser.iss = "https://evil.example"
        token_posts: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            token_posts.append(request)
            return httpx.Response(200, json={"access_token": "stolen"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with pytest.raises(OAuthFlowError, match="iss"):
                await run_authorization_flow(
                    http, auth_info(iss_parameter_supported=True), SERVER, "client-1", None
                )
        assert token_posts == []

    async def test_advertised_but_absent_iss_rejected(self, browser):
        """AS metadata advertised iss support; a callback WITHOUT iss is a
        downgrade and MUST be rejected (RFC 9207 §2.4 table row 2)."""
        browser.iss = None
        async with _token_http() as http:
            with pytest.raises(OAuthFlowError, match="iss"):
                await run_authorization_flow(
                    http, auth_info(iss_parameter_supported=True), SERVER, "client-1", None
                )

    async def test_present_iss_checked_even_when_not_advertised(self, browser):
        browser.iss = "https://evil.example"
        async with _token_http() as http:
            with pytest.raises(OAuthFlowError, match="iss"):
                await run_authorization_flow(
                    http, auth_info(iss_parameter_supported=False), SERVER, "client-1", None
                )

    async def test_exact_string_comparison_no_normalization(self, browser):
        """RFC 3986 §6.2.1: simple string compare — no case folding, no
        trailing-slash normalization. `https://auth.example/` != issuer."""
        browser.iss = ISSUER + "/"
        async with _token_http() as http:
            with pytest.raises(OAuthFlowError, match="iss"):
                await run_authorization_flow(
                    http, auth_info(iss_parameter_supported=True), SERVER, "client-1", None
                )

    async def test_error_responses_validated_too(self, browser):
        """On iss mismatch the client MUST NOT act on `error` either — the
        raised failure is about iss, not the attacker-chosen error text."""
        browser.iss = "https://evil.example"
        browser.error = "access_denied"
        browser.code = None
        async with _token_http() as http:
            with pytest.raises(OAuthFlowError, match="iss"):
                await run_authorization_flow(
                    http, auth_info(iss_parameter_supported=True), SERVER, "client-1", None
                )

    async def test_matching_iss_completes(self, browser):
        browser.iss = ISSUER
        async with _token_http() as http:
            tokens = await run_authorization_flow(
                http, auth_info(iss_parameter_supported=True), SERVER, "client-1", None
            )
        assert tokens["access_token"] == "at-1"

    async def test_no_iss_and_not_advertised_proceeds(self, browser):
        """Table row 4: nothing advertised, nothing present -> proceed
        (this is also what keeps legacy ASes working)."""
        browser.iss = None
        async with _token_http() as http:
            tokens = await run_authorization_flow(
                http, auth_info(iss_parameter_supported=False), SERVER, "client-1", None
            )
        assert tokens["access_token"] == "at-1"


# ---------------------------------------------------------------------------
# 2. RFC 8414 §3.3 metadata issuer validation + 5. registration https
# ---------------------------------------------------------------------------


def _discovery_http(prm: dict, metadata: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://srv.example/.well-known/oauth-protected-resource/mcp":
            return httpx.Response(200, json=prm)
        if url == f"{ISSUER}/.well-known/oauth-authorization-server":
            return httpx.Response(200, json=metadata)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestMetadataIssuerValidation:
    async def test_issuer_mismatch_rejected(self):
        """A document fetched for issuer X claiming to be issuer Y is an
        impersonation attempt — RFC 8414 §3.3 says MUST NOT use it."""
        forged = dict(AS_METADATA, issuer="https://honest-looking.example")
        async with _discovery_http(PRM, forged) as http:
            with pytest.raises(OAuthFlowError, match="issuer mismatch"):
                await discover_authorization_server(http, SERVER, None)

    async def test_matching_issuer_accepted_with_flags_captured(self):
        async with _discovery_http(PRM, AS_METADATA) as http:
            info = await discover_authorization_server(http, SERVER, None)
        assert info.issuer == ISSUER
        assert info.iss_parameter_supported is True
        assert info.scopes_supported == ["library:read"]

    async def test_http_registration_endpoint_rejected(self):
        """The registration endpoint mints our identity; it gets the same
        TLS-only rule as every other discovered endpoint."""
        weak = dict(AS_METADATA, registration_endpoint="http://auth.example/register")
        async with _discovery_http(PRM, weak) as http:
            with pytest.raises(OAuthFlowError, match=r"registration endpoint.*https"):
                await discover_authorization_server(http, SERVER, None)


class TestPrmResourceValidation:
    async def test_resource_mismatch_rejected(self):
        """RFC 9728: a PRM document describing a DIFFERENT resource must not
        steer this client's AS selection."""
        foreign = dict(PRM, resource="https://other.example/mcp")
        async with _discovery_http(foreign, AS_METADATA) as http:
            with pytest.raises(OAuthFlowError, match="resource"):
                await discover_authorization_server(http, SERVER, None)

    async def test_canonical_comparison_tolerates_case_and_slash(self):
        """Canonicalization: scheme/host case-insensitive, trailing slash
        trimmed, fragment dropped — but nothing more."""
        shouty = dict(PRM, resource="HTTPS://SRV.EXAMPLE/mcp/")
        async with _discovery_http(shouty, AS_METADATA) as http:
            info = await discover_authorization_server(http, SERVER, None)
        assert info.issuer == ISSUER

    async def test_absent_resource_field_is_tolerated(self):
        minimal = {"authorization_servers": [ISSUER]}
        async with _discovery_http(minimal, AS_METADATA) as http:
            info = await discover_authorization_server(http, SERVER, None)
        assert info.scopes_supported is None

    async def test_prm_discovered_at_root_when_path_inserted_404s(self):
        """RFC 9728 client MUST: when the WWW-Authenticate header omits the
        PRM URL, try the path-inserted well-known form FIRST, then fall back
        to the ROOT form. A server with a path (/mcp) may host its PRM only
        at the root; trying a single fixed URL breaks against it."""

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            # Path-inserted form 404s; PRM lives ONLY at the root form.
            if url == "https://srv.example/.well-known/oauth-protected-resource":
                return httpx.Response(200, json=PRM)
            if url == f"{ISSUER}/.well-known/oauth-authorization-server":
                return httpx.Response(200, json=AS_METADATA)
            return httpx.Response(404)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            info = await discover_authorization_server(http, SERVER, None)
        assert info.issuer == ISSUER


# ---------------------------------------------------------------------------
# 4. SEP-837 application_type
# ---------------------------------------------------------------------------


class TestDynamicRegistration:
    async def test_registration_sends_native_application_type(self):
        bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(201, json={"client_id": "dyn-client-1"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client_id, _ = await register_client(http, auth_info(), "http://127.0.0.1/callback")

        assert client_id == "dyn-client-1"
        # SEP-837: MUST specify application_type; a CLI with a loopback
        # redirect is a native app (OIDC would reject it as "web").
        assert bodies[0]["application_type"] == "native"
        assert bodies[0]["token_endpoint_auth_method"] == "none"


# ---------------------------------------------------------------------------
# 3 + 7 + 9. Full-flow tests through MCPOAuth (issuer keying, CIMD, scopes)
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch) -> TokenStore:
    monkeypatch.setattr(token_store_module, "FALLBACK_DIR", tmp_path / "secrets")
    return TokenStore(use_keyring=False)


class ASBackend:
    """A scripted AS + PRM backend for MCPOAuth full flows."""

    def __init__(self, issuer=ISSUER, metadata_extra=None, prm_extra=None):
        self.issuer = issuer
        self.register_calls: list[dict] = []
        self.token_calls: list[dict] = []
        self.metadata = {
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "registration_endpoint": f"{issuer}/register",
            "code_challenge_methods_supported": ["S256"],
            **(metadata_extra or {}),
        }
        self.prm = {
            "resource": SERVER,
            "authorization_servers": [issuer],
            **(prm_extra or {}),
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        issuer_host = urlsplit(self.issuer).netloc
        if url == "https://srv.example/.well-known/oauth-protected-resource/mcp":
            return httpx.Response(200, json=self.prm)
        if url == f"https://{issuer_host}/.well-known/oauth-authorization-server":
            return httpx.Response(200, json=self.metadata)
        if url == f"{self.issuer}/register":
            self.register_calls.append(json.loads(request.content))
            n = len(self.register_calls)
            return httpx.Response(201, json={"client_id": f"dyn-{issuer_host}-{n}"})
        if url == f"{self.issuer}/token":
            self.token_calls.append(dict(parse_qs(request.content.decode())))
            return httpx.Response(
                200, json={"access_token": f"at-{len(self.token_calls)}", "expires_in": 3600}
            )
        return httpx.Response(404)

    def http_factory(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


class TestIssuerKeyedCredentials:
    async def test_registration_reused_for_same_issuer(self, browser, store):
        backend = ASBackend()
        oauth = MCPOAuth(SERVER, store, http_factory=backend.http_factory)

        await oauth._full_authorization(None)
        await oauth._full_authorization(None)

        # SEP-2352: one registration, keyed by issuer, reused thereafter.
        assert len(backend.register_calls) == 1
        cached = store.load_client(ISSUER)
        assert cached["client_id"] == "dyn-auth.example-1"

    async def test_as_change_drops_tokens_and_reregisters(self, browser, store):
        backend_a = ASBackend()
        oauth = MCPOAuth(SERVER, store, http_factory=backend_a.http_factory)
        await oauth._full_authorization(None)
        assert store.load(SERVER)["issuer"] == ISSUER

        # The server's PRM now advertises a DIFFERENT authorization server.
        backend_b = ASBackend(issuer="https://auth-b.example")
        oauth._http_factory = backend_b.http_factory
        await oauth._full_authorization(None)

        # Credentials from AS-A were NOT replayed at AS-B: B saw a fresh
        # registration and the authorize request used B's client_id.
        assert len(backend_b.register_calls) == 1
        assert browser.authorize_query()["client_id"] == "dyn-auth-b.example-1"
        assert store.load_client("https://auth-b.example")["client_id"] == "dyn-auth-b.example-1"
        # And the stored tokens now record the new issuer.
        assert store.load(SERVER)["issuer"] == "https://auth-b.example"
        # AS-A's registration is still cached under ITS issuer (unharmed).
        assert store.load_client(ISSUER)["client_id"] == "dyn-auth.example-1"


class TestCimd:
    async def test_cimd_url_used_as_client_id_without_registration(self, browser, store):
        backend = ASBackend(metadata_extra={"client_id_metadata_document_supported": True})
        oauth = MCPOAuth(
            SERVER,
            store,
            client_metadata_url="https://app.example/oauth/client-metadata.json",
            http_factory=backend.http_factory,
        )
        await oauth._full_authorization(None)

        assert backend.register_calls == []  # no DCR round-trip
        assert (
            browser.authorize_query()["client_id"]
            == "https://app.example/oauth/client-metadata.json"
        )

    async def test_cimd_ignored_when_as_does_not_advertise(self, browser, store):
        backend = ASBackend()  # no client_id_metadata_document_supported
        oauth = MCPOAuth(
            SERVER,
            store,
            client_metadata_url="https://app.example/oauth/client-metadata.json",
            http_factory=backend.http_factory,
        )
        await oauth._full_authorization(None)
        assert len(backend.register_calls) == 1  # fell back to DCR

    async def test_cimd_url_must_be_https_with_path(self, browser, store):
        backend = ASBackend(metadata_extra={"client_id_metadata_document_supported": True})
        for bad in ("http://app.example/meta.json", "https://app.example", "https://app.example/"):
            oauth = MCPOAuth(
                SERVER, store, client_metadata_url=bad, http_factory=backend.http_factory
            )
            with pytest.raises(OAuthFlowError, match="client_metadata_url"):
                await oauth._full_authorization(None)


# ---------------------------------------------------------------------------
# 8. WWW-Authenticate parsing
# ---------------------------------------------------------------------------


class TestWwwAuthenticateParsing:
    def test_quoted_strings_with_commas_survive(self):
        header = (
            'Bearer error="insufficient_scope", '
            'error_description="Need scopes: files:read, files:write", '
            'scope="files:read files:write", '
            'resource_metadata="https://srv.example/.well-known/oauth-protected-resource"'
        )
        params = parse_www_authenticate(header)
        assert params["error"] == "insufficient_scope"
        assert params["error_description"] == "Need scopes: files:read, files:write"
        assert params["scope"] == "files:read files:write"
        assert params["resource_metadata"] == (
            "https://srv.example/.well-known/oauth-protected-resource"
        )

    def test_escaped_quotes_unescaped(self):
        params = parse_www_authenticate('Bearer error_description="say \\"hi\\" now"')
        assert params["error_description"] == 'say "hi" now'

    def test_unquoted_tokens_and_missing_header(self):
        assert parse_www_authenticate("Bearer realm=mcp, scope=files:read")["scope"] == "files:read"
        assert parse_www_authenticate(None) == {}

    def test_resource_metadata_found_after_comma_laden_param(self):
        header = (
            'Bearer error_description="a, b, c", '
            'resource_metadata="https://srv.example/.well-known/oauth-protected-resource/mcp"'
        )
        urls = parse_resource_metadata_url(header, SERVER)
        assert urls == ["https://srv.example/.well-known/oauth-protected-resource/mcp"]


# ---------------------------------------------------------------------------
# 9. Scope selection + step-up
# ---------------------------------------------------------------------------


class TestScopeSelection:
    def _oauth(self, store) -> MCPOAuth:
        return MCPOAuth(SERVER, store)

    def test_challenge_scope_has_priority_over_prm(self, store):
        scope = self._oauth(store)._select_scope(
            'Bearer scope="files:read", resource_metadata="https://x/.well-known/y"',
            auth_info(scopes_supported=["library:read", "library:write"]),
            None,
        )
        assert scope == "files:read"

    def test_prm_scopes_used_when_challenge_has_none(self, store):
        scope = self._oauth(store)._select_scope(
            "Bearer realm=mcp",
            auth_info(scopes_supported=["library:read", "library:write"]),
            None,
        )
        assert scope == "library:read library:write"

    def test_scope_omitted_when_nothing_advertised(self, store):
        assert self._oauth(store)._select_scope(None, auth_info(), None) is None

    async def test_authorize_url_omits_scope_when_unknown(self, browser, store):
        backend = ASBackend()  # PRM without scopes_supported
        oauth = MCPOAuth(SERVER, store, http_factory=backend.http_factory)
        await oauth._full_authorization(None)
        assert "scope" not in browser.authorize_query()

    async def test_step_up_unions_prior_and_challenged_scopes(self, browser, store):
        """403 insufficient_scope -> re-authorize with the UNION of held and
        challenged scopes (accumulation is the CLIENT's job), retry once."""
        backend = ASBackend()
        oauth = MCPOAuth(SERVER, store, http_factory=backend.http_factory)
        # An earlier grant is on file.
        store.save(
            SERVER,
            {
                "access_token": "at-old",
                "expires_at": 9999999999.0,
                "token_endpoint": f"{ISSUER}/token",
                "client_id": "dyn-1",
                "issuer": ISSUER,
                "scope": "library:read",
            },
        )

        flow = oauth.async_auth_flow(httpx.Request("POST", SERVER))
        first = await anext(flow)
        assert first.headers["Authorization"] == "Bearer at-old"

        challenge = (
            'Bearer error="insufficient_scope", scope="library:write", '
            'error_description="Write access required, please re-authorize", '
            'resource_metadata="https://srv.example/.well-known/oauth-protected-resource/mcp"'
        )
        retry = await flow.asend(
            httpx.Response(403, headers={"WWW-Authenticate": challenge}, request=first)
        )
        # Union, sorted: never lose what the user already granted.
        assert browser.authorize_query()["scope"] == "library:read library:write"
        assert retry.headers["Authorization"] == "Bearer at-1"
        with pytest.raises(StopAsyncIteration):
            await flow.asend(httpx.Response(200, request=retry))

    async def test_403_without_insufficient_scope_not_retried(self, store):
        oauth = MCPOAuth(SERVER, store)
        flow = oauth.async_auth_flow(httpx.Request("POST", SERVER))
        first = await anext(flow)
        with pytest.raises(StopAsyncIteration):
            await flow.asend(httpx.Response(403, request=first))
