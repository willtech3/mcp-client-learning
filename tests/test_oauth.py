"""Tests for the OAuth 2.1 layer — the security-critical core.

Network interactions use httpx.MockTransport: we test OUR protocol logic
(discovery chain, fallbacks, refusals, state validation), not the wire.
"""

import base64
import hashlib
from contextlib import asynccontextmanager

import httpx
import pytest

from mcp_client.callback_server import CallbackResult
from mcp_client.oauth import (
    AuthServerInfo,
    OAuthFlowError,
    discover_authorization_server,
    generate_pkce_pair,
    parse_resource_metadata_url,
    run_authorization_flow,
)


class TestPKCE:
    def test_challenge_is_s256_of_verifier(self):
        verifier, challenge = generate_pkce_pair()
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert challenge == expected

    def test_verifier_is_long_and_unique(self):
        pairs = {generate_pkce_pair()[0] for _ in range(50)}
        assert len(pairs) == 50  # no collisions
        assert all(len(v) >= 43 for v in pairs)  # RFC 7636 minimum length

    def test_challenge_is_urlsafe_without_padding(self):
        _, challenge = generate_pkce_pair()
        assert "=" not in challenge
        assert "+" not in challenge
        assert "/" not in challenge


class TestResourceMetadataDiscovery:
    def test_prefers_www_authenticate_header(self):
        header = (
            'Bearer error="invalid_token", '
            'resource_metadata="https://srv.example/.well-known/oauth-protected-resource/mcp"'
        )
        url = parse_resource_metadata_url(header, "https://srv.example/mcp")
        assert url == "https://srv.example/.well-known/oauth-protected-resource/mcp"

    def test_falls_back_to_well_known_with_path(self):
        """2025-11-25 made the header optional; the .well-known fallback
        must preserve the server's path component."""
        url = parse_resource_metadata_url(None, "https://srv.example/mcp")
        assert url == "https://srv.example/.well-known/oauth-protected-resource/mcp"

    def test_fallback_without_path(self):
        url = parse_resource_metadata_url(None, "https://srv.example")
        assert url == "https://srv.example/.well-known/oauth-protected-resource"


def _mock_http(responses: dict[str, dict | int]) -> httpx.AsyncClient:
    """AsyncClient whose GETs answer from a url->json (or status) table."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = str(request.url)
        if key in responses:
            value = responses[key]
            if isinstance(value, int):
                return httpx.Response(value)
            return httpx.Response(200, json=value)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


AS_METADATA = {
    "issuer": "https://auth.example",
    "authorization_endpoint": "https://auth.example/authorize",
    "token_endpoint": "https://auth.example/token",
    "registration_endpoint": "https://auth.example/register",
    "code_challenge_methods_supported": ["S256"],
}


class TestAuthServerDiscovery:
    async def test_full_chain(self):
        http = _mock_http(
            {
                "https://srv.example/.well-known/oauth-protected-resource/mcp": {
                    "resource": "https://srv.example/mcp",
                    "authorization_servers": ["https://auth.example"],
                },
                "https://auth.example/.well-known/oauth-authorization-server": AS_METADATA,
            }
        )
        info = await discover_authorization_server(http, "https://srv.example/mcp", None)
        assert info.token_endpoint == "https://auth.example/token"
        assert "S256" in info.code_challenge_methods

    async def test_falls_back_to_oidc_discovery(self):
        """RFC 8414 path missing -> OpenID Connect Discovery is required."""
        http = _mock_http(
            {
                "https://srv.example/.well-known/oauth-protected-resource/mcp": {
                    "authorization_servers": ["https://auth.example"],
                },
                "https://auth.example/.well-known/oauth-authorization-server": 404,
                "https://auth.example/.well-known/openid-configuration": AS_METADATA,
            }
        )
        info = await discover_authorization_server(http, "https://srv.example/mcp", None)
        assert info.issuer == "https://auth.example"

    async def test_refuses_server_without_s256(self):
        """No PKCE S256 advertised -> refuse, never downgrade."""
        weak = dict(AS_METADATA, code_challenge_methods_supported=["plain"])
        http = _mock_http(
            {
                "https://srv.example/.well-known/oauth-protected-resource/mcp": {
                    "authorization_servers": ["https://auth.example"],
                },
                "https://auth.example/.well-known/oauth-authorization-server": weak,
            }
        )
        with pytest.raises(OAuthFlowError, match="S256"):
            await discover_authorization_server(http, "https://srv.example/mcp", None)

    async def test_refuses_http_authorization_server(self):
        """A non-TLS authorization server is an attack, not a configuration."""
        http = _mock_http(
            {
                "https://srv.example/.well-known/oauth-protected-resource/mcp": {
                    "authorization_servers": ["http://auth.example"],
                },
            }
        )
        with pytest.raises(OAuthFlowError, match="https"):
            await discover_authorization_server(http, "https://srv.example/mcp", None)


class TestStateValidation:
    async def test_state_mismatch_aborts_flow(self, monkeypatch):
        """A callback with the wrong state is a CSRF attempt: hard abort."""

        @asynccontextmanager
        async def fake_callback_server():
            class FakeListener:
                redirect_uri = "http://127.0.0.1:9/callback"

                async def wait(self):
                    return CallbackResult(code="stolen-code", state="WRONG-STATE")

            yield FakeListener()

        monkeypatch.setattr("mcp_client.oauth.run_callback_server", fake_callback_server)
        monkeypatch.setattr("webbrowser.open", lambda *_: True)

        info = AuthServerInfo(
            issuer="https://auth.example",
            authorization_endpoint="https://auth.example/authorize",
            token_endpoint="https://auth.example/token",
            registration_endpoint=None,
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(500))
        ) as http:
            with pytest.raises(OAuthFlowError, match="State mismatch"):
                await run_authorization_flow(
                    http, info, "https://srv.example/mcp", "client-123", None
                )

    async def test_denied_authorization_surfaces_error(self, monkeypatch):
        @asynccontextmanager
        async def fake_callback_server():
            class FakeListener:
                redirect_uri = "http://127.0.0.1:9/callback"

                async def wait(self):
                    return CallbackResult(error="access_denied")

            yield FakeListener()

        monkeypatch.setattr("mcp_client.oauth.run_callback_server", fake_callback_server)
        monkeypatch.setattr("webbrowser.open", lambda *_: True)

        info = AuthServerInfo(
            issuer="https://auth.example",
            authorization_endpoint="https://auth.example/authorize",
            token_endpoint="https://auth.example/token",
            registration_endpoint=None,
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(500))
        ) as http:
            with pytest.raises(OAuthFlowError, match="access_denied"):
                await run_authorization_flow(
                    http, info, "https://srv.example/mcp", "client-123", None
                )
