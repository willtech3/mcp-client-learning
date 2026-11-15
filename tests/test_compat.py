"""Tests for era detection and the unified connect() fallback.

basic/versioning §4.6: a dual-era client attempts a modern request first and
inspects the failure body — only a *recognized modern* JSON-RPC error keeps
it on the modern path; anything else (bare 404, HTML 400, legacy JSON-RPC
errors) means fall back to the legacy initialize path.
"""

import json
from contextlib import asynccontextmanager

import httpx
import pytest

from mcp_client.config import ClientSettings
from mcp_client.modern import compat
from mcp_client.modern.session import ModernSession
from mcp_client.modern.transport import ModernTransport
from mcp_client.modern.types import DiscoverResult

SETTINGS = ClientSettings(_env_file=None)

# Published 2026-07-28 shape (server/discover page): identity rides in
# result _meta, NOT as a top-level field.
DISCOVER_RESULT = {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28", "2025-11-25"],
    "capabilities": {"tools": {}, "extensions": {"io.modelcontextprotocol/skills": {}}},
    "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "virtual-library", "version": "2.0.0"}},
    "ttlMs": 3600000,
    "cacheScope": "public",
}

# Intermediate-draft shape still emitted by some servers (top-level serverInfo).
DRAFT_DISCOVER_RESULT = {
    **{k: v for k, v in DISCOVER_RESULT.items() if k != "_meta"},
    "serverInfo": {"name": "draft-server", "version": "0.9.0"},
}


def make_transport(handler, url="http://srv.example/mcp") -> ModernTransport:
    return ModernTransport(httpx.AsyncClient(transport=httpx.MockTransport(handler)), url)


def rpc_error(request: httpx.Request, code: int, message: str, data=None, status=400):
    body = json.loads(request.content)
    return httpx.Response(
        status,
        json={"jsonrpc": "2.0", "id": body["id"],
              "error": {"code": code, "message": message, "data": data}},
    )


@pytest.fixture(autouse=True)
def _fresh_era_cache():
    compat.reset_era_cache()
    yield
    compat.reset_era_cache()


class TestProbeMatrix:
    async def test_discover_result_means_modern(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["method"] == "server/discover"
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": DISCOVER_RESULT}
            )

        era = await compat.probe_era(make_transport(handler), SETTINGS)
        assert era == ("modern", "2026-07-28")

    @pytest.mark.parametrize(
        "result",
        [
            DRAFT_DISCOVER_RESULT,
            # serverInfo is a SHOULD — a server that omits identity entirely
            # is still modern and must not be pushed onto the legacy path.
            {k: v for k, v in DISCOVER_RESULT.items() if k != "_meta"},
        ],
        ids=["draft-top-level-serverInfo", "no-serverInfo"],
    )
    async def test_discover_result_variants_mean_modern(self, result):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

        era = await compat.probe_era(make_transport(handler), SETTINGS)
        assert era == ("modern", "2026-07-28")

    async def test_unsupported_version_error_means_modern_with_advertised_version(self):
        """-32022 identifies a modern server; retry with data.supported,
        never fall back to initialize."""

        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_error(
                request, -32022, "Unsupported protocol version",
                data={"supported": ["2026-07-28"], "requested": "2026-07-28"},
            )

        era = await compat.probe_era(make_transport(handler), SETTINGS)
        assert era == ("modern", "2026-07-28")

    async def test_unsupported_version_with_no_mutual_version_falls_back(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_error(
                request, -32022, "Unsupported protocol version",
                data={"supported": ["2099-01-01"], "requested": "2026-07-28"},
            )

        era = await compat.probe_era(make_transport(handler), SETTINGS)
        assert era == ("legacy", None)

    async def test_bare_404_means_legacy(self):
        """A legacy HTTP+SSE-era server 404s unknown paths with no JSON-RPC
        body — the classic fall-back signal."""
        era = await compat.probe_era(
            make_transport(lambda _: httpx.Response(404, text="Not Found")), SETTINGS
        )
        assert era == ("legacy", None)

    async def test_400_with_unrecognized_jsonrpc_error_means_legacy(self):
        """Legacy SDK servers answer pre-initialize requests with generic
        JSON-RPC errors; the fallback MUST NOT be keyed to one code, and none
        of the legacy codes are in the spec-reserved modern range."""

        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_error(request, -32600, "Bad Request: server not initialized")

        era = await compat.probe_era(make_transport(handler), SETTINGS)
        assert era == ("legacy", None)

    async def test_400_with_non_json_body_means_legacy(self):
        era = await compat.probe_era(
            make_transport(lambda _: httpx.Response(400, text="<html>Bad Request</html>")),
            SETTINGS,
        )
        assert era == ("legacy", None)

    async def test_other_modern_reserved_code_means_modern(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_error(request, -32020, "Header mismatch")

        era = await compat.probe_era(make_transport(handler), SETTINGS)
        assert era[0] == "modern"


class TestConnect:
    def _patch_http(self, monkeypatch, handler):
        real_client = httpx.AsyncClient  # bind BEFORE patching (avoid recursion)

        def factory(**_kwargs):
            return real_client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(compat.httpx, "AsyncClient", factory)

    def _patch_legacy(self, monkeypatch, log: list):
        @asynccontextmanager
        async def fake_legacy(target, settings, *, anonymous=False):
            log.append(target)
            yield "legacy-session"

        monkeypatch.setattr(compat, "legacy_connect", fake_legacy)

    async def test_modern_server_yields_modern_session(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": DISCOVER_RESULT}
            )

        self._patch_http(monkeypatch, handler)
        legacy_calls: list = []
        self._patch_legacy(monkeypatch, legacy_calls)

        async with compat.connect("http://modern.example/mcp", SETTINGS, anonymous=True) as s:
            assert isinstance(s, ModernSession)
            assert s.protocol_version == "2026-07-28"
        assert legacy_calls == []

    async def test_legacy_server_falls_back_to_sdk_path(self, monkeypatch):
        self._patch_http(monkeypatch, lambda _: httpx.Response(404, text="nope"))
        legacy_calls: list = []
        self._patch_legacy(monkeypatch, legacy_calls)

        async with compat.connect("http://legacy.example/mcp", SETTINGS, anonymous=True) as s:
            assert s == "legacy-session"
        assert legacy_calls == ["http://legacy.example/mcp"]

    async def test_era_is_cached_per_origin(self, monkeypatch):
        """Era is a property of the server: probe once per origin, not per
        command (the spec says clients SHOULD cache the determination)."""
        probes: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if body["method"] == "server/discover":
                probes.append(body["method"])
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"],
                      "result": DISCOVER_RESULT if body["method"] == "server/discover"
                      else {"resultType": "complete", "tools": []}},
            )

        self._patch_http(monkeypatch, handler)

        async with compat.connect("http://cached.example/mcp", SETTINGS, anonymous=True):
            pass
        async with compat.connect("http://cached.example/mcp", SETTINGS, anonymous=True):
            pass
        assert len(probes) == 1

    async def test_modern_first_toggle_skips_probe(self, monkeypatch):
        settings = ClientSettings(_env_file=None, modern_first=False)
        legacy_calls: list = []
        self._patch_legacy(monkeypatch, legacy_calls)

        async with compat.connect("http://any.example/mcp", settings, anonymous=True) as s:
            assert s == "legacy-session"
        assert legacy_calls == ["http://any.example/mcp"]

    async def test_stdio_target_uses_legacy_path(self, monkeypatch):
        legacy_calls: list = []
        self._patch_legacy(monkeypatch, legacy_calls)

        async with compat.connect("uv run server", SETTINGS, anonymous=True) as s:
            assert s == "legacy-session"
        assert legacy_calls == ["uv run server"]


class TestDiscoverResultIdentity:
    def test_server_info_from_result_meta(self):
        info = DiscoverResult.model_validate(DISCOVER_RESULT).server_info
        assert info is not None
        assert (info.name, info.version) == ("virtual-library", "2.0.0")

    def test_server_info_falls_back_to_draft_field(self):
        info = DiscoverResult.model_validate(DRAFT_DISCOVER_RESULT).server_info
        assert info is not None
        assert info.name == "draft-server"

    def test_meta_wins_over_draft_field(self):
        both = {**DRAFT_DISCOVER_RESULT, "_meta": DISCOVER_RESULT["_meta"]}
        info = DiscoverResult.model_validate(both).server_info
        assert info is not None
        assert info.name == "virtual-library"

    def test_malformed_meta_identity_is_ignored(self):
        bad = {**DISCOVER_RESULT, "_meta": {"io.modelcontextprotocol/serverInfo": {"name": 1}}}
        assert DiscoverResult.model_validate(bad).server_info is None
