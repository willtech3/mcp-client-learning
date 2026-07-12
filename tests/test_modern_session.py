"""Tests for ModernSession: the MRTR driver, per-request _meta, and the
client-side x-mcp-header duties (SEP-2243).

Servers are scripted httpx.MockTransport handlers; input-request responders
are injected stubs so no terminal prompting happens.
"""

import json
import logging

import httpx
import pytest
from mcp import types as sdk_types

from mcp_client.handlers import SamplingHandler
from mcp_client.modern.session import (
    InvalidHeaderAnnotationError,
    ModernSession,
    ModernSessionError,
    collect_header_params,
    extract_param_headers,
)
from mcp_client.modern.transport import ModernTransport
from mcp_client.modern.types import (
    MAX_ROUND_TRIPS,
    META_CLIENT_CAPS,
    META_CLIENT_INFO,
    META_PROTOCOL_VERSION,
)

ENDPOINT = "http://srv.example/mcp"


def make_session(handler, **kwargs) -> ModernSession:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ModernSession(ModernTransport(client, ENDPOINT), **kwargs)


def rpc_result(request: httpx.Request, result: dict) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


async def accept_elicitation(_params: dict) -> dict:
    return {"action": "accept", "content": {"answer": "yes"}}


class TestPerRequestMeta:
    async def test_meta_carries_version_info_and_capabilities(self):
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return rpc_result(request, {"resultType": "complete", "tools": []})

        session = make_session(handler)
        await session.list_tools()

        meta = seen[0]["params"]["_meta"]
        assert meta[META_PROTOCOL_VERSION] == "2026-07-28"
        assert meta[META_CLIENT_INFO]["name"] == "mcp-client-learning"
        assert meta[META_CLIENT_INFO]["version"]
        # Elicitation (both modes) is always declared — the terminal handler
        # needs nothing external.
        assert meta[META_CLIENT_CAPS]["elicitation"] == {"form": {}, "url": {}}

    async def test_sampling_capability_only_with_api_key(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_result(request, {"resultType": "complete", "tools": []})

        # No API key -> handler has no client -> capability NOT declared.
        without_key = make_session(handler, sampling=SamplingHandler(api_key=None))
        assert "sampling" not in without_key._client_capabilities()

        # Key configured -> declared, including SEP-1577 tool support.
        with_key = make_session(handler, sampling=SamplingHandler(api_key="sk-test"))
        assert with_key._client_capabilities()["sampling"] == {"tools": {}}

    async def test_roots_capability_only_with_static_roots(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_result(request, {"resultType": "complete", "tools": []})

        assert "roots" not in make_session(handler)._client_capabilities()
        with_roots = make_session(handler, static_roots=["file:///tmp/proj"])
        assert with_roots._client_capabilities()["roots"] == {}


class TestMrtrLoop:
    async def test_multi_round_trip_echoes_state_and_uses_new_ids(self):
        """Two input_required rounds, then completion: ids must all differ,
        requestState must be echoed verbatim, responses keyed correctly."""
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body)
            round_number = len(calls)
            if round_number == 1:
                return rpc_result(
                    request,
                    {
                        "resultType": "input_required",
                        "inputRequests": {
                            "q1": {
                                "method": "elicitation/create",
                                "params": {"mode": "form", "message": "Round 1?",
                                           "requestedSchema": {"type": "object", "properties": {}}},
                            }
                        },
                        "requestState": "state-round-1",
                    },
                )
            if round_number == 2:
                return rpc_result(
                    request,
                    {
                        "resultType": "input_required",
                        "inputRequests": {
                            "q2": {
                                "method": "elicitation/create",
                                "params": {"message": "Round 2?",  # mode omitted = form
                                           "requestedSchema": {"type": "object", "properties": {}}},
                            }
                        },
                        "requestState": "state-round-2",
                    },
                )
            return rpc_result(request, {"resultType": "complete", "content": [
                {"type": "text", "text": "done"}], "isError": False})

        session = make_session(
            handler, input_responders={"elicitation/create": accept_elicitation}
        )
        result = await session.call_tool("checkout_book", {"isbn": "123"})

        assert len(calls) == 3
        # Client requirement 3: a NEW JSON-RPC id per retry.
        assert len({c["id"] for c in calls}) == 3
        # First call carries neither inputResponses nor requestState.
        assert "inputResponses" not in calls[0]["params"]
        assert "requestState" not in calls[0]["params"]
        # Retry 1 echoes round-1 state VERBATIM and answers key q1.
        assert calls[1]["params"]["requestState"] == "state-round-1"
        assert calls[1]["params"]["inputResponses"] == {
            "q1": {"action": "accept", "content": {"answer": "yes"}}
        }
        # Retry 2 echoes round-2 state and answers key q2 (not q1).
        assert calls[2]["params"]["requestState"] == "state-round-2"
        assert set(calls[2]["params"]["inputResponses"]) == {"q2"}
        # Original arguments repeat on every retry (it's the SAME request).
        assert all(c["params"]["arguments"] == {"isbn": "123"} for c in calls)
        assert result.isError is False
        assert isinstance(result.content[0], sdk_types.TextContent)

    async def test_state_only_result_retries_immediately(self):
        """requestState with no inputRequests (load shedding) is legal: the
        client MAY retry immediately, echoing the state."""
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                return rpc_result(
                    request, {"resultType": "input_required", "requestState": "opaque-blob"}
                )
            return rpc_result(request, {"resultType": "complete", "content": []})

        session = make_session(handler)
        await session.call_tool("slow_tool", {})
        assert calls[1]["params"]["requestState"] == "opaque-blob"
        assert "inputResponses" not in calls[1]["params"]

    async def test_absent_request_state_is_not_invented(self):
        """When the result omits requestState the retry MUST omit it too."""
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                return rpc_result(
                    request,
                    {
                        "resultType": "input_required",
                        "inputRequests": {
                            "k": {"method": "elicitation/create",
                                  "params": {"message": "?", "requestedSchema": {
                                      "type": "object", "properties": {}}}}
                        },
                    },
                )
            return rpc_result(request, {"resultType": "complete", "content": []})

        session = make_session(
            handler, input_responders={"elicitation/create": accept_elicitation}
        )
        await session.call_tool("t", {})
        assert "requestState" not in calls[1]["params"]
        assert "inputResponses" in calls[1]["params"]

    async def test_round_trip_cap(self):
        """A server that never completes gets cut off at MAX_ROUND_TRIPS."""
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content))
            return rpc_result(
                request, {"resultType": "input_required", "requestState": f"s{len(calls)}"}
            )

        session = make_session(handler)
        with pytest.raises(ModernSessionError, match="round trips"):
            await session.call_tool("t", {})
        assert len(calls) == MAX_ROUND_TRIPS

    async def test_unknown_result_type_is_an_error(self):
        """Unrecognized resultType values MUST be treated as invalid."""

        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_result(request, {"resultType": "partial", "content": []})

        session = make_session(handler)
        with pytest.raises(ModernSessionError, match="partial"):
            await session.call_tool("t", {})

    async def test_absent_result_type_treated_as_complete(self):
        """Back-compat rule: earlier-protocol servers omit resultType."""

        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_result(request, {"content": [{"type": "text", "text": "ok"}]})

        session = make_session(handler)
        result = await session.call_tool("t", {})
        assert result.result_type == "complete"

    async def test_input_required_without_requests_or_state_rejected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_result(request, {"resultType": "input_required"})

        session = make_session(handler)
        with pytest.raises(ModernSessionError, match="neither"):
            await session.call_tool("t", {})

    async def test_roots_input_request_served_from_static_config(self):
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                return rpc_result(
                    request,
                    {"resultType": "input_required",
                     "inputRequests": {"r": {"method": "roots/list"}},
                     "requestState": "s"},
                )
            return rpc_result(request, {"resultType": "complete", "content": []})

        session = make_session(handler, static_roots=["file:///home/me/proj"])
        await session.call_tool("t", {})
        assert calls[1]["params"]["inputResponses"]["r"] == {
            "roots": [{"uri": "file:///home/me/proj"}]
        }

    async def test_unsupported_input_request_method_aborts(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_result(
                request,
                {"resultType": "input_required",
                 "inputRequests": {"x": {"method": "who/knows"}}},
            )

        session = make_session(handler)
        with pytest.raises(ModernSessionError, match="who/knows"):
            await session.call_tool("t", {})

    async def test_sampling_without_key_does_not_retry_blind(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return rpc_result(
                request,
                {"resultType": "input_required",
                 "inputRequests": {"s": {"method": "sampling/createMessage",
                                         "params": {"messages": [], "maxTokens": 10}}}},
            )

        session = make_session(handler, sampling=SamplingHandler(api_key=None))
        with pytest.raises(ModernSessionError, match="API key"):
            await session.call_tool("t", {})

    async def test_mrtr_applies_to_resources_read_and_prompts_get(self):
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body)
            if body["method"] == "resources/read" and len(
                [c for c in calls if c["method"] == "resources/read"]
            ) == 1:
                return rpc_result(
                    request, {"resultType": "input_required", "requestState": "rs"}
                )
            if body["method"] == "resources/read":
                return rpc_result(request, {"resultType": "complete", "contents": [
                    {"uri": "library://books/1", "mimeType": "text/plain", "text": "hi"}]})
            return rpc_result(request, {"resultType": "complete", "messages": [
                {"role": "user", "content": {"type": "text", "text": "prompt"}}]})

        session = make_session(handler)
        read = await session.read_resource("library://books/1")
        assert read.contents[0].text == "hi"
        prompt = await session.get_prompt("review", {"genre": "scifi"})
        assert prompt.messages[0].content.text == "prompt"
        prompt_call = next(c for c in calls if c["method"] == "prompts/get")
        assert prompt_call["params"]["arguments"] == {"genre": "scifi"}


VALID_ANNOTATED_TOOL = {
    "name": "execute_sql",
    "description": "Run SQL",
    "inputSchema": {
        "type": "object",
        "properties": {
            "region": {"type": "string", "x-mcp-header": "Region"},
            "query": {"type": "string"},
            "options": {
                "type": "object",
                "properties": {
                    "row_limit": {"type": "integer", "x-mcp-header": "Row-Limit"},
                },
            },
        },
        "required": ["region", "query"],
    },
}


class TestHeaderAnnotationValidation:
    def test_valid_annotations_collected_with_paths(self):
        params = collect_header_params(VALID_ANNOTATED_TOOL["inputSchema"])
        assert params["Region"] == (("region",), "string")
        # Nested object properties are fine when every step is `properties`.
        assert params["Row-Limit"] == (("options", "row_limit"), "integer")

    def test_empty_value_rejected(self):
        schema = {"type": "object", "properties": {"a": {"type": "string", "x-mcp-header": ""}}}
        with pytest.raises(InvalidHeaderAnnotationError):
            collect_header_params(schema)

    def test_non_token_chars_rejected(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string", "x-mcp-header": "Bad Header"}},
        }
        with pytest.raises(InvalidHeaderAnnotationError, match="token"):
            collect_header_params(schema)

    def test_number_type_rejected(self):
        """`number` is explicitly NOT permitted (only integer/string/boolean)."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "number", "x-mcp-header": "Amount"}},
        }
        with pytest.raises(InvalidHeaderAnnotationError, match="number"):
            collect_header_params(schema)

    def test_case_insensitive_duplicates_rejected(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "x-mcp-header": "Region"},
                "b": {"type": "string", "x-mcp-header": "REGION"},
            },
        }
        with pytest.raises(InvalidHeaderAnnotationError, match="unique"):
            collect_header_params(schema)

    def test_annotation_through_items_rejected(self):
        """The property chain MUST NOT pass through array keywords."""
        schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string", "x-mcp-header": "Id"}},
                    },
                }
            },
        }
        with pytest.raises(InvalidHeaderAnnotationError, match="chain"):
            collect_header_params(schema)

    def test_annotation_through_one_of_rejected(self):
        schema = {
            "type": "object",
            "properties": {
                "cfg": {
                    "oneOf": [
                        {"type": "object",
                         "properties": {"x": {"type": "string", "x-mcp-header": "X"}}},
                    ]
                }
            },
        }
        with pytest.raises(InvalidHeaderAnnotationError, match="chain"):
            collect_header_params(schema)

    def test_value_extraction_and_omission(self):
        params = collect_header_params(VALID_ANNOTATED_TOOL["inputSchema"])
        headers = extract_param_headers(
            params, {"region": "us-west1", "options": {"row_limit": 42}}
        )
        assert headers == {"Region": "us-west1", "Row-Limit": "42"}
        # Absent and null values -> header omitted (server MUST NOT expect it).
        assert extract_param_headers(params, {"query": "SELECT 1"}) == {}
        assert extract_param_headers(params, {"region": None}) == {}

    def test_boolean_conversion_lowercase(self):
        schema = {
            "type": "object",
            "properties": {"dry_run": {"type": "boolean", "x-mcp-header": "Dry-Run"}},
        }
        params = collect_header_params(schema)
        assert extract_param_headers(params, {"dry_run": True}) == {"Dry-Run": "true"}
        assert extract_param_headers(params, {"dry_run": False}) == {"Dry-Run": "false"}


INVALID_ANNOTATED_TOOL = {
    "name": "broken_tool",
    "inputSchema": {
        "type": "object",
        "properties": {"a": {"type": "number", "x-mcp-header": "Amount"}},
    },
}


class TestHeaderMirroringInSession:

    def _handler(self, seen: dict):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if body["method"] == "tools/list":
                return rpc_result(
                    request,
                    {"resultType": "complete", "ttlMs": 300000, "cacheScope": "public",
                     "tools": [VALID_ANNOTATED_TOOL, INVALID_ANNOTATED_TOOL]},
                )
            seen["headers"] = request.headers
            return rpc_result(request, {"resultType": "complete", "content": []})

        return handler

    async def test_invalid_tool_excluded_with_warning(self, caplog):
        session = make_session(self._handler({}))
        with caplog.at_level(logging.WARNING):
            result = await session.list_tools()
        assert [t.name for t in result.tools] == ["execute_sql"]
        assert any("broken_tool" in r.message for r in caplog.records)
        # Cache hints surface on the result (SEP-2549).
        assert result.ttl_ms == 300000

    async def test_call_mirrors_annotated_params_from_latest_schema(self):
        seen: dict = {}
        session = make_session(self._handler(seen))
        await session.list_tools()
        await session.call_tool("execute_sql", {"region": "us-west1", "query": "SELECT 1"})
        assert seen["headers"]["mcp-param-region"] == "us-west1"
        assert "mcp-param-row-limit" not in seen["headers"]

    async def test_call_without_schema_sends_no_param_headers(self):
        """A client that never fetched the schema SHOULD send without
        Mcp-Param-* headers (spec fallback)."""
        seen: dict = {}
        session = make_session(self._handler(seen))
        await session.call_tool("execute_sql", {"region": "us-west1", "query": "SELECT 1"})
        assert not any(h.startswith("mcp-param-") for h in seen["headers"])


class TestListen:
    async def test_listen_yields_notifications(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["method"] == "subscriptions/listen"
            assert body["params"]["notifications"] == {"toolsListChanged": True}
            meta = {"io.modelcontextprotocol/subscriptionId": body["id"]}
            events = [
                {"jsonrpc": "2.0", "method": "notifications/subscriptions/acknowledged",
                 "params": {"_meta": meta, "notifications": {"toolsListChanged": True}}},
                {"jsonrpc": "2.0", "method": "notifications/tools/list_changed",
                 "params": {"_meta": meta}},
                {"jsonrpc": "2.0", "id": body["id"],
                 "result": {"resultType": "complete", "_meta": meta}},
            ]
            content = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=content.encode()
            )

        session = make_session(handler)
        messages = [m async for m in session.listen({"toolsListChanged": True})]
        assert [m["method"] for m in messages] == [
            "notifications/subscriptions/acknowledged",
            "notifications/tools/list_changed",
        ]
