"""Tests for the from-scratch 2026-07-28 Streamable HTTP transport.

All wire traffic is canned via httpx.MockTransport — we test OUR header
construction, sentinel encoding, SSE parsing, and error classification, not
the network.
"""

import json

import httpx
import pytest

from mcp_client.modern.transport import (
    McpProtocolError,
    ModernTransport,
    NotJsonRpcError,
    StreamClosedError,
)
from mcp_client.modern.types import decode_header_value, encode_header_value

ENDPOINT = "http://srv.example/mcp"


def make_transport(handler) -> ModernTransport:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ModernTransport(client, ENDPOINT)


def rpc_result(request: httpx.Request, result: dict) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


def sse_response(*events: dict | str, status: int = 200) -> httpx.Response:
    """Build an SSE body; dict events become data frames, str lines verbatim."""
    lines: list[str] = []
    for event in events:
        if isinstance(event, dict):
            lines.append(f"data: {json.dumps(event)}")
            lines.append("")
        else:
            lines.append(event)
    return httpx.Response(
        status,
        headers={"content-type": "text/event-stream"},
        content=("\n".join(lines) + "\n").encode(),
    )


class TestSentinelCodec:
    """streamable-http §2.8 examples, verbatim from the spec."""

    def test_plain_ascii_passes_through(self):
        assert encode_header_value("us-west1") == "us-west1"

    def test_non_ascii_is_encoded(self):
        assert encode_header_value("Hello, 世界") == "=?base64?SGVsbG8sIOS4lueVjA==?="

    def test_leading_trailing_whitespace_is_encoded(self):
        assert encode_header_value(" padded ") == "=?base64?IHBhZGRlZCA=?="

    def test_control_chars_are_encoded(self):
        assert encode_header_value("line1\nline2") == "=?base64?bGluZTEKbGluZTI=?="

    def test_literal_sentinel_is_reencoded(self):
        """Anti-ambiguity: a plain value matching the sentinel pattern MUST
        itself be encoded so decoders never have to guess."""
        assert encode_header_value("=?base64?literal?=") == "=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="

    @pytest.mark.parametrize(
        "value",
        ["us-west1", "Hello, 世界", " padded ", "line1\nline2", "=?base64?literal?=", "true", "42"],
    )
    def test_round_trip(self, value):
        assert decode_header_value(encode_header_value(value)) == value

    def test_decode_leaves_plain_values_alone(self):
        assert decode_header_value("plain-value") == "plain-value"

    def test_decode_rejects_malformed_base64(self):
        with pytest.raises(ValueError, match="sentinel"):
            decode_header_value("=?base64?!!!not-base64!!!?=")


class TestHeaderConstruction:
    async def test_required_headers_on_tools_call(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = request.headers
            seen["body"] = json.loads(request.content)
            return rpc_result(request, {"resultType": "complete"})

        transport = make_transport(handler)
        await transport.request("tools/call", {"name": "get_weather"}, name="get_weather")

        headers = seen["headers"]
        assert headers["mcp-protocol-version"] == "2026-07-28"
        assert headers["mcp-method"] == "tools/call"
        assert headers["mcp-name"] == "get_weather"
        # Client MUST accept both response content types.
        assert "application/json" in headers["accept"]
        assert "text/event-stream" in headers["accept"]
        # No session artifacts, ever (SEP-2567).
        assert "mcp-session-id" not in headers
        assert "last-event-id" not in headers

    async def test_mcp_name_uses_sentinel_for_non_ascii(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["name"] = request.headers["mcp-name"]
            return rpc_result(request, {"resultType": "complete"})

        transport = make_transport(handler)
        await transport.request("resources/read", {"uri": "library://livres/世界"}, name="library://livres/世界")
        assert seen["name"].startswith("=?base64?")
        assert decode_header_value(seen["name"]) == "library://livres/世界"

    async def test_no_name_header_on_list_methods(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = request.headers
            return rpc_result(request, {"resultType": "complete", "tools": []})

        transport = make_transport(handler)
        await transport.request("tools/list", {})
        assert "mcp-name" not in seen["headers"]
        assert seen["headers"]["mcp-method"] == "tools/list"

    async def test_name_required_for_mrtr_methods(self):
        transport = make_transport(lambda request: rpc_result(request, {}))
        with pytest.raises(ValueError, match="Mcp-Name"):
            await transport.request("prompts/get", {"name": "x"})

    async def test_param_headers_are_appended_and_encoded(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = request.headers
            return rpc_result(request, {"resultType": "complete"})

        transport = make_transport(handler)
        await transport.request(
            "tools/call",
            {"name": "execute_sql"},
            name="execute_sql",
            param_headers={"Region": "us-west1", "Greeting": "Hello, 世界"},
        )
        assert seen["headers"]["mcp-param-region"] == "us-west1"
        assert seen["headers"]["mcp-param-greeting"] == "=?base64?SGVsbG8sIOS4lueVjA==?="


class TestJsonResponses:
    async def test_json_rpc_error_raises_typed_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(
                400,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "error": {
                        "code": -32022,
                        "message": "Unsupported protocol version",
                        "data": {"supported": ["2026-07-28"], "requested": "1900-01-01"},
                    },
                },
            )

        transport = make_transport(handler)
        with pytest.raises(McpProtocolError) as excinfo:
            await transport.request("server/discover", {})
        assert excinfo.value.code == -32022
        assert excinfo.value.data["supported"] == ["2026-07-28"]
        assert excinfo.value.http_status == 400

    async def test_non_json_body_raises_notjsonrpc(self):
        transport = make_transport(lambda _: httpx.Response(404, text="Not Found"))
        with pytest.raises(NotJsonRpcError) as excinfo:
            await transport.request("server/discover", {})
        assert excinfo.value.status_code == 404

    async def test_json_but_not_jsonrpc_raises_notjsonrpc(self):
        transport = make_transport(lambda _: httpx.Response(400, json={"detail": "bad request"}))
        with pytest.raises(NotJsonRpcError):
            await transport.request("server/discover", {})


class TestSseParsing:
    async def test_notifications_then_final_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return sse_response(
                {"jsonrpc": "2.0", "method": "notifications/progress",
                 "params": {"progressToken": "t", "progress": 1}},
                {"jsonrpc": "2.0", "method": "notifications/message",
                 "params": {"level": "info", "data": "working"}},
                {"jsonrpc": "2.0", "id": body["id"], "result": {"resultType": "complete", "ok": True}},
            )

        notifications: list[dict] = []
        transport = make_transport(handler)
        result = await transport.request(
            "tools/call", {"name": "t"}, name="t", on_notification=notifications.append
        )
        assert result == {"resultType": "complete", "ok": True}
        assert [n["method"] for n in notifications] == [
            "notifications/progress",
            "notifications/message",
        ]

    async def test_comment_lines_are_ignored(self):
        """Keep-alive comment lines (leading colon) MUST be ignored per SSE."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return sse_response(
                ":keepalive",
                ": another comment",
                {"jsonrpc": "2.0", "id": body["id"], "result": {"resultType": "complete"}},
            )

        transport = make_transport(handler)
        result = await transport.request("tools/call", {"name": "t"}, name="t")
        assert result == {"resultType": "complete"}

    async def test_midstream_disconnect_raises_stream_closed(self):
        """A stream that ends without the final response = lost request.
        No Last-Event-ID resumption exists; callers re-issue with a new id."""

        def handler(_: httpx.Request) -> httpx.Response:
            return sse_response(
                {"jsonrpc": "2.0", "method": "notifications/progress",
                 "params": {"progressToken": "t", "progress": 1}},
                # ... and the connection drops before the response arrives.
            )

        transport = make_transport(handler)
        with pytest.raises(StreamClosedError, match="new id"):
            await transport.request("tools/call", {"name": "t"}, name="t")

    async def test_error_delivered_on_stream_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return sse_response(
                {"jsonrpc": "2.0", "id": body["id"],
                 "error": {"code": -32021, "message": "missing capability",
                           "data": {"requiredCapabilities": {"elicitation": {}}}}},
            )

        transport = make_transport(handler)
        with pytest.raises(McpProtocolError) as excinfo:
            await transport.request("tools/call", {"name": "t"}, name="t")
        assert excinfo.value.code == -32021


class TestListenStream:
    async def test_stream_yields_until_graceful_close(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            sub_id = body["id"]
            meta = {"io.modelcontextprotocol/subscriptionId": sub_id}
            return sse_response(
                {"jsonrpc": "2.0", "method": "notifications/subscriptions/acknowledged",
                 "params": {"_meta": meta, "notifications": {"toolsListChanged": True}}},
                ":keepalive",
                {"jsonrpc": "2.0", "method": "notifications/tools/list_changed",
                 "params": {"_meta": meta}},
                {"jsonrpc": "2.0", "id": sub_id,
                 "result": {"resultType": "complete", "_meta": meta}},
            )

        transport = make_transport(handler)
        received = [msg async for msg in transport.stream("subscriptions/listen", {})]
        assert [m["method"] for m in received] == [
            "notifications/subscriptions/acknowledged",
            "notifications/tools/list_changed",
        ]

    async def test_new_request_ids_per_post(self):
        ids: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            ids.append(body["id"])
            return rpc_result(request, {"resultType": "complete"})

        transport = make_transport(handler)
        await transport.request("tools/list", {})
        await transport.request("tools/list", {})
        await transport.request("tools/list", {})
        assert len(set(ids)) == 3
