"""Streamable HTTP for MCP 2026-07-28, hand-rolled on httpx.

The 2026-07-28 transport is radically simpler than its predecessors and this
module makes that visible:

- ONE HTTP POST per JSON-RPC message to a single MCP endpoint. There is no
  GET stream, no DELETE, and — crucially — **no session**: the
  ``Mcp-Session-Id`` header was removed outright (SEP-2567). We never mint,
  echo, or expect one.
- Required mirrored headers (SEP-2243): every POST carries
  ``MCP-Protocol-Version`` and ``Mcp-Method``; ``tools/call``,
  ``resources/read`` and ``prompts/get`` additionally carry ``Mcp-Name``
  (the tool/prompt name or resource URI). The body stays the source of
  truth; the headers exist so intermediaries can route without parsing JSON,
  and servers reject mismatches with ``-32020 HeaderMismatch``. Values that
  are not plain ASCII are Base64-sentinel encoded (types.encode_header_value).
- The response is either a single JSON object (``application/json``) or a
  request-scoped SSE stream (``text/event-stream``) carrying notifications
  followed by the final response. Clients MUST accept both, hence the
  two-valued ``Accept`` header on every POST.
- SSE is parsed by hand: ``data:`` lines accumulate an event, a blank line
  dispatches it, and comment lines (leading ``:``) — the keep-alive
  mechanism — are ignored per the SSE spec. There are **no** SSE event ids
  and no ``Last-Event-ID`` resumption (removed by SEP-2575): a broken stream
  simply loses the in-flight request, and this module surfaces that as
  :class:`StreamClosedError` so the caller re-issues the request with a NEW
  JSON-RPC id.
- Cancellation on HTTP is closing the response stream — there is no
  ``notifications/cancelled`` message on this transport.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from .types import PROTOCOL_VERSION, encode_header_value

logger = logging.getLogger(__name__)

#: Methods whose params.name / params.uri must be mirrored into Mcp-Name.
_NAME_HEADER_METHODS = {"tools/call", "resources/read", "prompts/get"}


class ModernTransportError(Exception):
    """Base class for modern-transport failures."""


class McpProtocolError(ModernTransportError):
    """The server answered with a JSON-RPC error object.

    Carries the code/message/data triple so callers (the MRTR driver, era
    detection) can react to specific spec codes such as -32022.
    """

    def __init__(self, code: int, message: str, data: Any = None, http_status: int | None = None):
        super().__init__(f"MCP error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data
        self.http_status = http_status


class NotJsonRpcError(ModernTransportError):
    """The server answered with something that is not a JSON-RPC message.

    Era detection treats this as the legacy signal (basic/versioning §4.6:
    a 400/404 whose body is NOT a recognized modern error means fall back).
    """

    def __init__(self, status_code: int, body: str):
        super().__init__(f"Non-JSON-RPC response (HTTP {status_code})")
        self.status_code = status_code
        self.body = body


class StreamClosedError(ModernTransportError):
    """An SSE response stream ended before the final JSON-RPC response.

    SEP-2575 removed resumability: there is no Last-Event-ID replay, so the
    in-flight request is simply LOST. Callers MUST re-issue it as a brand
    new request with a NEW JSON-RPC id — never retry the old id.
    """


async def _iter_sse_messages(response: httpx.Response) -> AsyncIterator[dict]:
    """Parse an SSE byte stream into JSON-RPC message dicts.

    Implements the subset of the SSE grammar the spec relies on:
    - ``data:`` field lines accumulate (multi-line data joins with newline);
    - an empty line dispatches the accumulated event;
    - lines starting with ``:`` are comments (the keep-alive convention —
      servers are encouraged to emit ``:keepalive`` periodically) and MUST
      be ignored;
    - ``event:``/``id:``/``retry:`` fields are ignored — 2026-07-28 assigns
      no meaning to event names and REMOVED event ids entirely.
    """
    data_lines: list[str] = []
    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r")
        if not line:  # blank line = dispatch event
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("Ignoring non-JSON SSE event: %.120s", payload)
            continue
        if line.startswith(":"):  # comment / keep-alive — ignore per SSE spec
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data_lines.append(value.removeprefix(" "))
        # any other field (event, id, retry) carries no meaning here
    # Stream ended; flush a final un-terminated event if the server closed
    # without the trailing blank line (lenient parse).
    if data_lines:
        try:
            yield json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            logger.warning("Ignoring trailing non-JSON SSE data")


class ModernTransport:
    """One MCP endpoint + one httpx client = the entire transport state.

    Compare with the legacy SDK transport, which tracked a session id, a GET
    stream, and resumption tokens. Here the only mutable state is a request
    id counter — everything else about a request is carried by the request.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        endpoint: str,
        protocol_version: str = PROTOCOL_VERSION,
    ):
        self._http = http
        self.endpoint = endpoint
        self.protocol_version = protocol_version
        self._ids = itertools.count(1)

    # -- header construction --------------------------------------------------

    def build_headers(
        self,
        method: str,
        name: str | None = None,
        param_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Assemble the required (and optional Mcp-Param-*) headers for a POST.

        Streamable HTTP §2.3/§2.6: Accept MUST list both response content
        types; MCP-Protocol-Version MUST match the body _meta value (the
        session builds both from the same attribute, so they cannot drift);
        Mcp-Method mirrors ``method``; Mcp-Name mirrors params.name/params.uri
        for the three MRTR-capable methods, sentinel-encoded when needed.
        """
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.protocol_version,
            "Mcp-Method": method,
        }
        if method in _NAME_HEADER_METHODS:
            if name is None:
                raise ValueError(f"{method} requires an Mcp-Name header value")
            headers["Mcp-Name"] = encode_header_value(name)
        for header_name, raw_value in (param_headers or {}).items():
            # SEP-2243 custom headers: Mcp-Param-{Name}, sentinel-encoded
            # under the same ASCII rules as Mcp-Name.
            headers[f"Mcp-Param-{header_name}"] = encode_header_value(raw_value)
        return headers

    # -- request/response ------------------------------------------------------

    async def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        name: str | None = None,
        param_headers: dict[str, str] | None = None,
        on_notification: Callable[[dict], None] | None = None,
    ) -> dict[str, Any]:
        """POST one JSON-RPC request; return the result object.

        Handles both response shapes: a plain JSON body, or an SSE stream
        whose notifications are forwarded to `on_notification` and whose
        final response resolves the call. JSON-RPC errors raise
        :class:`McpProtocolError` regardless of which shape carried them
        (modern servers put spec errors on 400/404 HTTP statuses with a
        JSON-RPC body — the body is what we trust).
        """
        request_id = next(self._ids)
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        headers = self.build_headers(method, name=name, param_headers=param_headers)

        try:
            async with self._http.stream(
                "POST", self.endpoint, json=message, headers=headers
            ) as response:
                content_type = response.headers.get("content-type", "")
                if content_type.startswith("text/event-stream"):
                    async for msg in _iter_sse_messages(response):
                        if "result" in msg or "error" in msg:
                            # The final response terminates the stream. Each
                            # request has its OWN stream, so correlation is
                            # trivial — no demultiplexing needed on HTTP.
                            return self._unwrap(msg, response.status_code)
                        if msg.get("method", "").startswith("notifications/") and on_notification:
                            on_notification(msg)
                    raise StreamClosedError(
                        "SSE stream closed before the response arrived; the request is "
                        "lost (no resumability in 2026-07-28) — re-issue with a new id"
                    )
                body = (await response.aread()).decode("utf-8", errors="replace")
                return self._parse_json_body(body, response.status_code)
        except httpx.TransportError as exc:
            # Mid-stream disconnects surface as httpx.ReadError etc.; map them
            # to the same typed error so callers have ONE re-issue signal.
            raise StreamClosedError(f"Connection broken during request: {exc}") from exc

    def _parse_json_body(self, body: str, status_code: int) -> dict[str, Any]:
        try:
            msg = json.loads(body)
        except json.JSONDecodeError:
            raise NotJsonRpcError(status_code, body) from None
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            raise NotJsonRpcError(status_code, body)
        return self._unwrap(msg, status_code)

    @staticmethod
    def _unwrap(msg: dict[str, Any], status_code: int) -> dict[str, Any]:
        if "error" in msg:
            err = msg["error"] or {}
            raise McpProtocolError(
                code=err.get("code", 0),
                message=err.get("message", ""),
                data=err.get("data"),
                http_status=status_code,
            )
        result = msg.get("result")
        if not isinstance(result, dict):
            raise NotJsonRpcError(status_code, json.dumps(msg))
        return result

    # -- long-lived streams (subscriptions/listen) -----------------------------

    async def stream(
        self,
        method: str,
        params: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """POST a request whose response is a long-lived SSE stream.

        Yields every JSON-RPC *notification* on the stream (the first will be
        ``notifications/subscriptions/acknowledged``). A JSON-RPC *response*
        is the server's graceful closure (SubscriptionsListenResult) and ends
        iteration; an abrupt disconnect raises :class:`StreamClosedError` so
        the caller can re-subscribe with a fresh request (there is no replay
        of missed notifications).
        """
        request_id = next(self._ids)
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        headers = self.build_headers(method)
        try:
            async with self._http.stream(
                "POST", self.endpoint, json=message, headers=headers
            ) as response:
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("text/event-stream"):
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    self._parse_json_body(body, response.status_code)
                    return  # a bare JSON response = immediately-closed stream
                async for msg in _iter_sse_messages(response):
                    if "result" in msg or "error" in msg:
                        # Graceful closure: the deferred response to the
                        # listen request itself (resultType complete).
                        self._unwrap(msg, response.status_code)
                        return
                    yield msg
        except httpx.TransportError as exc:
            raise StreamClosedError(f"Listen stream broken: {exc}") from exc
