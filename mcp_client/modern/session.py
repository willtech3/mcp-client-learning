"""ModernSession — the client request surface for MCP 2026-07-28.

Two big ideas from the revision live here:

1. **Per-request metadata (SEP-2575).** There is no handshake and no session,
   so *every* request carries the protocol version, client identity, and
   client capabilities in ``params._meta``. Servers MUST NOT remember what a
   client declared on an earlier request, which means the capability set we
   build in :meth:`ModernSession._client_capabilities` is attached to every
   single call — declaring elicitation once and dropping it later is a legal
   (if odd) thing for a client to do.

2. **Multi Round-Trip Requests (MRTR, SEP-2322).** Servers can no longer
   send JSON-RPC requests to the client. When a tool needs user input, an LLM
   completion, or the (deprecated) roots list, the server instead *returns*
   ``resultType: "input_required"`` with an ``inputRequests`` map of bare
   ``{method, params}`` objects and an opaque ``requestState``. The client
   answers the requests locally — routing them to the exact same handlers the
   legacy path uses (``handlers.elicitation_handler``/``SamplingHandler``) —
   then RETRIES the original request with ``inputResponses`` plus the
   ``requestState`` echoed verbatim, under a NEW JSON-RPC id. The server
   re-executes from scratch, finding the answers it needs in the retry. No
   server instance had to remember anything: that is what lets 2026-07-28
   servers run behind dumb load balancers.

This module also discharges the client-side ``x-mcp-header`` duty (SEP-2243):
tool parameters annotated with ``x-mcp-header`` in the *most recently
fetched* ``tools/list`` schema are mirrored into ``Mcp-Param-{Name}``
headers on ``tools/call``, and tool definitions whose annotations violate the
constraints are excluded from ``tools/list`` results entirely (a spec MUST,
with a logged warning) — a client that forwarded a malformed header name
would hand the server a header-splitting primitive.
"""

from __future__ import annotations

import logging
import string
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from mcp import types as sdk_types

from ..handlers import SamplingHandler, elicit_from_wire, sample_from_wire
from . import types as wire
from .transport import ModernTransport

logger = logging.getLogger(__name__)

#: Signature of an input-request responder: wire params in, wire result out.
InputResponder = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class ModernSessionError(Exception):
    """Client-side protocol failure (MRTR misuse, invalid resultType, ...)."""


# ---------------------------------------------------------------------------
# x-mcp-header (SEP-2243) — validation and value extraction
# ---------------------------------------------------------------------------

# RFC 9110 §5.1 token characters — the only ones legal in a header *name*.
_TCHAR = frozenset("!#$%&'*+-.^_`|~" + string.ascii_letters + string.digits)

# JavaScript safe-integer range: annotated integer values outside it cannot
# round-trip through header comparison on JS servers.
_JS_SAFE_MIN, _JS_SAFE_MAX = -(2**53) + 1, 2**53 - 1

_MISSING = object()


class InvalidHeaderAnnotationError(ValueError):
    """A tool's x-mcp-header annotations violate SEP-2243 — the whole tool
    definition is invalid and MUST be excluded from tools/list results."""


def collect_header_params(schema: dict[str, Any]) -> dict[str, tuple[tuple[str, ...], str]]:
    """Validate every ``x-mcp-header`` annotation in an inputSchema.

    Returns ``{header_name: (property_path, declared_type)}`` for the valid
    annotations, or raises :class:`InvalidHeaderAnnotationError` when ANY
    annotation breaks the SEP-2243 constraints (one bad annotation poisons
    the whole tool definition):

    - value non-empty, RFC 9110 token syntax, no control chars;
    - case-insensitively unique within the schema;
    - only on primitive parameters typed ``integer``/``string``/``boolean``
      (``number`` is explicitly NOT permitted — float serialization is
      ambiguous across intermediaries);
    - only on properties statically reachable from the schema root through a
      chain of ``properties`` keys — never through ``items``, combinators
      (``oneOf``/``anyOf``/``allOf``/``not``/``if``/``then``/``else``) or
      ``$ref``, because an intermediary must be able to find the value
      without evaluating the schema.
    """
    found: list[tuple[Any, tuple[str, ...] | None, dict[str, Any]]] = []

    def walk(node: Any, path: tuple[str, ...] | None, on_chain: bool) -> None:
        if isinstance(node, dict):
            if "x-mcp-header" in node:
                legal = on_chain and path is not None and len(path) > 0
                found.append((node["x-mcp-header"], path if legal else None, node))
            for key, child in node.items():
                if key == "x-mcp-header":
                    continue
                if key == "properties" and isinstance(child, dict) and on_chain:
                    for prop_name, prop_schema in child.items():
                        walk(prop_schema, (*(path or ()), prop_name), True)
                else:
                    # Any other keyword (items, anyOf, $ref, ...) breaks the
                    # statically-reachable chain for everything beneath it.
                    walk(child, None, False)
        elif isinstance(node, list):
            for item in node:
                walk(item, None, False)

    walk(schema, (), True)

    result: dict[str, tuple[tuple[str, ...], str]] = {}
    seen_lower: set[str] = set()
    for annotation, path, node in found:
        if path is None:
            raise InvalidHeaderAnnotationError(
                f"x-mcp-header {annotation!r} is not reachable via a properties-only chain"
            )
        if not isinstance(annotation, str) or not annotation:
            raise InvalidHeaderAnnotationError("x-mcp-header value must be a non-empty string")
        if not all(ch in _TCHAR for ch in annotation):
            raise InvalidHeaderAnnotationError(
                f"x-mcp-header {annotation!r} is not valid RFC 9110 token syntax"
            )
        if annotation.lower() in seen_lower:
            raise InvalidHeaderAnnotationError(
                f"x-mcp-header {annotation!r} is not case-insensitively unique"
            )
        seen_lower.add(annotation.lower())
        declared_type = node.get("type")
        if declared_type not in ("integer", "string", "boolean"):
            raise InvalidHeaderAnnotationError(
                f"x-mcp-header {annotation!r} on non-primitive type {declared_type!r} "
                "(only integer/string/boolean are permitted; number is not)"
            )
        result[annotation] = (path, declared_type)
    return result


def extract_param_headers(
    header_params: dict[str, tuple[tuple[str, ...], str]],
    arguments: dict[str, Any],
) -> dict[str, str]:
    """Read each annotated value at its exact property path in `arguments`.

    Per the SEP-2243 behavior table: value present -> header included; value
    ``null`` or property absent -> header OMITTED (the server must not expect
    it). Type conversion: strings as-is, integers as decimal strings,
    booleans as lowercase ``true``/``false``.
    """
    headers: dict[str, str] = {}
    for header_name, (path, declared_type) in header_params.items():
        value: Any = arguments
        for segment in path:
            if not isinstance(value, dict) or segment not in value:
                value = _MISSING
                break
            value = value[segment]
        if value is _MISSING or value is None:
            continue
        if declared_type == "boolean":
            headers[header_name] = "true" if value else "false"
        elif declared_type == "integer":
            as_int = int(value)
            if not _JS_SAFE_MIN <= as_int <= _JS_SAFE_MAX:
                # Outside the JS safe range the mirrored value cannot be
                # validated reliably; omit and let the body speak alone.
                continue
            headers[header_name] = str(as_int)
        else:
            headers[header_name] = str(value)
    return headers


# ---------------------------------------------------------------------------
# Content-block parsing — reuse the SDK's item models so downstream code
# (the CLI's isinstance checks on TextContent, etc.) works on both eras.
# ---------------------------------------------------------------------------

_CONTENT_TYPES: dict[str, type] = {
    "text": sdk_types.TextContent,
    "image": sdk_types.ImageContent,
    "audio": sdk_types.AudioContent,
    "resource_link": sdk_types.ResourceLink,
    "resource": sdk_types.EmbeddedResource,
}


def _parse_content_block(block: Any) -> Any:
    if isinstance(block, dict):
        model = _CONTENT_TYPES.get(block.get("type", ""))
        if model is not None:
            try:
                return model.model_validate(block)
            except Exception:  # malformed block: keep the raw dict
                return block
    return block


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class ModernSession:
    """Stateless-protocol client session (the only state here is OURS).

    The attributes on this object — cached tool schemas, the transport's id
    counter — are client-side conveniences. Nothing about this object exists
    on the server, which is exactly the 2026-07-28 design goal.
    """

    def __init__(
        self,
        transport: ModernTransport,
        *,
        client_name: str = "mcp-client-learning",
        client_version: str = "0.1.0",
        sampling: SamplingHandler | None = None,
        static_roots: Sequence[str] = (),
        on_notification: Callable[[dict[str, Any]], None] | None = None,
        input_responders: dict[str, InputResponder] | None = None,
    ):
        self._transport = transport
        self.client_name = client_name
        self.client_version = client_version
        self._sampling = sampling
        self._static_roots = list(static_roots)
        self._on_notification = on_notification or _log_notification
        # Most-recent tools/list schemas, used for Mcp-Param-* mirroring.
        self._tool_headers: dict[str, dict[str, tuple[tuple[str, ...], str]]] = {}
        # Test seam: override how individual inputRequest methods are answered.
        self._responders: dict[str, InputResponder] = {
            "elicitation/create": self._respond_elicitation,
            "sampling/createMessage": self._respond_sampling,
            "roots/list": self._respond_roots,
            **(input_responders or {}),
        }

    @property
    def protocol_version(self) -> str:
        return self._transport.protocol_version

    # -- per-request metadata --------------------------------------------------

    def _client_capabilities(self) -> dict[str, Any]:
        """Capabilities declared on EVERY request (SEP-2575).

        Elicitation (both modes) is always available — the terminal handler
        needs no external service. Sampling is declared only when an
        Anthropic API key is configured: declaring it without being able to
        serve it would invite servers into MRTR loops we must abort.
        `sampling.tools` rides along (SEP-1577 tool-enabled sampling), and
        `roots` (deprecated, SEP-2577) only when static roots are configured.
        """
        caps: dict[str, Any] = {"elicitation": {"form": {}, "url": {}}}
        if self._sampling is not None and self._sampling.client is not None:
            caps["sampling"] = {"tools": {}}
        if self._static_roots:
            caps["roots"] = {}
        return caps

    def _meta(self) -> dict[str, Any]:
        return {
            wire.META_PROTOCOL_VERSION: self._transport.protocol_version,
            wire.META_CLIENT_INFO: {"name": self.client_name, "version": self.client_version},
            wire.META_CLIENT_CAPS: self._client_capabilities(),
        }

    async def _request(
        self,
        method: str,
        extra_params: dict[str, Any] | None = None,
        *,
        name: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"_meta": self._meta(), **(extra_params or {})}
        return await self._transport.request(
            method, params, name=name, on_notification=self._on_notification
        )

    # -- discovery & lists -----------------------------------------------------

    async def discover(self) -> wire.DiscoverResult:
        return wire.DiscoverResult.model_validate(await self._request("server/discover"))

    async def list_tools(self) -> wire.ListToolsResult:
        raw = await self._request("tools/list")
        valid_tools: list[dict[str, Any]] = []
        for tool in raw.get("tools", []):
            tool_name = tool.get("name", "<unnamed>")
            try:
                header_params = collect_header_params(tool.get("inputSchema") or {})
            except InvalidHeaderAnnotationError as exc:
                # SEP-2243: clients MUST exclude tools with invalid
                # x-mcp-header annotations from tools/list results and
                # SHOULD log a warning naming tool and reason.
                logger.warning("Excluding tool %r from tools/list: %s", tool_name, exc)
                continue
            self._tool_headers[tool_name] = header_params
            valid_tools.append(tool)
        return wire.ListToolsResult.model_validate({**raw, "tools": valid_tools})

    async def list_resources(self) -> wire.ListResourcesResult:
        return wire.ListResourcesResult.model_validate(await self._request("resources/list"))

    async def list_resource_templates(self) -> wire.ListResourceTemplatesResult:
        return wire.ListResourceTemplatesResult.model_validate(
            await self._request("resources/templates/list")
        )

    async def list_prompts(self) -> wire.ListPromptsResult:
        return wire.ListPromptsResult.model_validate(await self._request("prompts/list"))

    # -- MRTR-capable requests ---------------------------------------------------

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> wire.CallToolResult:
        arguments = arguments or {}
        # Mirror annotated params into Mcp-Param-* using the MOST RECENT
        # tools/list schema (spec MUST). A client that never fetched the
        # schema SHOULD send the request without Mcp-Param-* headers.
        header_params = self._tool_headers.get(name)
        param_headers = extract_param_headers(header_params, arguments) if header_params else None
        result = await self._run_mrtr(
            "tools/call",
            {"name": name, "arguments": arguments},
            name_header=name,
            param_headers=param_headers,
        )
        result["content"] = [_parse_content_block(b) for b in result.get("content") or []]
        return wire.CallToolResult.model_validate(result)

    async def read_resource(self, uri: str) -> wire.ReadResourceResult:
        result = await self._run_mrtr("resources/read", {"uri": uri}, name_header=uri)
        return wire.ReadResourceResult.model_validate(result)

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> wire.GetPromptResult:
        params: dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments
        result = await self._run_mrtr("prompts/get", params, name_header=name)
        return wire.GetPromptResult.model_validate(result)

    # -- the MRTR driver ---------------------------------------------------------

    async def _run_mrtr(
        self,
        method: str,
        base_params: dict[str, Any],
        *,
        name_header: str,
        param_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Issue a request and drive it to completion through MRTR retries.

        Client requirements implemented here (basic/patterns/mrtr §5):
        1. When `inputRequests` is present we construct every requested input
           before retrying (absent inputRequests = retry immediately — the
           "load shedding" variant).
        2. `requestState` is echoed VERBATIM: never inspected, parsed, or
           modified. When the result omits it, the retry omits it too.
        3. Every retry is an independent request — the transport assigns a
           NEW JSON-RPC id each time.
        4. The collected responses/state apply only to the retry of THIS
           request; the loop variables never leak to other calls.
        """
        input_responses: dict[str, Any] | None = None
        request_state: str | None = None

        for _ in range(wire.MAX_ROUND_TRIPS):
            params: dict[str, Any] = {"_meta": self._meta(), **base_params}
            if input_responses:
                params["inputResponses"] = input_responses
            if request_state is not None:
                params["requestState"] = request_state

            result = await self._transport.request(
                method,
                params,
                name=name_header,
                param_headers=param_headers,
                on_notification=self._on_notification,
            )

            # SEP-2322: resultType is required from 2026-07-28 servers, but an
            # ABSENT field (earlier-protocol server) MUST be read as complete;
            # any UNRECOGNIZED value MUST be considered invalid.
            result_type = result.get("resultType", "complete")
            if result_type == "complete":
                return result
            if result_type != "input_required":
                raise ModernSessionError(
                    f"Server returned unrecognized resultType {result_type!r}; "
                    "the spec says unknown values must be treated as invalid"
                )

            request_state = result.get("requestState")
            input_requests = result.get("inputRequests") or {}
            if not input_requests and request_state is None:
                # Server requirement 6: at least one of the two MUST be set.
                raise ModernSessionError(
                    "input_required result carried neither inputRequests nor requestState"
                )
            input_responses = {}
            for key, request in input_requests.items():
                input_responses[key] = await self._answer_input_request(key, request)

        raise ModernSessionError(
            f"{method} still input_required after {wire.MAX_ROUND_TRIPS} round trips; "
            "aborting (clients are not obligated to retry forever)"
        )

    async def _answer_input_request(self, key: str, request: dict[str, Any]) -> dict[str, Any]:
        """Route one bare {method, params} inputRequest to its handler.

        The values are NOT JSON-RPC requests — no id, no jsonrpc field; the
        map key is the only correlation identifier and the response we build
        is a bare result object (no resultType either).
        """
        method = request.get("method")
        responder = self._responders.get(method or "")
        if responder is None:
            # Servers MUST NOT send request types we did not declare; if one
            # does anyway, refusing to retry is the spec-sanctioned response.
            raise ModernSessionError(
                f"Server sent unsupported inputRequest method {method!r} (key {key!r})"
            )
        return await responder(request.get("params") or {})

    # -- default input responders -------------------------------------------------

    async def _respond_elicitation(self, params: dict[str, Any]) -> dict[str, Any]:
        # Same terminal UX as the legacy era: handlers.elicitation_handler,
        # via the wire adapter that bridges dict params to SDK param models.
        return await elicit_from_wire(params)

    async def _respond_sampling(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._sampling is None or self._sampling.client is None:
            # We only declare `sampling` when a key is configured, so a server
            # sending this violated capability gating; don't retry blind.
            raise ModernSessionError(
                "Server requested sampling/createMessage but no Anthropic API key "
                "is configured (capability was not declared)"
            )
        return await sample_from_wire(self._sampling, params)

    async def _respond_roots(self, _params: dict[str, Any]) -> dict[str, Any]:
        # Roots are deprecated (SEP-2577) and informational-only; we serve
        # the statically configured list. An empty list is the polite refusal.
        return {"roots": [{"uri": root} for root in self._static_roots]}

    # -- subscriptions -------------------------------------------------------------

    async def listen(self, notifications: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Open a subscriptions/listen stream and yield its notifications.

        `notifications` is the SubscriptionFilter: opt-in booleans for
        toolsListChanged / promptsListChanged / resourcesListChanged plus a
        resourceSubscriptions URI list. The first yielded message is the
        server's `notifications/subscriptions/acknowledged` (carrying the
        subset it agreed to honor); iteration ends on graceful server closure
        and raises StreamClosedError on abrupt disconnect — re-subscribing is
        a brand-new request, no notification replay exists.
        """
        params = {"_meta": self._meta(), "notifications": notifications}
        async for message in self._transport.stream("subscriptions/listen", params):
            yield message


def _log_notification(message: dict[str, Any]) -> None:
    """Default sink for request-scoped notifications (progress / log)."""
    method = message.get("method", "")
    params = message.get("params", {})
    if method == "notifications/message":
        logger.info("server[%s]: %s", params.get("level", "info"), params.get("data"))
    elif method == "notifications/progress":
        logger.info(
            "progress: %s/%s %s",
            params.get("progress"),
            params.get("total", "?"),
            params.get("message", ""),
        )
    else:
        logger.debug("notification %s: %s", method, params)
