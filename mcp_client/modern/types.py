"""Wire-level constants and small models for MCP 2026-07-28.

Revision 2026-07-28 made the protocol stateless (SEP-2575): there is no
``initialize`` handshake, so everything a server needs to process a request
travels *inside* the request — protocol version, client identity, and client
capabilities all ride in ``params._meta`` under reserved
``io.modelcontextprotocol/*`` keys. This module centralizes those key names,
the renumbered error codes (the spec now owns ``-32020..-32099``), and the
Base64 sentinel codec that Streamable HTTP uses when a mirrored header value
cannot be represented as plain ASCII (SEP-2243).

The pydantic models here are deliberately small: just enough shape to give
the CLI dot-access to results (``DiscoverResult``, list results, call
results). Item types (tools, resources, prompts, content blocks) reuse the
official SDK's ``mcp.types`` models — the *items* did not change shape in
2026-07-28, only the envelope around them did.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any, Literal

from mcp import types as sdk_types
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Protocol versions (basic/versioning)
# ---------------------------------------------------------------------------

#: The revision this package implements. On the wire the version string is
#: always "2026-07-28" — the spec's /draft URL never appears in messages.
PROTOCOL_VERSION = "2026-07-28"

#: Versions this package can speak on the modern (per-request-metadata) path.
MODERN_VERSIONS = (PROTOCOL_VERSION,)

#: Legacy (initialize-handshake) revisions — handled by the official SDK via
#: compat.connect()'s fallback, never by this package.
LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

# ---------------------------------------------------------------------------
# _meta keys (basic/index §3; SEP-2575)
# ---------------------------------------------------------------------------

META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_LOG_LEVEL = "io.modelcontextprotocol/logLevel"  # deprecated, SEP-2577
META_SUBSCRIPTION_ID = "io.modelcontextprotocol/subscriptionId"
META_PROGRESS_TOKEN = "progressToken"  # unprefixed reserved key!  # noqa: S105 - a _meta key name, not a secret

# ---------------------------------------------------------------------------
# Error codes. JSON-RPC standard codes plus the MCP-reserved sub-range
# -32020..-32099 (allocated sequentially; renumbered from -3200x late in the
# 2026-07-28 draft — beware stale intermediate material).
# ---------------------------------------------------------------------------

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

HEADER_MISMATCH = -32020
MISSING_REQUIRED_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022

#: Error codes that positively identify a *modern* (2026-07-28+) server
#: during era detection: legacy implementations cannot emit codes from the
#: spec-reserved -32020..-32099 range (basic/versioning §4.6).
MODERN_ERROR_CODES = frozenset(
    {HEADER_MISMATCH, MISSING_REQUIRED_CLIENT_CAPABILITY, UNSUPPORTED_PROTOCOL_VERSION}
)

#: The MRTR loop's client-side safety valve. The spec sets no protocol-level
#: limit on how often a server may answer with input_required (server req. 8
#: even allows repeated re-prompting), so a well-behaved client bounds the
#: loop itself.
MAX_ROUND_TRIPS = 8

# ---------------------------------------------------------------------------
# Base64 sentinel codec (streamable-http §2.8, SEP-2243)
#
# Header values mirrored from the body (Mcp-Name, Mcp-Param-*) must be plain
# ASCII per RFC 9110. When the body value is not — non-ASCII, control chars,
# or leading/trailing whitespace — the client MUST send
# ``=?base64?{base64(utf-8 bytes)}?=`` instead. The markers are lowercase and
# case-sensitive, and a *plain* value that happens to look like the sentinel
# must itself be encoded (anti-ambiguity rule) so decoders never guess.
# ---------------------------------------------------------------------------

_SENTINEL_PREFIX = "=?base64?"
_SENTINEL_SUFFIX = "?="


def _is_header_safe(value: str) -> bool:
    """True when every char is RFC 9110 field-value ASCII (VCHAR, SP, HTAB)
    and the value has no leading/trailing whitespace."""
    if value != value.strip(" \t"):
        return False
    return all(ch == "\t" or 0x20 <= ord(ch) <= 0x7E for ch in value)


def _looks_like_sentinel(value: str) -> bool:
    return value.startswith(_SENTINEL_PREFIX) and value.endswith(_SENTINEL_SUFFIX)


def encode_header_value(value: str) -> str:
    """Encode `value` for use as an Mcp-Name / Mcp-Param-* header value.

    Plain header-safe ASCII passes through untouched; anything else (and any
    literal value matching the sentinel pattern) becomes
    ``=?base64?...?=`` of the UTF-8 bytes.
    """
    if _is_header_safe(value) and not _looks_like_sentinel(value):
        return value
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"{_SENTINEL_PREFIX}{encoded}{_SENTINEL_SUFFIX}"


def decode_header_value(value: str) -> str:
    """Inverse of :func:`encode_header_value` (used by servers/intermediaries;
    implemented here so both directions of the codec live side by side)."""
    if not _looks_like_sentinel(value):
        return value
    payload = value[len(_SENTINEL_PREFIX) : -len(_SENTINEL_SUFFIX)]
    try:
        return base64.b64decode(payload.encode("ascii"), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Malformed base64 sentinel header value: {value!r}") from exc


# ---------------------------------------------------------------------------
# Result envelopes. Every 2026-07-28 result carries a required `resultType`
# (SEP-2322) and list/read results additionally carry the SEP-2549 cache
# hints (`ttlMs` — required, max-age analog — and `cacheScope`).
# ---------------------------------------------------------------------------


class _WireModel(BaseModel):
    """Base for wire results: tolerate unknown fields (forward compat) and
    accept both snake_case names and the camelCase wire aliases."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class ServerInfo(_WireModel):
    name: str
    version: str
    title: str | None = None
    description: str | None = None


class DiscoverResult(_WireModel):
    """`server/discover` result — the modern era's capability surface.

    Servers MUST implement server/discover (basic/versioning, SEP-2575); a
    successful DiscoverResult is also the strongest era signal a dual-era
    client can get, so compat.py probes with it.
    """

    result_type: str = Field(alias="resultType", default="complete")
    supported_versions: list[str] = Field(alias="supportedVersions")
    capabilities: dict[str, Any] = Field(default_factory=dict)
    server_info: ServerInfo = Field(alias="serverInfo")
    instructions: str | None = None
    ttl_ms: int | None = Field(alias="ttlMs", default=None)
    cache_scope: Literal["public", "private"] | None = Field(alias="cacheScope", default=None)


class ListToolsResult(_WireModel):
    tools: list[sdk_types.Tool] = Field(default_factory=list)
    next_cursor: str | None = Field(alias="nextCursor", default=None)
    ttl_ms: int | None = Field(alias="ttlMs", default=None)
    cache_scope: str | None = Field(alias="cacheScope", default=None)


class ListResourcesResult(_WireModel):
    resources: list[sdk_types.Resource] = Field(default_factory=list)
    next_cursor: str | None = Field(alias="nextCursor", default=None)
    ttl_ms: int | None = Field(alias="ttlMs", default=None)
    cache_scope: str | None = Field(alias="cacheScope", default=None)


class ListResourceTemplatesResult(_WireModel):
    resource_templates: list[sdk_types.ResourceTemplate] = Field(
        alias="resourceTemplates", default_factory=list
    )
    next_cursor: str | None = Field(alias="nextCursor", default=None)
    ttl_ms: int | None = Field(alias="ttlMs", default=None)
    cache_scope: str | None = Field(alias="cacheScope", default=None)

    @property
    def resourceTemplates(self) -> list[sdk_types.ResourceTemplate]:  # noqa: N802
        """CamelCase accessor so CLI code written for the SDK keeps working."""
        return self.resource_templates


class CallToolResult(_WireModel):
    content: list[Any] = Field(default_factory=list)
    structured_content: dict[str, Any] | None = Field(alias="structuredContent", default=None)
    is_error: bool | None = Field(alias="isError", default=None)
    result_type: str = Field(alias="resultType", default="complete")

    @property
    def structuredContent(self) -> dict[str, Any] | None:  # noqa: N802
        return self.structured_content

    @property
    def isError(self) -> bool | None:  # noqa: N802
        return self.is_error


class ReadResourceResult(_WireModel):
    contents: list[sdk_types.TextResourceContents | sdk_types.BlobResourceContents] = Field(
        default_factory=list
    )
    ttl_ms: int | None = Field(alias="ttlMs", default=None)
    cache_scope: str | None = Field(alias="cacheScope", default=None)


class GetPromptResult(_WireModel):
    description: str | None = None
    messages: list[sdk_types.PromptMessage] = Field(default_factory=list)


class ListPromptsResult(_WireModel):
    prompts: list[sdk_types.Prompt] = Field(default_factory=list)
    next_cursor: str | None = Field(alias="nextCursor", default=None)
    ttl_ms: int | None = Field(alias="ttlMs", default=None)
    cache_scope: str | None = Field(alias="cacheScope", default=None)
