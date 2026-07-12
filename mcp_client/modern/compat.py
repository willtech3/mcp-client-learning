"""Era detection and the unified connect() — dual-era client plumbing.

MCP 2026-07-28 defines two *eras* (basic/versioning): **modern** servers
speak per-request metadata and implement ``server/discover``; **legacy**
servers (2025-11-25 and earlier) require an ``initialize`` handshake. A
dual-era client like this one must figure out which kind it is talking to —
without a handshake to ask.

The HTTP recipe (basic/versioning §4.6 + streamable-http §2.11): attempt a
modern request first and *inspect the failure*. We probe with
``server/discover`` because it is the one method modern servers MUST
implement, and classify the outcome:

1. A valid ``DiscoverResult``            -> modern. Pick a mutual version
   from ``supportedVersions``.
2. ``UnsupportedProtocolVersionError``   -> modern, but our preferred version
   (-32022, a spec-reserved code no        lost; retry with a version from
   legacy server can emit)                 ``data.supported`` — do NOT fall
                                           back to ``initialize``.
3. Anything else — a 404, a 400 whose body is not a recognized modern
   JSON-RPC error, an HTML error page — -> legacy. Fall back to the official
   SDK's ``initialize`` path (mcp_client.session.connect), which this module
   wraps unchanged.

Era determination is a property of the *server*, not of a request, so the
verdict is cached per origin for the lifetime of this process (the spec says
clients SHOULD cache it and MAY persist it, re-probing if the assumption
later fails).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from ..config import ClientSettings
from ..handlers import SamplingHandler
from ..oauth import MCPOAuth
from ..session import connect as legacy_connect
from ..token_store import TokenStore
from .session import ModernSession
from .transport import (
    McpProtocolError,
    ModernTransport,
    NotJsonRpcError,
    StreamClosedError,
)
from .types import (
    META_CLIENT_CAPS,
    META_CLIENT_INFO,
    META_PROTOCOL_VERSION,
    MODERN_ERROR_CODES,
    MODERN_VERSIONS,
    PROTOCOL_VERSION,
    UNSUPPORTED_PROTOCOL_VERSION,
    DiscoverResult,
)

logger = logging.getLogger(__name__)

#: origin -> ("modern", negotiated_version) | ("legacy", None). Process-wide:
#: era is a server property, so one probe per origin is enough.
_ERA_CACHE: dict[str, tuple[str, str | None]] = {}


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def reset_era_cache() -> None:
    """Forget cached era verdicts (used by tests and manual re-probing)."""
    _ERA_CACHE.clear()


async def probe_era(  # noqa: PLR0911 - one return per cell of the spec's classification matrix
    transport: ModernTransport, settings: ClientSettings
) -> tuple[str, str | None]:
    """POST server/discover and classify the server's era from the outcome."""
    probe_meta = {
        META_PROTOCOL_VERSION: transport.protocol_version,
        META_CLIENT_INFO: {"name": settings.client_name, "version": settings.client_version},
        # Empty object = "no optional capabilities"; the probe needs none.
        META_CLIENT_CAPS: {},
    }
    try:
        raw = await transport.request("server/discover", {"_meta": probe_meta})
    except McpProtocolError as exc:
        if exc.code == UNSUPPORTED_PROTOCOL_VERSION:
            # Outcome 2: definitively modern. Choose a mutually supported
            # version from data.supported; if there is none we can still be
            # sure initialize would fare no better on a modern-only server,
            # but a DUAL-era server never -32022s a version it can serve
            # legacy-side, so falling back is the pragmatic total answer.
            supported = (exc.data or {}).get("supported") or []
            for version in supported:
                if version in MODERN_VERSIONS:
                    return ("modern", version)
            logger.warning(
                "Server is modern but supports none of our versions (%s); "
                "falling back to the legacy path",
                supported,
            )
            return ("legacy", None)
        if exc.code in MODERN_ERROR_CODES:
            # -32020/-32021 are spec-reserved codes only a modern server can
            # emit; treat the server as modern and let the real request
            # surface the underlying problem.
            return ("modern", transport.protocol_version)
        # Any other JSON-RPC error (e.g. a legacy server's -32601/-32600 to
        # an unknown pre-initialize method) is NOT a modern signal. The spec
        # forbids keying the fallback to one specific error code.
        return ("legacy", None)
    except (NotJsonRpcError, StreamClosedError):
        # Empty/HTML/unrecognized body on a 4xx — the classic legacy shape.
        return ("legacy", None)

    try:
        discover = DiscoverResult.model_validate(raw)
    except ValidationError:
        return ("legacy", None)
    for version in discover.supported_versions:
        if version in MODERN_VERSIONS:
            return ("modern", version)
    # It answered server/discover, so it IS modern — even if we share no
    # version, initialize would be refused too; keep the modern verdict and
    # let -32022 from real requests explain the mismatch to the user.
    return ("modern", transport.protocol_version)


@asynccontextmanager
async def connect(
    target: str,
    settings: ClientSettings,
    *,
    anonymous: bool = False,
):
    """Open a session against `target`, modern era first, legacy fallback.

    Yields either a :class:`ModernSession` (2026-07-28, from-scratch) or the
    official SDK's ``ClientSession`` (legacy initialize path) — both expose
    the list_*/call_tool/read_resource surface the CLI uses. stdio targets
    always take the legacy SDK path (a stdio-modern probe is documented
    future work; the local Virtual Library server speaks legacy on stdio).
    """
    is_http = target.startswith(("http://", "https://"))
    if not is_http or not settings.modern_first:
        async with legacy_connect(target, settings, anonymous=anonymous) as session:
            yield session
        return

    origin = _origin(target)
    era = _ERA_CACHE.get(origin)

    if era is None or era[0] == "modern":
        auth = (
            None
            if anonymous
            else MCPOAuth(target, TokenStore(), client_metadata_url=settings.client_metadata_url)
        )
        # read=None keeps subscriptions/listen streams open indefinitely;
        # request-level cancellation on HTTP is closing the stream, which is
        # exactly what happens when this client is interrupted.
        timeout = httpx.Timeout(30.0, read=None)
        async with httpx.AsyncClient(auth=auth, timeout=timeout) as http:
            transport = ModernTransport(http, target)
            if era is None:
                era = await probe_era(transport, settings)
                _ERA_CACHE[origin] = era
                logger.info("Era detected for %s: %s", origin, era[0])
            if era[0] == "modern":
                transport.protocol_version = era[1] or PROTOCOL_VERSION
                sampling = SamplingHandler(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                    auto_approve=settings.auto_approve_sampling,
                )
                yield ModernSession(
                    transport,
                    client_name=settings.client_name,
                    client_version=settings.client_version,
                    sampling=sampling,
                    static_roots=settings.static_roots,
                )
                return

    async with legacy_connect(target, settings, anonymous=anonymous) as session:
        yield session


def detect_era_label(session: Any) -> str:
    """Human-readable era note for CLI output."""
    if isinstance(session, ModernSession):
        return f"modern (protocol {session.protocol_version}, stateless)"
    return "legacy (initialize handshake via official SDK)"
