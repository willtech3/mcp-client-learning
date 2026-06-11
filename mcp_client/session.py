"""Session orchestration: transport + capabilities + handlers in one place.

This is where a "full" MCP client comes together:

- Transport: Streamable HTTP (remote, with OAuth 2.1) or stdio (local
  child process). Both yield the same (read, write) stream pair — the
  protocol above is identical, which is the whole point of MCP transports.
- Capabilities: we advertise sampling (WITH tool support, SEP-1577) and
  elicitation by registering handlers; the SDK derives the capability
  flags it sends during initialize from what we provide.
- Lifecycle: initialize() negotiates protocol version and exchanges
  capability sets before any feature request flows.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .config import ClientSettings
from .handlers import SamplingHandler, elicitation_handler, logging_handler
from .oauth import MCPOAuth
from .token_store import TokenStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def connect(
    target: str,
    settings: ClientSettings,
    *,
    anonymous: bool = False,
):
    """Open an initialized ClientSession against `target`.

    `target` is either an http(s) URL (Streamable HTTP) or a shell-style
    command line (stdio child process). `anonymous=True` skips OAuth for
    servers that allow unauthenticated access (local development).
    """
    sampling = SamplingHandler(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        auto_approve=settings.auto_approve_sampling,
    )

    def build_session(read, write) -> ClientSession:
        return ClientSession(
            read,
            write,
            read_timeout_seconds=timedelta(seconds=120),
            sampling_callback=sampling,
            elicitation_callback=elicitation_handler,
            logging_callback=logging_handler,
            client_info=types.Implementation(
                name=settings.client_name,
                version=settings.client_version,
                title="Educational MCP client with OAuth 2.1, sampling, and elicitation",
            ),
            # Advertise SEP-1577 tool-enabled sampling: without this the
            # server must not send tools with createMessage requests.
            sampling_capabilities=types.SamplingCapability(tools=types.SamplingToolsCapability()),
        )

    if target.startswith(("http://", "https://")):
        auth = None if anonymous else MCPOAuth(target, TokenStore())
        async with (
            streamablehttp_client(target, auth=auth) as (read, write, _get_session_id),
            build_session(read, write) as session,
        ):
            result = await session.initialize()
            _log_server(result)
            yield session
    else:
        command, *args = target.split()
        params = StdioServerParameters(command=command, args=args)
        async with (
            stdio_client(params) as (read, write),
            build_session(read, write) as session,
        ):
            result = await session.initialize()
            _log_server(result)
            yield session


def _log_server(result: types.InitializeResult) -> None:
    caps = result.capabilities
    features = [
        name
        for name, present in (
            ("tools", caps.tools),
            ("resources", caps.resources),
            ("prompts", caps.prompts),
            ("logging", caps.logging),
        )
        if present
    ]
    logger.info(
        "Connected to %s v%s (protocol %s; server capabilities: %s)",
        result.serverInfo.name,
        result.serverInfo.version,
        result.protocolVersion,
        ", ".join(features) or "none",
    )
