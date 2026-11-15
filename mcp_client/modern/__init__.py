"""mcp_client.modern — a from-scratch client for MCP protocol 2026-07-28.

The official ``mcp`` SDK 1.x (pinned here for the legacy path) speaks the
*legacy* eras of the protocol: an ``initialize`` handshake, connection-scoped
sessions, and server-initiated JSON-RPC requests. Revision 2026-07-28 removes all of that (SEP-2567 sessions,
SEP-2575 statelessness, SEP-2322 MRTR), so this package implements the modern
era by hand — every wire byte is code you can read:

- ``types``     — protocol constants, ``_meta`` keys, error codes, the
                  Base64 sentinel header codec, and small result models.
- ``transport`` — Streamable HTTP binding: one POST per message, required
                  mirrored headers, and a hand-rolled SSE parser.
- ``session``   — ``ModernSession``: the request surface plus the full
                  Multi Round-Trip Request (MRTR) driver.
- ``compat``    — era detection (``server/discover`` probe) and a unified
                  ``connect()`` that falls back to the legacy SDK path.
"""

from .compat import connect
from .session import ModernSession
from .types import PROTOCOL_VERSION

__all__ = ["PROTOCOL_VERSION", "ModernSession", "connect"]
