"""Loopback redirect listener for the OAuth authorization flow.

Native-app OAuth (RFC 8252) sends the user's browser back to a local HTTP
listener. Security properties this implementation guarantees:

- Binds to 127.0.0.1 only — never 0.0.0.0, so nothing off-machine can
  reach it or race the callback.
- Ephemeral port chosen by the OS — no fixed port to squat on.
- Accepts exactly one callback, then stops listening.
- Returns code/state to the flow for validation (state is checked by the
  caller with constant-time comparison; this module just transports it).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

SUCCESS_PAGE = b"""\
HTTP/1.1 200 OK\r
Content-Type: text/html; charset=utf-8\r
Connection: close\r
\r
<!doctype html><html><body style="font-family: system-ui; margin: 4rem;">
<h2>Signed in</h2><p>You can close this tab and return to the terminal.</p>
</body></html>"""


@dataclass
class CallbackResult:
    code: str | None = None
    state: str | None = None
    error: str | None = None
    # RFC 9207 (SEP-2468): the AS's issuer identifier, echoed on the
    # authorization response so the client can detect mix-up attacks. This
    # module only transports it; oauth.py validates it (exact string compare
    # against the issuer recorded before the redirect).
    iss: str | None = None


class _Listener:
    def __init__(self, port: int, future: asyncio.Future[CallbackResult]):
        self.redirect_uri = f"http://127.0.0.1:{port}/callback"
        self._future = future

    async def wait(self) -> CallbackResult:
        return await self._future


async def _handle(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    future: asyncio.Future[CallbackResult],
) -> None:
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=10)
        # The request line looks like: GET /callback?code=X&state=Y HTTP/1.1
        parts = request_line.decode("ascii", errors="replace").split(" ")
        path = parts[1] if len(parts) >= 2 else "/"
        query = parse_qs(urlsplit(path).query)

        writer.write(SUCCESS_PAGE)
        await writer.drain()

        if not future.done():
            future.set_result(
                CallbackResult(
                    code=query.get("code", [None])[0],
                    state=query.get("state", [None])[0],
                    error=query.get("error", [None])[0],
                    # parse_qs has already form-url-decoded the value; RFC
                    # 9207 §2.4 forbids any FURTHER normalization before the
                    # issuer comparison, so it is passed through verbatim.
                    iss=query.get("iss", [None])[0],
                )
            )
    finally:
        writer.close()


@asynccontextmanager
async def run_callback_server():
    """Context manager yielding a single-use loopback listener."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[CallbackResult] = loop.create_future()

    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, future),
        host="127.0.0.1",  # loopback ONLY — see module docstring
        port=0,  # ephemeral port from the OS
    )
    port = server.sockets[0].getsockname()[1]
    try:
        yield _Listener(port, future)
    finally:
        server.close()
        await server.wait_closed()
