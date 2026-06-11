"""Tests for the loopback OAuth callback listener."""

import asyncio

import httpx

from mcp_client.callback_server import run_callback_server


class TestCallbackServer:
    async def test_binds_loopback_with_ephemeral_port(self):
        async with run_callback_server() as listener:
            assert listener.redirect_uri.startswith("http://127.0.0.1:")
            port = int(listener.redirect_uri.split(":")[2].split("/")[0])
            assert port > 0

    async def test_captures_code_and_state(self):
        async with run_callback_server() as listener:
            async with httpx.AsyncClient() as http:
                response = await http.get(
                    listener.redirect_uri, params={"code": "abc123", "state": "xyz789"}
                )
            assert response.status_code == 200
            result = await asyncio.wait_for(listener.wait(), timeout=5)
        assert result.code == "abc123"
        assert result.state == "xyz789"
        assert result.error is None

    async def test_captures_error_response(self):
        async with run_callback_server() as listener:
            async with httpx.AsyncClient() as http:
                await http.get(listener.redirect_uri, params={"error": "access_denied"})
            result = await asyncio.wait_for(listener.wait(), timeout=5)
        assert result.error == "access_denied"
        assert result.code is None
