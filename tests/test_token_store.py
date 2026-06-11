"""Tests for token persistence (file-fallback mode).

The keyring path depends on the host OS; tests exercise the fallback,
including the property that matters most: file permissions.
"""

import stat

import pytest

import mcp_client.token_store as token_store_module
from mcp_client.token_store import TokenStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(token_store_module, "FALLBACK_DIR", tmp_path / "secrets")
    return TokenStore(use_keyring=False)


SERVER = "https://library.example.app/mcp"
TOKENS = {"access_token": "at-123", "refresh_token": "rt-456", "expires_at": 1234567890.0}


class TestFileFallback:
    def test_round_trip(self, store):
        store.save(SERVER, TOKENS)
        assert store.load(SERVER) == TOKENS

    def test_missing_returns_none(self, store):
        assert store.load("https://never-seen.example") is None

    def test_clear_removes_tokens(self, store):
        store.save(SERVER, TOKENS)
        store.clear(SERVER)
        assert store.load(SERVER) is None

    def test_file_is_owner_only(self, store, tmp_path):
        """0600: tokens must not be readable by other local users."""
        store.save(SERVER, TOKENS)
        files = list((tmp_path / "secrets").glob("tokens-*.json"))
        assert len(files) == 1
        mode = stat.S_IMODE(files[0].stat().st_mode)
        assert mode == 0o600

    def test_directory_is_owner_only(self, store, tmp_path):
        store.save(SERVER, TOKENS)
        mode = stat.S_IMODE((tmp_path / "secrets").stat().st_mode)
        assert mode == 0o700

    def test_distinct_servers_do_not_collide(self, store):
        store.save(SERVER, TOKENS)
        store.save("https://other.example/mcp", {"access_token": "other"})
        assert store.load(SERVER)["access_token"] == "at-123"
        assert store.load("https://other.example/mcp")["access_token"] == "other"
