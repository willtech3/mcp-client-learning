"""Token persistence: OS keychain first, locked-down file as fallback.

OAuth tokens are credentials. The safest place for them on a workstation
is the operating system keychain (Keychain on macOS, Secret Service on
Linux, Credential Locker on Windows) — encrypted at rest, ACL'd to the
user, and never accidentally committed.

Where no keychain exists (headless boxes, some CI), we fall back to a JSON
file created with 0600 permissions under ~/.config. The fallback is
explicit and logged, never silent.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SERVICE_NAME = "mcp-client-learning"
FALLBACK_DIR = Path.home() / ".config" / "mcp-client-learning"


class TokenStore:
    """Per-server-URL token storage."""

    def __init__(self, use_keyring: bool = True):
        self._keyring = None
        if use_keyring:
            try:
                # Lazily imported: keyring probes platform backends at import
                # time, which we only want when keychain use is requested.
                import keyring  # noqa: PLC0415
                from keyring.backends.fail import Keyring as FailKeyring  # noqa: PLC0415

                if not isinstance(keyring.get_keyring(), FailKeyring):
                    self._keyring = keyring
            except Exception:  # pragma: no cover - platform-specific
                self._keyring = None
        if self._keyring is None:
            logger.warning(
                "OS keychain unavailable - falling back to %s (0600 permissions)", FALLBACK_DIR
            )

    # -- public API ----------------------------------------------------------

    def save(self, server_url: str, tokens: dict) -> None:
        payload = json.dumps(tokens)
        if self._keyring:
            self._keyring.set_password(SERVICE_NAME, server_url, payload)
            return
        path = self._fallback_path(server_url)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)  # enforce even if the file pre-existed looser
        path.write_text(payload)

    def load(self, server_url: str) -> dict | None:
        if self._keyring:
            payload = self._keyring.get_password(SERVICE_NAME, server_url)
            return json.loads(payload) if payload else None
        path = self._fallback_path(server_url)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def clear(self, server_url: str) -> None:
        if self._keyring:
            with contextlib.suppress(Exception):  # nothing stored is fine
                self._keyring.delete_password(SERVICE_NAME, server_url)
            return
        self._fallback_path(server_url).unlink(missing_ok=True)

    # -- issuer-keyed client credentials (SEP-2352) ----------------------------

    def save_client(self, issuer: str, client: dict) -> None:
        """Persist dynamically-registered client credentials KEYED BY ISSUER.

        SEP-2352 (Authorization Server Binding): credentials obtained via DCR
        MUST be associated with the specific authorization server that issued
        them, keyed by its `issuer` identifier — a client_id minted by AS-A
        MUST never be presented to AS-B, even if both protect the same MCP
        server. Tokens stay keyed by server URL (they authorize a resource);
        client registrations are keyed here (they authenticate to an AS).
        """
        self.save(self._client_key(issuer), client)

    def load_client(self, issuer: str) -> dict | None:
        return self.load(self._client_key(issuer))

    def clear_client(self, issuer: str) -> None:
        self.clear(self._client_key(issuer))

    @staticmethod
    def _client_key(issuer: str) -> str:
        return f"oauth-client::{issuer}"

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _fallback_path(server_url: str) -> Path:
        digest = hashlib.sha256(server_url.encode()).hexdigest()[:16]
        return FALLBACK_DIR / f"tokens-{digest}.json"
