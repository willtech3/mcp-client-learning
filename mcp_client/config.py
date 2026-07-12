"""Client configuration via environment / .env.

Secrets policy: the Anthropic key is read from the standard
ANTHROPIC_API_KEY variable (never a CLI flag — flags leak into shell
history and process lists). OAuth tokens live in the OS keychain via
token_store, not here.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClientSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MCP_CLIENT_", env_file=".env", extra="ignore", case_sensitive=False
    )

    default_server: str = Field(
        default="http://127.0.0.1:8080/mcp",
        description="Server URL used when --server is not passed",
    )

    auto_approve_sampling: bool = Field(
        default=False,
        description="Skip the human confirmation before serving sampling requests",
    )

    client_name: str = Field(default="mcp-client-learning")
    client_version: str = Field(default="0.1.0")

    modern_first: bool = Field(
        default=True,
        description=(
            "Probe HTTP servers with server/discover (protocol 2026-07-28) before "
            "falling back to the legacy SDK initialize path"
        ),
    )

    client_metadata_url: str | None = Field(
        default=None,
        description=(
            "HTTPS URL of a hosted CIMD client metadata document "
            "(draft-ietf-oauth-client-id-metadata-document). When set and the "
            "authorization server advertises client_id_metadata_document_supported, "
            "this URL is used as the OAuth client_id and dynamic registration is skipped"
        ),
    )

    static_roots: list[str] = Field(
        default_factory=list,
        description=(
            "file:// URIs served in answer to MRTR roots/list input requests "
            "(roots are deprecated in 2026-07-28, SEP-2577; informational only). "
            'JSON list in env, e.g. MCP_CLIENT_STATIC_ROOTS=\'["file:///home/me/proj"]\''
        ),
    )


def get_settings() -> ClientSettings:
    return ClientSettings()
