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


def get_settings() -> ClientSettings:
    return ClientSettings()
