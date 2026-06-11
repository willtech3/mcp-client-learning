"""mcp-client — a terminal MCP client that exercises the whole protocol.

Every command opens a session, performs one protocol interaction, and
prints the result. `demo` chains them into a guided tour designed to be
shown live: search (structured output), checkout (elicitation), insights
(sampling), import (progress), maintenance (notifications).
"""

from __future__ import annotations

import asyncio
import json as jsonlib
import logging

import typer
from mcp import types
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .session import connect
from .token_store import TokenStore

app = typer.Typer(help=__doc__, no_args_is_help=True)
console = Console()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

ServerOption = typer.Option(None, "--server", "-s", help="Server URL or stdio command line")
AnonymousOption = typer.Option(
    False, "--anonymous", help="Skip OAuth (for servers allowing unauthenticated access)"
)


def _target(server: str | None) -> str:
    return server or get_settings().default_server


@app.command()
def tools(server: str = ServerOption, anonymous: bool = AnonymousOption):
    """List the server's tools with their schemas and annotations."""

    async def run():
        async with connect(_target(server), get_settings(), anonymous=anonymous) as session:
            result = await session.list_tools()
            table = Table(title="Tools", show_lines=True)
            table.add_column("Name", style="bold")
            table.add_column("Parameters")
            table.add_column("Hints")
            for tool in result.tools:
                params = ", ".join((tool.inputSchema or {}).get("properties", {}).keys())
                hints = []
                if tool.annotations:
                    if tool.annotations.readOnlyHint:
                        hints.append("read-only")
                    if tool.annotations.idempotentHint:
                        hints.append("idempotent")
                table.add_row(tool.name, params or "—", ", ".join(hints) or "—")
            console.print(table)

    asyncio.run(run())


@app.command()
def call(
    tool: str,
    args: str = typer.Option("{}", "--args", "-a", help="Tool arguments as JSON"),
    server: str = ServerOption,
    anonymous: bool = AnonymousOption,
):
    """Call a tool. Elicitation/sampling requests are handled interactively."""

    async def run():
        async with connect(_target(server), get_settings(), anonymous=anonymous) as session:
            result = await session.call_tool(tool, jsonlib.loads(args))
            for block in result.content or []:
                if isinstance(block, types.TextContent):
                    console.print(Panel(block.text, title=f"{tool} → text"))
            if result.structuredContent:
                console.print(
                    Panel(jsonlib.dumps(result.structuredContent, indent=2), title="structured")
                )
            if result.isError:
                raise typer.Exit(code=1)

    asyncio.run(run())


@app.command()
def resources(server: str = ServerOption, anonymous: bool = AnonymousOption):
    """List resources and resource templates."""

    async def run():
        async with connect(_target(server), get_settings(), anonymous=anonymous) as session:
            fixed = await session.list_resources()
            templates = await session.list_resource_templates()
            table = Table(title="Resources")
            table.add_column("URI / Template", style="bold")
            table.add_column("Name")
            for r in fixed.resources:
                table.add_row(str(r.uri), r.name)
            for t in templates.resourceTemplates:
                table.add_row(t.uriTemplate, t.name)
            console.print(table)

    asyncio.run(run())


@app.command()
def read(uri: str, server: str = ServerOption, anonymous: bool = AnonymousOption):
    """Read a resource by URI."""

    async def run():
        async with connect(_target(server), get_settings(), anonymous=anonymous) as session:
            result = await session.read_resource(uri)  # type: ignore[arg-type]
            for item in result.contents:
                text = getattr(item, "text", None)
                if text:
                    body = text if len(text) < 4000 else text[:4000] + "\n... (truncated)"
                    console.print(Panel(body, title=str(item.uri)))

    asyncio.run(run())


@app.command()
def prompts(server: str = ServerOption, anonymous: bool = AnonymousOption):
    """List prompt templates."""

    async def run():
        async with connect(_target(server), get_settings(), anonymous=anonymous) as session:
            result = await session.list_prompts()
            table = Table(title="Prompts")
            table.add_column("Name", style="bold")
            table.add_column("Arguments")
            for p in result.prompts:
                args = ", ".join(a.name for a in (p.arguments or []))
                table.add_row(p.name, args or "—")
            console.print(table)

    asyncio.run(run())


@app.command()
def logout(server: str = ServerOption):
    """Forget stored OAuth tokens for a server."""
    target = _target(server)
    TokenStore().clear(target.rstrip("/"))
    console.print(f"Cleared stored tokens for {target}")


@app.command()
def demo(server: str = ServerOption, anonymous: bool = AnonymousOption):
    """Guided tour: every MCP feature against the Virtual Library server."""

    async def run():
        settings = get_settings()
        async with connect(_target(server), settings, anonymous=anonymous) as session:
            console.rule("[bold]1. Tools with real schemas")
            tools_result = await session.list_tools()
            console.print(f"Server exposes [bold]{len(tools_result.tools)}[/bold] tools")

            console.rule("[bold]2. search_catalog — structured output")
            search = await session.call_tool("search_catalog", {"query": "dune"})
            if search.structuredContent:
                for book in search.structuredContent.get("books", [])[:3]:
                    console.print(f"  • {book['title']} ({book['publication_year']})")

            console.rule("[bold]3. Resource read — popular books")
            popular = await session.read_resource("library://stats/popular/365/5")  # type: ignore[arg-type]
            text = getattr(popular.contents[0], "text", "")
            console.print(Panel(text[:600], title="library://stats/popular/365/5"))

            console.rule("[bold]4. renew_membership — elicitation (answer the prompt!)")
            renewal = await session.call_tool("renew_membership", {"patron_id": "patron_00002"})
            if renewal.structuredContent:
                console.print(renewal.structuredContent.get("message", ""))

            console.rule("[bold]5. generate_book_insights — sampling via YOUR Claude")
            if search.structuredContent and search.structuredContent.get("books"):
                isbn = search.structuredContent["books"][0]["isbn"]
                insights = await session.call_tool(
                    "generate_book_insights", {"isbn": isbn, "insight_type": "summary"}
                )
                for block in insights.content or []:
                    if isinstance(block, types.TextContent):
                        console.print(Panel(block.text[:800], title="AI insight"))

            console.rule("[bold]6. regenerate_catalog — progress + notifications")
            done = await session.call_tool("regenerate_catalog", {})
            for block in done.content or []:
                if isinstance(block, types.TextContent):
                    console.print(Panel(block.text, title="maintenance report"))

            console.rule("[bold]Demo complete")

    asyncio.run(run())


if __name__ == "__main__":
    app()
