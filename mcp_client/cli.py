"""mcp-client — a terminal MCP client that exercises the whole protocol.

Every command opens a session, performs one protocol interaction, and
prints the result. `demo` chains them into a guided tour designed to be
shown live: search (structured output), checkout (elicitation), insights
(sampling), import (progress), maintenance (notifications).

Every command connects through `modern.compat.connect`, which probes HTTP
servers with `server/discover` (MCP 2026-07-28, SEP-2575) and falls back to
the official SDK's legacy `initialize` path when the server predates the
modern era — so the same commands work against both kinds of server.
Modern-only surfaces get their own commands: `discover` (the capability
document), `skills` (SEP-2640 skills with digest verification), and
`listen` (subscriptions/listen notification streaming).
"""

from __future__ import annotations

import asyncio
import hashlib
import json as jsonlib
import logging
import re

import typer
import yaml
from mcp import types
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .modern.compat import connect, detect_era_label
from .modern.session import ModernSession
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
def discover(server: str = ServerOption, anonymous: bool = AnonymousOption):
    """Pretty-print the server/discover capability document (2026-07-28)."""

    async def run():
        async with connect(_target(server), get_settings(), anonymous=anonymous) as session:
            if not isinstance(session, ModernSession):
                console.print(
                    "[yellow]Legacy server[/yellow] — server/discover does not exist before "
                    "protocol 2026-07-28; this server negotiated via initialize instead."
                )
                return
            result = await session.discover()
            info = result.server_info
            if info is None:
                console.print("[dim]Server did not identify itself (serverInfo is a SHOULD)[/dim]")
            else:
                console.print(
                    Panel(
                        f"[bold]{info.name}[/bold] v{info.version}"
                        + (f"\n{info.title}" if info.title else ""),
                        title="serverInfo",
                    )
                )
            console.print(f"Supported versions: [bold]{', '.join(result.supported_versions)}[/bold]")
            console.print(
                f"Cache hints: ttlMs={result.ttl_ms} cacheScope={result.cache_scope} "
                "(SEP-2549 — clients may reuse this document until it goes stale)"
            )
            table = Table(title="Capabilities")
            table.add_column("Capability", style="bold")
            table.add_column("Settings")
            for cap_name, cap_value in sorted(result.capabilities.items()):
                if cap_name == "extensions":
                    continue
                table.add_row(cap_name, jsonlib.dumps(cap_value))
            for ext_id, ext_value in sorted((result.capabilities.get("extensions") or {}).items()):
                table.add_row(f"extension: {ext_id}", jsonlib.dumps(ext_value))
            console.print(table)
            if result.instructions:
                console.print(Panel(result.instructions, title="instructions"))

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Skills (SEP-2640): index verification + display
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)


def parse_skill_frontmatter(text: str) -> dict | None:
    """Extract the YAML frontmatter dict from a SKILL.md, or None."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None
    try:
        loaded = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    return loaded if isinstance(loaded, dict) else None


def verify_skill_document(text: str, entry: dict) -> tuple[bool, str]:
    """SEP-2640 host obligations for a skill fetched via its index entry.

    1. The SHA-256 of the SKILL.md bytes MUST match the index `digest`
       (format sha256:<64 lowercase hex>); on mismatch the content MUST NOT
       be used. (resources/read returns text; the digest covers its UTF-8
       bytes.)
    2. After the digest check, the parsed frontmatter MUST match the index
       entry's `frontmatter` FIELD BY FIELD; any discrepancy is a
       verification failure and the skill MUST NOT be loaded.

    A pass proves index/content consistency only — digests are unsigned and
    same-origin, so this is NOT a security boundary (spec's own words); the
    content stays untrusted model input either way.
    """
    computed = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    declared = entry.get("digest")
    if computed != declared:
        return False, f"digest mismatch: computed {computed}, index declares {declared}"
    frontmatter = parse_skill_frontmatter(text)
    if frontmatter is None:
        return False, "SKILL.md carries no parseable YAML frontmatter"
    expected = entry.get("frontmatter")
    if frontmatter != expected:
        return False, "frontmatter differs from the index entry (field-by-field mismatch)"
    return True, "verified"


def _resource_text(result) -> str:
    for item in result.contents:
        text = getattr(item, "text", None)
        if text is not None:
            return text
    raise typer.Exit(code=1)


@app.command()
def skills(
    server: str = ServerOption,
    anonymous: bool = AnonymousOption,
    show: str = typer.Option(None, "--show", help="Print the SKILL.md of one skill by name"),
):
    """List SEP-2640 skills from skill://index.json, verifying every digest."""

    async def run():
        async with connect(_target(server), get_settings(), anonymous=anonymous) as session:
            index_raw = await session.read_resource("skill://index.json")  # type: ignore[arg-type]
            index = jsonlib.loads(_resource_text(index_raw))

            # SEP-2640: a name is a LABEL, not an identifier — a skill is
            # identified by its url, and hosts MUST NOT assume names are
            # unique within an index. So we key verified content by url and
            # track which urls each name maps to, to disambiguate on --show
            # instead of silently preferring the last same-named entry.
            verified: dict[str, str] = {}  # url -> SKILL.md text (verified only)
            names_to_urls: dict[str, list[str]] = {}
            table = Table(title="Skills (skill://index.json)")
            table.add_column("Name", style="bold")
            table.add_column("URL")
            table.add_column("Description")
            table.add_column("Verification")
            for entry in index.get("skills", []):
                frontmatter = entry.get("frontmatter") or {}
                name = frontmatter.get("name", "?")
                url = entry.get("url", "")
                doc = await session.read_resource(url)  # type: ignore[arg-type]
                text = _resource_text(doc)
                ok, reason = verify_skill_document(text, entry)
                status = "[green]verified[/green]" if ok else f"[red]{reason}[/red]"
                if ok:
                    verified[url] = text
                    names_to_urls.setdefault(name, []).append(url)
                table.add_row(name, url, frontmatter.get("description", ""), status)
            console.print(table)

            if show:
                # Accept either a url or a (verified, unambiguous) name.
                if show in verified:
                    target_url = show
                else:
                    candidates = names_to_urls.get(show, [])
                    if len(candidates) > 1:
                        # Intra-index name collision: MUST disambiguate, not
                        # silently prefer one (SEP-2640 §3.1).
                        console.print(
                            f"[yellow]{show!r} is ambiguous — {len(candidates)} verified "
                            "skills share that name. Re-run --show with one of these URLs:"
                            "[/yellow]"
                        )
                        for url in candidates:
                            console.print(f"  {url}")
                        raise typer.Exit(code=2)
                    target_url = candidates[0] if candidates else None
                if target_url is None:
                    # SEP-2640: content failing digest/frontmatter verification
                    # MUST NOT be used — refusal, not a warning.
                    console.print(
                        f"[red]Refusing to display {show!r}: not present or failed "
                        "verification against the index.[/red]"
                    )
                    raise typer.Exit(code=1)
                console.print(Panel(verified[target_url], title=f"skill: {target_url}"))

    asyncio.run(run())


@app.command()
def listen(
    server: str = ServerOption,
    anonymous: bool = AnonymousOption,
    uri: list[str] = typer.Option(  # noqa: B008 - typer option factory
        [], "--uri", help="Resource URI(s) to watch for notifications/resources/updated"
    ),
):
    """Stream change notifications (subscriptions/listen) until interrupted."""

    async def run():
        async with connect(_target(server), get_settings(), anonymous=anonymous) as session:
            if not isinstance(session, ModernSession):
                console.print(
                    "[yellow]Legacy server[/yellow] — subscriptions/listen requires "
                    "protocol 2026-07-28."
                )
                raise typer.Exit(code=1)
            notifications: dict = {
                "toolsListChanged": True,
                "promptsListChanged": True,
                "resourcesListChanged": True,
            }
            if uri:
                notifications["resourceSubscriptions"] = list(uri)
            console.print("Listening (Ctrl-C to stop; closing the stream IS the cancel signal)…")
            async for message in session.listen(notifications):
                method = message.get("method", "?")
                params = {
                    k: v for k, v in (message.get("params") or {}).items() if k != "_meta"
                }
                console.print(f"[cyan]{method}[/cyan] {jsonlib.dumps(params)}")
            console.print("Server closed the subscription gracefully.")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\nStream closed (disconnect = cancellation on Streamable HTTP).")


@app.command()
def demo(server: str = ServerOption, anonymous: bool = AnonymousOption):
    """Guided tour: every MCP feature against the Virtual Library server."""

    async def run():
        settings = get_settings()
        async with connect(_target(server), settings, anonymous=anonymous) as session:
            console.print(f"Protocol era: [bold]{detect_era_label(session)}[/bold]")
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
