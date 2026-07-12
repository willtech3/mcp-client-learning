"""Handlers for server-initiated MCP requests.

MCP reverses direction for three request types — the SERVER asks the
CLIENT — and this module implements all of them:

- sampling/createMessage: the server wants an LLM completion. We bridge
  to the Anthropic API, honoring model preference hints, and support
  tool-enabled sampling (SEP-1577): the server may attach tools the LLM
  can call; we surface Claude's tool_use blocks back to the server, which
  executes them and continues the loop.
- elicitation/create: the server needs input from the human. We render
  the requested schema as terminal prompts (strings, booleans, enums) and
  return accept/decline/cancel — the user always has an explicit out.
- notifications/message: server log messages, rendered to the console.

Human-in-the-loop: the spec says users SHOULD control sampling. Every
sampling request is shown and confirmed before tokens are spent, unless
auto-approve is configured.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
from mcp import types
from mcp.shared.context import RequestContext
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

logger = logging.getLogger(__name__)
console = Console()

# Models we recognize when the server hints a preference. Hints are
# advisory substrings per the MCP spec; anything unrecognized falls back
# to the default.
KNOWN_MODELS = [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-fable-5",
]
DEFAULT_MODEL = "claude-opus-4-8"


def choose_model(preferences: types.ModelPreferences | None) -> str:
    """Map the server's model *hints* onto a model we actually have.

    Hints are substrings evaluated in order ("claude-opus" matches
    claude-opus-4-8). The client always retains final say — that's the
    spec's trust boundary, not a courtesy.
    """
    if preferences and preferences.hints:
        for hint in preferences.hints:
            if not hint.name:
                continue
            for model in KNOWN_MODELS:
                if hint.name in model or model in hint.name:
                    return model
    return DEFAULT_MODEL


def _mcp_content_to_anthropic(content: Any) -> list[dict]:
    """Convert one MCP sampling content block (or list) to Anthropic blocks."""
    blocks = content if isinstance(content, list) else [content]
    converted: list[dict] = []
    for block in blocks:
        if isinstance(block, types.TextContent):
            converted.append({"type": "text", "text": block.text})
        elif isinstance(block, types.ImageContent):
            converted.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mimeType,
                        "data": block.data,
                    },
                }
            )
        elif isinstance(block, types.ToolUseContent):
            converted.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
            )
        elif isinstance(block, types.ToolResultContent):
            converted.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.toolUseId,
                    "content": [
                        {"type": "text", "text": c.text}
                        for c in (block.content or [])
                        if isinstance(c, types.TextContent)
                    ],
                    "is_error": bool(block.isError),
                }
            )
    return converted


_STOP_REASON_MAP = {
    "end_turn": "endTurn",
    "max_tokens": "maxTokens",
    "stop_sequence": "stopSequence",
    "tool_use": "toolUse",
    "refusal": "endTurn",
}


def _build_anthropic_request(
    params: types.CreateMessageRequestParams, model: str
) -> dict[str, Any]:
    """Translate an MCP sampling request into Anthropic Messages kwargs.

    temperature is intentionally NOT forwarded — sampling parameters are
    removed on the current Claude generation (Opus 4.7+/Fable) and would
    fail the request.
    """
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": min(params.maxTokens or 1024, 4096),
        "messages": [
            {"role": m.role, "content": _mcp_content_to_anthropic(m.content)}
            for m in params.messages
        ],
    }
    if params.systemPrompt:
        request["system"] = params.systemPrompt
    if params.stopSequences:
        request["stop_sequences"] = params.stopSequences

    # SEP-1577: tools attached by the server become Anthropic tools.
    if params.tools:
        request["tools"] = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in params.tools
        ]
    return request


class SamplingHandler:
    """Bridges sampling/createMessage to the Anthropic API."""

    def __init__(self, api_key: str | None = None, auto_approve: bool = False):
        self.client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else None
        self.auto_approve = auto_approve

    async def __call__(
        self,
        context: RequestContext,
        params: types.CreateMessageRequestParams,
    ) -> types.CreateMessageResult | types.CreateMessageResultWithTools | types.ErrorData:
        if self.client is None:
            return types.ErrorData(
                code=types.INTERNAL_ERROR,
                message="Client has no ANTHROPIC_API_KEY configured; sampling unavailable.",
            )

        model = choose_model(params.modelPreferences)

        # HUMAN IN THE LOOP: show what the server wants before spending
        # tokens on its behalf.
        preview = ""
        if params.messages:
            first = params.messages[0].content
            if isinstance(first, types.TextContent):
                preview = first.text[:300]
        console.print(
            Panel(
                f"[bold]Server requests an LLM completion[/bold] (model: {model})\n\n{preview}...",
                title="sampling/createMessage",
                border_style="yellow",
            )
        )
        if not self.auto_approve and not Confirm.ask("Allow this sampling request?", default=True):
            return types.ErrorData(code=types.INVALID_REQUEST, message="User rejected sampling")

        request = _build_anthropic_request(params, model)

        try:
            response = await self.client.messages.create(**request)
        except anthropic.APIError as e:
            logger.warning("Anthropic API error during sampling: %s", e)
            return types.ErrorData(code=types.INTERNAL_ERROR, message=f"Sampling failed: {e}")

        # Convert Claude's response blocks back to MCP sampling content.
        out_blocks: list[types.TextContent | types.ToolUseContent] = []
        for block in response.content:
            if block.type == "text" and block.text:
                out_blocks.append(types.TextContent(type="text", text=block.text))
            elif block.type == "tool_use":
                out_blocks.append(
                    types.ToolUseContent(
                        type="tool_use", id=block.id, name=block.name, input=dict(block.input)
                    )
                )

        content: Any = out_blocks if len(out_blocks) != 1 else out_blocks[0]
        if not out_blocks:
            content = types.TextContent(type="text", text="")
        stop_reason = _STOP_REASON_MAP.get(response.stop_reason or "end_turn", "endTurn")

        # SEP-1577 defines a wider result type whose content may include
        # tool_use blocks; plain CreateMessageResult forbids them. Use the
        # wide type whenever the exchange involves tools.
        has_tool_use = any(isinstance(b, types.ToolUseContent) for b in out_blocks)
        if params.tools or has_tool_use:
            return types.CreateMessageResultWithTools(
                role="assistant",
                content=content,
                model=response.model,
                stopReason=stop_reason,
            )
        return types.CreateMessageResult(
            role="assistant",
            content=content,
            model=response.model,
            stopReason=stop_reason,
        )


async def elicitation_handler(
    context: RequestContext,
    params: types.ElicitRequestParams,  # URL or Form variant
) -> types.ElicitResult:
    """Render an elicitation request as terminal prompts.

    Supports the 2025-11-25 surface: plain approval, typed fields
    (string/number/boolean), single-select enums, defaults, and URL-mode
    elicitation (the server asks the user to visit a link).
    """
    console.print(Panel(params.message, title="Server asks", border_style="cyan"))

    # URL mode (SEP-1036): hand the URL to the human, confirm completion.
    if isinstance(params, types.ElicitRequestURLParams):
        console.print(f"The server asks you to visit:\n  [bold blue]{params.url}[/bold blue]")
        done = Confirm.ask("Did you complete it?", default=False)
        return types.ElicitResult(action="accept" if done else "decline")

    schema = params.requestedSchema
    properties: dict[str, Any] = (schema or {}).get("properties", {}) if schema else {}

    # Approval-only: no fields requested, just consent.
    if not properties:
        choice = Prompt.ask("Respond", choices=["accept", "decline", "cancel"], default="accept")
        return types.ElicitResult(action=choice)  # type: ignore[arg-type]

    answers: dict[str, Any] = {}
    for name, prop in properties.items():
        title = prop.get("title", name)
        default = prop.get("default")

        # Enum constraint -> menu (covers SEP-1330 single-select enums,
        # including the anyOf/const encoding pydantic emits).
        choices = prop.get("enum")
        if choices is None and "anyOf" in prop:
            consts = [v.get("const") for v in prop["anyOf"] if "const" in v]
            choices = consts or None

        if choices:
            answers[name] = Prompt.ask(
                title, choices=[str(c) for c in choices], default=str(default) if default else None
            )
        elif prop.get("type") == "boolean":
            answers[name] = Confirm.ask(title, default=bool(default))
        elif prop.get("type") in ("number", "integer"):
            raw = Prompt.ask(title, default=str(default) if default is not None else None) or "0"
            answers[name] = int(raw) if prop["type"] == "integer" else float(raw)
        else:
            answers[name] = Prompt.ask(title, default=default)

    if not Confirm.ask("Submit these answers?", default=True):
        return types.ElicitResult(action="cancel")
    return types.ElicitResult(action="accept", content=answers)


async def logging_handler(params: types.LoggingMessageNotificationParams) -> None:
    """Render server log notifications with severity-appropriate styling."""
    styles = {"error": "red", "warning": "yellow", "info": "dim", "debug": "dim blue"}
    style = styles.get(str(params.level), "dim")
    console.print(f"[{style}]server[{params.level}]: {params.data}[/{style}]")


# ---------------------------------------------------------------------------
# MRTR wire adapters (MCP 2026-07-28, SEP-2322)
#
# Under the modern era, servers no longer SEND elicitation/sampling as
# JSON-RPC requests; they return them as bare {method, params} objects inside
# an InputRequiredResult, and the client answers them locally before retrying
# the original request. These adapters bridge those wire-level dicts to the
# exact same handlers above, so the human-facing UX (terminal prompts,
# sampling approval) is identical in both eras.
# ---------------------------------------------------------------------------

# The handlers above take a RequestContext because the legacy SDK provides
# one; under MRTR there is no server-initiated request and hence no context.
_NO_CONTEXT: Any = None


class WireSamplingError(Exception):
    """Sampling could not be served client-side. Under MRTR the server is NOT
    waiting for an answer (there is no error channel for nested requests) —
    the client simply may not retry, so this propagates and ends the loop."""


async def elicit_from_wire(params: dict) -> dict:
    """Answer one MRTR ``elicitation/create`` inputRequest.

    2026-07-28 notes handled here: `mode` is optional and defaults to
    "form" (clients MUST treat a mode-less request as form mode), and the
    URL-mode `elicitationId` field was REMOVED — correlation now lives in
    the server's opaque requestState — so we synthesize one purely to
    satisfy the legacy SDK model the terminal handler consumes.
    """
    if params.get("mode") == "url":
        sdk_params: types.ElicitRequestParams = types.ElicitRequestURLParams(
            mode="url",
            message=params.get("message", ""),
            url=params.get("url", ""),
            elicitationId=params.get("elicitationId", "mrtr"),
        )
    else:
        sdk_params = types.ElicitRequestFormParams(
            message=params.get("message", ""),
            requestedSchema=params.get("requestedSchema")
            or {"type": "object", "properties": {}},
        )
    result = await elicitation_handler(_NO_CONTEXT, sdk_params)
    # Bare InputResponse object: {action, content?} — no resultType, no id.
    return result.model_dump(by_alias=True, exclude_none=True, exclude={"meta"})


async def sample_from_wire(handler: SamplingHandler, params: dict) -> dict:
    """Answer one MRTR ``sampling/createMessage`` inputRequest.

    The wire params validate directly into the SDK's request model (the
    sampling *payload* did not change in 2026-07-28 — only its envelope did).
    An ErrorData outcome (no API key, user rejection, API failure) becomes an
    exception: under MRTR the spec-sanctioned move is to stop retrying.
    """
    sdk_params = types.CreateMessageRequestParams.model_validate(params)
    result = await handler(_NO_CONTEXT, sdk_params)
    if isinstance(result, types.ErrorData):
        raise WireSamplingError(result.message)
    return result.model_dump(by_alias=True, exclude_none=True, exclude={"meta"})
