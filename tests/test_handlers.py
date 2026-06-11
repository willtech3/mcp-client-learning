"""Tests for the server-initiated request handlers.

The Anthropic client is replaced with a stub: these tests verify OUR
mapping logic (MCP <-> Anthropic shapes, model hint resolution, SEP-1577
tool bridging, human-in-the-loop), not Anthropic's API.
"""

import pytest
from mcp import types

from mcp_client.handlers import (
    DEFAULT_MODEL,
    SamplingHandler,
    choose_model,
    elicitation_handler,
)


class TestModelSelection:
    def test_default_without_preferences(self):
        assert choose_model(None) == DEFAULT_MODEL

    def test_hint_substring_matches_known_model(self):
        prefs = types.ModelPreferences(hints=[types.ModelHint(name="claude-sonnet-4-6")])
        assert choose_model(prefs) == "claude-sonnet-4-6"

    def test_partial_hint_resolves(self):
        prefs = types.ModelPreferences(hints=[types.ModelHint(name="claude-opus-4-8")])
        assert choose_model(prefs) == "claude-opus-4-8"

    def test_unknown_hint_falls_back_to_default(self):
        prefs = types.ModelPreferences(hints=[types.ModelHint(name="gpt-99-ultra")])
        assert choose_model(prefs) == DEFAULT_MODEL

    def test_generic_claude_hint(self):
        prefs = types.ModelPreferences(hints=[types.ModelHint(name="claude")])
        assert choose_model(prefs) == DEFAULT_MODEL  # generic hint -> our default Claude


class _FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResponse:
    def __init__(self, content, stop_reason="end_turn", model="claude-opus-4-8"):
        self.content = content
        self.stop_reason = stop_reason
        self.model = model


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._response


class _FakeAnthropic:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def _params(**overrides) -> types.CreateMessageRequestParams:
    base = {
        "messages": [
            types.SamplingMessage(
                role="user", content=types.TextContent(type="text", text="Summarize Dune")
            )
        ],
        "maxTokens": 400,
    }
    base.update(overrides)
    return types.CreateMessageRequestParams(**base)


@pytest.fixture
def handler():
    h = SamplingHandler(api_key=None, auto_approve=True)
    h.client = _FakeAnthropic(_FakeResponse([_FakeBlock(type="text", text="A spice-laden epic.")]))
    return h


class TestSamplingHandler:
    async def test_text_round_trip(self, handler):
        result = await handler(None, _params())
        assert isinstance(result, types.CreateMessageResult)
        assert result.content.text == "A spice-laden epic."
        assert result.stopReason == "endTurn"

    async def test_no_api_key_returns_error_data(self):
        h = SamplingHandler(api_key=None, auto_approve=True)  # client stays None
        result = await h(None, _params())
        assert isinstance(result, types.ErrorData)

    async def test_temperature_never_forwarded(self, handler):
        """Sampling params are removed on current Claude models; forwarding
        the server's temperature would 400 every request."""
        await handler(None, _params(temperature=0.9))
        assert "temperature" not in handler.client.messages.requests[0]

    async def test_system_prompt_and_stop_sequences_forwarded(self, handler):
        await handler(None, _params(systemPrompt="be brief", stopSequences=["END"]))
        request = handler.client.messages.requests[0]
        assert request["system"] == "be brief"
        assert request["stop_sequences"] == ["END"]

    async def test_sep1577_tools_mapped_to_anthropic(self, handler):
        tool = types.Tool(
            name="search_library_catalog",
            description="Search holdings",
            inputSchema={"type": "object", "properties": {"genre": {"type": "string"}}},
        )
        await handler(None, _params(tools=[tool]))
        request = handler.client.messages.requests[0]
        assert request["tools"] == [
            {
                "name": "search_library_catalog",
                "description": "Search holdings",
                "input_schema": {"type": "object", "properties": {"genre": {"type": "string"}}},
            }
        ]

    async def test_tool_use_response_becomes_tool_use_content(self):
        h = SamplingHandler(api_key=None, auto_approve=True)
        h.client = _FakeAnthropic(
            _FakeResponse(
                [_FakeBlock(type="tool_use", id="tu_1", name="search", input={"genre": "Fantasy"})],
                stop_reason="tool_use",
            )
        )
        result = await h(None, _params())
        assert isinstance(result, types.CreateMessageResultWithTools)
        assert isinstance(result.content, types.ToolUseContent)
        assert result.content.name == "search"
        assert result.stopReason == "toolUse"

    async def test_tool_result_content_maps_back_to_anthropic(self, handler):
        """Round 2 of a tool loop: the server sends the tool result back."""
        params = _params(
            messages=[
                types.SamplingMessage(
                    role="user", content=types.TextContent(type="text", text="find books")
                ),
                types.SamplingMessage(
                    role="assistant",
                    content=types.ToolUseContent(
                        type="tool_use", id="tu_1", name="search", input={}
                    ),
                ),
                types.SamplingMessage(
                    role="user",
                    content=types.ToolResultContent(
                        type="tool_result",
                        toolUseId="tu_1",
                        content=[types.TextContent(type="text", text='["Dune"]')],
                    ),
                ),
            ]
        )
        await handler(None, params)
        sent = handler.client.messages.requests[0]["messages"]
        assert sent[1]["content"][0]["type"] == "tool_use"
        assert sent[2]["content"][0]["type"] == "tool_result"
        assert sent[2]["content"][0]["tool_use_id"] == "tu_1"


class TestElicitationHandler:
    async def test_approval_only_accept(self, monkeypatch):
        monkeypatch.setattr("rich.prompt.Prompt.ask", staticmethod(lambda *a, **k: "accept"))
        params = types.ElicitRequestFormParams(
            message="Proceed despite fines?", requestedSchema={"type": "object", "properties": {}}
        )
        result = await elicitation_handler(None, params)
        assert result.action == "accept"

    async def test_enum_selection(self, monkeypatch):
        answers = iter(["12 months", True])
        monkeypatch.setattr("rich.prompt.Prompt.ask", staticmethod(lambda *a, **k: next(answers)))
        monkeypatch.setattr("rich.prompt.Confirm.ask", staticmethod(lambda *a, **k: True))
        params = types.ElicitRequestFormParams(
            message="Which term?",
            requestedSchema={
                "type": "object",
                "properties": {
                    "value": {"title": "Term", "enum": ["6 months", "12 months", "24 months"]}
                },
            },
        )
        result = await elicitation_handler(None, params)
        assert result.action == "accept"
        assert result.content == {"value": "12 months"}

    async def test_url_mode_elicitation(self, monkeypatch):
        """SEP-1036: the server may ask the user to visit a URL."""
        monkeypatch.setattr("rich.prompt.Confirm.ask", staticmethod(lambda *a, **k: True))
        params = types.ElicitRequestURLParams(
            mode="url",
            message="Complete verification",
            url="https://example.com/verify",
            elicitationId="el_1",
        )
        result = await elicitation_handler(None, params)
        assert result.action == "accept"

    async def test_cancel_on_final_confirm(self, monkeypatch):
        monkeypatch.setattr("rich.prompt.Prompt.ask", staticmethod(lambda *a, **k: "anything"))
        monkeypatch.setattr("rich.prompt.Confirm.ask", staticmethod(lambda *a, **k: False))
        params = types.ElicitRequestFormParams(
            message="Name?",
            requestedSchema={
                "type": "object",
                "properties": {"name": {"type": "string", "title": "Name"}},
            },
        )
        result = await elicitation_handler(None, params)
        assert result.action == "cancel"
