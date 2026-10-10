"""Local models see past tool calls in the same ``<code>`` format the tool prompt asks for.

Chat templates otherwise render history tool calls in the model's native format
(e.g. Qwen's ``<tool_call>`` XML); the model copies that and its later calls are
no longer parsed.
"""

from __future__ import annotations

from typing import Any

import pytest
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
)

from speech_to_speech.LLM import language_model
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.LLM.language_model import LanguageModelHandler, StreamContext
from speech_to_speech.LLM.tool_call.function_call import extract_function_calls_from_text
from speech_to_speech.LLM.tool_call.tool_prompt import (
    build_block_regex,
    format_tool_call_block,
    tool_calls_as_code_blocks,
)


def _tool_call_message(*calls: tuple[str, dict[str, Any]], content: str = "") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"type": "function", "id": f"call_{i}", "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(calls)
        ],
    }


def test_format_tool_call_block_round_trips_through_parser() -> None:
    args = {"query": "台北 今天 天气", "limit": 3, "fresh": True}

    block = format_tool_call_block("web_search", args)
    _, calls = extract_function_calls_from_text(block, build_block_regex())

    assert block == "<code>web_search(query='台北 今天 天气', limit=3, fresh=True)</code>"
    assert [(c.function_name, c.parameters) for c in calls] == [("web_search", args)]


def test_format_tool_call_block_without_arguments() -> None:
    assert format_tool_call_block("get_time", {}) == "<code>get_time()</code>"


def test_tool_calls_as_code_blocks_rewrites_only_tool_call_messages() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "search two things"},
        _tool_call_message(("web_search", {"query": "a"}), ("web_search", {"query": "b"}), content="Let me check."),
        {"role": "tool", "tool_call_id": "call_0", "name": "web_search", "content": "result a"},
        {"role": "assistant", "content": "Done."},
    ]

    converted = tool_calls_as_code_blocks(messages)

    assert converted[2] == {
        "role": "assistant",
        "content": "Let me check.<code>web_search(query='a')</code><code>web_search(query='b')</code>",
    }
    assert converted[:2] == messages[:2]
    assert converted[3:] == messages[3:]


def test_tool_calls_as_code_blocks_merges_spoken_preamble_into_call_turn() -> None:
    messages = [
        {"role": "user", "content": "查一下台北天气"},
        {"role": "assistant", "content": "我帮你查看看啦。"},
        _tool_call_message(("web_search", {"query": "台北 天气"})),
        {"role": "tool", "tool_call_id": "call_0", "name": "web_search", "content": "晴天"},
        {"role": "assistant", "content": "台北今天晴天。"},
    ]

    converted = tool_calls_as_code_blocks(messages)

    assert converted == [
        {"role": "user", "content": "查一下台北天气"},
        {"role": "assistant", "content": "我帮你查看看啦。<code>web_search(query='台北 天气')</code>"},
        messages[3],
        messages[4],
    ]
    assert messages[1] == {"role": "assistant", "content": "我帮你查看看啦。"}


@pytest.mark.skipif(not language_model.HAS_MLX, reason="mlx_lm not installed")
def test_mlx_prompt_history_uses_code_format(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[dict[str, Any]]] = []

    class _Tokenizer:
        def apply_chat_template(self, messages: Any, tokenize: bool = False, **kwargs: Any) -> Any:
            seen.append(messages)
            return [1] if tokenize else "prompt"

    class _Stop(Exception):
        pass

    def fake_stream_generate(*args: Any, **kwargs: Any) -> Any:
        raise _Stop

    monkeypatch.setattr(language_model, "mlx_stream_generate", fake_stream_generate)
    handler = object.__new__(LanguageModelHandler)
    handler.backend = "mlx"
    handler.model = object()
    handler.tokenizer = _Tokenizer()
    handler.gen_kwargs = {"max_new_tokens": 8}

    chat = Chat(size=8)
    chat.add_item(make_user_message("帮我搜索一下今天台北的天气"))
    chat.add_item(
        RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_call_1",
            call_id="call_1",
            name="web_search",
            arguments='{"query": "台北 今天 天气"}',
        )
    )
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(type="function_call_output", call_id="call_1", output="晴天")
    )
    chat.add_item(make_assistant_message("今天台北晴天。"))
    chat.add_item(make_user_message("搜索一下最新的苹果股价"))

    with pytest.raises(_Stop):
        next(iter(handler._generate(chat, None, None, StreamContext())))

    assert seen, "prompt was never rendered"
    for messages in seen:
        assert not any("tool_calls" in m for m in messages)
        assert {"role": "assistant", "content": "<code>web_search(query='台北 今天 天气')</code>"} in messages
        assert any(m["role"] == "tool" and m["content"] == "晴天" for m in messages)
