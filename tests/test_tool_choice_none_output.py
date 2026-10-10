"""With tools disabled for one response (tool_choice="none"), tool-call text is removed, not spoken or run."""

from types import SimpleNamespace

from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.language_model import LanguageModelHandler, StreamContext
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.tool_prompt import END_CODE, ENTER_CODE, build_block_regex

WEB_SEARCH = FunctionTool(
    type="function",
    name="web_search",
    description="Search.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)


def test_tool_call_text_is_dropped_when_tools_are_disabled():
    handler = object.__new__(LanguageModelHandler)
    handler.stream_batch_sentences = 1
    ctx = StreamContext(
        function_tools=[WEB_SEARCH],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
        discard_tool_calls=True,
    )
    text = f"我幫你查查看啦。{ENTER_CODE}web_search(query='手歌'){END_CODE}抱歉，我沒有找到。"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    spoken = "".join(chunk.text for chunk in chunks) + remaining
    assert ENTER_CODE not in spoken and "web_search" not in spoken
    assert tools == []
    assert all(chunk.tools == [] for chunk in chunks)


def _apply(tool_choice):
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext()
    tools = [SimpleNamespace(type="function", model_dump=lambda: WEB_SEARCH.model_dump())]
    handler._apply_instructions(Chat(4), "Be Alex.", tools, tool_choice, ctx)
    return ctx


def test_apply_instructions_marks_disabled_tools_for_stripping():
    disabled = _apply("none")
    assert disabled.discard_tool_calls and disabled.block_regex is not None
    enabled = _apply("auto")
    assert not enabled.discard_tool_calls and enabled.block_regex is not None


def test_no_tools_means_nothing_to_strip():
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext()
    handler._apply_instructions(Chat(4), "Be Alex.", [], "none", ctx)
    assert not ctx.discard_tool_calls and ctx.block_regex is None


def test_response_params_accept_tool_choice_none():
    assert RealtimeResponseCreateParams(tool_choice="none").tool_choice == "none"
