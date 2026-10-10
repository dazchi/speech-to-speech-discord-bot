"""
Optional system-prompt builder that instructs a local LLM to output tool calls
inside delimited code blocks (e.g. ``<code>func(arg='val')</code>``).

The prompt is rendered from a Jinja2 template and relies on
``FunctionTool.to_code_prompt()`` to expose each tool's Python-style signature.
"""

import re
from typing import Any

from jinja2 import Template

from speech_to_speech.LLM.tool_call.function_tool import FunctionTool

# ---------------------------------------------------------------------------
# Default delimiters
# ---------------------------------------------------------------------------
ENTER_CODE = "<code>"
END_CODE = "</code>"

# ---------------------------------------------------------------------------
# Jinja2 template
# ---------------------------------------------------------------------------
# ``enter_code`` / ``end_code`` are the block delimiters the model must wrap
# every tool call in.  ``tools`` is a list of FunctionTool instances whose
# ``.to_code_prompt()`` is called inside the template.
# ---------------------------------------------------------------------------

TOOL_PROMPT_TEMPLATE = Template(
    """\
Available tools:

{% for tool in tools %}
{{ tool.to_code_prompt() }}

{% endfor %}
To call tools, put each named-argument function call inside its own {{ enter_code }}...{{ end_code }} block:
{{ enter_code }}function_name(required_arg='value'){{ end_code }}

Rules:
- Keep tags out of prose. Keep prose outside tags brief, and do not claim tool results before a tool result is available.
- Use named arguments only; quote strings. Omit optional args instead of placeholder values like "random", "none", "", or null.
- Keep every tool call in a separate block and preserve the intended text/tool order.\
""",
    keep_trailing_newline=True,
)

# Text-channel variant: same call format and structural rules, with explicit
# guidance to skip a preamble before a tool call.
TEXT_TOOL_PROMPT_TEMPLATE = Template(
    """\
Available tools:

{% for tool in tools %}
{{ tool.to_code_prompt() }}

{% endfor %}
To call tools, put each named-argument function call inside its own {{ enter_code }}...{{ end_code }} block:
{{ enter_code }}function_name(required_arg='value'){{ end_code }}

Rules:
- Call a tool directly when it helps fulfill the request; no preamble sentence is required.
- Do not mention tags, functions, or tools in your prose, and do not claim tool results before a tool result is available.
- Use named arguments only; quote strings. Omit optional args instead of placeholder values like "random", "none", "", or null.
- Keep every tool call in a separate block and preserve the intended text/tool order.\
""",
    keep_trailing_newline=True,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_tool_system_prompt(
    tools: list[FunctionTool],
    enter_code: str = ENTER_CODE,
    end_code: str = END_CODE,
    *,
    text_only: bool = False,
) -> str:
    """Render the tool-calling system-prompt section.

    Returns an empty string when *tools* is empty so it can be
    unconditionally appended to a base system prompt.  When *text_only* is set,
    the written-channel variant adds explicit no-preamble guidance.
    """
    if not tools:
        return ""

    template = TEXT_TOOL_PROMPT_TEMPLATE if text_only else TOOL_PROMPT_TEMPLATE
    return template.render(
        tools=tools,
        enter_code=enter_code,
        end_code=end_code,
    )


def build_block_regex(
    enter_code: str = ENTER_CODE,
    end_code: str = END_CODE,
) -> str:
    """Build a regex that matches a single code block (non-greedy).

    >>> build_block_regex("<code>", "</code>")
    '<code>.*?</code>'
    """
    return f"{re.escape(enter_code)}.*?{re.escape(end_code)}"


def format_tool_call_block(
    name: str,
    arguments: dict[str, Any],
    enter_code: str = ENTER_CODE,
    end_code: str = END_CODE,
) -> str:
    """Render a tool call in the same delimited format the tool prompt asks for.

    >>> format_tool_call_block("web_search", {"query": "weather"})
    "<code>web_search(query='weather')</code>"
    """
    args = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{enter_code}{name}({args}){end_code}"


def tool_calls_as_code_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite structured assistant ``tool_calls`` as ``<code>`` text.

    Chat templates render past tool calls in the model's native format (e.g.
    Qwen's ``<tool_call>`` XML), which contradicts the ``<code>`` format the tool
    prompt asks for; models then copy the history and stop emitting parseable
    calls. Keeping history in the prompted format avoids that drift.

    Spoken text stored just before a call (e.g. "Let me check.") is merged into
    the same assistant turn: as separate turns it teaches the model to end its
    reply after the preamble, so the call is never generated.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if message.get("role") == "assistant" and tool_calls:
            blocks = "".join(
                format_tool_call_block(call["function"]["name"], call["function"]["arguments"]) for call in tool_calls
            )
            content = f"{message.get('content') or ''}{blocks}"
            previous = converted[-1] if converted else None
            if previous and previous.get("role") == "assistant" and isinstance(previous.get("content"), str):
                converted[-1] = {**previous, "content": f"{previous['content']}{content}"}
            else:
                converted.append({"role": "assistant", "content": content})
        else:
            converted.append(message)
    return converted
