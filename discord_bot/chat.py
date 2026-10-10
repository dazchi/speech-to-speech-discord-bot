"""Text-channel questions: "@AI Alex ..." is answered out loud in the voice channel (and in chat)."""

from __future__ import annotations

import re
from collections.abc import Mapping

MAX_QUESTION_CHARS = 600
DISCORD_MESSAGE_LIMIT = 2000
_USER_MENTION = re.compile(r"<@!?(\d+)>")


def clean_question(content: str, bot_id: int, names: Mapping[int, str]) -> str:
    """Drop the bot's own mention and turn other user mentions into readable names."""

    def replace(match: re.Match[str]) -> str:
        user_id = int(match.group(1))
        if user_id == bot_id:
            return " "
        return f"@{names[user_id]}" if user_id in names else match.group(0)

    text = _USER_MENTION.sub(replace, content)
    text = " ".join(text.split()).strip(" ,，:：")
    return text[:MAX_QUESTION_CHARS]


def chat_label(name: str, question: str) -> str:
    return f"[{name} wrote in chat] {question}"


def split_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split a reply into Discord-sized messages, preferring line and sentence boundaries."""
    chunks: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        window = rest[:limit]
        cut = max(window.rfind("\n"), *(window.rfind(p) for p in "。！？.!?"))
        cut = cut + 1 if cut > limit // 2 else limit
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        chunks.append(rest)
    return chunks
