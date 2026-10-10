import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("discord")

from discord_bot.bot import SessionHandlers  # noqa: E402
from discord_bot.config import BotConfig  # noqa: E402
from discord_bot.conversation import ConversationController  # noqa: E402
from discord_bot.realtime import RealtimeSession, speaker_label  # noqa: E402


def make_bot():
    announced: list[str] = []

    async def announce_speaker(name):
        announced.append(name)

    bot = SimpleNamespace(
        config=BotConfig(discord_token="x"),
        controller=ConversationController(wake_timeout_s=6, followup_s=8),
        speaker=SimpleNamespace(stop=lambda: None),
        loop=asyncio.get_running_loop(),
        session=SimpleNamespace(announce_speaker=announce_speaker, set_turn_note=lambda note: asyncio.sleep(0)),
        pick_turn_note=lambda: None,
    )
    return bot, SessionHandlers(bot), announced


async def test_each_turn_is_labelled_once_with_the_active_speaker():
    bot, handlers, announced = make_bot()
    bot.controller.wake(1, "Alice")
    handlers.on_speech_started()
    handlers.on_speech_started()  # VAD restarts within the same turn: no second label
    await asyncio.sleep(0.01)
    handlers.on_user_transcript("what's the weather?")
    handlers.on_speech_started()  # follow-up question: labelled again
    await asyncio.sleep(0.01)
    assert announced == ["Alice", "Alice"]


async def test_no_label_when_nobody_is_active():
    _, handlers, announced = make_bot()
    handlers.on_speech_started()
    await asyncio.sleep(0.01)
    assert announced == []


async def test_label_is_a_user_text_item():
    sent = []

    class WS:
        async def send(self, raw):
            import json

            sent.append(json.loads(raw))

    session = RealtimeSession("ws://x", "Be Alex.", None, SimpleNamespace(definitions=[]), SimpleNamespace())
    session._ws = WS()
    await session.announce_speaker("Bob")
    assert sent == [
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "[Bob is speaking]"}],
            },
        }
    ]
    assert speaker_label("Bob") == "[Bob is speaking]"
