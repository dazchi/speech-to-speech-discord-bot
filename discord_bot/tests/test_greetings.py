import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("discord")

from discord_bot.bot import VoiceBot  # noqa: E402
from discord_bot.config import BotConfig  # noqa: E402
from discord_bot.conversation import ConversationController  # noqa: E402
from discord_bot.realtime import RealtimeSession  # noqa: E402

CHANNEL = SimpleNamespace(id=10, name="一般")
OTHER = SimpleNamespace(id=20, name="other")


def member(user_id=5, name="lylla", bot=False):
    return SimpleNamespace(id=user_id, name=name, display_name=name, bot=bot, guild=SimpleNamespace(id=1))


def state(channel):
    return SimpleNamespace(channel=channel)


def make_bot(**config):
    bot = object.__new__(VoiceBot)
    bot.config = BotConfig(discord_token="x", auto_join=False, **config)
    bot.controller = ConversationController(wake_timeout_s=6, followup_s=8)
    bot.greet_queue = asyncio.Queue()
    bot._last_greeted, bot._joined_at = {}, {}
    bot.voice = SimpleNamespace(channel=CHANNEL)
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    bot._serves = lambda guild: True
    bot._rebalance_task = None
    bot._auto_join_paused = False
    return bot


def queued(bot):
    items = []
    while not bot.greet_queue.empty():
        items.append(bot.greet_queue.get_nowait()[1])
    return items


async def test_greets_a_person_who_joins_the_bots_channel():
    bot = make_bot()
    await bot.on_voice_state_update(member(), state(None), state(CHANNEL))
    assert queued(bot) == ["lylla"]


async def test_ignores_bots_other_channels_and_mute_toggles():
    bot = make_bot()
    await bot.on_voice_state_update(member(6, "MusicBot", bot=True), state(None), state(CHANNEL))
    await bot.on_voice_state_update(member(7, "bob"), state(None), state(OTHER))
    await bot.on_voice_state_update(member(8, "carol"), state(CHANNEL), state(CHANNEL))  # mute/deafen
    assert queued(bot) == []


async def test_cooldown_prevents_greeting_a_quick_rejoin():
    bot = make_bot(greet_cooldown_s=300)
    await bot.on_voice_state_update(member(), state(None), state(CHANNEL))
    await bot.on_voice_state_update(member(), state(CHANNEL), state(None))
    await bot.on_voice_state_update(member(), state(None), state(CHANNEL))
    assert queued(bot) == ["lylla"]


async def test_can_be_turned_off():
    bot = make_bot(greet_on_join=False)
    await bot.on_voice_state_update(member(), state(None), state(CHANNEL))
    assert queued(bot) == []


def test_greeting_prompt_fills_in_the_name():
    config = BotConfig(discord_token="x", greeting_prompt="Say hi to {name}!")
    assert config.greeting_instruction("lylla") == "Say hi to lylla!"


async def test_greeting_is_out_of_band_and_without_tools():
    sent = []

    class WS:
        async def send(self, raw):
            sent.append(json.loads(raw))

    session = RealtimeSession(
        "ws://x",
        "Be Alex. Search a lot.",
        None,
        SimpleNamespace(definitions=[]),
        SimpleNamespace(),
        answer_instructions="Be Alex.",
    )
    session._ws = WS()
    await session.greet("lylla", "Greet lylla.")
    response = sent[0]["response"]
    assert response["conversation"] == "none"  # not written into the shared history
    assert response["tool_choice"] == "none"
    assert "Search a lot" not in response["instructions"] and response["instructions"].endswith("Greet lylla.")
    assert response["input"][0]["content"][0]["text"] == "[lylla joined the voice channel]"
