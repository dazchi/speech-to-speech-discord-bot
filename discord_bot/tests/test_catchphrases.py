import asyncio
import json
import random
from types import SimpleNamespace

from discord_bot.config import BotConfig
from discord_bot.realtime import RealtimeSession


def test_no_catchphrases_means_no_rule_and_no_note():
    config = BotConfig(discord_token="x")
    assert "catchphrase" not in config.session_instructions([])
    assert config.catchphrase_note(True) is None


def test_catchphrases_are_listed_and_noted_per_reply():
    config = BotConfig(discord_token="x", catchphrases=("我跟你說喔", "真的假的"))
    text = config.session_instructions(["web_search"])
    assert "「我跟你說喔」、「真的假的」" in text and "never inside the fixed sentence" in text
    assert "naturally include" in config.catchphrase_note(True)
    assert "do not use" in config.catchphrase_note(False)


def test_rate_controls_how_often_a_catchphrase_is_requested():
    from discord_bot.bot import VoiceBot

    bot = object.__new__(VoiceBot)
    bot.config = BotConfig(discord_token="x", catchphrases=("吼",), catchphrase_rate=0.3)
    bot.rng = random.Random(0)
    notes = [bot.pick_turn_note() for _ in range(2000)]
    share = sum("naturally include" in n for n in notes) / len(notes)
    assert 0.25 < share < 0.35


async def test_turn_note_is_sent_with_the_instructions_once():
    sent = []

    class WS:
        async def send(self, raw):
            sent.append(json.loads(raw))

    session = RealtimeSession("ws://x", "Be Alex.", None, SimpleNamespace(definitions=[]), SimpleNamespace())
    session._ws = WS()
    await session.set_turn_note("For this reply: naturally include one of your catchphrases.")
    await session.set_turn_note("For this reply: naturally include one of your catchphrases.")
    await asyncio.sleep(0)
    assert len(sent) == 1
    assert sent[0]["session"]["instructions"].endswith("naturally include one of your catchphrases.")


def test_catchphrases_from_env(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.setattr("discord_bot.config._load_dotenv", lambda: None)
    monkeypatch.setenv("CATCHPHRASES", "我跟你說喔 | 盆栽要剪，女人要扁")
    monkeypatch.setenv("CATCHPHRASE_RATE", "2")
    config = BotConfig.from_env()
    assert config.catchphrases == ("我跟你說喔", "盆栽要剪，女人要扁")  # "|" separates, commas stay
    assert config.catchphrase_rate == 1.0  # clamped
