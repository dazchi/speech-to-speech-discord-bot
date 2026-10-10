import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("discord")

from discord_bot.bot import SessionHandlers, UserAudio, VoiceBot  # noqa: E402
from discord_bot.config import BotConfig  # noqa: E402
from discord_bot.conversation import ConversationController  # noqa: E402
from discord_bot.realtime import build_session_update  # noqa: E402


class Session:
    def __init__(self):
        self.audio = []

    async def send_audio(self, pcm):
        self.audio.append(pcm)


def make_bot(allow: bool) -> VoiceBot:
    bot = object.__new__(VoiceBot)
    bot.config = BotConfig(discord_token="x", allow_interruptions=allow)
    bot.controller = ConversationController(wake_timeout_s=6, followup_s=8)
    bot.session = Session()
    bot.controller.wake(1, "alice")
    return bot


@pytest.mark.parametrize("allow", [True, False])
async def test_speaker_audio_while_alex_is_busy(allow):
    bot = make_bot(allow)
    state = UserAudio()
    await bot._route(1, state, b"question", np.zeros(320, np.float32))
    bot.controller.response_started()  # Alex is thinking / speaking
    await bot._route(1, state, b"talking over alex", np.zeros(320, np.float32))
    expected = [b"question", b"talking over alex"] if allow else [b"question"]
    assert bot.session.audio == expected


async def test_speech_does_not_cut_alex_off_when_interruptions_are_disabled():
    stopped = []
    bot = SimpleNamespace(
        config=BotConfig(discord_token="x", allow_interruptions=False),
        controller=ConversationController(wake_timeout_s=6, followup_s=8),
        speaker=SimpleNamespace(stop=lambda: stopped.append(True)),
        loop=asyncio.get_running_loop(),
        session=SimpleNamespace(announce_speaker=lambda name: asyncio.sleep(0)),
    )
    SessionHandlers(bot).on_speech_started()
    assert stopped == []


def test_server_is_told_not_to_cancel_replies():
    vad = build_session_update("x", None, [], allow_interruptions=False)["session"]["audio"]["input"]["turn_detection"]
    assert vad["interrupt_response"] is False
    assert build_session_update("x", None, [])["session"]["audio"]["input"]["turn_detection"]["interrupt_response"]


def test_assistant_busy_covers_thinking_tools_and_playback():
    c = ConversationController(wake_timeout_s=6, followup_s=8)
    assert not c.assistant_busy
    c.speech_started()
    assert not c.assistant_busy  # the user talking is not Alex being busy
    c.response_started()
    assert c.assistant_busy
    c.playback_started()
    c.response_finished(tools_pending=False)
    assert c.assistant_busy  # still speaking
    c.playback_finished()
    c.speech_stopped()
    assert not c.assistant_busy


def test_option_from_env(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.setattr("discord_bot.config._load_dotenv", lambda: None)
    monkeypatch.setenv("ALLOW_INTERRUPTIONS", "false")
    assert BotConfig.from_env().allow_interruptions is False
    monkeypatch.delenv("ALLOW_INTERRUPTIONS")
    assert BotConfig.from_env().allow_interruptions is True
