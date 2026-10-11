"""Regression: a reply that ends with response.done but no output_audio.done must still close the window."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("discord")

from discord_bot.bot import SessionHandlers  # noqa: E402
from discord_bot.config import BotConfig  # noqa: E402
from discord_bot.conversation import ConversationController  # noqa: E402
from discord_bot.playback import Speaker  # noqa: E402


class PlayingVoiceClient:
    """Plays a source in real time on a thread, like discord.py's AudioPlayer."""

    def __init__(self) -> None:
        self._thread = None
        self._stop = threading.Event()

    def is_connected(self):
        return True

    def is_playing(self):
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def stop(self):
        self._stop.set()

    def play(self, source, *, after=None):
        self._stop = stop = threading.Event()

        def run():
            while not stop.is_set() and source.read() != b"":
                time.sleep(0.002)
            after(None)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()


async def test_window_closes_after_reply_without_output_audio_done():
    controller = ConversationController(wake_timeout_s=6, followup_s=0)
    vc = PlayingVoiceClient()
    speaker = Speaker(lambda: vc, asyncio.get_running_loop(), controller.playback_finished)

    async def announce_speaker(name):
        pass

    bot = SimpleNamespace(
        config=BotConfig(discord_token="x"),
        controller=controller,
        speaker=speaker,
        output=SimpleNamespace(convert=lambda pcm: pcm),
        ducker=None,
        loop=asyncio.get_running_loop(),
        session=SimpleNamespace(announce_speaker=announce_speaker, set_turn_note=lambda note: asyncio.sleep(0)),
        pick_turn_note=lambda: None,
    )
    handlers = SessionHandlers(bot)

    controller.wake(1, "alice")
    handlers.on_speech_started()
    handlers.on_speech_stopped()
    handlers.on_response_created()
    for _ in range(10):
        handlers.on_audio(bytes(3840))
    handlers.on_response_done(tools_pending=False)  # no on_audio_done(), as for a cancelled response

    for _ in range(200):
        await asyncio.sleep(0.01)
        if controller.expire() is not None:
            break
    assert controller.is_idle, "the conversation window never closed"
