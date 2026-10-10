import asyncio

import pytest

pytest.importorskip("discord")

from discord_bot.audio import DISCORD_FRAME_BYTES  # noqa: E402
from discord_bot.playback import Speaker, StreamingSource  # noqa: E402


class FakeVoiceClient:
    def __init__(self) -> None:
        self.source = None
        self.after = None
        self.playing = False

    def is_connected(self):
        return True

    def is_playing(self):
        return self.playing

    def play(self, source, *, after=None):
        self.source, self.after, self.playing = source, after, True

    def stop(self):
        self.end()

    def end(self):
        """Simulate the player thread finishing the current source."""
        if self.playing:
            self.playing = False
            self.after(None)

    def drain(self):
        while self.playing:
            if self.source.read() == b"":
                self.end()


def test_source_buffers_then_plays_then_ends():
    source = StreamingSource()
    source.feed(bytes(DISCORD_FRAME_BYTES * 6))
    source.finish()
    frames = []
    while (frame := source.read()) != b"":
        frames.append(frame)
    assert len(frames) == 6 and source.closed


async def test_speaker_reports_when_playback_really_ends():
    vc, stopped = FakeVoiceClient(), []
    speaker = Speaker(lambda: vc, asyncio.get_running_loop(), lambda: stopped.append(True))
    assert speaker.play(bytes(DISCORD_FRAME_BYTES * 10))
    speaker.finish()
    await asyncio.sleep(0)
    assert stopped == []  # finish() only marks the end of the stream
    vc.drain()
    await asyncio.sleep(0)
    assert stopped == [True]


async def test_stale_stream_end_is_ignored():
    vc, stopped = FakeVoiceClient(), []
    speaker = Speaker(lambda: vc, asyncio.get_running_loop(), lambda: stopped.append(True))
    speaker.play(bytes(DISCORD_FRAME_BYTES))
    old_after = vc.after
    vc.playing = False  # old stream ended, its callback not delivered yet
    speaker.play(bytes(DISCORD_FRAME_BYTES))  # a new stream starts
    old_after(None)
    await asyncio.sleep(0)
    assert stopped == []


async def test_barge_in_stop_reports_end():
    vc, stopped = FakeVoiceClient(), []
    speaker = Speaker(lambda: vc, asyncio.get_running_loop(), lambda: stopped.append(True))
    speaker.play(bytes(DISCORD_FRAME_BYTES * 50))
    speaker.stop()
    await asyncio.sleep(0)
    assert stopped == [True]


def test_play_without_voice_connection_returns_false():
    assert Speaker(lambda: None).play(b"\0" * 4) is False


class RecvVoiceClient(FakeVoiceClient):
    """Like discord-ext-voice-recv's client: stop() also stops listening."""

    def __init__(self) -> None:
        super().__init__()
        self.listening = True

    def stop(self):
        self.listening = False
        super().stop()

    def stop_playing(self):
        FakeVoiceClient.stop(self)


async def test_barge_in_stops_playback_but_keeps_listening():
    vc, stopped = RecvVoiceClient(), []
    speaker = Speaker(lambda: vc, asyncio.get_running_loop(), lambda: stopped.append(True))
    speaker.play(bytes(DISCORD_FRAME_BYTES * 50))
    speaker.stop()
    await asyncio.sleep(0)
    assert stopped == [True]
    assert vc.listening, "interrupting Alex must not stop receiving audio"
