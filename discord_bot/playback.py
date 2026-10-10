"""Streams assistant audio into the Discord voice channel."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import discord

from discord_bot.audio import DISCORD_FRAME_BYTES

# Wait for this much audio before starting, so small network gaps do not cause stutter.
START_BUFFER_BYTES = DISCORD_FRAME_BYTES * 5  # 100 ms


class StreamingSource(discord.AudioSource):
    """Audio source fed from another thread. Plays silence on underrun until :meth:`finish`."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._finished = False
        self._started = False
        self.closed = False

    def feed(self, pcm: bytes) -> None:
        """Queue audio. New audio reopens a source that was finished but has not ended yet."""
        with self._lock:
            self._buffer.extend(pcm)
            self._finished = False

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._finished = True

    def read(self) -> bytes:
        with self._lock:
            if not self._started and len(self._buffer) < START_BUFFER_BYTES and not self._finished:
                return bytes(DISCORD_FRAME_BYTES)
            self._started = True
            if len(self._buffer) >= DISCORD_FRAME_BYTES:
                frame = bytes(self._buffer[:DISCORD_FRAME_BYTES])
                del self._buffer[:DISCORD_FRAME_BYTES]
                return frame
            if self._finished:
                if self._buffer:
                    frame = bytes(self._buffer).ljust(DISCORD_FRAME_BYTES, b"\0")
                    self._buffer.clear()
                    return frame
                self.closed = True
                return b""
            return bytes(DISCORD_FRAME_BYTES)

    def is_opus(self) -> bool:
        return False


def _stop_playback(vc: discord.VoiceClient) -> None:
    """Stop playing only. ``VoiceRecvClient.stop()`` would also stop *receiving* audio."""
    stop_playing = getattr(vc, "stop_playing", None)
    if callable(stop_playing):
        stop_playing()
    else:
        vc.stop()


class Speaker:
    """Owns the current :class:`StreamingSource` and restarts playback when needed.

    ``on_stopped`` runs on the event loop when the current stream has really stopped playing
    (drained after :meth:`finish`, or cut off by :meth:`stop`).
    """

    def __init__(
        self,
        voice_client_getter: Callable[[], discord.VoiceClient | None],
        loop: asyncio.AbstractEventLoop | None = None,
        on_stopped: Callable[[], None] | None = None,
    ) -> None:
        self._get_vc = voice_client_getter
        self.loop = loop
        self._on_stopped = on_stopped
        self._source: StreamingSource | None = None

    def _ensure_source(self) -> StreamingSource | None:
        vc = self._get_vc()
        if vc is None or not vc.is_connected():
            return None
        if self._source is None or self._source.closed or not vc.is_playing():
            if vc.is_playing():
                _stop_playback(vc)
            source = self._source = StreamingSource()
            vc.play(source, after=lambda _error: self._player_ended(source))
        return self._source

    def _player_ended(self, source: StreamingSource) -> None:
        """Runs on Discord's player thread."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._stream_stopped, source)

    def _stream_stopped(self, source: StreamingSource) -> None:
        # A stream that was already replaced by a newer one does not count.
        if source is self._source or self._source is None:
            if source is self._source:
                self._source = None
            if self._on_stopped is not None:
                self._on_stopped()

    def play(self, pcm: bytes) -> bool:
        """Queue audio; returns False if the bot is not connected to a voice channel."""
        source = self._ensure_source()
        if source is None:
            return False
        source.feed(pcm)
        return True

    @property
    def is_playing(self) -> bool:
        """True while a stream is queued or playing in the voice channel."""
        return self._source is not None and not self._source.closed

    def finish(self) -> None:
        if self._source is not None:
            self._source.finish()

    def stop(self) -> None:
        """Barge-in: drop whatever is still queued."""
        if self._source is not None:
            self._source.clear()
        vc = self._get_vc()
        if vc is not None and vc.is_playing():
            _stop_playback(vc)  # the player's ``after`` callback reports the stop
        else:
            self._source = None
            if self._on_stopped is not None:
                self._on_stopped()
