"""Audio format conversion between Discord and the speech-to-speech server.

Discord: 48 kHz, 2-channel, signed 16-bit PCM in 20 ms frames (3840 bytes).
Server:  24 kHz, mono, signed 16-bit PCM (declared as ``audio/pcm`` at 24 kHz).
Wake word: 16 kHz mono float32.
"""

from __future__ import annotations

import numpy as np
import soxr

DISCORD_RATE = 48_000
SERVER_RATE = 24_000
WAKE_RATE = 16_000
DISCORD_FRAME_BYTES = 3840  # 20 ms of 48 kHz stereo s16le


def discord_pcm_to_mono(pcm: bytes) -> np.ndarray:
    """48 kHz stereo s16le bytes -> 48 kHz mono float32 in [-1, 1]."""
    stereo = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2).astype(np.float32)
    return stereo.mean(axis=1) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def mono_pcm16_to_discord(pcm: bytes) -> bytes:
    """48 kHz mono s16le -> 48 kHz stereo s16le."""
    mono = np.frombuffer(pcm, dtype=np.int16)
    return np.repeat(mono, 2).tobytes()


class StreamResampler:
    """Stateful mono float32 resampler, so 20 ms chunks join without clicks."""

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self._stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32")

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self._stream.resample_chunk(np.ascontiguousarray(samples, dtype=np.float32))


class InputConverter:
    """Per-user converter from Discord frames to server and wake-word audio."""

    def __init__(self) -> None:
        self._to_server = StreamResampler(DISCORD_RATE, SERVER_RATE)
        self._to_wake = StreamResampler(DISCORD_RATE, WAKE_RATE)

    def convert(self, pcm: bytes) -> tuple[bytes, np.ndarray]:
        mono = discord_pcm_to_mono(pcm)
        return float_to_pcm16(self._to_server(mono)), self._to_wake(mono)


class OutputConverter:
    """Server audio (24 kHz mono s16le) -> Discord audio (48 kHz stereo s16le)."""

    def __init__(self) -> None:
        self._to_discord = StreamResampler(SERVER_RATE, DISCORD_RATE)

    def convert(self, pcm: bytes) -> bytes:
        mono = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return mono_pcm16_to_discord(float_to_pcm16(self._to_discord(mono)))


def silence_server(seconds: float) -> bytes:
    return bytes(int(SERVER_RATE * seconds) * 2)


def chime(rising: bool = True) -> bytes:
    """A short two-note chime as Discord audio (48 kHz stereo s16le)."""
    notes = (660.0, 880.0) if rising else (880.0, 660.0)
    parts = []
    for freq in notes:
        t = np.arange(int(DISCORD_RATE * 0.09)) / DISCORD_RATE
        envelope = np.minimum(1.0, np.minimum(t, t[::-1]) * 60.0)
        parts.append(0.18 * np.sin(2 * np.pi * freq * t) * envelope)
    return mono_pcm16_to_discord(float_to_pcm16(np.concatenate(parts)))
