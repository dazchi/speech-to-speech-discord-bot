"""Optional audio diagnostics (AUDIO_DEBUG=1): per-user stats in the log, and recordings of what the bot hears.

Recordings are raw 16 kHz mono s16le (``.pcm``) so they can be read while the bot is still running, e.g.
``sox -t raw -r 16000 -e signed -b 16 -c 1 user.pcm user.wav``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from discord_bot.audio import WAKE_RATE

logger = logging.getLogger(__name__)

REPORT_EVERY_S = 5.0


@dataclass
class _Stats:
    frames: int = 0
    empty: int = 0
    sum_squares: float = 0.0
    samples: int = 0
    peak: float = 0.0


class AudioDebugger:
    def __init__(self, record_dir: Path) -> None:
        self.record_dir = record_dir
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._stats: dict[int, _Stats] = {}
        self._writers: dict[int, BinaryIO] = {}
        self._last_report = time.monotonic()
        logger.info("AUDIO_DEBUG on: recording what the bot hears to %s", record_dir)

    def empty_frame(self, user_id: int) -> None:
        """A packet arrived without audio (typically a DAVE decryption failure)."""
        self._stats.setdefault(user_id, _Stats()).empty += 1

    def frame(self, user_id: int, name: str, wake_samples: np.ndarray) -> None:
        stats = self._stats.setdefault(user_id, _Stats())
        stats.frames += 1
        stats.sum_squares += float(np.sum(wake_samples.astype(np.float64) ** 2))
        stats.samples += len(wake_samples)
        if len(wake_samples):
            stats.peak = max(stats.peak, float(np.abs(wake_samples).max()))
        writer = self._writers.get(user_id)
        if writer is None:
            safe = "".join(c if c.isalnum() else "_" for c in name) or str(user_id)
            path = self.record_dir / f"{safe}_{user_id}_{WAKE_RATE // 1000}k.pcm"
            writer = self._writers[user_id] = path.open("ab")
            logger.info("Recording %s to %s", name, path)
        writer.write((np.clip(wake_samples, -1, 1) * 32767).astype(np.int16).tobytes())

    def maybe_report(self, names: dict[int, str]) -> None:
        now = time.monotonic()
        if now - self._last_report < REPORT_EVERY_S:
            return
        self._last_report = now
        for user_id, stats in self._stats.items():
            if stats.frames == 0 and stats.empty == 0:
                continue
            rms = math.sqrt(stats.sum_squares / stats.samples) if stats.samples else 0.0
            level = f"{20 * math.log10(rms):.0f} dBFS" if rms > 0 else "silent"
            peak = f"{20 * math.log10(stats.peak):.0f} dBFS" if stats.peak > 0 else "silent"
            logger.info(
                "Heard %s: %d audio frames, %d empty/undecryptable frames, level %s (peak %s) in the last %.0fs",
                names.get(user_id, user_id),
                stats.frames,
                stats.empty,
                level,
                peak,
                REPORT_EVERY_S,
            )
            self._stats[user_id] = _Stats()
        for writer in self._writers.values():
            writer.flush()

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
