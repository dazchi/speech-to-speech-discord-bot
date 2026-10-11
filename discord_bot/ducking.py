"""Turn the music down while Alex speaks, and back up once it is done.

The music is lowered when Alex's reply audio starts, and stays down while Alex is still busy with the
turn (thinking, waiting on a tool, or speaking), so it does not bounce up between the sentence before
a web search and the answer. It comes back up ``release_delay_s`` after Alex is done.

The music player restores the volume by itself after ``hold_s`` unless renewed, so the music does not
stay quiet if the bot stops or loses the connection mid-reply.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class MusicDucker:
    def __init__(
        self,
        send: Callable[[bool], Awaitable[None]],
        busy: Callable[[], bool],
        *,
        release_delay_s: float = 1.0,
        hold_s: float = 30.0,
        poll_s: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._send = send  # True: lower the music (for hold_s), False: back to normal
        self._busy = busy
        self.release_delay_s = release_delay_s
        self.hold_s = hold_s
        self.poll_s = poll_s
        self._clock = clock
        self._task: asyncio.Task[None] | None = None
        self._failed = False

    @property
    def ducked(self) -> bool:
        return self._task is not None and not self._task.done()

    def speaking(self) -> None:
        """Alex's reply audio is playing: lower the music if it is not already down."""
        if not self.ducked:
            self._task = asyncio.get_running_loop().create_task(self._hold())

    async def _hold(self) -> None:
        if not await self._call(True):
            return
        renewed = self._clock()
        quiet_since: float | None = None
        try:
            while True:
                await asyncio.sleep(self.poll_s)
                now = self._clock()
                if self._busy():
                    quiet_since = None
                elif quiet_since is None:
                    quiet_since = now
                elif now - quiet_since >= self.release_delay_s:
                    break
                if now - renewed >= self.hold_s / 2:
                    await self._call(True)
                    renewed = now
        finally:
            await asyncio.shield(self._call(False))

    async def _call(self, down: bool) -> bool:
        try:
            await self._send(down)
        except Exception as exc:  # the music bot is optional: never disturb the conversation
            if not self._failed:
                logger.warning("Could not %s the music: %s", "lower" if down else "restore", exc)
            self._failed = True
            return False
        if self._failed:
            logger.info("Music ducking works again")
        self._failed = False
        return True

    async def close(self) -> None:
        """Bring the music back up now (on shutdown or leaving the voice channel)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
