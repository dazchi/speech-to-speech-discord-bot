"""Wake-word conversation state: who the bot is listening to, and when it goes back to sleep.

Pure logic with an injectable clock, so it can be tested without Discord or a server.

- IDLE: every user's audio goes to the wake-word detector; nothing reaches the server.
- ACTIVE(user): only that user's audio is forwarded. The window stays open while the user
  is speaking, a response is being generated, or its audio is still playing in the channel,
  and closes ``followup_s`` after the reply has finished playing (or ``wake_timeout_s`` after
  the wake word if the user never speaks).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

# After the user stops talking, wait at least this long for the server to start a response.
AWAIT_RESPONSE_S = 5.0


@dataclass
class ActiveSpeaker:
    user_id: int
    name: str


class ConversationController:
    def __init__(
        self,
        *,
        wake_timeout_s: float,
        followup_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.wake_timeout_s = wake_timeout_s
        self.followup_s = followup_s
        self._clock = clock
        self.active: ActiveSpeaker | None = None
        self._deadline: float | None = None
        self._user_speaking = False
        self._responses_in_flight = 0
        self._tools_pending = False
        self._audio_playing = False
        # From a tool call until the response with the answer is done (searching can take a while).
        self._searching = False

    @property
    def is_idle(self) -> bool:
        return self.active is None

    def is_active_user(self, user_id: int) -> bool:
        return self.active is not None and self.active.user_id == user_id

    @property
    def busy(self) -> bool:
        """True while a turn is in progress: user speaking, response being generated or still playing."""
        return self._user_speaking or self._responses_in_flight > 0 or self._tools_pending or self._audio_playing

    def wake(self, user_id: int, name: str) -> bool:
        """A wake word was heard. Returns True if this starts a new conversation window.

        Only one person is served at a time: wake words from others are ignored until the
        active window closes. The active user saying it again just extends their window.
        """
        if self.active is not None and self.active.user_id != user_id:
            return False
        started = self.active is None
        self.active = ActiveSpeaker(user_id, name)
        if not self.busy:
            self._deadline = self._clock() + self.wake_timeout_s
        return started

    def speech_started(self) -> None:
        self._user_speaking = True
        self._deadline = None

    def speech_stopped(self) -> None:
        self._user_speaking = False
        if self._responses_in_flight == 0 and not self._audio_playing:
            # The server commits the turn and normally starts a response right away.
            self._deadline = self._clock() + max(self.followup_s, AWAIT_RESPONSE_S)

    def response_started(self) -> None:
        self._responses_in_flight += 1
        self._tools_pending = False
        self._deadline = None

    def response_finished(self, *, tools_pending: bool = False) -> None:
        """A response ended. ``tools_pending``: tool results will trigger a follow-up response."""
        self._responses_in_flight = max(0, self._responses_in_flight - 1)
        self._tools_pending = self._tools_pending or tools_pending
        if not tools_pending:
            self._searching = False
        self._restart_followup()

    def tool_call_started(self) -> None:
        self._searching = True

    @property
    def searching(self) -> bool:
        """True from a tool call (e.g. web_search) until the response with the answer is done."""
        return self._searching

    @property
    def assistant_busy(self) -> bool:
        """True while Alex is thinking (generating or waiting on a tool) or its reply is still playing."""
        return self._responses_in_flight > 0 or self._tools_pending or self._audio_playing

    @property
    def playing(self) -> bool:
        return self._audio_playing

    def playback_started(self) -> None:
        """Reply audio started playing in the voice channel."""
        self._audio_playing = True
        self._deadline = None

    def playback_finished(self) -> None:
        """Reply audio finished playing (or was interrupted): the follow-up window starts now."""
        if self._audio_playing:
            self._audio_playing = False
            self._restart_followup()

    def tools_done_without_follow_up(self) -> None:
        self._tools_pending = False
        self._searching = False
        self._restart_followup()

    def _restart_followup(self) -> None:
        if not self.busy and self.active is not None:
            self._deadline = self._clock() + self.followup_s

    def expire(self) -> ActiveSpeaker | None:
        """Close the window if its deadline passed. Returns the speaker that was released."""
        if self.active is None or self.busy or self._deadline is None or self._clock() < self._deadline:
            return None
        released, self.active, self._deadline = self.active, None, None
        return released

    def reset(self) -> None:
        self.active = None
        self._deadline = None
        self._user_speaking = False
        self._responses_in_flight = 0
        self._tools_pending = False
        self._audio_playing = False
        self._searching = False
