"""One persistent Realtime session with the speech-to-speech server, shared by every Discord user."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import websockets

from discord_bot.audio import SERVER_RATE
from discord_bot.tools import ToolOutcome, ToolRegistry

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 5.0
# Instructions for the last response of a question once searching is over (tools disabled).
FINAL_ANSWER_NOTE = (
    "The web search is finished. Answer now in one to three short spoken sentences using the search results "
    "above. If they do not answer the question, say briefly that you could not find it. Do not call any tools "
    "and do not write tool-call syntax."
)
# If the follow-up response to a tool result has not started by then, stop waiting for it.
FOLLOW_UP_TIMEOUT_S = 15.0


class SessionEvents(Protocol):
    def on_connected(self) -> None: ...
    def on_speech_started(self) -> None: ...
    def on_speech_stopped(self) -> None: ...
    def on_response_created(self) -> None: ...
    def on_response_done(self, *, tools_pending: bool) -> None: ...
    def on_tool_call(self, name: str) -> None: ...
    def on_tools_done_without_follow_up(self) -> None: ...
    def on_audio(self, pcm: bytes) -> None: ...
    def on_audio_done(self) -> None: ...
    def on_user_transcript(self, text: str) -> None: ...
    def on_assistant_transcript(self, text: str) -> None: ...


@dataclass
class _PendingCalls:
    order: list[str] = field(default_factory=list)
    tasks: dict[str, asyncio.Task[ToolOutcome]] = field(default_factory=dict)


def speaker_label(name: str) -> str:
    return f"[{name} is speaking]"


def build_session_update(
    instructions: str, voice: str | None, tools: list[dict[str, Any]], *, allow_interruptions: bool = True
) -> dict[str, Any]:
    pcm = {"type": "audio/pcm", "rate": SERVER_RATE}
    vad = {"type": "server_vad", "interrupt_response": allow_interruptions}
    session: dict[str, Any] = {
        "type": "realtime",
        "instructions": instructions,
        "audio": {
            "input": {"format": pcm, "turn_detection": vad},
            "output": {"format": pcm, **({"voice": voice} if voice else {})},
        },
    }
    if tools:
        session["tools"] = tools
        session["tool_choice"] = "auto"
    return {"type": "session.update", "session": session}


class RealtimeSession:
    def __init__(
        self,
        url: str,
        instructions: str,
        voice: str | None,
        tools: ToolRegistry,
        events: SessionEvents,
        connect: Callable[[str], Awaitable[Any]] | None = None,
        max_tool_rounds: int = 2,
        allow_interruptions: bool = True,
        answer_instructions: str | None = None,
    ) -> None:
        self.url = url if "?" in url else f"{url}?model=local"
        self.base_instructions = instructions
        self.voice = voice
        self.tools = tools
        self.events = events
        self._connect = connect or (lambda u: websockets.connect(u, max_size=None))
        self._ws: Any = None
        self._speaker: str | None = None
        self._turn_note: str | None = None
        self._calls: dict[str, _PendingCalls] = {}
        self._closed = False
        self._awaiting_follow_up = False
        # Tool round trips since the user last spoke; past ``max_tool_rounds`` the model must answer.
        self.max_tool_rounds = max_tool_rounds
        # Base instructions for final answers: without the search guide, so the model is not cued to search.
        self.answer_instructions = answer_instructions or instructions
        self._searched: set[str] = set()  # tool calls already made for the current question
        self._force_answer = False
        self.allow_interruptions = allow_interruptions
        self._tool_rounds = 0

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def instructions_for(self, speaker: str | None, base: str | None = None, *, with_note: bool = True) -> str:
        # The model does not know the date; it needs it for "today", and for search queries.
        today = datetime.now().astimezone().strftime("%Y-%m-%d (%A)")
        text = f"{base or self.base_instructions}\n\nToday's date: {today}."
        if speaker:
            text += f"\nCurrent speaker: {speaker}."
        if self._turn_note and with_note:
            text += f"\n{self._turn_note}"
        return text

    async def run(self) -> None:
        """Connect and keep the session alive until :meth:`close`."""
        while not self._closed:
            try:
                async with await self._connect(self.url) as ws:
                    first = json.loads(await ws.recv())
                    if first.get("type") == "error":
                        message = (first.get("error") or {}).get("message", "unknown error")
                        logger.warning("Speech server refused the session: %s", message)
                    else:
                        self._ws = ws
                        await self._send(
                            build_session_update(
                                self.instructions_for(self._speaker),
                                self.voice,
                                self.tools.definitions,
                                allow_interruptions=self.allow_interruptions,
                            )
                        )
                        logger.info("Connected to speech server at %s", self.url)
                        self.events.on_connected()
                        async for raw in ws:
                            await self._dispatch(json.loads(raw))
            except (OSError, websockets.WebSocketException) as exc:
                logger.warning("Speech server connection lost: %s", exc)
            finally:
                self._ws = None
                self._cancel_all_calls()
            if not self._closed:
                await asyncio.sleep(RECONNECT_DELAY_S)

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            await self._ws.close()

    async def _send(self, event: dict[str, Any]) -> None:
        if self._ws is not None:
            await self._ws.send(json.dumps(event, ensure_ascii=False))

    async def send_audio(self, pcm: bytes) -> None:
        if pcm:
            await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

    async def greet(self, name: str, instruction: str) -> None:
        """Speak a greeting without touching the shared history (an out-of-band response)."""
        base = self.instructions_for(None, self.answer_instructions, with_note=False)
        await self._send(
            {
                "type": "response.create",
                "response": {
                    "conversation": "none",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": f"[{name} joined the voice channel]"}],
                        }
                    ],
                    "instructions": f"{base}\n\n{instruction}",
                    "tool_choice": "none",
                },
            }
        )

    async def ask_text(self, text: str) -> None:
        """A typed question: add it to the shared history and ask for a (spoken) response."""
        self._new_question()
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
            }
        )
        await self._send({"type": "response.create"})

    async def set_speaker(self, name: str | None) -> None:
        if name == self._speaker:
            return
        self._speaker = name
        await self._send_instructions()

    async def set_turn_note(self, note: str | None) -> None:
        """A note for the next reply only (e.g. whether to use a catchphrase)."""
        if note == self._turn_note:
            return
        self._turn_note = note
        await self._send_instructions()

    async def _send_instructions(self) -> None:
        await self._send(
            {
                "type": "session.update",
                "session": {"type": "realtime", "instructions": self.instructions_for(self._speaker)},
            }
        )

    async def announce_speaker(self, name: str) -> None:
        """Add "[name is speaking]" to the shared history, right before that person's spoken turn.

        A user-role text item: it does not trigger a response, and unlike a system item it
        does not replace the session instructions.
        """
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": speaker_label(name)}],
                },
            }
        )

    async def _dispatch(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "input_audio_buffer.speech_started":
            self._new_question()
            self.events.on_speech_started()
        elif kind == "input_audio_buffer.speech_stopped":
            self.events.on_speech_stopped()
        elif kind == "response.created":
            if self._awaiting_follow_up:
                self._awaiting_follow_up = False
                logger.info("Follow-up response to the tool result started")
            self.events.on_response_created()
        elif kind == "response.output_audio.delta":
            self.events.on_audio(base64.b64decode(event.get("delta", "")))
        elif kind == "response.output_audio.done":
            self.events.on_audio_done()
        elif kind == "conversation.item.input_audio_transcription.completed":
            self.events.on_user_transcript(event.get("transcript", ""))
        elif kind == "response.output_audio_transcript.done":
            self.events.on_assistant_transcript(event.get("transcript", ""))
        elif kind == "response.function_call_arguments.done":
            self._start_call(event)
        elif kind == "response.done":
            await self._finish_response(event.get("response") or {})
        elif kind == "error":
            logger.warning("Speech server error: %s", (event.get("error") or {}).get("message"))

    def _new_question(self) -> None:
        self._tool_rounds = 0
        self._searched.clear()
        self._force_answer = False

    def _start_call(self, event: dict[str, Any]) -> None:
        response_id = event.get("response_id") or ""
        call_id = event.get("call_id") or ""
        name = event.get("name") or ""
        arguments = event.get("arguments") or "{}"
        pending = self._calls.setdefault(response_id, _PendingCalls())
        pending.order.append(call_id)
        self.events.on_tool_call(name)
        key = f"{name}:{_normalized_arguments(arguments)}"
        if key in self._searched:
            # The model repeated a search it already has results for: do not run it again,
            # and make the next response answer instead of looping.
            logger.info("Repeated tool call %s(%s); not running it again", name, arguments)
            self._force_answer = True
            pending.tasks[call_id] = asyncio.create_task(_already_done(name))
            return
        self._searched.add(key)
        logger.info("Tool call %s(%s)", name, arguments)
        pending.tasks[call_id] = asyncio.create_task(self.tools.run(name, arguments))

    async def _finish_response(self, response: dict[str, Any]) -> None:
        pending = self._calls.pop(response.get("id") or "", None)
        if pending is None:
            self.events.on_response_done(tools_pending=False)
            return
        # Results are delivered even when the response was cut off (e.g. the user talked over it),
        # so the server is never left waiting for an output; only completed responses get a follow-up.
        completed = response.get("status") == "completed"
        self.events.on_response_done(tools_pending=completed)
        asyncio.create_task(self._deliver(pending, follow_up=completed))

    async def _deliver(self, pending: _PendingCalls, *, follow_up: bool) -> None:
        create_response = False
        try:
            for call_id in pending.order:
                outcome = await pending.tasks[call_id]
                await self._send(
                    {
                        "type": "conversation.item.create",
                        "item": {"type": "function_call_output", "call_id": call_id, "output": outcome.output},
                    }
                )
                logger.info("Sent tool result for %s (%d chars)", call_id, len(outcome.output))
                create_response = create_response or outcome.create_response
            if follow_up and create_response:
                self._awaiting_follow_up = True
                self._tool_rounds += 1
                if self._tool_rounds >= self.max_tool_rounds or self._force_answer:
                    # Stop search loops: the follow-up must answer with what it has, under instructions
                    # without the search guide (so it is not cued to search or say the search sentence).
                    instructions = self.instructions_for(
                        self._speaker, f"{self.answer_instructions}\n\n{FINAL_ANSWER_NOTE}", with_note=False
                    )
                    await self._send(
                        {"type": "response.create", "response": {"tool_choice": "none", "instructions": instructions}}
                    )
                    logger.info("Searching is over (%d rounds); requested the final answer", self._tool_rounds)
                else:
                    await self._send({"type": "response.create"})
                    logger.info("Requested the follow-up response")
                asyncio.create_task(self._follow_up_watchdog())
                return
        except asyncio.CancelledError:
            pass
        except (OSError, websockets.WebSocketException) as exc:
            logger.warning("Could not deliver tool results: %s", exc)
        if follow_up:
            self.events.on_tools_done_without_follow_up()

    async def _follow_up_watchdog(self) -> None:
        await asyncio.sleep(FOLLOW_UP_TIMEOUT_S)
        if self._awaiting_follow_up:
            self._awaiting_follow_up = False
            logger.warning("The follow-up response never started; no longer waiting for it")
            self.events.on_tools_done_without_follow_up()

    def _cancel_all_calls(self) -> None:
        for pending in self._calls.values():
            for task in pending.tasks.values():
                task.cancel()
        self._calls.clear()
        self._awaiting_follow_up = False


def _normalized_arguments(raw: str) -> str:
    try:
        value = json.loads(raw)
    except ValueError:
        return " ".join(raw.lower().split())
    if isinstance(value, dict):
        value = {k: " ".join(v.lower().split()) if isinstance(v, str) else v for k, v in value.items()}
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


async def _already_done(name: str) -> ToolOutcome:
    return ToolOutcome(
        json.dumps(
            {"note": f"You already called {name} with these arguments; the results are above. Answer from them."}
        )
    )
