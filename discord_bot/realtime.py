"""One persistent Realtime session with the speech-to-speech server, shared by every Discord user."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import websockets

from discord_bot.audio import SERVER_RATE
from discord_bot.tools import ToolOutcome, ToolRegistry

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 5.0
# Instructions for the last response of a question once tool calls are over (tools disabled).
FINAL_ANSWER_NOTE = (
    "Your tool calls are finished. Answer now in one to three short spoken sentences using the results above. "
    "If they do not answer the question, say briefly that you could not find it. Do not call any tools and do "
    "not write tool-call syntax."
)
# Asked once more when the answer after tool calls said nothing (e.g. the model wrote another tool call
# while tools were off, which the server removes), or only the search sentence again.
EMPTY_ANSWER_NOTE = (
    "Your previous reply did not answer: it only tried to call a tool or repeated the search sentence. Tools "
    "are off now. Answer in plain spoken words, from the results above, even if they are incomplete, and do not "
    "say the search sentence."
)
# Asked once when a reply only said the search sentence and ended without calling the tool.
SEARCH_RETRY_NOTE = (
    "Your previous reply said you would look it up, but it called no tool. Call the tool the request needs now, "
    "in the tool-call format, without saying anything."
)
# Tools the search sentence is said before (not tools that act, such as music controls).
SEARCH_TOOLS = frozenset({"web_search"})
# Where the server puts its own timing of a turn in response.done (speech_to_speech.pipeline.turn_latency).
SERVER_LATENCY_KEY = "speech_to_speech.turn_latency"
# Longest wait for the search sentence to be spoken before the tool results are handed over.
PREAMBLE_WAIT_S = 6.0
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
    names: set[str] = field(default_factory=set)
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
        clock: Callable[[], float] = time.monotonic,
        max_tool_rounds: int = 2,
        allow_interruptions: bool = True,
        answer_instructions: str | None = None,
        search_preambles: tuple[str, ...] = (),
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
        self._follow_ups: set[str] = set()  # responses that answer tool results
        self._said: dict[str, str] = {}  # what each response has said so far
        self._empty_retry_used = False
        # Sentences said right before a search: a reply that is only one of them promised a search.
        self._preamble_sentences = tuple(p for p in search_preambles if _bare(p))
        self._preambles = {_bare(p) for p in self._preamble_sentences}
        self._search_retry_used = False
        self._search_requested = False  # the awaited response is the retry asking for the search
        # The search sentence for the current question: said (by the model or the bot) before its first tool
        # round. When the bot said it, the model is told so, so that the answer does not repeat it.
        self.preamble: str | None = None  # this question's search sentence, if the bot picked one
        self._preamble_done = False
        self._preamble_said_by_bot: str | None = None
        self._preamble_expected = False  # the next response is the bot's search sentence
        self._preamble_response: str | None = None
        self._preamble_finished: asyncio.Event | None = None
        self._preamble_status: str | None = None
        # Response timing for the logs: from the end of the user's speech (or the tool results being sent).
        self._clock = clock
        self._input_done_at: float | None = None
        self._timings: dict[str, _Timing] = {}

    @property
    def connected(self) -> bool:
        return self._ws is not None

    @property
    def speaker(self) -> str | None:
        """Who Alex is talking to now."""
        return self._speaker

    def instructions_for(self, speaker: str | None, base: str | None = None, *, with_note: bool = True) -> str:
        # The model does not know the date; it needs it for "today", and for search queries.
        today = datetime.now().astimezone().strftime("%Y-%m-%d (%A)")
        text = f"{base or self.base_instructions}\n\nToday's date: {today}."
        if speaker:
            text += f"\nCurrent speaker: {speaker}."
        if self._turn_note and with_note:
            text += f"\n{self._turn_note}"
        if self._preamble_said_by_bot:
            text += (
                f'\nYou already said "{self._preamble_said_by_bot}" for this question: do not say it again, '
                "just answer."
            )
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
        if self._new_question():
            await self._send_instructions()
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
            if self._new_question():
                await self._send_instructions()
            self.events.on_speech_started()
        elif kind == "input_audio_buffer.speech_stopped":
            self._input_done_at = self._clock()
            self.events.on_speech_stopped()
        elif kind == "response.created":
            if self._preamble_expected:
                self._preamble_expected = False
                self._preamble_response = (event.get("response") or {}).get("id") or ""
            elif self._awaiting_follow_up:
                self._awaiting_follow_up = False
                if self._search_requested:
                    # Asked to search, not to answer tool results: no empty-answer retry for it.
                    self._search_requested = False
                else:
                    self._follow_ups.add((event.get("response") or {}).get("id") or "")
                    logger.info("Follow-up response to the tool result started")
            self._response_started((event.get("response") or {}).get("id") or "")
            self.events.on_response_created()
        elif kind in ("response.output_audio_transcript.delta", "response.output_text.delta"):
            response_id = event.get("response_id") or ""
            self._said[response_id] = self._said.get(response_id, "") + str(event.get("delta", ""))
        elif kind == "response.output_audio.delta":
            self._first_audio(event.get("response_id") or "")
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
            response = event.get("response") or {}
            self._response_finished(response)
            if self._preamble_response is not None and response.get("id") == self._preamble_response:
                self._end_preamble(response)
            else:
                await self._finish_response(response)
        elif kind == "error":
            if self._preamble_expected and self._preamble_finished is not None:
                self._preamble_status = "refused"  # the server would not say it: hand over the results now
                self._preamble_finished.set()
            logger.warning("Speech server error: %s", (event.get("error") or {}).get("message"))

    def _response_started(self, response_id: str) -> None:
        if response_id == self._preamble_response:
            kind = "search sentence"
        elif response_id in self._follow_ups:
            kind = "answer to the tool results"
        else:
            kind = "reply"
        now = self._clock()
        self._timings[response_id] = _Timing(kind, now)
        waited = f" {now - self._input_done_at:.2f}s after the input ended" if self._input_done_at else ""
        logger.info("LLM started the %s (%s)%s", kind, response_id, waited)

    def _first_audio(self, response_id: str) -> None:
        timing = self._timings.get(response_id)
        if timing is None or timing.first_audio_at is not None:
            return
        timing.first_audio_at = self._clock()
        since_input = (
            f", {timing.first_audio_at - self._input_done_at:.2f}s after the input ended"
            if self._input_done_at and self._input_done_at <= timing.started_at
            else ""
        )
        logger.info(
            "First audio of the %s after %.2fs%s", timing.kind, timing.first_audio_at - timing.started_at, since_input
        )

    def _response_finished(self, response: dict[str, Any]) -> None:
        timing = self._timings.pop(response.get("id") or "", None)
        if timing is None:
            return
        server = _server_latency(response)
        logger.info(
            "LLM finished the %s (%s) in %.2fs%s",
            timing.kind,
            response.get("status") or "unknown",
            self._clock() - timing.started_at,
            f"; server: {server}" if server else "",
        )

    def _new_question(self) -> bool:
        """Forget the previous question. Returns True if the instructions must be sent again."""
        self._tool_rounds = 0
        self._searched.clear()
        self._force_answer = False
        self._empty_retry_used = False
        self._search_retry_used = False
        self._preamble_done = False
        stale_note, self._preamble_said_by_bot = self._preamble_said_by_bot, None
        return stale_note is not None

    def _start_call(self, event: dict[str, Any]) -> None:
        response_id = event.get("response_id") or ""
        call_id = event.get("call_id") or ""
        name = event.get("name") or ""
        arguments = event.get("arguments") or "{}"
        pending = self._calls.setdefault(response_id, _PendingCalls())
        pending.order.append(call_id)
        pending.names.add(name)
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
        response_id = response.get("id") or ""
        pending = self._calls.pop(response_id, None)
        follow_up, said = response_id in self._follow_ups, self._said.pop(response_id, "")
        # An answer to tool results that is only the search sentence again says nothing either.
        spoken = bool(said.strip()) and not (follow_up and _bare(said) in self._preambles)
        self._follow_ups.discard(response_id)
        if pending is None:
            if not follow_up and self._only_promised_a_search(said) and response.get("status") == "completed":
                # The model said the search sentence, then ended without a tool call (or wrote one the
                # server could not parse): ask once for the search, so the question is not left hanging.
                self._search_retry_used = True
                self.events.on_response_done(tools_pending=True)
                logger.warning("The reply promised a search but called no tool; asking for the search")
                await self._request_search()
                return
            if follow_up and not spoken and response.get("status") == "completed":
                if not self._empty_retry_used:
                    # Answering tool results produced nothing to say: ask once more, so the user is not
                    # left with silence and a conversation window that closes.
                    self._empty_retry_used = True
                    self.events.on_response_done(tools_pending=True)
                    logger.warning("The answer to the tool results was empty; asking for it again")
                    await self._request_final_answer(EMPTY_ANSWER_NOTE)
                    return
                logger.warning("The answer to the tool results was empty again; giving up on this question")
            self.events.on_response_done(tools_pending=False)
            return
        # Results are delivered even when the response was cut off (e.g. the user talked over it),
        # so the server is never left waiting for an output; only completed responses get a follow-up.
        completed = response.get("status") == "completed"
        self.events.on_response_done(tools_pending=completed)
        searching = bool(pending.names & SEARCH_TOOLS)
        if completed and searching and not spoken and not self._preamble_done and self._preamble_sentences:
            # The model went straight to the search without the search sentence: say it while it runs.
            asyncio.create_task(self._say_preamble_then_deliver(pending))
            return
        self._preamble_done = self._preamble_done or (completed and searching)
        asyncio.create_task(self._deliver(pending, follow_up=completed))

    async def _say_preamble_then_deliver(self, pending: _PendingCalls) -> None:
        self._preamble_done = True
        sentence = self.preamble or random.choice(self._preamble_sentences)
        finished = self._preamble_finished = asyncio.Event()
        self._preamble_expected = True
        self._preamble_status = None
        try:
            await self._send(
                {
                    "type": "response.create",
                    "response": {
                        "conversation": "none",
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": "[Say the sentence]"}],
                            }
                        ],
                        "instructions": f'Say exactly this sentence and nothing else: "{sentence}"',
                        "tool_choice": "none",
                    },
                }
            )
            logger.info("The reply called a tool without the search sentence; saying %r", sentence)
            await asyncio.wait_for(finished.wait(), PREAMBLE_WAIT_S)
        except TimeoutError:
            logger.warning("The search sentence was not spoken in time; handing over the tool results")
        except (OSError, websockets.WebSocketException) as exc:
            logger.warning("Could not ask for the search sentence: %s", exc)
        finally:
            self._preamble_expected = False
            self._preamble_response = None
            self._preamble_finished = None
        if self._preamble_status not in (None, "completed", "refused"):
            # Someone talked over the sentence: hand over the results, but do not answer the old question.
            await self._deliver(pending, follow_up=False)
            return
        if self._preamble_status != "refused":
            self._preamble_said_by_bot = sentence
            try:
                await self._send_instructions()
            except (OSError, websockets.WebSocketException):
                pass
        await self._deliver(pending, follow_up=True)

    def _end_preamble(self, response: dict[str, Any]) -> None:
        self._said.pop(response.get("id") or "", None)
        self._preamble_status = response.get("status")
        # When it was spoken, the tool results are still to come.
        self.events.on_response_done(tools_pending=self._preamble_status == "completed")
        if self._preamble_finished is not None:
            self._preamble_finished.set()

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
                self._input_done_at = self._clock()
                create_response = create_response or outcome.create_response
            if follow_up and create_response:
                self._awaiting_follow_up = True
                self._tool_rounds += 1
                if self._tool_rounds >= self.max_tool_rounds or self._force_answer:
                    # Stop search loops: the follow-up must answer with what it has.
                    await self._request_final_answer()
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

    def _only_promised_a_search(self, said: str) -> bool:
        return not self._search_retry_used and bool(said.strip()) and _bare(said) in self._preambles

    async def _request_search(self) -> None:
        instructions = self.instructions_for(self._speaker, f"{self.base_instructions}\n\n{SEARCH_RETRY_NOTE}")
        self._awaiting_follow_up = True
        self._search_requested = True
        await self._send({"type": "response.create", "response": {"instructions": instructions}})
        asyncio.create_task(self._follow_up_watchdog())

    async def _request_final_answer(self, extra_note: str = "") -> None:
        """Ask for the answer with tools off, under instructions without the search guide (so the model is
        not cued to search or say the search sentence)."""
        note = f"{FINAL_ANSWER_NOTE}\n{extra_note}" if extra_note else FINAL_ANSWER_NOTE
        instructions = self.instructions_for(self._speaker, f"{self.answer_instructions}\n\n{note}", with_note=False)
        self._awaiting_follow_up = True
        await self._send({"type": "response.create", "response": {"tool_choice": "none", "instructions": instructions}})
        asyncio.create_task(self._follow_up_watchdog())

    async def _follow_up_watchdog(self) -> None:
        await asyncio.sleep(FOLLOW_UP_TIMEOUT_S)
        if self._awaiting_follow_up:
            self._awaiting_follow_up = False
            self._search_requested = False
            logger.warning("The follow-up response never started; no longer waiting for it")
            self.events.on_tools_done_without_follow_up()

    def _cancel_all_calls(self) -> None:
        for pending in self._calls.values():
            for task in pending.tasks.values():
                task.cancel()
        self._calls.clear()
        self._awaiting_follow_up = False
        self._search_requested = False
        self._follow_ups.clear()
        self._said.clear()
        self._timings.clear()
        self._preamble_expected = False
        self._preamble_response = None


@dataclass
class _Timing:
    kind: str
    started_at: float
    first_audio_at: float | None = None


def _server_latency(response: dict[str, Any]) -> str:
    """The server's own timing of the turn (STT, LLM, first TTS audio, end to end), if it sent one."""
    raw = (response.get("metadata") or {}).get(SERVER_LATENCY_KEY)
    try:
        values = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        return ""
    names = (("stt_s", "stt"), ("llm_s", "llm"), ("tts_ttfa_s", "tts_first_audio"), ("e2e_s", "e2e"))
    parts = [f"{label}={values[key]:.2f}s" for key, label in names if isinstance(values.get(key), (int, float))]
    return " ".join(parts)


def _bare(text: str) -> str:
    """Text without spaces or punctuation, to compare what was said with a configured sentence."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


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
