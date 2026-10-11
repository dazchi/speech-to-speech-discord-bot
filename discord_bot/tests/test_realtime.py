import asyncio
import base64
import json

from discord_bot.realtime import RealtimeSession, build_session_update
from discord_bot.tools import ToolOutcome


class Events:
    def __init__(self) -> None:
        self.log: list[tuple] = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.log.append((name, args, kwargs))


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


class Tools:
    definitions = [{"type": "function", "name": "web_search"}]

    async def run(self, name, raw_arguments):
        await asyncio.sleep(0)
        return ToolOutcome(json.dumps({"tool": name, "args": json.loads(raw_arguments)}))


def make_session() -> tuple[RealtimeSession, FakeWS, Events]:
    events = Events()
    session = RealtimeSession("ws://x/v1/realtime", "Be Alex.", None, Tools(), events)
    ws = FakeWS()
    session._ws = ws
    return session, ws, events


def test_session_update_declares_24k_pcm_vad_and_tools():
    update = build_session_update("Be Alex.", "aiden", [{"name": "t"}])["session"]
    assert update["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert update["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert update["audio"]["output"] == {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "aiden"}
    assert update["tool_choice"] == "auto" and update["instructions"] == "Be Alex."


def test_url_gets_model_query():
    assert make_session()[0].url == "ws://x/v1/realtime?model=local"


async def test_set_speaker_updates_instructions_once():
    session, ws, _ = make_session()
    await session.set_speaker("Alice")
    await session.set_speaker("Alice")
    assert len(ws.sent) == 1
    assert ws.sent[0]["session"]["instructions"].endswith("Current speaker: Alice.")


async def test_audio_and_transcript_events_are_forwarded():
    session, _, events = make_session()
    await session._dispatch({"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x01\x02").decode()})
    await session._dispatch({"type": "conversation.item.input_audio_transcription.completed", "transcript": "hi"})
    assert ("on_audio", (b"\x01\x02",), {}) in events.log
    assert ("on_user_transcript", ("hi",), {}) in events.log


async def test_tool_calls_are_answered_in_order_then_follow_up_is_requested():
    session, ws, events = make_session()
    for call_id, query in (("c1", "a"), ("c2", "b")):
        await session._dispatch(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "r1",
                "call_id": call_id,
                "name": "web_search",
                "arguments": json.dumps({"query": query}),
            }
        )
    await session._dispatch({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
    await asyncio.sleep(0.05)

    assert ("on_response_done", (), {"tools_pending": True}) in events.log
    outputs = [e["item"]["call_id"] for e in ws.sent if e["type"] == "conversation.item.create"]
    assert outputs == ["c1", "c2"]
    assert ws.sent[-1] == {"type": "response.create"}


async def test_cancelled_response_still_delivers_results_without_follow_up():
    session, ws, events = make_session()
    await session._dispatch(
        {
            "type": "response.function_call_arguments.done",
            "response_id": "r1",
            "call_id": "c1",
            "name": "web_search",
            "arguments": "{}",
        }
    )
    await session._dispatch({"type": "response.done", "response": {"id": "r1", "status": "cancelled"}})
    await asyncio.sleep(0.05)
    assert [e["type"] for e in ws.sent] == ["conversation.item.create"]  # result sent, no response.create
    assert ("on_response_done", (), {"tools_pending": False}) in events.log


async def test_follow_up_that_never_starts_stops_blocking(monkeypatch):
    import discord_bot.realtime as realtime

    monkeypatch.setattr(realtime, "FOLLOW_UP_TIMEOUT_S", 0.05)
    session, ws, events = make_session()
    await session._dispatch(
        {
            "type": "response.function_call_arguments.done",
            "response_id": "r1",
            "call_id": "c1",
            "name": "web_search",
            "arguments": "{}",
        }
    )
    await session._dispatch({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
    await asyncio.sleep(0.2)  # the server rejected response.create: no response.created arrives
    assert ("on_tools_done_without_follow_up", (), {}) in events.log


async def test_follow_up_that_starts_is_not_timed_out(monkeypatch):
    import discord_bot.realtime as realtime

    monkeypatch.setattr(realtime, "FOLLOW_UP_TIMEOUT_S", 0.05)
    session, ws, events = make_session()
    await session._dispatch(
        {
            "type": "response.function_call_arguments.done",
            "response_id": "r1",
            "call_id": "c1",
            "name": "web_search",
            "arguments": "{}",
        }
    )
    await session._dispatch({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
    await asyncio.sleep(0.02)
    await session._dispatch({"type": "response.created"})
    await asyncio.sleep(0.1)
    assert ("on_tools_done_without_follow_up", (), {}) not in events.log


async def _one_search_round(session, response_id):
    await session._dispatch(
        {
            "type": "response.function_call_arguments.done",
            "response_id": response_id,
            "call_id": f"c_{response_id}",
            "name": "web_search",
            "arguments": json.dumps({"query": f"query {response_id}"}),
        }
    )
    await session._dispatch({"type": "response.done", "response": {"id": response_id, "status": "completed"}})
    await asyncio.sleep(0.02)
    await session._dispatch({"type": "response.created"})


async def test_search_loop_is_cut_off_after_max_rounds():
    session, ws, _ = make_session()
    session.max_tool_rounds = 2
    await _one_search_round(session, "r1")
    await _one_search_round(session, "r2")
    creates = [e for e in ws.sent if e["type"] == "response.create"]
    assert creates[0] == {"type": "response.create"}  # first follow-up may search again
    final = creates[1]["response"]  # then it must answer, with tools off and answer-only instructions
    assert final["tool_choice"] == "none"
    assert "Your tool calls are finished" in final["instructions"]


async def test_new_question_resets_the_search_limit():
    session, ws, _ = make_session()
    session.max_tool_rounds = 2
    await _one_search_round(session, "r1")
    await session._dispatch({"type": "input_audio_buffer.speech_started"})  # the user asks something new
    await _one_search_round(session, "r2")
    creates = [e for e in ws.sent if e["type"] == "response.create"]
    assert creates == [{"type": "response.create"}, {"type": "response.create"}]


async def test_repeated_search_is_not_run_and_forces_an_answer():
    session, ws, _ = make_session()
    session.max_tool_rounds = 5
    ran = []

    async def run(name, raw):
        ran.append(raw)
        return ToolOutcome('{"results": []}')

    session.tools.run = run
    for rid in ("r1", "r2"):
        await session._dispatch(
            {
                "type": "response.function_call_arguments.done",
                "response_id": rid,
                "call_id": f"c_{rid}",
                "name": "web_search",
                "arguments": json.dumps({"query": "手歌 歌手 專輯 " if rid == "r2" else "手歌 歌手 專輯"}),
            }
        )
        await session._dispatch({"type": "response.done", "response": {"id": rid, "status": "completed"}})
        await asyncio.sleep(0.02)
        await session._dispatch({"type": "response.created"})

    assert len(ran) == 1  # the identical second search was not run
    outputs = [e["item"]["output"] for e in ws.sent if e["type"] == "conversation.item.create"]
    assert "already called web_search" in outputs[1]
    creates = [e for e in ws.sent if e["type"] == "response.create"]
    assert creates[1]["response"]["tool_choice"] == "none"  # answer now, well before the limit of 5


async def test_final_answer_leaves_out_the_per_reply_note():
    session, ws, _ = make_session()
    session.max_tool_rounds = 1
    session._turn_note = 'If you search for this question, first say exactly "等我一下喔".'
    await _one_search_round(session, "r1")
    final = [e for e in ws.sent if e["type"] == "response.create"][0]["response"]
    assert "等我一下喔" not in final["instructions"]


async def test_tool_calls_are_reported_to_the_bot():
    session, _, events = make_session()
    await session._dispatch(
        {
            "type": "response.function_call_arguments.done",
            "response_id": "r",
            "call_id": "c",
            "name": "web_search",
            "arguments": "{}",
        }
    )
    assert ("on_tool_call", ("web_search",), {}) in events.log


async def _tool_round_then_follow_up(session, follow_up_id):
    await session._dispatch(
        {
            "type": "response.function_call_arguments.done",
            "response_id": "r1",
            "call_id": "c1",
            "name": "read_channel",
            "arguments": "{}",
        }
    )
    await session._dispatch({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
    await asyncio.sleep(0.02)
    await session._dispatch({"type": "response.created", "response": {"id": follow_up_id}})


async def test_an_empty_answer_after_tool_calls_is_asked_for_once_more():
    session, ws, events = make_session()
    session.max_tool_rounds = 1
    await _tool_round_then_follow_up(session, "f1")
    await session._dispatch({"type": "response.done", "response": {"id": "f1", "status": "completed"}})  # said nothing
    creates = [e for e in ws.sent if e["type"] == "response.create"]
    assert len(creates) == 2 and creates[1]["response"]["tool_choice"] == "none"
    assert "Your previous reply was empty" in creates[1]["response"]["instructions"]
    assert events.log[-1] == ("on_response_done", (), {"tools_pending": True})  # the window stays open

    await session._dispatch({"type": "response.created", "response": {"id": "f2"}})
    await session._dispatch({"type": "response.done", "response": {"id": "f2", "status": "completed"}})
    assert len([e for e in ws.sent if e["type"] == "response.create"]) == 2  # only one retry
    assert events.log[-1] == ("on_response_done", (), {"tools_pending": False})


async def test_a_spoken_answer_after_tool_calls_is_not_repeated():
    session, ws, events = make_session()
    session.max_tool_rounds = 1
    await _tool_round_then_follow_up(session, "f1")
    await session._dispatch(
        {"type": "response.output_audio_transcript.delta", "response_id": "f1", "delta": "台北今天晴天"}
    )
    await session._dispatch({"type": "response.done", "response": {"id": "f1", "status": "completed"}})
    assert len([e for e in ws.sent if e["type"] == "response.create"]) == 1
    assert events.log[-1] == ("on_response_done", (), {"tools_pending": False})


async def test_an_ordinary_silent_reply_is_not_retried():
    session, ws, _ = make_session()
    await session._dispatch({"type": "response.created", "response": {"id": "x"}})
    await session._dispatch({"type": "response.done", "response": {"id": "x", "status": "completed"}})
    assert [e for e in ws.sent if e["type"] == "response.create"] == []
