import json
from types import SimpleNamespace

import httpx
import pytest
from discord_bot.config import DJ_ALEX_MODULE, BotConfig
from discord_bot.tools import ToolRegistry, dj_alex

STATUS = {
    "ok": True,
    "status": "playing",
    "current": {"title": "晴天", "artist": "周杰倫", "position": 30, "duration": 269},
    "upNext": ["七里香 by 周杰倫"],
    "queueLength": 1,
    "volume": 60,
    "repeat": "off",
    "voiceChannel": "Lounge",
    "listeners": ["Darren"],
}


@pytest.fixture
def dj(monkeypatch):
    """DJ Alex's API, recording the requests the tools make."""
    monkeypatch.setenv("DJ_ALEX_URL", "http://dj:3000/")
    monkeypatch.setenv("DJ_ALEX_TOKEN", "secret")
    requests: list[httpx.Request] = []
    replies: dict[str, httpx.Response] = {}
    state = SimpleNamespace(down=False)

    def handle(request: httpx.Request) -> httpx.Response:
        if state.down:
            raise httpx.ConnectError("refused")
        requests.append(request)
        if request.url.path in replies:
            return replies[request.url.path]
        if request.url.path == "/api/control/search":
            songs = (
                [{"title": "晴天", "artist": "周杰倫", "album": "葉惠美"}]
                if request.url.params["q"] == "周杰倫"
                else []
            )
            return httpx.Response(200, json={"ok": True, "songs": songs, "total": len(songs)})
        return httpx.Response(
            200,
            json={
                **STATUS,
                "track": {"title": "晴天", "artist": "周杰倫", "album": "葉惠美"},
                "placement": "now playing",
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        dj_alex.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw)
    )
    bot = SimpleNamespace(
        user=SimpleNamespace(display_name="Alex"),
        session=SimpleNamespace(speaker="Darren"),
        voice=SimpleNamespace(channel=SimpleNamespace(id=42)),
    )
    dj_alex.bind(bot)
    yield SimpleNamespace(requests=requests, replies=replies, bot=bot, state=state)
    dj_alex.bind(None)


def body(request: httpx.Request) -> dict:
    return json.loads(request.content)


async def test_play_music_queues_for_the_speaker_in_alexs_channel(dj):
    result = await dj_alex.execute_tool("play_music", {"query": "晴天"})
    request = dj.requests[0]
    assert (request.method, str(request.url)) == ("POST", "http://dj:3000/api/control/play")
    assert request.headers["authorization"] == "Bearer secret"
    assert body(request) == {"actor": "Alex (for Darren)", "channelId": "42", "position": "now", "query": "晴天"}
    assert result["track"]["artist"] == "周杰倫" and result["current"]["title"] == "晴天"
    assert "exactly as given" in result["note"]
    assert "listeners" not in result and "voiceChannel" not in result


async def test_play_an_album_at_the_end_of_the_queue(dj):
    await dj_alex.execute_tool("play_music", {"query": "七里香", "album": True, "when": "end"})
    assert dj.requests[0].url.path == "/api/control/album"
    assert body(dj.requests[0])["name"] == "七里香" and body(dj.requests[0])["position"] == "end"


async def test_relative_volume_reads_the_current_volume_first(dj):
    await dj_alex.execute_tool("music_control", {"action": "volume", "volume_change": -15})
    assert [r.url.path for r in dj.requests] == ["/api/control/status", "/api/control/volume"]
    assert body(dj.requests[1])["volume"] == 45


async def test_volume_is_clamped_and_repeat_is_checked(dj):
    await dj_alex.execute_tool("music_control", {"action": "volume", "volume": 150})
    assert body(dj.requests[0])["volume"] == 100
    assert "error" in await dj_alex.execute_tool("music_control", {"action": "repeat", "repeat": "twice"})
    await dj_alex.execute_tool("music_control", {"action": "repeat", "repeat": "one"})
    assert body(dj.requests[-1])["mode"] == "one"


async def test_dj_alex_errors_are_passed_on(dj):
    dj.replies["/api/control/play"] = httpx.Response(400, json={"ok": False, "error": 'No songs match "x"'})
    assert await dj_alex.execute_tool("play_music", {"query": "x"}) == {"error": 'No songs match "x"'}
    dj.replies["/api/control/skip"] = httpx.Response(401, json={"ok": False, "error": "Unauthorized"})
    assert "CONTROL_TOKEN" in (await dj_alex.execute_tool("music_control", {"action": "skip"}))["error"]


async def test_dj_alex_out_of_voice_is_reported(dj):
    dj.replies["/api/control/status"] = httpx.Response(200, json={**STATUS, "voiceChannel": None, "status": "waiting"})
    result = await dj_alex.execute_tool("music_status", {})
    assert result["voiceChannel"] is None and result["status"] == "waiting"


async def test_without_a_voice_channel_or_speaker(dj):
    dj.bot.voice = None
    dj.bot.session.speaker = None
    await dj_alex.execute_tool("music_control", {"action": "pause"})
    assert body(dj.requests[0]) == {"actor": "Alex"}


async def test_search_music_lists_matching_songs(dj):
    found = await dj_alex.execute_tool("search_music", {"query": "周杰倫"})
    assert found["songs"] == [{"title": "晴天", "artist": "周杰倫", "album": "葉惠美"}]
    assert "exactly as given" in found["note"]
    assert dj.requests[0].headers["authorization"] == "Bearer secret"
    assert dj.requests[0].url.params["limit"] == "8"
    assert (await dj_alex.execute_tool("search_music", {"query": "nobody"}))["note"]


async def test_unreachable_dj_alex(dj):
    dj.state.down = True
    assert await dj_alex.execute_tool("music_status", {}) == {"error": "DJ Alex is not reachable right now."}


def test_tools_are_on_by_default_only_with_url_and_token(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.delenv("TOOL_MODULES", raising=False)
    monkeypatch.setenv("DJ_ALEX_URL", "http://dj:3000")
    monkeypatch.setenv("DJ_ALEX_TOKEN", "")
    assert DJ_ALEX_MODULE not in BotConfig.from_env().tool_modules
    monkeypatch.setenv("DJ_ALEX_TOKEN", "secret")
    config = BotConfig.from_env()
    assert DJ_ALEX_MODULE in config.tool_modules
    names = [t["name"] for t in ToolRegistry(config.tool_modules).definitions]
    assert {"play_music", "music_control", "music_status", "search_music"} <= set(names)
    assert "DJ Alex is the music bot" in config.session_instructions(names)


async def test_duck_and_release_in_alexs_channel(dj):
    await dj_alex.duck(30, 30)
    await dj_alex.duck(None, 30)
    assert [r.url.path for r in dj.requests] == ["/api/control/duck"] * 2
    assert body(dj.requests[0]) == {"level": 30, "seconds": 30, "channelId": "42"}
    assert body(dj.requests[1]) == {"level": 100, "seconds": 0, "channelId": "42"}


async def test_duck_raises_when_refused(dj):
    dj.replies["/api/control/duck"] = httpx.Response(401, json={"ok": False})
    with pytest.raises(httpx.HTTPStatusError):
        await dj_alex.duck(30, 30)


def test_guide_says_to_try_again_after_an_earlier_failure():
    config = BotConfig(discord_token="x")
    guide = config.session_instructions(["play_music"])
    assert "even if an earlier call in this conversation failed" in guide


async def test_play_music_without_a_song_lets_dj_alex_pick_one(dj):
    await dj_alex.execute_tool("play_music", {})
    assert dj.requests[0].url.path == "/api/control/play"
    assert body(dj.requests[0])["query"] == ""


def test_tool_descriptions_name_no_example_songs():
    text = json.dumps(dj_alex.TOOLS, ensure_ascii=False)
    assert "晴天" not in text and "周杰倫" not in text
    play = next(t for t in dj_alex.TOOLS if t["name"] == "play_music")
    assert "query" not in play["parameters"].get("required", [])
    assert "never pick a song yourself" in BotConfig(discord_token="x").session_instructions(["play_music"])


async def test_volume_changes_carry_no_library_note(dj):
    dj.replies["/api/control/volume"] = httpx.Response(200, json={"ok": True, "volume": 30, "voiceChannel": "Lounge"})
    assert "note" not in await dj_alex.execute_tool("music_control", {"action": "volume", "volume": 30})
