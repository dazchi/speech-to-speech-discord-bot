"""Control DJ Alex, the music bot (https://github.com/dazchi/dj-alex), through its control API.

Needs DJ_ALEX_URL (e.g. http://192.168.1.20:3000) and DJ_ALEX_TOKEN (the CONTROL_TOKEN in DJ Alex's .env).
Music is queued as "Alex (for <speaker>)", and DJ Alex joins Alex's voice channel if it is in none.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

TIMEOUT_S = 10.0
SEARCH_RESULTS = 8
# Added to results that name songs: the library's tags are what is playing, even for well-known songs
# (e.g. a cover or an AI version), so the model must not swap in the artist it remembers.
LIBRARY_NOTE = (
    "Titles, artists and albums come from DJ Alex's library. Say them exactly as given here, even if you "
    "remember a different singer for that song."
)
ACTIONS = ("pause", "resume", "skip", "previous", "stop", "shuffle", "clear", "volume", "repeat", "leave")

TOOLS = [
    {
        "type": "function",
        "name": "play_music",
        "description": (
            "Play a song or album on DJ Alex, the music bot in this server, from its music library, or add it to "
            "the queue. Use it when someone asks to play, put on or queue music. Pass the song title and/or artist exactly as they said "
            "it. If they did not name a song, album or artist, leave query out: DJ Alex then plays a random song "
            "from its library. For the newest or most recently added songs, use latest instead of query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Only words the person said: song title, artist or both, album name, or an audio URL. "
                        "Leave it out when they did not name anything."
                    ),
                },
                "latest": {
                    "type": "integer",
                    "description": (
                        "Play the most recently added songs, newest first: 1 for the latest song, or how many "
                        "they asked for (up to 20). Leave query out when using it."
                    ),
                },
                "album": {"type": "boolean", "description": "true to queue the whole album named in query."},
                "when": {
                    "type": "string",
                    "enum": ["now", "next", "end"],
                    "description": (
                        "now: play it right away, replacing the current song (when they say 'play'); next: "
                        "play it after the current song ('play X next', '下一首放'); end: add it to the end of "
                        "the queue ('add X to the queue', '加到播放清單', '排一首', 'queue X')."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "music_control",
        "description": (
            "Control DJ Alex's playback: pause, resume, skip to the next song, go back to the previous one, "
            "stop, shuffle or clear the queue, set the volume, set repeat, or make DJ Alex leave the voice channel."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "volume": {"type": "integer", "description": "For action volume: the new volume, 0-100."},
                "volume_change": {
                    "type": "integer",
                    "description": "For action volume: change the volume by this much instead (e.g. -20, +10).",
                },
                "repeat": {
                    "type": "string",
                    "enum": ["off", "one", "all"],
                    "description": "For action repeat: one repeats the current song, all the whole queue.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "music_status",
        "description": "What DJ Alex is playing now, what is up next in the queue, the volume and the repeat mode.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "search_music",
        "description": (
            "Search DJ Alex's music library by title, artist or album, e.g. to answer 'what songs by X do we "
            "have?' or to find the right song before play_music. Each song comes with the day it was added. "
            "With newest, lists the most recently added songs first (leave query out for the newest in the library)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Title, artist or album keywords."},
                "newest": {"type": "boolean", "description": "true to list the most recently added songs first."},
            },
            "additionalProperties": False,
        },
    },
]

_bot: Any = None


def bind(bot: Any) -> None:
    global _bot
    _bot = bot


def _settings() -> tuple[str, str]:
    url = os.environ.get("DJ_ALEX_URL", "").strip().rstrip("/")
    token = os.environ.get("DJ_ALEX_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError("DJ_ALEX_URL and DJ_ALEX_TOKEN must be set to control DJ Alex")
    return url, token


def _requester() -> dict[str, Any]:
    """Who asked (shown in DJ Alex's activity feed) and the voice channel DJ Alex should join."""
    body: dict[str, Any] = {}
    if _bot is None:
        return body
    own = getattr(getattr(_bot, "user", None), "display_name", None) or "Alex"
    speaker = getattr(getattr(_bot, "session", None), "speaker", None)
    body["actor"] = f"{own} (for {speaker})" if speaker else own
    voice = getattr(_bot, "voice", None)
    channel = getattr(voice, "channel", None)
    if channel is not None:
        body["channelId"] = str(channel.id)
    return body


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    """The parts of DJ Alex's reply worth telling the model."""
    keep = (
        "track",
        "placement",
        "songs",
        "album",
        "tracks",
        "status",
        "current",
        "upNext",
        "queueLength",
        "volume",
        "repeat",
    )
    result = {key: data[key] for key in keep if key in data and data[key] not in (None, [])}
    if not data.get("voiceChannel"):
        result["voiceChannel"] = None  # DJ Alex is not in a voice channel: nobody hears the music
    if any(key in result for key in ("track", "songs", "album", "current", "upNext")):
        result["note"] = LIBRARY_NOTE
    return result


async def _call(method: str, path: str, body: dict[str, Any] | None = None, **params: Any) -> dict[str, Any]:
    """DJ Alex's full reply, or {"error": ...} for the model."""
    url, token = _settings()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.request(
                method,
                f"{url}{path}",
                json=body,
                params=params or None,
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.RequestError:
        return {"error": "DJ Alex is not reachable right now."}
    if response.status_code == 401:
        return {"error": "DJ Alex refused the request (DJ_ALEX_TOKEN does not match its CONTROL_TOKEN)."}
    try:
        data = response.json()
    except ValueError:
        return {"error": f"DJ Alex answered with HTTP {response.status_code}."}
    if not data.get("ok"):
        return {"error": data.get("error") or f"DJ Alex answered with HTTP {response.status_code}."}
    return data


async def _request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = await _call(method, path, body)
    return data if "error" in data else _summary(data)


async def play_music(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    when = arguments.get("when") if arguments.get("when") in ("now", "next", "end") else "now"
    body = {**_requester(), "position": when}
    if arguments.get("latest") is not None:
        try:
            latest = max(1, min(20, int(arguments["latest"])))
        except (TypeError, ValueError):
            latest = 1
        return await _request("POST", "/api/control/play", {**body, "latest": latest})
    if arguments.get("album"):
        return await _request("POST", "/api/control/album", {**body, "name": query})
    return await _request("POST", "/api/control/play", {**body, "query": query})


async def music_control(arguments: dict[str, Any]) -> dict[str, Any]:
    action = arguments.get("action")
    if action not in ACTIONS:
        return {"error": f"action must be one of {', '.join(ACTIONS)}"}
    body = _requester()
    if action == "volume":
        if arguments.get("volume") is not None:
            volume = int(arguments["volume"])
        elif arguments.get("volume_change") is not None:
            status = await _request("GET", "/api/control/status")
            if "error" in status:
                return status
            volume = int(status.get("volume", 0)) + int(arguments["volume_change"])
        else:
            return {"error": "volume or volume_change is required for action volume"}
        body["volume"] = max(0, min(100, volume))
    elif action == "repeat":
        if arguments.get("repeat") not in ("off", "one", "all"):
            return {"error": "repeat must be off, one or all"}
        body["mode"] = arguments["repeat"]
    result = await _request("POST", f"/api/control/{action}", body)
    if action in ("skip", "previous") and "error" not in result:
        # Only the song that plays now: with the queue listed too, the model announced the song after it
        # as "the next song".
        result.pop("upNext", None)
        result.pop("queueLength", None)
        current = result.get("current")
        result["nowPlaying"] = f"{current['title']} by {current['artist']}" if current else None
    return result


async def search_music(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    newest = bool(arguments.get("newest"))
    if not query and not newest:
        return {"error": "give a query, or newest: true for the latest songs"}
    params: dict[str, Any] = {"q": query, "limit": SEARCH_RESULTS}
    if newest:
        params["sort"] = "recent"
    page = await _call("GET", "/api/control/search", **params)
    if "error" in page:
        return page
    songs = [{key: t[key] for key in ("title", "artist", "album", "added") if key in t} for t in page.get("songs", [])]
    result: dict[str, Any] = {"query": query, "total": page.get("total", len(songs)), "songs": songs}
    if newest:
        result["order"] = "newest first"
    result["note"] = LIBRARY_NOTE if songs else "No songs in the library match."
    return result


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "play_music":
        return await play_music(arguments)
    if name == "music_control":
        return await music_control(arguments)
    if name == "music_status":
        return await _request("GET", "/api/control/status")
    if name == "search_music":
        return await search_music(arguments)
    raise ValueError(f"Unknown tool: {name}")


async def duck(level: int | None, seconds: float) -> None:
    """Lower DJ Alex's music to ``level`` percent while Alex speaks (``level`` None: back to normal).

    Only when the music plays in Alex's voice channel. Not a tool: the bot calls it around replies.
    Raises httpx.HTTPError if DJ Alex cannot be reached.
    """
    url, token = _settings()
    body: dict[str, Any] = {"level": 100 if level is None else level, "seconds": 0 if level is None else seconds}
    channel = getattr(getattr(_bot, "voice", None), "channel", None)
    if channel is not None:
        body["channelId"] = str(channel.id)
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        response = await client.post(f"{url}/api/control/duck", json=body, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()


def configured() -> bool:
    try:
        _settings()
    except RuntimeError:
        return False
    return True
