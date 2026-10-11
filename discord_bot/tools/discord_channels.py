"""Read the Discord server the bot is in: list its channels and read recent messages.

Covers every channel the bot can see: text and announcement channels, voice and stage channels
(their chat), forum channels (their posts) and threads.

Reading message text needs the privileged Message Content intent: enable it in the Discord
Developer Portal (your application -> Bot -> Privileged Gateway Intents -> Message Content Intent).
The bot only sees channels its role is allowed to view; hide a channel from the role to keep it private.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
SEARCH_SCAN = 500  # messages scanned when looking for a keyword
MAX_MESSAGE_CHARS = 300
MAX_OUTPUT_CHARS = 6000  # keep tool results small: the model re-reads them every turn
MAX_CHANNELS = 100
MAX_THREADS_LISTED = 10  # active threads shown under their channel by list_channels
MAX_FORUM_POSTS = 30
ACTIVITY_PER_CHANNEL = 5  # recent_activity: newest messages shown per active channel
ACTIVITY_MAX_CHANNELS = 15
DISCORD_EPOCH_MS = 1420070400000
# Discord channel types, as the model sees them. Categories are shown as groups instead.
KINDS = {
    "text": "text",
    "news": "announcement",
    "voice": "voice",
    "stage_voice": "stage",
    "forum": "forum",
    "media": "forum",
    "public_thread": "thread",
    "private_thread": "thread",
    "news_thread": "thread",
}

TOOLS = [
    {
        "type": "function",
        "name": "list_channels",
        "description": (
            "List every channel of this Discord server the bot can see, grouped by category: text, "
            "announcement, voice and stage channels (with who is in them), forums and active threads. Use it to "
            "find the right channel name before read_channel, or when someone asks what channels there are or "
            "who is in a voice channel."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "read_channel",
        "description": (
            "Read recent messages from a channel or thread of this Discord server, oldest first, with time and "
            "author. Use it when someone asks what was said or posted in a channel, asks about recent "
            "messages or wants a summary. For a forum it returns the forum's posts; read a post by its name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel, thread or forum-post name, e.g. 'general' or '聊天'. Close matches work.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"How many messages to return (1-{MAX_LIMIT}, default {DEFAULT_LIMIT}).",
                },
                "contains": {
                    "type": "string",
                    "description": "Only messages containing this text (case-insensitive). Optional.",
                },
                "hours": {
                    "type": "number",
                    "description": "Only messages from the last this many hours. Optional.",
                },
            },
            "required": ["channel"],
            "additionalProperties": False,
        },
    },
]

TOOLS.append(
    {
        "type": "function",
        "name": "recent_activity",
        "description": (
            "What has been happening across the whole Discord server: the newest messages from every channel "
            "and thread with activity in the last hours, most recent first. Use it for questions like 'what is "
            "everyone talking about?' or 'anything new on the server?' instead of reading channels one by one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hours": {"type": "number", "description": "How far back to look, in hours (default 24)."},
            },
            "additionalProperties": False,
        },
    }
)

_bot: Any = None


def bind(bot: Any) -> None:
    global _bot
    _bot = bot


def _key(name: str) -> str:
    """Channel names compared by letters and digits only (no emoji, separators or case)."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


TEXT_LIKE = ("text", "announcement", "thread")


def _container(channel: Any) -> str | None:
    """The category of a channel, or the parent channel of a thread."""
    owner = getattr(channel, "parent", None) or getattr(channel, "category", None)
    return getattr(owner, "name", None)


def describe(channel: Any) -> str:
    """A candidate name the model can tell apart from same-named channels."""
    container = _container(channel)
    details = [kind(channel) or "channel", *([f"in {container}"] if container else []), f"id {channel.id}"]
    return f"{channel.name} ({', '.join(details)})"


def _prefer_text(matches: list[Any]) -> list[Any]:
    """Same-named channels: reading "一般" means the text channel, not the voice channel's chat."""
    if len(matches) > 1:
        text_like = [c for c in matches if kind(c) in TEXT_LIKE]
        if len(text_like) == 1:
            return text_like
    return matches


def find_channel(channels: list[Any], query: str) -> tuple[Any | None, list[str]]:
    """The channel ``query`` names, or ``(None, candidate descriptions)`` if there is no single match.

    ``query`` may be a name, ``#name``, an ID or mention, or include the category or parent channel
    ("文字頻道/一般", "語音頻道 一般").
    """
    query = query.strip()
    digits = query.strip("<#>")
    if digits.isdigit():
        by_id = [c for c in channels if str(c.id) == digits]
        if by_id:
            return by_id[0], []
    wanted = _key(query)
    if not wanted:
        return None, [describe(c) for c in channels][:MAX_CHANNELS]
    exact = [c for c in channels if _key(c.name) == wanted]
    with_container = [c for c in channels if _key((_container(c) or "") + c.name) == wanted]
    partial = [c for c in channels if _key(c.name) and (wanted in _key(c.name) or _key(c.name) in wanted)]
    for matches in (exact, with_container, partial):
        narrowed = _prefer_text(matches)
        if len(narrowed) == 1:
            return narrowed[0], []
    pool = exact or with_container or partial or channels
    return None, [describe(c) for c in pool][:MAX_CHANNELS]


def message_entry(message: Any) -> dict[str, Any]:
    text = (message.clean_content or "").strip()
    if len(text) > MAX_MESSAGE_CHARS:
        text = text[:MAX_MESSAGE_CHARS] + "…"
    extras = [f"[attachment: {a.filename}]" for a in message.attachments]
    extras += [f"[sticker: {s.name}]" for s in getattr(message, "stickers", [])]
    if message.embeds and not text:
        extras.append("[embed]")
    author = message.author
    entry: dict[str, Any] = {
        "time": message.created_at.astimezone().strftime("%Y-%m-%d %H:%M"),
        "author": getattr(author, "display_name", None) or author.name,
        "text": " ".join([text, *extras]).strip(),
    }
    if author.bot:
        entry["bot"] = True
    return entry


def _guild() -> Any:
    if _bot is None:
        raise RuntimeError("The Discord tools are not connected to the bot")
    guild_id = _bot.config.guild_id
    if guild_id is not None:
        guild = _bot.get_guild(guild_id)
    elif getattr(_bot, "voice", None) is not None:
        guild = _bot.voice.guild
    else:
        guild = next(iter(_bot.guilds), None)
    if guild is None:
        raise RuntimeError("The bot is not in a Discord server")
    return guild


def kind(channel: Any) -> str | None:
    """The channel's kind as the model sees it, or None for categories and unknown types."""
    return KINDS.get(getattr(channel.type, "name", str(channel.type)))


def _can(channel: Any, me: Any) -> tuple[bool, bool]:
    """(can see it, can read its message history)."""
    permissions = channel.permissions_for(me)
    return bool(permissions.view_channel), bool(permissions.view_channel and permissions.read_message_history)


def readable_channels(guild: Any) -> list[Any]:
    """Channels and threads the bot may read, plus forums (read_channel lists their posts)."""
    me = guild.me
    channels = [c for _category, group in guild.by_category() for c in group]
    candidates = [c for c in [*channels, *getattr(guild, "threads", [])] if kind(c) is not None]
    return [c for c in candidates if _can(c, me)[0 if kind(c) == "forum" else 1]]


async def _archived_posts(forum: Any, limit: int) -> list[Any]:
    posts = []
    try:
        async for thread in forum.archived_threads(limit=limit):
            posts.append(thread)
    except Exception:  # missing permission or Discord error: show the active posts only
        pass
    return posts


async def _forum_posts(forum: Any) -> list[Any]:
    posts = list(forum.threads)
    known = {p.id for p in posts}
    posts += [p for p in await _archived_posts(forum, MAX_FORUM_POSTS) if p.id not in known]
    return posts[:MAX_FORUM_POSTS]


def _content_note() -> dict[str, str]:
    if _bot is not None and not _bot.intents.message_content:
        return {"note": "Message text is unavailable: the bot lacks the Message Content intent."}
    return {}


def _channel_entry(channel: Any, me: Any) -> dict[str, Any] | None:
    channel_kind = kind(channel)
    if channel_kind is None or channel_kind == "thread":
        return None
    visible, readable = _can(channel, me)
    if not visible:
        return None
    entry: dict[str, Any] = {"name": channel.name, "type": channel_kind}
    if getattr(channel, "topic", None):
        entry["topic"] = channel.topic[:150]
    if channel_kind in ("voice", "stage"):
        entry["people"] = [m.display_name for m in channel.members if not m.bot]
    threads = [t.name for t in getattr(channel, "threads", []) if _can(t, me)[1]]
    if threads:
        entry["threads" if channel_kind != "forum" else "active_posts"] = threads[:MAX_THREADS_LISTED]
    if not readable and channel_kind != "forum":
        entry["readable"] = False
    return entry


def list_channels() -> dict[str, Any]:
    guild = _guild()
    groups: list[dict[str, Any]] = []
    shown = 0
    for category, channels in guild.by_category():
        entries = [e for e in (_channel_entry(c, guild.me) for c in channels) if e is not None]
        entries = entries[: max(0, MAX_CHANNELS - shown)]
        shown += len(entries)
        if entries:
            groups.append({**({"category": category.name} if category else {}), "channels": entries})
    return {"server": guild.name, "channels_by_category": groups}


async def _find_archived_post(guild: Any, query: str) -> Any | None:
    """Archived forum posts are not cached; look them up when nothing else matches."""
    posts = []
    for forum in getattr(guild, "forums", []):
        if _can(forum, guild.me)[1]:
            posts += await _archived_posts(forum, 50)
    post, _ = find_channel(posts, query)
    return post


async def _why_empty(channel: Any, hours: float, contains: str) -> str:
    """Say what was searched, and when the channel last had a message, so "empty" is not misread."""
    latest = None
    async for message in channel.history(limit=1):
        latest = message
    if latest is None:
        return "This channel has no messages."
    when = latest.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
    searched = []
    if contains:
        searched.append(f"containing {contains!r} in the last {SEARCH_SCAN} messages")
    if hours > 0:
        searched.append(f"in the last {hours:g} hours")
    scope = " ".join(searched) or "with text"
    return f"No messages {scope}. The latest message in this channel is from {when}."


async def read_channel(arguments: dict[str, Any]) -> dict[str, Any]:
    guild = _guild()
    query = str(arguments.get("channel", "")).strip()
    if not query:
        return {"error": "channel must be a channel name"}
    channel, candidates = find_channel(readable_channels(guild), query)
    if channel is None:
        channel = await _find_archived_post(guild, query)
    if channel is None:
        return {"error": f"No single readable channel matches {query!r}", "channels": candidates}
    if kind(channel) == "forum":
        posts = await _forum_posts(channel)
        return {
            "forum": channel.name,
            "posts": [p.name for p in posts],
            "note": "This is a forum: call read_channel with a post name to read that post.",
        }
    limit = min(MAX_LIMIT, max(1, int(arguments.get("limit") or DEFAULT_LIMIT)))
    contains = str(arguments.get("contains") or "").strip().lower()
    hours = float(arguments.get("hours") or 0)
    after = datetime.now(timezone.utc) - timedelta(hours=hours) if hours > 0 else None

    entries: list[dict[str, Any]] = []
    async for message in channel.history(limit=SEARCH_SCAN if contains else limit, after=after, oldest_first=False):
        entry = message_entry(message)
        if not entry["text"] or (contains and contains not in entry["text"].lower()):
            continue
        entries.append(entry)
        if len(entries) >= limit:
            break
    entries.reverse()  # oldest first, like reading the channel
    while len(entries) > 1 and len(str(entries)) > MAX_OUTPUT_CHARS:
        entries.pop(0)
    result: dict[str, Any] = {"channel": channel.name, "messages": entries, **_content_note()}
    if not entries and "note" not in result:
        result["note"] = await _why_empty(channel, hours, contains)
    if kind(channel) == "thread" and getattr(channel, "parent", None) is not None:
        result["in"] = channel.parent.name
    return result


def _last_activity(channel: Any) -> datetime | None:
    """When the channel's latest message was posted, from its ID (no API call)."""
    last_id = getattr(channel, "last_message_id", None)
    if not last_id:
        return None
    return datetime.fromtimestamp(((int(last_id) >> 22) + DISCORD_EPOCH_MS) / 1000, tz=timezone.utc)


async def _recent_messages(channel: Any, after: datetime) -> list[dict[str, Any]]:
    entries = []
    async for message in channel.history(limit=ACTIVITY_PER_CHANNEL * 2, after=after, oldest_first=False):
        entry = message_entry(message)
        if entry["text"]:
            entries.append(entry)
        if len(entries) >= ACTIVITY_PER_CHANNEL:
            break
    return entries[::-1]


async def recent_activity(arguments: dict[str, Any]) -> dict[str, Any]:
    guild = _guild()
    hours = float(arguments.get("hours") or 24)
    after = datetime.now(timezone.utc) - timedelta(hours=hours)
    channels = [c for c in readable_channels(guild) if kind(c) != "forum"]
    # Channels whose latest message is older than the window need no API call at all.
    active = [c for c in channels if (_last_activity(c) or after) > after]
    active.sort(key=lambda c: _last_activity(c) or after, reverse=True)
    active = active[:ACTIVITY_MAX_CHANNELS]
    results = await asyncio.gather(*(_recent_messages(c, after) for c in active), return_exceptions=True)
    groups = []
    for channel, messages in zip(active, results):
        if isinstance(messages, BaseException) or not messages:
            continue
        group: dict[str, Any] = {"channel": channel.name, "messages": messages}
        container = _container(channel)
        if container:
            group["in"] = container
        groups.append(group)
    while groups and len(str(groups)) > MAX_OUTPUT_CHARS:  # trim the oldest messages of the busiest lists
        longest = max(groups, key=lambda g: len(g["messages"]))
        longest["messages"].pop(0)
        groups = [g for g in groups if g["messages"]]
    result: dict[str, Any] = {"hours": hours, "active_channels": groups, **_content_note()}
    if not groups:
        result["note"] = result.get("note") or f"No messages anywhere on the server in the last {hours:g} hours."
    return result


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "recent_activity":
        return await recent_activity(arguments)
    if name == "list_channels":
        return list_channels()
    if name == "read_channel":
        return await read_channel(arguments)
    raise ValueError(f"Unknown tool: {name}")
