import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from discord_bot.config import DISCORD_TOOLS_MODULE, BotConfig
from discord_bot.tools import ToolRegistry
from discord_bot.tools import discord_channels as tools

NOW = datetime(2026, 10, 11, 9, 0, tzinfo=timezone.utc)


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


def perms(view=True, history=True):
    return SimpleNamespace(view_channel=view, read_message_history=history)


def member(name, bot=False):
    return SimpleNamespace(display_name=name, name=name.lower(), bot=bot)


def message(author, text, minutes_ago, attachments=()):
    return SimpleNamespace(
        author=author,
        clean_content=text,
        created_at=NOW - timedelta(minutes=minutes_ago),
        attachments=[SimpleNamespace(filename=f) for f in attachments],
        embeds=[],
        stickers=[],
    )


class Channel:
    def __init__(
        self, id, name, type="text", messages=(), topic=None, members=(), allowed=None, parent=None, category=None
    ):
        self.id, self.name, self.topic, self.members, self.parent = id, name, topic, list(members), parent
        self.category = SimpleNamespace(name=category) if category else None
        self.type = SimpleNamespace(name=type)
        self._messages = list(messages)  # newest first, as Discord returns them
        self._allowed = allowed or perms()
        self.threads: list = []
        self.archived: list = []
        self.history_calls: list = []

    def permissions_for(self, _me):
        return self._allowed

    async def history(self, limit, after=None, oldest_first=None):
        self.history_calls.append({"limit": limit, "after": after, "oldest_first": oldest_first})
        count = 0
        for msg in self._messages:
            if after is not None and msg.created_at <= after:
                continue
            if count >= limit:
                return
            count += 1
            yield msg

    async def archived_threads(self, limit):
        for thread in self.archived[:limit]:
            yield thread


alice, bob, alex = member("Alice"), member("Bob"), member("Alex", bot=True)


@pytest.fixture
def server(monkeypatch):
    general = Channel(
        1,
        "💬｜general",
        topic="Anything goes",
        messages=[
            message(bob, "明天去看戰鬥陀螺比賽嗎？", 5),
            message(alex, "我他媽的幫你查了", 20),
            message(alice, "", 30, attachments=["photo.png"]),
            message(alice, "Beyblade X 新版本出來了", 3 * 60),
        ],
    )
    news = Channel(5, "announcements", type="news", messages=[message(alice, "Server rules updated", 60)])
    secret = Channel(2, "admins", allowed=perms(history=False))
    hidden = Channel(6, "mods-only", allowed=perms(view=False))
    games = Channel(3, "game-news")
    lobby = Channel(4, "Lobby", type="voice", members=[alice, bob, alex])
    stage = Channel(7, "Town Hall", type="stage_voice")
    forum = Channel(8, "beyblade-forum", type="forum")
    active_post = Channel(
        9, "Best launcher?", type="public_thread", parent=forum, messages=[message(bob, "Dran Sword", 10)]
    )
    old_post = Channel(
        10, "Tournament 2025 results", type="public_thread", parent=forum, messages=[message(alice, "I won!", 9000)]
    )
    forum.threads, forum.archived = [active_post], [old_post]
    thread = Channel(11, "trip planning", type="public_thread", parent=general, messages=[message(bob, "Okinawa?", 15)])
    general.threads = [thread]
    category = lambda name: SimpleNamespace(name=name)  # noqa: E731
    guild = SimpleNamespace(
        name="Test Server",
        me=object(),
        by_category=lambda: [
            (None, [news]),
            (category("Text"), [general, secret, hidden, games]),
            (category("Voice"), [lobby, stage]),
            (category("Community"), [forum]),
        ],
        threads=[thread, active_post],
        forums=[forum],
    )
    bot = SimpleNamespace(
        config=SimpleNamespace(guild_id=42),
        get_guild=lambda gid: guild if gid == 42 else None,
        intents=SimpleNamespace(message_content=True),
        guilds=[guild],
    )
    tools.bind(bot)
    monkeypatch.setattr(tools, "datetime", FrozenDatetime)
    yield SimpleNamespace(bot=bot, general=general)
    tools.bind(None)


def run(name, **arguments):
    return asyncio.run(tools.execute_tool(name, arguments))


def test_find_channel_by_name_ignoring_emoji_case_and_separators():
    general, games, lobby = Channel(1, "💬｜general"), Channel(3, "game-news"), Channel(4, "Lobby")
    channels = [general, games, lobby]
    assert tools.find_channel(channels, "General")[0] is general
    assert tools.find_channel(channels, "#game news")[0] is games
    assert tools.find_channel(channels, "<#4>")[0] is lobby
    channel, candidates = tools.find_channel(channels, "music")
    assert channel is None and candidates == [
        "💬｜general (text, id 1)",
        "game-news (text, id 3)",
        "Lobby (text, id 4)",
    ]


def test_list_channels_shows_every_visible_channel_by_category(server):
    result = run("list_channels")
    groups = {g.get("category"): g["channels"] for g in result["channels_by_category"]}
    assert groups[None] == [{"name": "announcements", "type": "announcement"}]
    text = {c["name"]: c for c in groups["Text"]}
    assert text["💬｜general"] == {
        "name": "💬｜general",
        "type": "text",
        "topic": "Anything goes",
        "threads": ["trip planning"],
    }
    assert text["admins"]["readable"] is False
    assert "mods-only" not in text  # hidden from the bot's role
    assert groups["Voice"] == [
        {"name": "Lobby", "type": "voice", "people": ["Alice", "Bob"]},
        {"name": "Town Hall", "type": "stage", "people": []},
    ]
    assert groups["Community"] == [{"name": "beyblade-forum", "type": "forum", "active_posts": ["Best launcher?"]}]


def test_read_channel_returns_recent_messages_oldest_first(server):
    result = run("read_channel", channel="general", limit=3)
    assert result["channel"] == "💬｜general"
    assert [m["author"] for m in result["messages"]] == ["Alice", "Alex", "Bob"]
    assert result["messages"][0]["text"] == "[attachment: photo.png]"
    assert result["messages"][1]["bot"] is True and "bot" not in result["messages"][2]
    assert "note" not in result


def test_announcements_threads_and_voice_chats_can_be_read(server):
    assert run("read_channel", channel="announcements")["messages"][0]["text"] == "Server rules updated"
    thread = run("read_channel", channel="trip planning")
    assert thread["messages"][0]["text"] == "Okinawa?" and thread["in"] == "💬｜general"


def test_a_forum_lists_its_posts_and_archived_posts_can_be_read(server):
    forum = run("read_channel", channel="beyblade forum")
    assert forum["posts"] == ["Best launcher?", "Tournament 2025 results"]
    assert run("read_channel", channel="Best launcher")["messages"][0]["text"] == "Dran Sword"
    archived = run("read_channel", channel="tournament 2025 results")  # not cached: looked up in the forum
    assert archived["messages"][0]["text"] == "I won!" and archived["in"] == "beyblade-forum"


def test_contains_scans_further_and_filters(server):
    result = run("read_channel", channel="general", contains="beyblade")
    assert [m["text"] for m in result["messages"]] == ["Beyblade X 新版本出來了"]
    assert server.general.history_calls[0]["limit"] == tools.SEARCH_SCAN


def test_hours_limits_the_time_window(server):
    result = run("read_channel", channel="general", hours=1)
    assert len(result["messages"]) == 3
    assert server.general.history_calls[0]["after"] == NOW - timedelta(hours=1)


def test_channels_the_bot_cannot_read_are_not_offered(server):
    result = run("read_channel", channel="admins")
    assert "error" in result and "admins" not in result["channels"] and "mods-only" not in result["channels"]


def test_missing_message_content_intent_is_explained(server):
    server.bot.intents.message_content = False
    assert "Message Content intent" in run("read_channel", channel="general")["note"]


def test_long_logs_are_trimmed_from_the_oldest_end(server, monkeypatch):
    monkeypatch.setattr(tools, "MAX_OUTPUT_CHARS", 200)
    result = run("read_channel", channel="general", limit=4)
    assert result["messages"][-1]["author"] == "Bob" and len(result["messages"]) < 4


def test_registry_binds_the_bot_and_config_enables_the_intent(monkeypatch):
    registry = ToolRegistry((DISCORD_TOOLS_MODULE,))
    marker = object()
    registry.bind(marker)
    assert tools._bot is marker
    tools.bind(None)
    assert [t["name"] for t in registry.definitions] == ["list_channels", "read_channel", "recent_activity"]
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.delenv("TOOL_MODULES", raising=False)
    config = BotConfig.from_env()
    assert config.reads_discord_messages
    assert "Reading this Discord server" in config.session_instructions(["list_channels", "read_channel"])
    assert "Reading this Discord server" not in config.session_instructions([])


def test_same_named_text_and_voice_channels_like_a_new_server():
    # A new Discord server has a text channel and a voice channel both named 一般.
    text = Channel(123, "一般", category="文字頻道", messages=[message(bob, "hi", 30)])
    voice = Channel(124, "一般", type="voice", category="語音頻道")
    dating = Channel(125, "約會", type="voice")
    channels = [text, voice, dating]
    assert tools.find_channel(channels, "一般")[0] is text  # reading 一般 means the text channel
    assert tools.find_channel(channels, "#一般")[0] is text
    assert tools.find_channel(channels, "文字頻道一般")[0] is text
    assert tools.find_channel(channels, "語音頻道/一般")[0] is voice
    assert tools.find_channel(channels, "124")[0] is voice
    both_voice = [voice, Channel(126, "一般", type="voice", category="其他")]
    channel, candidates = tools.find_channel(both_voice, "一般")
    assert channel is None
    assert candidates == ["一般 (voice, in 語音頻道, id 124)", "一般 (voice, in 其他, id 126)"]


def test_an_empty_time_window_says_when_the_last_message_was(server):
    result = run("read_channel", channel="general", hours=0.05)
    assert result["messages"] == []
    assert result["note"].startswith("No messages in the last 0.05 hours. The latest message in this channel is from ")
    assert "2026-10-11" in result["note"]
    result = run("read_channel", channel="general", contains="okinawa")
    assert "containing 'okinawa' in the last 500 messages" in result["note"]


def snowflake(minutes_ago):
    return int((NOW - timedelta(minutes=minutes_ago)).timestamp() * 1000 - tools.DISCORD_EPOCH_MS) << 22


def test_recent_activity_covers_the_whole_server_most_recent_first(server, monkeypatch):
    general = server.general
    general.last_message_id = snowflake(5)
    guild = server.bot.get_guild(42)
    news = guild.by_category()[0][1][0]
    news.last_message_id = snowflake(60)
    games = guild.by_category()[1][1][3]
    games.last_message_id = snowflake(60 * 72)  # quiet for 3 days: not even fetched
    thread = guild.threads[0]
    thread.last_message_id = snowflake(15)
    result = run("recent_activity", hours=24)
    assert [g["channel"] for g in result["active_channels"]] == ["💬｜general", "trip planning", "announcements"]
    assert result["active_channels"][1]["in"] == "💬｜general"
    assert len(result["active_channels"][0]["messages"]) == tools.ACTIVITY_PER_CHANNEL - 1  # 4 with text in 24 h
    assert games.history_calls == []


def test_recent_activity_on_a_quiet_server(server):
    result = run("recent_activity", hours=1)
    assert result["active_channels"] == [] and "No messages anywhere" in result["note"]
