import asyncio
from types import SimpleNamespace

import pytest
from discord_bot.chat import chat_label, clean_question, split_message

BOT = 111


def test_clean_question_removes_the_bot_mention_and_names_other_users():
    assert clean_question(f"<@{BOT}> 今天台北天氣如何？", BOT, {}) == "今天台北天氣如何？"
    assert clean_question(f"<@!{BOT}>, what did <@222> say?", BOT, {222: "lylla"}) == "what did @lylla say?"
    assert clean_question(f"<@{BOT}>", BOT, {}) == ""


def test_chat_label_marks_typed_messages():
    assert chat_label("lylla", "hi") == "[lylla wrote in chat] hi"


def test_split_message_respects_discord_limit():
    text = ("這是一句話。" * 500).strip()
    chunks = split_message(text)
    assert all(len(c) <= 2000 for c in chunks)
    assert "".join(chunks) == text
    assert split_message("short") == ["short"]


# ---- bot flow ----

pytest.importorskip("discord")


class FakeMessage:
    def __init__(self, content, author_id=5, name="lylla", mentions=()):
        self.content = content
        self.author = SimpleNamespace(id=author_id, bot=False, name=name, display_name=name)
        self.guild = SimpleNamespace(id=1)
        self.mentions = list(mentions)
        self.reactions, self.replies = [], []
        self.channel = SimpleNamespace(send=self._send)

    async def add_reaction(self, emoji):
        self.reactions.append(emoji)

    async def reply(self, text, **kwargs):
        self.replies.append(text)

    async def _send(self, text):
        self.replies.append(text)


class FakeSession:
    connected = True

    def __init__(self):
        self.asked = []

    async def set_speaker(self, name):
        pass

    async def ask_text(self, text):
        self.asked.append(text)

    async def set_turn_note(self, note):
        pass


def make_bot():
    from discord_bot.bot import SessionHandlers, VoiceBot
    from discord_bot.config import BotConfig
    from discord_bot.conversation import ConversationController

    bot = object.__new__(VoiceBot)
    bot.config = BotConfig(discord_token="x")
    bot.controller = ConversationController(wake_timeout_s=6, followup_s=8)
    bot.session = FakeSession()
    bot.chat_queue = asyncio.Queue()
    bot.chat_origin = None
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=BOT))  # discord.Client.user
    bot.speaker = SimpleNamespace(stop=lambda: None)
    bot.handlers = SessionHandlers(bot)
    bot.loop = asyncio.get_running_loop()
    bot._serves = lambda guild: True
    return bot


async def test_mention_is_asked_out_loud_and_answer_posted_in_chat():
    bot = make_bot()
    me = SimpleNamespace(id=BOT, display_name="AI Alex", name="AI Alex")
    message = FakeMessage(f"<@{BOT}> 今天台北天氣如何？", mentions=[me])
    await bot.on_message(message)
    worker = asyncio.create_task(bot._chat_worker())
    await asyncio.sleep(0.05)

    assert message.reactions == ["👀"]
    assert bot.session.asked == ["[lylla wrote in chat] 今天台北天氣如何？"]
    assert bot.controller.is_active_user(5)  # the asker can follow up by voice

    bot.handlers.on_assistant_transcript("台北今天晴天。")
    await asyncio.sleep(0.05)
    assert message.replies == ["台北今天晴天。"]
    worker.cancel()


async def test_messages_without_a_mention_are_ignored():
    bot = make_bot()
    await bot.on_message(FakeMessage("hello everyone"))
    assert bot.chat_queue.empty()


async def test_chat_question_waits_for_a_voice_conversation_to_finish():
    bot = make_bot()
    bot.controller.wake(9, "darren")  # someone is talking to Alex by voice
    me = SimpleNamespace(id=BOT, display_name="AI Alex", name="AI Alex")
    message = FakeMessage(f"<@{BOT}> hi", mentions=[me])
    await bot.on_message(message)
    worker = asyncio.create_task(bot._chat_worker())
    await asyncio.sleep(0.3)
    assert "⏳" in message.reactions and bot.session.asked == []
    bot.controller.reset()  # the voice window closed
    await asyncio.sleep(0.3)
    assert bot.session.asked == ["[lylla wrote in chat] hi"]
    worker.cancel()


async def test_spoken_turn_stops_chat_replies():
    bot = make_bot()
    bot.chat_origin = FakeMessage("x")
    bot.handlers.on_speech_started()
    assert bot.chat_origin is None
