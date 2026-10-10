"""Discord voice bot: wake word -> speech-to-speech server -> voice reply.

Run from the repository root:  python -m discord_bot
"""

from __future__ import annotations

import asyncio
import ctypes.util
import logging
import random
import time
from collections.abc import Awaitable, Callable

import discord
import numpy as np
from discord import app_commands
from discord.ext import voice_recv

from discord_bot.audio import WAKE_RATE, InputConverter, OutputConverter, chime, silence_server
from discord_bot.channels import busiest_channel
from discord_bot.chat import chat_label, clean_question, split_message
from discord_bot.config import BotConfig
from discord_bot.conversation import ConversationController
from discord_bot.debug import AudioDebugger
from discord_bot.playback import Speaker
from discord_bot.realtime import RealtimeSession
from discord_bot.tools import ToolRegistry
from discord_bot.wake_word import WakeWordRecognizer

logger = logging.getLogger("discord_bot")

PUMP_INTERVAL_S = 0.05
# Discord stops sending packets while someone is silent; a gap this long ends an utterance.
UTTERANCE_GAP_S = 0.35
# Long utterances are also checked after this much audio, so the wake word is not held back.
EARLY_CHECK_S = 3.0
# An utterance this long without the wake word is dropped (the wake word must come first).
MAX_UTTERANCE_S = 15.0
REBALANCE_DELAY_S = 2.0
# A greeting waits at most this long for Alex to be free, then it is skipped.
GREET_WAIT_S = 20.0
# When the bot moves into a channel, greet people who joined it this recently.
GREET_RECENT_JOIN_S = 15.0


class UserAudio:
    """Per-user audio state: format converters and the current utterance (for wake-word checks)."""

    def __init__(self) -> None:
        self.converter = InputConverter()
        self.last_packet = 0.0
        self.wake_audio: list[np.ndarray] = []  # 16 kHz float32, for transcription
        self.wake_samples = 0
        self.server_audio = bytearray()  # the same utterance as 24 kHz PCM, forwarded on wake
        self.early_checked = False
        self.checking = False

    def append(self, server_pcm: bytes, wake_samples: np.ndarray) -> None:
        self.wake_audio.append(wake_samples)
        self.wake_samples += len(wake_samples)
        self.server_audio.extend(server_pcm)

    def consume(self, wake_samples: int, server_bytes: int) -> None:
        """Drop the first part of the utterance (already checked); keep audio that arrived since."""
        rest = np.concatenate(self.wake_audio)[wake_samples:] if self.wake_audio else np.zeros(0, np.float32)
        self.wake_audio = [rest] if len(rest) else []
        self.wake_samples = len(rest)
        del self.server_audio[:server_bytes]
        self.early_checked = False

    def clear(self) -> None:
        self.consume(self.wake_samples, len(self.server_audio))


class VoiceBot(discord.Client):
    def __init__(self, config: BotConfig) -> None:
        intents = discord.Intents.default()
        intents.voice_states = True
        super().__init__(intents=intents)
        self.config = config
        self.tree = app_commands.CommandTree(self)
        self.wake = WakeWordRecognizer(config.models_dir, config.wake_words, config.wake_aliases)
        self.controller = ConversationController(wake_timeout_s=config.wake_timeout_s, followup_s=config.followup_s)
        self.speaker = Speaker(lambda: self.voice, on_stopped=self.controller.playback_finished)
        self.handlers = SessionHandlers(self)
        tools = ToolRegistry(config.tool_modules)
        self.session = RealtimeSession(
            config.s2s_url,
            config.session_instructions([t["name"] for t in tools.definitions]),
            config.voice,
            tools,
            self.handlers,
            max_tool_rounds=config.max_tool_rounds,
            allow_interruptions=config.allow_interruptions,
            answer_instructions=config.session_instructions([]),
        )
        self.voice: voice_recv.VoiceRecvClient | None = None
        self.user_audio: dict[int, UserAudio] = {}
        self.names: dict[int, str] = {}
        self.audio_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        self.output = OutputConverter()
        self._rebalance_task: asyncio.Task[None] | None = None
        self._rebalance_pending = False
        self._auto_join_paused = False  # set by /leave, cleared by /join
        self._switching = asyncio.Lock()  # one voice-channel switch at a time
        self.debug = AudioDebugger(config.debug_dir) if config.audio_debug else None
        # Questions typed in text channels by mentioning the bot, answered one at a time.
        self.chat_queue: asyncio.Queue[tuple[discord.Message, str]] = asyncio.Queue()
        self.chat_origin: discord.Message | None = None  # where to post the text of the current answer
        # Greeting people who join the voice channel.
        self.greet_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        self._last_greeted: dict[int, float] = {}
        self._joined_at: dict[int, float] = {}
        self._register_commands()

    # ---- startup and commands -------------------------------------------------------------
    async def setup_hook(self) -> None:
        self.speaker.loop = self.loop
        self.loop.create_task(self.session.run())
        self.loop.create_task(_keep_running("audio worker", self._audio_worker))
        self.loop.create_task(_keep_running("pump", self._pump))
        self.loop.create_task(_keep_running("chat worker", self._chat_worker))
        self.loop.create_task(_keep_running("greeter", self._greet_worker))

    def _serves(self, guild: discord.Guild | None) -> bool:
        return guild is not None and (self.config.guild_id is None or guild.id == self.config.guild_id)

    async def on_ready(self) -> None:
        guilds = [g for g in self.guilds if self._serves(g)]
        for guild in guilds:
            await self._sync_commands(guild)
        logger.info("Logged in as %s, serving %s", self.user, [g.name for g in guilds])
        if self.config.guild_id is not None and not guilds:
            logger.warning("The bot is not in server %s yet; invite it with the link below.", self.config.guild_id)
        logger.info("Invite link: %s", self.invite_url())
        self._schedule_rebalance(delay=0)

    def invite_url(self) -> str:
        """OAuth2 link that adds the bot with the voice permissions it needs."""
        assert self.user is not None
        permissions = discord.Permissions(
            view_channel=True,
            connect=True,
            speak=True,
            use_voice_activation=True,
            send_messages=True,
            read_message_history=True,
            add_reactions=True,
        )
        guild = discord.Object(self.config.guild_id) if self.config.guild_id is not None else discord.utils.MISSING
        return discord.utils.oauth_url(
            self.application_id or self.user.id,
            permissions=permissions,
            guild=guild,
            scopes=("bot", "applications.commands"),
        )

    # ---- questions typed in chat ----------------------------------------------------------
    async def on_message(self, message: discord.Message) -> None:
        """ "@AI Alex <question>" in a text channel: answered out loud (and as a chat reply)."""
        if message.author.bot or self.user is None or not self._serves(message.guild):
            return
        if not any(user.id == self.user.id for user in message.mentions):
            return
        names = {user.id: getattr(user, "display_name", user.name) for user in message.mentions}
        question = clean_question(message.content, self.user.id, names)
        if not question:
            await _try(message.reply("Ask me something after the mention, e.g. `@Alex what time is it?`"))
            return
        await _try(message.add_reaction("👀"))
        if not self.controller.is_idle or not self.chat_queue.empty():
            await _try(message.add_reaction("⏳"))  # someone is talking to Alex; answered afterwards
        self.chat_queue.put_nowait((message, question))

    async def _chat_worker(self) -> None:
        while True:
            message, question = await self.chat_queue.get()
            name = getattr(message.author, "display_name", message.author.name)
            # Wait until nobody is talking to Alex by voice, then take the floor like a wake word.
            while not (self.session.connected and self.controller.is_idle):
                await asyncio.sleep(0.2)
            self.controller.wake(message.author.id, name)
            self.handlers.speaker_announced = False
            self.chat_origin = message if self.config.mention_text_reply else None
            logger.info("Chat question from %s: %s", name, question)
            await self.session.set_speaker(name)
            await self.session.set_turn_note(self.pick_turn_note())
            await self.session.ask_text(chat_label(name, question))

    def pick_turn_note(self) -> str | None:
        """Per-reply choices: whether to use a catchphrase (CATCHPHRASE_RATE), and which search sentence."""
        rng = getattr(self, "rng", random)
        notes = [self.config.catchphrase_note(rng.random() < self.config.catchphrase_rate)]
        if self.config.search_preambles:
            notes.append(self.config.preamble_note(rng.choice(self.config.search_preambles)))
        return "\n".join(n for n in notes if n) or None

    async def reply_in_chat(self, text: str) -> None:
        origin = self.chat_origin
        if origin is None or not text.strip():
            return
        for i, chunk in enumerate(split_message(text)):
            await _try(origin.reply(chunk, mention_author=False) if i == 0 else origin.channel.send(chunk))

    async def on_guild_join(self, guild: discord.Guild) -> None:
        if not self._serves(guild):
            return
        await self._sync_commands(guild)
        logger.info("Added to server %s", guild.name)
        self._schedule_rebalance(delay=0)

    async def on_voice_state_update(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
    ) -> None:
        if not self._serves(member.guild):
            return
        if self.user is not None and member.id == self.user.id:
            # Our own moves come from join()/leave(); only react to being disconnected from outside.
            if after.channel is None and not self._switching.locked():
                self._forget_voice()
                self._schedule_rebalance()
            return
        if not member.bot and after.channel is not None and after.channel != before.channel:
            self._joined_at[member.id] = time.monotonic()
            if self.voice is not None and after.channel == self.voice.channel:
                self._queue_greeting(member)
        self._schedule_rebalance()

    # ---- greeting people who join ----------------------------------------------------------
    def _queue_greeting(self, member: discord.abc.User) -> None:
        if not self.config.greet_on_join or getattr(member, "bot", False):
            return
        now = time.monotonic()
        if now - self._last_greeted.get(member.id, -1e9) < self.config.greet_cooldown_s:
            return  # greeted recently (e.g. a quick leave and rejoin)
        self._last_greeted[member.id] = now
        self.greet_queue.put_nowait((member.id, getattr(member, "display_name", None) or member.name))

    async def _greet_worker(self) -> None:
        while True:
            _user_id, name = await self.greet_queue.get()
            if not await self._wait_until_free(GREET_WAIT_S):
                logger.info("Skipped greeting %s: Alex was busy", name)
                continue
            logger.info("Greeting %s", name)
            await self.session.greet(name, self.config.greeting_instruction(name))
            await asyncio.sleep(1.0)  # let the greeting start before checking for the next one
            await self._wait_until_free(GREET_WAIT_S)

    async def _wait_until_free(self, timeout: float) -> bool:
        """Wait until nobody is talking to Alex and it is not speaking; False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.session.connected and self.controller.is_idle and not self.controller.assistant_busy:
                return True
            await asyncio.sleep(0.2)
        return False

    # ---- following the busiest channel -----------------------------------------------------
    def _schedule_rebalance(self, delay: float = REBALANCE_DELAY_S) -> None:
        """Re-pick the voice channel shortly after people move (debounced)."""
        if not self.config.auto_join or self._auto_join_paused:
            return
        # Only the waiting period is debounced; a switch that already started is never cancelled.
        if self._rebalance_task is not None:
            self._rebalance_task.cancel()
        self._rebalance_task = self.loop.create_task(self._rebalance_after(delay))

    async def _rebalance_after(self, delay: float) -> None:
        await asyncio.sleep(delay)
        self.loop.create_task(self._rebalance())

    async def _rebalance(self) -> None:
        async with self._switching:
            await self._rebalance_locked()

    async def _rebalance_locked(self) -> None:
        if not self.controller.is_idle:
            self._rebalance_pending = True  # never move in the middle of a conversation
            return
        self._rebalance_pending = False
        guild = next((g for g in self.guilds if self._serves(g)), None)
        if guild is None:
            return
        me = guild.me
        counts = {
            channel.id: sum(not m.bot for m in channel.members)
            for channel in guild.voice_channels
            if channel.permissions_for(me).connect
        }
        current = self.voice.channel.id if self.voice is not None and self.voice.channel else None
        target = busiest_channel(counts, current)
        try:
            if target is None:
                if self.voice is not None:
                    logger.info("Everyone left; leaving the voice channel")
                    await self.leave()
            elif target != current:
                channel = guild.get_channel(target)
                if isinstance(channel, discord.VoiceChannel):
                    await self.join(channel)
        except (discord.DiscordException, asyncio.TimeoutError) as exc:
            logger.warning("Could not switch voice channel: %s", exc)

    async def _sync_commands(self, guild: discord.Guild) -> None:
        # Per-server sync makes /join and /leave available immediately.
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)

    def _register_commands(self) -> None:
        @self.tree.command(name="join", description="Join your voice channel and listen for 'Hey Alex'.")
        async def join_cmd(interaction: discord.Interaction) -> None:
            member = interaction.user
            channel = getattr(getattr(member, "voice", None), "channel", None)
            if channel is None:
                await interaction.response.send_message("Join a voice channel first.", ephemeral=True)
                return
            await interaction.response.defer(ephemeral=True)
            self._auto_join_paused = False
            async with self._switching:
                await self.join(channel)
            await interaction.followup.send(f'Joined **{channel.name}**. Say "Hey Alex" to talk to me.', ephemeral=True)

        @self.tree.command(name="leave", description="Leave the voice channel.")
        async def leave_cmd(interaction: discord.Interaction) -> None:
            self._auto_join_paused = True
            async with self._switching:
                await self.leave()
            note = " Use /join to bring me back." if self.config.auto_join else ""
            await interaction.response.send_message(f"Left the voice channel.{note}", ephemeral=True)

    async def join(self, channel: discord.VoiceChannel) -> None:
        if self.voice is not None:
            if self.voice.channel == channel and self.voice.is_connected():
                return
            await self.leave()
        stale = channel.guild.voice_client
        if stale is not None:  # left over from an interrupted connection attempt
            await stale.disconnect(force=True)
        self.voice = await channel.connect(cls=voice_recv.VoiceRecvClient)
        self.voice.listen(voice_recv.BasicSink(self._on_voice_packet))
        logger.info("Listening in #%s", channel.name)
        # The bot followed people here: greet those who just arrived (not everyone already sitting there).
        now = time.monotonic()
        for member in channel.members:
            if now - self._joined_at.get(member.id, -1e9) < GREET_RECENT_JOIN_S:
                self._queue_greeting(member)

    async def leave(self) -> None:
        if self.voice is not None:
            await self.voice.disconnect()
        self._forget_voice()

    def _forget_voice(self) -> None:
        self.voice = None
        self._release_active(chime_down=False)
        self.user_audio.clear()

    # ---- receiving audio ------------------------------------------------------------------
    def _on_voice_packet(self, user: discord.abc.User | None, data: voice_recv.VoiceData) -> None:
        """Runs on the voice receive thread: hand frames to the event loop."""
        if user is None:
            if self.debug is not None:  # sender not identified yet: cannot decrypt or attribute the audio
                self.loop.call_soon_threadsafe(self.debug.empty_frame, 0)
            return
        if user.bot:
            return
        if not data.pcm:
            if self.debug is not None:
                self.loop.call_soon_threadsafe(self.debug.empty_frame, user.id)
            return
        self.names[user.id] = getattr(user, "display_name", None) or user.name
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, (user.id, data.pcm))

    async def _audio_worker(self) -> None:
        while True:
            user_id, pcm = await self.audio_queue.get()
            state = self.user_audio.get(user_id)
            if state is None:
                state = self.user_audio[user_id] = UserAudio()
            server_pcm, wake_samples = state.converter.convert(pcm)
            if self.debug is not None:
                self.debug.frame(user_id, self.names.get(user_id, str(user_id)), wake_samples)
            state.last_packet = time.monotonic()
            await self._route(user_id, state, server_pcm, wake_samples)

    async def _route(self, user_id: int, state: UserAudio, server_pcm: bytes, wake_samples: np.ndarray) -> None:
        if self.controller.is_active_user(user_id):
            if self.config.allow_interruptions or not self.controller.assistant_busy:
                await self.session.send_audio(server_pcm)
            # else: ALLOW_INTERRUPTIONS=false and Alex is thinking or speaking: the server never hears it
            return
        if not self.controller.is_idle:
            state.clear()  # someone else has the floor
            return
        state.append(server_pcm, wake_samples)
        if state.wake_samples > MAX_UTTERANCE_S * WAKE_RATE and not state.checking:
            state.clear()
        elif not state.early_checked and not state.checking and state.wake_samples >= EARLY_CHECK_S * WAKE_RATE:
            state.early_checked = True
            self._start_wake_check(user_id, state, final=False)

    def _start_wake_check(self, user_id: int, state: UserAudio, *, final: bool) -> None:
        state.checking = True  # set before scheduling so the pump cannot start a second check
        task = self.loop.create_task(self._check_wake(user_id, state, final=final))
        task.add_done_callback(_log_task_error)

    async def _check_wake(self, user_id: int, state: UserAudio, *, final: bool) -> None:
        """Transcribe the utterance so far and wake up if it starts with the wake phrase."""
        try:
            checked_wake, checked_server = state.wake_samples, len(state.server_audio)
            if checked_wake == 0:
                return
            samples = np.concatenate(state.wake_audio)[:checked_wake]
            text = await asyncio.to_thread(self.wake.transcribe, samples)
        finally:
            state.checking = False
        name = self.names.get(user_id, str(user_id))
        remainder = self.wake.find_wake(text)
        if self.debug is not None or remainder is not None:
            logger.info("Heard %s say %r%s", name, text, " -> wake word" if remainder is not None else "")
        if remainder is not None and self.controller.is_idle:
            await self._on_wake(user_id, state, name, send_utterance=bool(remainder))
        elif final:
            state.consume(checked_wake, checked_server)

    async def _on_wake(self, user_id: int, state: UserAudio, name: str, *, send_utterance: bool) -> None:
        if not self.session.connected:
            logger.warning("Wake word from %s, but the speech server is not connected", name)
            state.clear()
            return
        if not self.controller.wake(user_id, name):
            return
        logger.info("Listening to %s", name)
        self.handlers.speaker_announced = False  # a new person: label their next turn
        await self.session.set_speaker(name)
        await self.session.clear_input()
        self.speaker.play(chime(rising=True))
        self.speaker.finish()
        # "Hey Alex, what time is it?" is sent whole; a bare "Hey Alex" just opens the window.
        if send_utterance:
            await self.session.send_audio(bytes(state.server_audio))
        state.clear()

    async def _pump(self) -> None:
        """Feeds silence during pauses and closes the conversation window when it expires."""
        while True:
            await asyncio.sleep(PUMP_INTERVAL_S)
            now = time.monotonic()
            for user_id, state in list(self.user_audio.items()):
                gap = now - state.last_packet
                if self.controller.is_active_user(user_id):
                    if gap >= PUMP_INTERVAL_S * 2:  # let the server's VAD see the pause
                        await self.session.send_audio(silence_server(PUMP_INTERVAL_S))
                elif gap >= UTTERANCE_GAP_S and state.wake_samples and not state.checking:
                    self._start_wake_check(user_id, state, final=True)
            if self.debug is not None:
                self.debug.maybe_report({0: "unknown speaker", **self.names})
            self._ensure_listening()
            if self.controller.playing and not self.speaker.is_playing:
                self.controller.playback_finished()  # safety net: never wait on a stream that is gone
            released = self.controller.expire()
            if released is not None:
                self._window_closed(released.name, chime_down=True)

    def _ensure_listening(self) -> None:
        """Self-heal: if we are in a voice channel but no longer receiving audio, listen again."""
        vc = self.voice
        if vc is not None and vc.is_connected() and not vc.is_listening():
            logger.warning("Voice receive had stopped; listening again")
            vc.listen(voice_recv.BasicSink(self._on_voice_packet))

    def _release_active(self, *, chime_down: bool) -> None:
        released = self.controller.active
        self.controller.reset()
        if released is not None:
            self._window_closed(released.name, chime_down=chime_down)
        elif self._rebalance_pending:
            self._schedule_rebalance()

    def _window_closed(self, name: str, *, chime_down: bool) -> None:
        logger.info("Conversation window with %s closed; waiting for the wake word", name)
        self.chat_origin = None
        if chime_down:
            self.speaker.play(chime(rising=False))
            self.speaker.finish()
        if self._rebalance_pending:
            self._schedule_rebalance()


class SessionHandlers:
    """Speech-server callbacks, kept off ``discord.Client`` so they cannot clash with ``on_<event>`` dispatch."""

    def __init__(self, bot: VoiceBot) -> None:
        self.bot = bot
        # True once "[name is speaking]" was added for the turn now starting; cleared when the
        # turn's transcript arrives, so a false start does not leave duplicate labels behind.
        self.speaker_announced = False

    def on_connected(self) -> None:
        self.bot.controller.reset()
        self.speaker_announced = False

    def on_speech_started(self) -> None:
        self.bot.controller.speech_started()
        self.bot.chat_origin = None  # a spoken turn: answers are no longer posted to the chat
        if self.bot.config.allow_interruptions:
            self.bot.speaker.stop()  # barge-in
        active = self.bot.controller.active
        if active is not None and not self.speaker_announced:
            self.speaker_announced = True
            task = self.bot.loop.create_task(self._start_turn(active.name))
            task.add_done_callback(_log_task_error)

    async def _start_turn(self, name: str) -> None:
        await self.bot.session.set_turn_note(self.bot.pick_turn_note())
        await self.bot.session.announce_speaker(name)

    def on_speech_stopped(self) -> None:
        self.bot.controller.speech_stopped()

    def on_response_created(self) -> None:
        self.bot.controller.response_started()

    def on_response_done(self, *, tools_pending: bool) -> None:
        # The server does not send output_audio.done for every response (e.g. cancelled ones),
        # so close the audio stream here too; otherwise it would wait for more audio forever.
        self.bot.speaker.finish()
        self.bot.controller.response_finished(tools_pending=tools_pending)

    def on_tools_done_without_follow_up(self) -> None:
        self.bot.controller.tools_done_without_follow_up()

    def on_audio(self, pcm: bytes) -> None:
        if self.bot.speaker.play(self.bot.output.convert(pcm)):
            self.bot.controller.playback_started()  # the follow-up timer waits for playback to end

    def on_audio_done(self) -> None:
        self.bot.speaker.finish()

    def on_user_transcript(self, text: str) -> None:
        self.speaker_announced = False
        active = self.bot.controller.active
        logger.info("%s: %s", active.name if active else "user", text)

    def on_assistant_transcript(self, text: str) -> None:
        logger.info("Alex: %s", text)
        if self.bot.chat_origin is not None:
            task = self.bot.loop.create_task(self.bot.reply_in_chat(text))
            task.add_done_callback(_log_task_error)


async def _keep_running(name: str, loop_fn: Callable[[], Awaitable[None]]) -> None:
    """Run a background loop forever; log and restart it if it crashes instead of dying silently."""
    while True:
        try:
            await loop_fn()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("The %s crashed; restarting it", name)
            await asyncio.sleep(1.0)


async def _try(action: Awaitable[object]) -> None:
    """Chat side effects (replies, reactions) must never break the bot, e.g. without permission."""
    try:
        await action
    except discord.DiscordException as exc:
        logger.warning("Could not post to the chat: %s", exc)


def _log_task_error(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.error("Background task failed", exc_info=task.exception())


def load_opus(path: str | None) -> None:
    if discord.opus.is_loaded():
        return
    candidates = [path, ctypes.util.find_library("opus"), "/opt/homebrew/lib/libopus.dylib", "libopus.so.0"]
    for candidate in filter(None, candidates):
        try:
            discord.opus.load_opus(candidate)
            return
        except OSError:
            continue
    raise SystemExit("Could not load the Opus library. Install it (macOS: brew install opus) or set OPUS_LIBRARY.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("discord.ext.voice_recv").setLevel(logging.WARNING)
    config = BotConfig.from_env()
    load_opus(config.opus_library)
    VoiceBot(config).run(config.discord_token, log_handler=None)
