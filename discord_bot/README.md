# Discord voice bot ("Hey Alex")

A Discord bot that sits in one voice channel and talks back using this project's
speech-to-speech server. It only listens after a wake word:

- **Anyone** in the channel can say **"Hey Alex"**, **"OK Alex"**, **"Hi Alex"** or **「艾利同學」**.
  Chinese wake phrases are matched by pronunciation (pinyin), so 艾丽同学, 爱丽同学 or 艾莉同學 all work.
- The bot plays a chime and then listens **only to that person** until the exchange is over:
  the window stays open while they talk and while Alex is thinking or speaking, and closes
  `FOLLOWUP_SECONDS` (default 8 s) after Alex's voice **finishes playing** in the channel, so
  follow-up questions need no wake word. A falling chime means it went back to sleep. Set
  `FOLLOWUP_SECONDS=0` to require the wake word for every question.
- **Typing works too:** mention the bot in any text channel of the server, e.g.
  `@AI Alex 今天台北天氣如何？`. It reacts with 👀, answers **out loud in the voice channel**,
  and also replies with the text (`MENTION_TEXT_REPLY=false` for voice only). Questions typed
  while someone is talking to Alex get ⏳ and are answered when that conversation ends.
- Everyone shares **one conversation**: the bot keeps a single session with the server, and
  tells the model who is speaking each time.
- Nothing is sent to the speech server until someone says the wake word. Wake-word detection
  runs locally: each short utterance is transcribed with
  [Qwen3-ASR 1.7B](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit) on MLX (~0.2 s per
  utterance, ~2.5 GB, downloaded on first start), given the wake phrases as context, and matched
  against them. The wake phrase has to come first, and very short sounds are ignored. Set
  `WAKE_MODEL=mlx-community/Qwen3-ASR-0.6B-8bit` for a smaller model, or `WAKE_MODEL=sensevoice` for
  [SenseVoice](https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html) on CPU (~50 ms, but it
  missed about 4 in 10 wake phrases in real Discord audio). "Hey Alex, what time is it?" is sent to the
  server whole; a bare "Hey Alex" just opens the window and waits for the question.

```
Discord voice channel ──per-user audio──▶ bot ──wake word?──▶ speech-to-speech server
        ▲                                   │                  (VAD → STT → LLM → TTS)
        └──────────── reply audio ◀─────────┘◀──────────────────────────┘
```

## Setup

1. **Create the bot** in the [Discord Developer Portal](https://discord.com/developers/applications):
   New Application → Bot → copy the token. Invite it with the `bot` and
   `applications.commands` scopes and the **View Channels**, **Connect**, **Speak**, **Use Voice
   Activity**, **Send Messages**, **Read Message History** and **Add Reactions** permissions (the
   invite link the bot prints at startup includes them).
2. **Install** (Python 3.11, separate from the server's environment):
   ```bash
   brew install opus            # macOS; Linux: apt install libopus0
   conda create -n s2s-discord python=3.11 && conda activate s2s-discord
   pip install -r discord_bot/requirements.txt
   ```
3. **Configure:** `cp discord_bot/.env.example discord_bot/.env` and set `DISCORD_TOKEN` and
   `DISCORD_GUILD_ID` (see the file for the other options). Never commit `.env`.
4. **Start the speech server** in another terminal, with `serve`, for example:
   ```bash
   speech-to-speech serve --mac-optimal-settings --stt mlx-audio-qwen3-asr \
     --mlx_audio_qwen3_asr_prompt "Alex, 艾丽同学, Discord" \
     --model_name mlx-community/Qwen3.6-35B-A3B-4bit
   ```
   `mlx-audio-qwen3-asr` transcribes Mandarin from Discord much more accurately than
   `mlx-audio-whisper`, and is as fast.
5. **Run the bot** from the repository root:
   ```bash
   python -m discord_bot
   ```
6. Join a voice channel in that server. With `AUTO_JOIN=true` (the default) the bot joins the
   voice channel with the most people and follows it as people move, but never in the middle of
   a conversation; it leaves when everyone is gone. Say "Hey Alex, ...".
   `/join` pulls it into your channel; `/leave` makes it leave and pauses auto-join until the next `/join`.

## Notes

- **Voice receive and Discord's DAVE encryption.** Since March 2026 Discord requires DAVE
  end-to-end encryption for all calls. `discord.py` 2.7 supports it for sending audio, but
  no released Python library decrypts *received* audio yet, so `requirements.txt` pins
  `discord-ext-voice-recv` to [PR #54](https://github.com/imayhaveborkedit/discord-ext-voice-recv/pull/54).
  Switch to the official release once it is merged.
- **One server session.** The speech server serves one session at a time, so close the
  browser demo and don't run `speech-to-speech local` while the bot is connected. The bot
  reconnects automatically if the server restarts.
- **Greetings.** When someone (not a bot) joins the bot's voice channel, Alex greets them by name in one
  short sentence. Greetings are not added to the conversation history, wait until Alex is free (or are
  skipped after 20 s), and the same person is not greeted again within `GREET_COOLDOWN_SECONDS` (300).
  Customize with `GREETING_PROMPT` (use `{name}`), or turn off with `GREET_ON_JOIN=false`.
- **Catchphrases.** Give Alex pet phrases with `CATCHPHRASES=我跟你說喔|真的假的` (phrases are separated by `|`, as in `SEARCH_PREAMBLE`, so they can contain commas). The bot decides for
  each reply whether to use one (`CATCHPHRASE_RATE`, default 0.3 = about one reply in three) and tells
  the model, so they are not repeated in every answer.
- **Interruptions.** By default the active speaker can cut Alex off by talking. Set
  `ALLOW_INTERRUPTIONS=false` to make Alex finish: while it is thinking, searching or speaking,
  the speaker's audio is not sent to the server, so it neither stops the reply nor queues an
  extra question. Web searches are never interrupted, even with `ALLOW_INTERRUPTIONS=true`: from the
  tool call until the answer is ready, what the speaker says is dropped; the answer itself can be
  talked over as usual.
- **Tools.** `web_search` is enabled when `SERPER_API_KEY` is set. Add your own modules with
  `TOOL_MODULES` using the same format as the server's `--tool-module`
  (`TOOLS` + async `execute_tool(name, arguments)`).
- **Tuning the wake word.** Set `AUDIO_DEBUG=1` to log what the bot hears from each person
  (`Heard <name> say '...'`) and record it to `discord_bot/debug_audio/`. If the name is
  misheard consistently (e.g. "Alec"), add that spelling to `WAKE_ALIASES`. Change the phrases
  with `WAKE_WORDS`; Chinese forms such as 嘿艾力克斯 are matched too.

## Tests

```bash
conda activate s2s-discord
python -m pytest discord_bot/tests
```
The wake-word tests synthesize speech with macOS `say` and are skipped elsewhere.
