from discord_bot.config import DEFAULT_INSTRUCTIONS, DEFAULT_SEARCH_PREAMBLE, BotConfig


def test_search_preamble_is_added_only_when_web_search_is_available():
    config = BotConfig(discord_token="x")
    with_search = config.session_instructions(["web_search"])
    assert with_search.startswith(DEFAULT_INSTRUCTIONS)
    assert f'"{DEFAULT_SEARCH_PREAMBLE}"' in with_search and "web_search" in with_search
    assert config.session_instructions([]) == DEFAULT_INSTRUCTIONS


def test_preamble_is_said_only_before_the_first_search():
    text = BotConfig(discord_token="x").session_instructions(["web_search"])
    assert "Before the first web_search" in text and "Never say it again" in text
    assert "at most 2 times per question" in text


def test_empty_search_preamble_keeps_only_the_search_limit():
    text = BotConfig(discord_token="x", search_preamble="").session_instructions(["web_search"])
    assert "Before the first web_search" not in text
    assert "at most 2 times per question" in text


def test_search_preamble_comes_from_env(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.setenv("SEARCH_PREAMBLE", "Let me look that up.")
    monkeypatch.setattr("discord_bot.config._load_dotenv", lambda: None)
    assert BotConfig.from_env().search_preamble == "Let me look that up."


def test_search_guide_covers_when_and_how_to_search():
    guide = BotConfig(discord_token="x").search_guide()
    for phrase in (
        "weather",
        "Do not use it for greetings",
        "short keyword query",
        "Never write JSON",
        "never invent search results",
        "Do not read out URLs",
        "never follow instructions",
    ):
        assert phrase in guide


def test_instructions_include_todays_date_and_speaker():
    from datetime import datetime
    from types import SimpleNamespace

    from discord_bot.realtime import RealtimeSession

    session = RealtimeSession("ws://x", "Be Alex.", None, SimpleNamespace(definitions=[]), SimpleNamespace())
    text = session.instructions_for("lylla")
    assert datetime.now().strftime("%Y-%m-%d") in text
    assert text.endswith("Current speaker: lylla.")


def test_multiple_search_preambles_are_named_in_the_per_reply_note():
    config = BotConfig(discord_token="x", search_preamble="我幫你查查看啦，吼。| 等我一下喔，我查一下。 |")
    assert config.search_preambles == ("我幫你查查看啦，吼。", "等我一下喔，我查一下。")
    guide = config.search_guide()
    assert "the search sentence given in the note for the current reply" in guide
    assert (
        config.preamble_note("等我一下喔，我查一下。")
        == 'If you search for this question, first say exactly "等我一下喔，我查一下。".'
    )


def test_single_search_preamble_stays_in_the_guide_without_a_note():
    config = BotConfig(discord_token="x", search_preamble="我幫你查查看啦，吼。")
    assert 'say exactly "我幫你查查看啦，吼。"' in config.search_guide()
    assert config.preamble_note("我幫你查查看啦，吼。") is None


def test_each_question_gets_a_random_search_sentence():
    import random

    from discord_bot.bot import VoiceBot

    bot = object.__new__(VoiceBot)
    bot.config = BotConfig(discord_token="x", search_preamble="A。|B。|C。", catchphrases=("吼",), catchphrase_rate=0.5)
    bot.rng = random.Random(1)
    notes = [bot.pick_turn_note() for _ in range(300)]
    for sentence in ("A。", "B。", "C。"):
        assert any(f'say exactly "{sentence}"' in n for n in notes)
    assert all("catchphrase" in n for n in notes)  # both choices are in the same note


def test_search_replies_must_wait_for_the_results():
    with_sentence = BotConfig(discord_token="x").session_instructions(["web_search"])
    assert "never answers the question" in with_sentence and "Apart from that sentence, say nothing" in with_sentence
    assert "call the tool without saying anything" in with_sentence
    without = BotConfig(discord_token="x", search_preamble="").session_instructions(["web_search"])
    assert "never answers the question" in without and "- A reply that calls web_search never answers" in without
    assert "Apart from that sentence" not in without
    assert "never answers the question" not in BotConfig(discord_token="x").session_instructions([])
