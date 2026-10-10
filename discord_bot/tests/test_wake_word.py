import shutil
import subprocess

import numpy as np
import pytest
from discord_bot.config import DEFAULT_WAKE_ALIASES, DEFAULT_WAKE_WORDS
from discord_bot.wake_word import build_wake_pattern, find_wake

PATTERN = build_wake_pattern(DEFAULT_WAKE_WORDS, DEFAULT_WAKE_ALIASES)


@pytest.mark.parametrize(
    ("text", "remainder"),
    [
        ("Hey, Alex, what time is it?", "what time is it"),
        ("Hi, Alex.", ""),
        ("OK Alex play some music", "play some music"),
        ("okay alec, help me", "help me"),
        ("Hello Alex", ""),
        ("嘿艾力克斯，现在几点？", "现在几点"),
        ("嗨，亞歷克斯", ""),
        ("Hey 艾力克斯 what's up", "what's up"),
    ],
)
def test_wake_phrases_match_and_return_the_rest(text, remainder):
    assert find_wake(PATTERN, text) == remainder


@pytest.mark.parametrize(
    "text",
    [
        "Hey Alexa, turn on the lights",
        "Alexa, turn on the lights.",
        "Hey Alice, come over here",
        "I talked to Alexander yesterday",
        "hey alexander",
        "Hey everyone, let's start",
        "我今天跟Alex吃饭",
        "So yesterday I was telling my friend about the game and then hey alex",
        "Heyex.",
    ],
)
def test_other_speech_does_not_wake(text):
    assert find_wake(PATTERN, text) is None


# ---- end to end on synthesized speech (macOS `say`); downloads the ~240 MB model once ----


@pytest.fixture(scope="module")
def recognizer():
    pytest.importorskip("sherpa_onnx")
    if shutil.which("say") is None or shutil.which("afconvert") is None:
        pytest.skip("needs macOS say/afconvert")
    from discord_bot.config import BOT_DIR
    from discord_bot.wake_word import WakeWordRecognizer

    return WakeWordRecognizer(BOT_DIR / "models", DEFAULT_WAKE_WORDS, DEFAULT_WAKE_ALIASES)


def _speak(tmp_path, text: str, voice: str = "Samantha") -> np.ndarray:
    import soundfile as sf

    aiff, wav = tmp_path / "s.aiff", tmp_path / "s.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
    return sf.read(wav, dtype="float32")[0]


@pytest.mark.parametrize("voice", ["Samantha", "Daniel", "Meijia"])
@pytest.mark.parametrize("phrase", ["Hey Alex", "OK Alex", "Hi Alex"])
def test_detects_spoken_wake_phrase(recognizer, tmp_path, phrase, voice):
    text = recognizer.transcribe(_speak(tmp_path, f"{phrase}, what's the weather today?", voice))
    assert recognizer.find_wake(text) is not None, text


@pytest.mark.parametrize(
    "text", ["Hey Alice, come over here.", "Alexa, turn on the lights.", "Hey everyone, let's go."]
)
def test_ignores_spoken_similar_phrases(recognizer, tmp_path, text):
    heard = recognizer.transcribe(_speak(tmp_path, text))
    assert recognizer.find_wake(heard) is None, heard


# ---- Chinese wake phrases, matched by pronunciation ----


@pytest.fixture(scope="module")
def matcher():
    pytest.importorskip("pypinyin")
    from discord_bot.wake_word import WakeMatcher

    return WakeMatcher(DEFAULT_WAKE_WORDS, DEFAULT_WAKE_ALIASES)


@pytest.mark.parametrize(
    ("text", "remainder"),
    [
        ("艾丽同学，现在几点？", "现在几点"),
        ("爱丽同学。", ""),  # different characters, same sound
        ("艾莉同學幫我查一下天氣", "幫我查一下天氣"),  # traditional characters
        ("嘿，艾丽同学，你好", "你好"),
        ("喂艾利同学", ""),
        ("Hey Alex, what time is it?", "what time is it"),  # English phrases still work
    ],
)
def test_chinese_wake_phrase_matches_by_sound(matcher, text, remainder):
    assert matcher.find(text) == remainder


@pytest.mark.parametrize("text", ["我今天头好痛", "同学们，大家好。", "我跟你说那个艾丽同学她今天没来", "Hey Alexa"])
def test_chinese_wake_phrase_needs_to_come_first(matcher, text):
    assert matcher.find(text) is None


@pytest.mark.parametrize("voice", ["Meijia", "Tingting"])
@pytest.mark.parametrize("phrase", ["艾丽同学，现在几点？", "艾丽同学"])
def test_detects_spoken_chinese_wake_phrase(recognizer, tmp_path, voice, phrase):
    text = recognizer.transcribe(_speak(tmp_path, phrase, voice))
    assert recognizer.find_wake(text) is not None, text


@pytest.mark.parametrize("text", ["我今天头好痛", "同学们大家好"])
def test_ignores_spoken_chinese_speech(recognizer, tmp_path, text):
    heard = recognizer.transcribe(_speak(tmp_path, text, "Meijia"))
    assert recognizer.find_wake(heard) is None, heard
