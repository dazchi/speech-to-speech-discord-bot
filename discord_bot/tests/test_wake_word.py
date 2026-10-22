import shutil
import subprocess

import numpy as np
import pytest
from discord_bot.config import DEFAULT_WAKE_ALIASES, DEFAULT_WAKE_WORDS
from discord_bot.wake_word import build_wake_pattern, find_wake, speech_seconds, wake_context

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
        ("hialex 你知道战斗陀螺吗？", "你知道战斗陀螺吗"),  # greeting glued to the name
        ("嘿alex,你现在心情怎么样？", "你现在心情怎么样"),
        ("Hello, hello. Hey Alex. Hi Alex.", "Hi Alex"),
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


def test_speech_gate_ignores_clicks_but_not_short_words():
    rng = np.random.default_rng(0)
    click = np.zeros(16000, np.float32)
    click[:3200] = rng.normal(0, 0.3, 3200)  # 0.2 s burst
    word = np.zeros(16000, np.float32)
    word[:7200] = rng.normal(0, 0.05, 7200)  # 0.45 s of quiet speech
    assert speech_seconds(click) < 0.3 <= speech_seconds(word)


def test_wake_context_writes_phrases_out():
    assert wake_context(["hey alex", "ok alex", "艾利同學", " "]) == "Hey Alex, OK Alex, 艾利同學"


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


@pytest.mark.parametrize(
    "text",
    [
        "我觉得要再调一下，你不然说 Hi Alex 好了。",  # talking about the bot, not to it
        "你你你等下试着叫那个 Hey Alex，然后看他会不会回。",
        "Hey Alex, OK Alex, Hi Alex, 艾丽同学。",  # the recognizer repeating its context on noise
        "是的，OK Alex, Hi Alex, 艾丽同学。",
    ],
)
def test_mentions_and_context_echoes_do_not_wake(matcher, text):
    assert matcher.find(text) is None


def test_repeating_the_same_wake_phrase_still_wakes(matcher):
    assert matcher.find("Hi Alex, Hi Alex, Hey Alex.") is not None


@pytest.mark.parametrize("voice", ["Meijia", "Tingting"])
@pytest.mark.parametrize("phrase", ["艾丽同学，现在几点？", "艾丽同学"])
def test_detects_spoken_chinese_wake_phrase(recognizer, tmp_path, voice, phrase):
    text = recognizer.transcribe(_speak(tmp_path, phrase, voice))
    assert recognizer.find_wake(text) is not None, text


@pytest.mark.parametrize("text", ["我今天头好痛", "同学们大家好"])
def test_ignores_spoken_chinese_speech(recognizer, tmp_path, text):
    heard = recognizer.transcribe(_speak(tmp_path, text, "Meijia"))
    assert recognizer.find_wake(heard) is None, heard


@pytest.fixture(scope="module")
def speech_detector():
    pytest.importorskip("sherpa_onnx")
    from discord_bot.config import BOT_DIR
    from discord_bot.wake_word import SpeechDetector

    return SpeechDetector(BOT_DIR / "models")


def test_speech_detector_hears_speech_but_not_noise(speech_detector, tmp_path):
    rng = np.random.default_rng(0)
    assert speech_detector.speech_seconds(rng.normal(0, 0.05, 32000).astype(np.float32)) < 0.3  # loud hiss
    assert speech_detector.speech_seconds(np.zeros(32000, np.float32)) == 0
    if shutil.which("say") is None or shutil.which("afconvert") is None:
        pytest.skip("needs macOS say/afconvert")
    assert speech_detector.speech_seconds(_speak(tmp_path, "Hey Alex")) >= 0.3
