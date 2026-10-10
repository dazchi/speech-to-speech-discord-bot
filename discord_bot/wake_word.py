"""Wake-word detection by transcribing short utterances and matching the text.

A tiny keyword-spotting model missed real Discord voices (accents, Opus compression), so each
utterance is transcribed with SenseVoice (sherpa-onnx, CPU, ~50 ms per utterance; understands
English, Mandarin, Cantonese, Japanese and Korean) and the text is matched against the wake phrases:
English phrases by words, Chinese phrases by pronunciation (pinyin).
"""

from __future__ import annotations

import logging
import re
import tarfile
import threading
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from discord_bot.audio import WAKE_RATE

logger = logging.getLogger(__name__)

MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL_NAME}.tar.bz2"

# Chinese greetings and renderings of the name that a recognizer may produce for "Hey Alex".
CJK_GREETINGS = ("嘿", "嗨", "哈囉", "哈喽", "欸", "诶", "哎")
CJK_NAMES = ("艾力克斯", "艾利克斯", "亞歷克斯", "亚历克斯", "阿力克斯", "艾力斯", "艾力")
# How far into the utterance the wake phrase may start (characters of normalized text).
WAKE_SEARCH_CHARS = 40
# How far into the utterance a Chinese wake phrase may start (Chinese syllables).
WAKE_SEARCH_SYLLABLES = 2  # e.g. 「嘿，艾丽同学」, but not mid-sentence
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _has_cjk(text: str) -> bool:
    return _CJK.search(text) is not None


def _normalize_syllable(syllable: str) -> str:
    """Fold sounds that Taiwanese-accented Mandarin and recognizers often merge."""
    syllable = syllable.replace("ü", "v")
    for retroflex in ("zh", "ch", "sh"):
        if syllable.startswith(retroflex):
            syllable = syllable[0] + syllable[2:]
    if syllable.startswith("n"):
        syllable = "l" + syllable[1:]
    if syllable.endswith("ng") and syllable[-3] in "ie":
        syllable = syllable[:-1]  # ing -> in, eng -> en
    return syllable


def _syllables(text: str) -> list[tuple[str, int]]:
    """(normalized toneless pinyin, character index) for each Chinese character in ``text``."""
    from pypinyin import lazy_pinyin

    out = []
    for index, char in enumerate(text):
        if _CJK.match(char):
            out.append((_normalize_syllable(lazy_pinyin(char)[0]), index))
    return out


class WakeMatcher:
    """Finds a wake phrase near the start of a transcript.

    English phrases ("hey alex") match words, allowing known mishearings of the name. Chinese phrases
    ("艾丽同学") match by pronunciation, so 爱丽同学, 艾莉同學 or 艾力同学 count too.
    """

    def __init__(self, phrases: Iterable[str], name_aliases: Iterable[str]) -> None:
        phrases = [p.strip() for p in phrases if p.strip()]
        self.pattern = build_wake_pattern([p for p in phrases if not _has_cjk(p)], name_aliases)
        self.chinese = [[s for s, _ in _syllables(p)] for p in phrases if _has_cjk(p)]

    def find(self, text: str) -> str | None:
        """Return what was said after the wake phrase ("" if nothing), or None if there is no wake phrase."""
        normalized = " ".join(text.split())
        ends = []
        match = self.pattern.search(normalized)
        if match is not None and match.start() <= WAKE_SEARCH_CHARS:
            ends.append(match.end())
        if self.chinese:
            syllables = _syllables(normalized)
            sounds = [s for s, _ in syllables]
            for phrase in self.chinese:
                for start in range(min(WAKE_SEARCH_SYLLABLES, len(sounds) - len(phrase)) + 1):
                    if sounds[start : start + len(phrase)] == phrase and syllables[start][1] <= WAKE_SEARCH_CHARS:
                        ends.append(syllables[start + len(phrase) - 1][1] + 1)
                        break
        if not ends:
            return None
        return re.sub(r"^[\W_]+|[\W_]+$", "", normalized[min(ends) :])


def build_wake_pattern(phrases: Iterable[str], name_aliases: Iterable[str]) -> re.Pattern[str]:
    """Regex for "<greeting> <name>" near the start of a transcript.

    ``phrases`` like "hey alex" contribute their first word as a greeting and the rest as the name;
    ``name_aliases`` add common mishearings of the name ("alec", "alix").
    """
    greetings: set[str] = set()
    names: set[str] = {a.lower() for a in name_aliases}
    for phrase in phrases:
        words = phrase.lower().split()
        if len(words) >= 2:
            greetings.add(words[0])
            names.add(" ".join(words[1:]))
        elif words:
            names.add(words[0])
    latin_greeting = "|".join(sorted(map(re.escape, greetings), key=len, reverse=True)) or "(?!)"
    latin_name = "|".join(sorted(map(re.escape, names), key=len, reverse=True)) or "(?!)"
    cjk_greeting = "|".join(map(re.escape, CJK_GREETINGS))
    cjk_name = "|".join(map(re.escape, CJK_NAMES))
    return re.compile(
        rf"(?:\b(?:{latin_greeting})\b\W*\b(?:{latin_name})\b)"
        rf"|(?:(?:{cjk_greeting}|\b(?:{latin_greeting})\b)\W*(?:{cjk_name}))",
        re.IGNORECASE,
    )


def find_wake(pattern: re.Pattern[str], text: str) -> str | None:
    """English-only matching with a prebuilt pattern (see :class:`WakeMatcher` for the full matcher)."""
    normalized = " ".join(text.split())
    match = pattern.search(normalized)
    if match is None or match.start() > WAKE_SEARCH_CHARS:
        return None
    return re.sub(r"^[\W_]+|[\W_]+$", "", normalized[match.end() :])


def ensure_model(models_dir: Path) -> Path:
    model_dir = models_dir / MODEL_NAME
    if (model_dir / "model.int8.onnx").exists():
        return model_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    archive = models_dir / f"{MODEL_NAME}.tar.bz2"
    logger.info("Downloading the wake-word speech model (~240 MB) to %s", models_dir)
    urllib.request.urlretrieve(MODEL_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(models_dir, filter="data")
    archive.unlink()
    return model_dir


class WakeWordRecognizer:
    def __init__(self, models_dir: Path, phrases: Iterable[str], name_aliases: Iterable[str]) -> None:
        import sherpa_onnx

        model_dir = ensure_model(models_dir)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            use_itn=True,
            num_threads=2,
        )
        self._lock = threading.Lock()
        self.matcher = WakeMatcher(phrases, name_aliases)

    def transcribe(self, samples: np.ndarray) -> str:
        """Blocking; call from a worker thread. ``samples``: 16 kHz mono float32."""
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(WAKE_RATE, samples)
            self._recognizer.decode_stream(stream)
            return str(stream.result.text).strip()

    def find_wake(self, text: str) -> str | None:
        return self.matcher.find(text)
