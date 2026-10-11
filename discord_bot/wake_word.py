"""Wake-word detection by transcribing short utterances and matching the text.

A tiny keyword-spotting model missed real Discord voices (accents, Opus compression), so each
utterance is transcribed and the text is matched against the wake phrases: English phrases by words,
Chinese phrases by pronunciation (pinyin).

The default recognizer is Qwen3-ASR 1.7B (MLX, ~0.2 s per utterance), given the wake phrases as
context so it spells the name the way the matcher expects. On recordings from Discord it caught 41 of
41 wake phrases, where SenseVoice (sherpa-onnx, CPU) caught 24. SenseVoice stays available with
``WAKE_MODEL=sensevoice``.
"""

from __future__ import annotations

import logging
import re
import tarfile
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from discord_bot.audio import WAKE_RATE

logger = logging.getLogger(__name__)

DEFAULT_WAKE_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
SENSEVOICE = "sensevoice"
MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL_NAME}.tar.bz2"

# Chinese greetings and renderings of the name that a recognizer may produce for "Hey Alex".
CJK_GREETINGS = ("嘿", "嗨", "哈囉", "哈喽", "欸", "诶", "哎")
CJK_NAMES = ("艾力克斯", "艾利克斯", "亞歷克斯", "亚历克斯", "阿力克斯", "艾力斯", "艾力")
# The wake phrase must come first: at most this much may be said before it, e.g. "Hello, hello. Hey
# Alex" or 「嘿，艾丽同学」, but not "...你就说 Hi Alex 好了" in the middle of a conversation.
MAX_WORDS_BEFORE = 4
MAX_CJK_BEFORE = 2
# A transcript with this many different wake phrases is the recognizer repeating its context.
ECHO_PHRASES = 3
# Utterances with less speech than this are not transcribed: on clicks and breaths the recognizer
# may "hear" the wake phrases it was given as context. The shortest real "Hey Alex" had 0.46 s.
MIN_SPEECH_S = 0.3
SPEECH_RMS = 0.01  # per 20 ms frame, on [-1, 1] audio
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")


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


def _near_start(before: str) -> bool:
    return len(_LATIN_WORD.findall(before)) <= MAX_WORDS_BEFORE and len(_CJK.findall(before)) <= MAX_CJK_BEFORE


def _rest(text: str) -> str:
    return re.sub(r"^[\W_]+|[\W_]+$", "", text)


def speech_seconds(samples: np.ndarray, rate: int = WAKE_RATE) -> float:
    """How long the audio is louder than quiet background noise."""
    frame = rate // 50
    usable = len(samples) // frame * frame
    if usable == 0:
        return 0.0
    rms = np.sqrt(np.mean(samples[:usable].reshape(-1, frame) ** 2, axis=1))
    return float(np.count_nonzero(rms > SPEECH_RMS)) / 50


def wake_context(phrases: Iterable[str]) -> str:
    """The wake phrases written out, e.g. "Hey Alex, OK Alex, 艾利同學", as context for the recognizer."""

    def written(phrase: str) -> str:
        if _has_cjk(phrase):
            return phrase
        return " ".join(w.upper() if w in ("ok",) else w.capitalize() for w in phrase.lower().split())

    return ", ".join(written(p) for p in phrases if p.strip())


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
        hits: list[tuple[int, int, str]] = []  # (start, end, which phrase)
        for match in self.pattern.finditer(normalized):
            hits.append((match.start(), match.end(), " ".join(re.findall(r"\w+", match.group(0).lower()))))
        if self.chinese:
            syllables = _syllables(normalized)
            sounds = [s for s, _ in syllables]
            for index, phrase in enumerate(self.chinese):
                for start in range(len(sounds) - len(phrase) + 1):
                    if sounds[start : start + len(phrase)] == phrase:
                        hits.append((syllables[start][1], syllables[start + len(phrase) - 1][1] + 1, f"zh{index}"))
        if not hits or len({phrase for _, _, phrase in hits}) >= ECHO_PHRASES:
            return None
        start, end, _ = min(hits)
        if not _near_start(normalized[:start]):
            return None
        return _rest(normalized[end:])


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
        rf"(?:(?:\b(?:{latin_greeting})|{cjk_greeting})\W*(?:{latin_name})\b)"  # "hey alex", "hialex", "嘿 Alex"
        rf"|(?:(?:{cjk_greeting}|\b(?:{latin_greeting})\b)\W*(?:{cjk_name}))",
        re.IGNORECASE,
    )


def find_wake(pattern: re.Pattern[str], text: str) -> str | None:
    """English-only matching with a prebuilt pattern (see :class:`WakeMatcher` for the full matcher)."""
    normalized = " ".join(text.split())
    match = pattern.search(normalized)
    if match is None or not _near_start(normalized[: match.start()]):
        return None
    return _rest(normalized[match.end() :])


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


class _SenseVoice:
    def __init__(self, models_dir: Path) -> None:
        import sherpa_onnx

        model_dir = ensure_model(models_dir)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            use_itn=True,
            num_threads=2,
        )

    def transcribe(self, samples: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(WAKE_RATE, samples)
        self._recognizer.decode_stream(stream)
        return str(stream.result.text)


class _Qwen3ASR:
    def __init__(self, model_name: str, context: str) -> None:
        from mlx_audio.stt import load

        logger.info("Loading the wake-word speech model %s (downloaded on first start)", model_name)
        self._model = load(model_name)
        self._context = context or None

    def transcribe(self, samples: np.ndarray) -> str:
        result = self._model.generate(samples, max_tokens=128, system_prompt=self._context)
        return " ".join(str(result.text).split())


class WakeWordRecognizer:
    def __init__(
        self,
        models_dir: Path,
        phrases: Iterable[str],
        name_aliases: Iterable[str],
        model: str = DEFAULT_WAKE_MODEL,
    ) -> None:
        phrases = tuple(phrases)
        self.matcher = WakeMatcher(phrases, name_aliases)
        # One thread owns the model: MLX models are used from the thread that loaded them.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wake-word")

        def load() -> _SenseVoice | _Qwen3ASR:
            if model.strip().lower() == SENSEVOICE:
                return _SenseVoice(models_dir)
            return _Qwen3ASR(model, wake_context(phrases))

        self._backend = self._executor.submit(load).result()
        self._executor.submit(self._backend.transcribe, np.zeros(WAKE_RATE, np.float32)).result()  # warm up

    def transcribe(self, samples: np.ndarray) -> str:
        """Blocking; call from a worker thread. ``samples``: 16 kHz mono float32."""
        if speech_seconds(samples) < MIN_SPEECH_S:
            return ""
        return self._executor.submit(self._backend.transcribe, samples).result().strip()

    def find_wake(self, text: str) -> str | None:
        return self.matcher.find(text)
