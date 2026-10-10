"""``--qwen3_tts_t2s`` converts Traditional Chinese to Simplified for synthesis only."""

import sys
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import speech_to_speech.TTS.qwen3_tts_handler as qwen3_tts_module
from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.backend_registry import normalize_dataclass_config
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

TRADITIONAL = "這種髒話我絕對不學。"
SIMPLIFIED = "这种脏话我绝对不学。"


class _FakeConverter:
    def convert(self, text: str) -> str:
        return text.replace(TRADITIONAL, SIMPLIFIED)


def _handler(monkeypatch: pytest.MonkeyPatch, printed: list[str], synthesized: list[str]) -> Qwen3TTSHandler:
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.speaker = None
    handler.instruct = None
    handler.language = "chinese"
    handler.backend = "mlx"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(config=SimpleNamespace(tts_model_type="base"))
    handler._apply_session_voice_override = lambda model_type, runtime_config=None, response=None: None
    handler._process_voice_clone = lambda text, language=None: (
        synthesized.append(text),
        iter([np.zeros(512, dtype=np.int16)]),
    )[1]
    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda message, *a, **k: printed.append(message))
    return handler


def test_process_synthesizes_simplified_but_prints_original(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []
    synthesized: list[str] = []
    handler = _handler(monkeypatch, printed, synthesized)
    handler._t2s_converter = _FakeConverter()

    assert list(handler.process(TTSInput(text=TRADITIONAL)))

    assert synthesized == [SIMPLIFIED]
    assert printed == [f"[green]ASSISTANT: {TRADITIONAL}"]


def test_process_leaves_text_unchanged_without_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []
    synthesized: list[str] = []
    handler = _handler(monkeypatch, printed, synthesized)

    assert list(handler.process(TTSInput(text=TRADITIONAL)))

    assert synthesized == [TRADITIONAL]


def test_t2s_flag_defaults_off_and_reaches_setup_kwargs() -> None:
    assert normalize_dataclass_config(Qwen3TTSHandlerArguments(), "qwen3_tts")["t2s"] is False
    assert normalize_dataclass_config(Qwen3TTSHandlerArguments(qwen3_tts_t2s=True), "qwen3_tts")["t2s"] is True


def test_load_t2s_converter_explains_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "opencc", None)

    with pytest.raises(ImportError, match=r"speech-to-speech\[t2s\]"):
        Qwen3TTSHandler._load_t2s_converter()


def test_load_t2s_converter_converts_with_opencc() -> None:
    pytest.importorskip("opencc")

    converter = Qwen3TTSHandler._load_t2s_converter()

    assert converter.convert(TRADITIONAL) == SIMPLIFIED
