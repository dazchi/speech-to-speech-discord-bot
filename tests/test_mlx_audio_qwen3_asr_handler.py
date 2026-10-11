"""MLX Qwen3-ASR STT backend (mlx-audio). The model is faked, so no MLX install is needed."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.STT.mlx_audio_qwen3_asr_handler import MLXAudioQwen3ASRSTTHandler
from speech_to_speech.utils import mlx_lock


def _lock_held_elsewhere() -> bool:
    got: list[bool] = []

    def probe() -> None:
        got.append(mlx_lock._mlx_lock.acquire(blocking=False))
        if got[0]:
            mlx_lock._mlx_lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return not got[0]


class FakeModel:
    def __init__(self, text: str = "你好", language: Any = None) -> None:
        self.text = text
        self.language = language
        self.calls: list[dict[str, Any]] = []
        self.lock_held: list[bool] = []

    def generate(self, audio: np.ndarray, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        self.lock_held.append(_lock_held_elsewhere())
        return SimpleNamespace(text=f" {self.text} ", language=self.language)


def make_handler(model: FakeModel, *, language: str = "auto", prompt: str | None = None) -> MLXAudioQwen3ASRSTTHandler:
    handler = object.__new__(MLXAudioQwen3ASRSTTHandler)
    handler.prompt = prompt
    handler.max_tokens = 256
    handler.gen_kwargs = {}
    handler.configure_language(language)
    handler.model = model
    handler._record_final_stt = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return handler


def transcribe(handler: MLXAudioQwen3ASRSTTHandler) -> Transcription:
    (message,) = handler.process(VADAudio(audio=np.zeros(16000, np.float32), turn_id="t", turn_revision=0))
    assert isinstance(message, Transcription)
    return message


def test_forced_language_is_passed_by_name_with_the_prompt_under_the_mlx_lock() -> None:
    model = FakeModel("今天天气怎么样？")
    message = transcribe(make_handler(model, language="zh", prompt="Alex, 艾丽同学"))
    assert model.calls == [
        {"max_tokens": 256, "verbose": False, "language": "Chinese", "system_prompt": "Alex, 艾丽同学"}
    ]
    assert model.lock_held == [True]
    assert (message.text, message.language_code) == ("今天天气怎么样？", "zh")


def test_auto_reports_the_detected_language() -> None:
    model = FakeModel("Hey Alex.", language=["English"])
    message = transcribe(make_handler(model))
    assert "language" not in model.calls[0] and "system_prompt" not in model.calls[0]
    assert message.language_code == "en-auto"


def test_auto_falls_back_to_the_last_language_when_none_is_detected() -> None:
    handler = make_handler(FakeModel(language=["Chinese"]))
    transcribe(handler)
    handler.model = FakeModel(language=[])
    assert transcribe(handler).language_code == "zh-auto"


def test_setup_turns_max_new_tokens_into_max_tokens(monkeypatch: Any) -> None:
    pytest.importorskip("mlx_audio.stt.generate")

    model = FakeModel()
    monkeypatch.setattr("mlx_audio.stt.generate.load_model", lambda name: model)
    handler = object.__new__(MLXAudioQwen3ASRSTTHandler)
    handler.setup(model_name="m", language="zh", gen_kwargs={"max_new_tokens": 64})
    assert handler.max_tokens == 64 and model.calls[0]["max_tokens"] == 64  # warmup ran


def test_cli_selects_the_backend_and_keeps_the_language_flag() -> None:
    args = parse_arguments(["--stt", "mlx-audio-qwen3-asr", "--language", "zh", "--mlx_audio_qwen3_asr_prompt", "Alex"])
    assert args.stt_backend.name == "mlx-audio-qwen3-asr"
    assert args.stt_backend.config == {
        "model_name": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "language": "zh",
        "prompt": "Alex",
        "gen_kwargs": {"max_new_tokens": 256},
    }
