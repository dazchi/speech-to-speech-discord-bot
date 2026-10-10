"""MLX LM backend honours ``--llm_gen_temperature`` through an mlx-lm sampler."""

from __future__ import annotations

from typing import Any

import pytest

from speech_to_speech.LLM import language_model
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import LanguageModelHandler, StreamContext

pytestmark = pytest.mark.skipif(not language_model.HAS_MLX, reason="mlx_lm not installed")


class _StopGeneration(Exception):
    pass


class _FakeTokenizer:
    def apply_chat_template(self, messages: Any, tokenize: bool = False, **kwargs: Any) -> Any:
        return [1, 2, 3] if tokenize else "prompt"


def _load_mlx_handler(monkeypatch: pytest.MonkeyPatch, gen_kwargs: dict[str, Any]) -> tuple[Any, list[float]]:
    temps: list[float] = []
    monkeypatch.setattr(language_model, "mlx_load", lambda name: (object(), _FakeTokenizer()))

    def fake_make_sampler(temp: float = 0.0) -> Any:
        temps.append(temp)
        return ("sampler", temp)

    monkeypatch.setattr(language_model, "mlx_make_sampler", fake_make_sampler)
    handler = object.__new__(LanguageModelHandler)
    handler.backend = "mlx"
    handler._load_model("fake/model", "mps", "float16", gen_kwargs)
    return handler, temps


@pytest.mark.parametrize(
    ("gen_kwargs", "expected_temp"),
    [
        ({"max_new_tokens": 16, "temperature": 0.7}, 0.7),
        ({"max_new_tokens": 16, "temperature": 0.0}, 0.0),
        ({"max_new_tokens": 16}, 0.0),
    ],
)
def test_mlx_load_builds_sampler_from_temperature(
    monkeypatch: pytest.MonkeyPatch, gen_kwargs: dict[str, Any], expected_temp: float
) -> None:
    handler, temps = _load_mlx_handler(monkeypatch, gen_kwargs)

    assert temps == [expected_temp]
    assert handler._mlx_sampler == ("sampler", expected_temp)


def test_mlx_generate_passes_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, _ = _load_mlx_handler(monkeypatch, {"max_new_tokens": 16, "temperature": 0.7})
    calls: list[dict[str, Any]] = []

    def fake_stream_generate(model: Any, tokenizer: Any, prompt: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        raise _StopGeneration

    monkeypatch.setattr(language_model, "mlx_stream_generate", fake_stream_generate)
    chat = Chat(size=4)
    chat.add_item(make_user_message("hello"))

    with pytest.raises(_StopGeneration):
        next(iter(handler._generate(chat, None, None, StreamContext())))

    assert calls == [{"max_tokens": 16, "sampler": ("sampler", 0.7)}]
