"""History compaction on the MLX backend must not let thinking models reason first.

With thinking on, Qwen3-style templates open the reply with ``<think>``: the summary then spends
its token budget reasoning while holding the MLX lock, stalling replies, and can run out of
tokens before the JSON summary, so compaction fails and is retried on the next turn.
"""

from __future__ import annotations

from typing import Any

import pytest

from speech_to_speech.LLM import language_model
from speech_to_speech.LLM.language_model import LanguageModelHandler

pytestmark = pytest.mark.skipif(not language_model.HAS_MLX, reason="mlx_lm not installed")


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return "prompt"


def test_mlx_compaction_disables_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(language_model, "mlx_generate", lambda model, tokenizer, **kwargs: '{"a": 1}')
    handler = object.__new__(LanguageModelHandler)
    handler.backend = "mlx"
    handler.model = object()
    handler.tokenizer = tokenizer = _RecordingTokenizer()

    assert handler._build_compaction_generate_fn()("system", "user") == '{"a": 1}'
    assert tokenizer.kwargs["enable_thinking"] is False
    assert tokenizer.kwargs["add_generation_prompt"] is True
