from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from speech_to_speech.STT.qwen3_asr_handler import LANGUAGE_CODE_TO_NAME, Qwen3ASRSTTHandler, language_to_code
from speech_to_speech.utils.mlx_lock import MLXLockContext

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
DEFAULT_MAX_TOKENS = 256


class MLXAudioQwen3ASRSTTHandler(Qwen3ASRSTTHandler):
    """Speech to text with a Qwen3-ASR checkpoint on MLX through mlx-audio (Apple Silicon).

    Same language policy as :class:`Qwen3ASRSTTHandler`; only loading and generation differ.
    """

    def setup(  # type: ignore[override]
        self,
        model_name: str = DEFAULT_MODEL,
        language: Optional[str] = "auto",
        prompt: Optional[str] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        from mlx_audio.stt.generate import load_model

        logger.info("Loading MLX Qwen3-ASR STT model: %s", model_name)
        self.model_name = model_name
        self.prompt = prompt or None
        gen_kwargs = dict(gen_kwargs or {})
        self.max_tokens = int(gen_kwargs.pop("max_new_tokens", DEFAULT_MAX_TOKENS))
        self.gen_kwargs = gen_kwargs
        self.configure_language(language)
        self.model = load_model(model_name)
        self.warmup()

    def transcribe_text(
        self, audio: np.ndarray, language: Optional[str] = None, prompt: Optional[str] = None
    ) -> tuple[str, Optional[str]]:
        """Transcribe one clip outside the conversation (``POST /v1/transcribe``).

        Safe to call from another thread: inference holds the global MLX lock. ``language`` is an ISO
        code or name (``None`` or ``"auto"`` to detect it); ``prompt`` replaces the configured context.
        Returns the text and the language code (forced or detected, ``None`` if unknown).
        """
        requested = (language or "").strip()
        forced = None if requested.lower() in ("", "auto") else (language_to_code(requested) or requested)
        text, detected = self._transcribe(np.asarray(audio, dtype=np.float32), forced, prompt=prompt)
        return text, forced or language_to_code(detected)

    def _transcribe(
        self, audio: np.ndarray, language: Optional[str], prompt: Optional[str] = None
    ) -> tuple[str, Optional[str]]:
        """Run one generation. Return the text and the language name the model identified (auto mode only)."""
        kwargs: dict[str, Any] = {"max_tokens": self.max_tokens, "verbose": False, **self.gen_kwargs}
        if language is not None:
            # mlx-audio takes the language name the model was trained with ("Chinese"), not the code.
            kwargs["language"] = LANGUAGE_CODE_TO_NAME.get(language, language)
        context = prompt if prompt is not None else self.prompt
        if context:
            kwargs["system_prompt"] = context
        # MLX models share one Metal command queue: every MLX inference in the pipeline holds this lock.
        with MLXLockContext(handler_name=self.__class__.__name__):
            result = self.model.generate(audio, **kwargs)
        text = str(getattr(result, "text", "")).strip()
        if language is not None:
            return text, None
        detected = getattr(result, "language", None)
        if isinstance(detected, list):  # one entry per audio chunk
            detected = next((d for d in detected if d), None)
        return text, detected if isinstance(detected, str) else None
