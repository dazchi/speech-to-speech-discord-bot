"""``POST /v1/transcribe``: transcribe one short clip with the pipeline's loaded STT model.

Lets a companion process (such as the Discord bot's wake-word check) reuse the server's speech
recognizer instead of loading a second copy of the model. The clip does not enter any
conversation. Only STT handlers with a ``transcribe_text(audio, language, prompt)`` method can
serve it (``--stt mlx-audio-qwen3-asr``); other backends answer 501.

Request body: 16 kHz mono audio, either 16-bit little-endian PCM (``Content-Type: audio/pcm``)
or a 16-bit PCM WAV file (``audio/wav``). Optional query parameters: ``language`` (ISO code,
name or ``auto``) and ``prompt`` (context or hotwords for this clip). Response:
``{"text": "...", "language": "zh"}``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from speech_to_speech.pipeline.transcript_logging import log_exception

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0
PCM_TYPES = ("audio/pcm", "audio/l16", "application/octet-stream")
WAV_TYPES = ("audio/wav", "audio/x-wav", "audio/wave")


class AudioFormatError(ValueError):
    pass


def decode_audio(body: bytes, content_type: str) -> np.ndarray:
    """16 kHz mono float32 samples from a raw PCM or WAV request body."""
    media_type = content_type.split(";")[0].strip().lower()
    if media_type in WAV_TYPES:
        try:
            with wave.open(io.BytesIO(body)) as wav:
                if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getframerate() != SAMPLE_RATE:
                    raise AudioFormatError("WAV audio must be 16-bit mono at 16 kHz")
                body = wav.readframes(wav.getnframes())
        except (wave.Error, EOFError) as exc:
            raise AudioFormatError(f"Invalid WAV file: {exc}") from exc
    elif media_type not in PCM_TYPES:
        raise AudioFormatError("Send 16 kHz mono 16-bit audio as audio/pcm or audio/wav")
    if len(body) % 2:
        raise AudioFormatError("16-bit PCM must have an even number of bytes")
    samples = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    if len(samples) > MAX_SECONDS * SAMPLE_RATE:
        raise AudioFormatError(f"Clips longer than {MAX_SECONDS:.0f} s are not accepted")
    return samples


def find_transcriber(handler_groups: Iterable[Iterable[Any]]) -> Optional[Any]:
    """The first handler that can transcribe a standalone clip, if any."""
    for handlers in handler_groups:
        for handler in handlers:
            if callable(getattr(handler, "transcribe_text", None)):
                return handler
    return None


def mount_transcribe(app: FastAPI, handler_groups: Iterable[Iterable[Any]]) -> None:
    groups = [list(handlers) for handlers in handler_groups]

    @app.post("/v1/transcribe")
    async def transcribe_endpoint(
        request: Request, language: Optional[str] = None, prompt: Optional[str] = None
    ) -> Response:
        transcriber = find_transcriber(groups)
        if transcriber is None:
            return JSONResponse(
                {
                    "error": "The configured STT backend cannot transcribe standalone clips; use --stt mlx-audio-qwen3-asr."
                },
                status_code=501,
            )
        try:
            audio = decode_audio(await request.body(), request.headers.get("content-type", ""))
        except AudioFormatError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if len(audio) == 0:
            return JSONResponse({"text": "", "language": None})
        try:
            text, language_code = await asyncio.to_thread(transcriber.transcribe_text, audio, language, prompt)
        except Exception as exc:
            log_exception(logger, "Standalone transcription failed", exc)
            return JSONResponse({"error": "Transcription failed"}, status_code=500)
        return JSONResponse({"text": text, "language": language_code})
