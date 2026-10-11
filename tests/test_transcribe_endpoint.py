"""``POST /v1/transcribe``: standalone clips through the pipeline's loaded STT model."""

from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from speech_to_speech.api.openai_realtime.transcribe_endpoint import mount_transcribe


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, Any, Any]] = []

    def transcribe_text(self, audio: np.ndarray, language: Any = None, prompt: Any = None) -> tuple[str, str]:
        self.calls.append((audio, language, prompt))
        return "艾丽同学，现在几点？", "zh"


def client_for(*handlers: Any) -> TestClient:
    app = FastAPI()
    mount_transcribe(app, [list(handlers)])
    return TestClient(app)


def pcm(samples: np.ndarray) -> bytes:
    return (samples * 32767).astype("<i2").tobytes()


def test_transcribes_raw_pcm_with_language_and_prompt() -> None:
    transcriber = FakeTranscriber()
    client = client_for(object(), transcriber)  # the STT handler is found among the pipeline's handlers
    audio = np.full(1600, 0.5, np.float32)
    response = client.post(
        "/v1/transcribe",
        params={"language": "zh", "prompt": "Hey Alex, 艾丽同学"},
        content=pcm(audio),
        headers={"Content-Type": "audio/pcm"},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "艾丽同学，现在几点？", "language": "zh"}
    (received, language, prompt), *_ = transcriber.calls
    assert len(received) == 1600 and abs(received[0] - 0.5) < 1e-3
    assert (language, prompt) == ("zh", "Hey Alex, 艾丽同学")


def test_accepts_16k_mono_wav() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm(np.zeros(800, np.float32)))
    transcriber = FakeTranscriber()
    response = client_for(transcriber).post(
        "/v1/transcribe", content=buffer.getvalue(), headers={"Content-Type": "audio/wav"}
    )
    assert response.status_code == 200 and len(transcriber.calls[0][0]) == 800


def test_rejects_wav_at_other_rates_and_unknown_types() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(b"\0\0" * 10)
    client = client_for(FakeTranscriber())
    assert (
        client.post("/v1/transcribe", content=buffer.getvalue(), headers={"Content-Type": "audio/wav"}).status_code
        == 400
    )
    assert client.post("/v1/transcribe", content=b"\0\0", headers={"Content-Type": "audio/mpeg"}).status_code == 400


def test_backends_without_standalone_transcription_answer_501() -> None:
    response = client_for(object()).post("/v1/transcribe", content=b"\0\0", headers={"Content-Type": "audio/pcm"})
    assert response.status_code == 501


def test_transcription_errors_return_500_without_details() -> None:
    class Broken:
        def transcribe_text(self, audio: np.ndarray, language: Any = None, prompt: Any = None) -> tuple[str, str]:
            raise RuntimeError("secret transcript")

    response = client_for(Broken()).post("/v1/transcribe", content=b"\0\0", headers={"Content-Type": "audio/pcm"})
    assert response.status_code == 500 and "secret" not in response.text
