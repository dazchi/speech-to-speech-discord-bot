import logging
from pathlib import Path

import httpx
import numpy as np
import pytest
from discord_bot import wake_word
from discord_bot.config import DEFAULT_WAKE_ALIASES, DEFAULT_WAKE_WORDS
from discord_bot.wake_word import WakeWordRecognizer, transcribe_url


def test_transcribe_url_comes_from_the_realtime_url():
    assert transcribe_url("ws://127.0.0.1:8765/v1/realtime?model=local") == "http://127.0.0.1:8765/v1/transcribe"
    assert transcribe_url("wss://box.lan:9000/v1/realtime") == "https://box.lan:9000/v1/transcribe"


def make(monkeypatch, handler) -> WakeWordRecognizer:
    real_client = httpx.Client
    monkeypatch.setattr(
        wake_word.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )
    return WakeWordRecognizer(Path("unused"), DEFAULT_WAKE_WORDS, DEFAULT_WAKE_ALIASES, model="server")


def speech(seconds: float = 1.0) -> np.ndarray:
    return np.random.default_rng(0).normal(0, 0.1, int(16000 * seconds)).astype(np.float32)


def test_sends_pcm_with_the_wake_phrases_as_context(monkeypatch):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"text": "嘿，艾丽同学，现在几点？", "language": "zh"})

    recognizer = make(monkeypatch, handler)
    assert requests == []  # no warm-up call: the server may not be up yet
    text = recognizer.transcribe(speech())
    assert recognizer.find_wake(text) == "现在几点"
    (request,) = requests
    assert request.url.path == "/v1/transcribe" and request.headers["content-type"] == "audio/pcm"
    assert "艾利同學" in request.url.params["prompt"] and "Hey Alex" in request.url.params["prompt"]
    assert len(request.content) == 32000  # 1 s of 16-bit samples


def test_short_sounds_are_not_sent(monkeypatch):
    requests = []
    recognizer = make(monkeypatch, lambda r: requests.append(r) or httpx.Response(200, json={"text": ""}))
    assert recognizer.transcribe(speech(0.1)) == "" and requests == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(501, json={"error": "use --stt mlx-audio-qwen3-asr"}),
        httpx.Response(500, text="boom"),
    ],
)
def test_server_errors_do_not_wake_and_are_logged_once(monkeypatch, caplog, response):
    recognizer = make(monkeypatch, lambda r: response)
    with caplog.at_level(logging.WARNING):
        assert recognizer.transcribe(speech()) == ""
        assert recognizer.transcribe(speech()) == ""
    assert len([r for r in caplog.records if "Wake-word check" in r.message]) == 1


def test_unreachable_server_does_not_wake(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    assert make(monkeypatch, handler).transcribe(speech()) == ""
