import json

import numpy as np
import pytest
from discord_bot.audio import (
    DISCORD_FRAME_BYTES,
    SERVER_RATE,
    InputConverter,
    OutputConverter,
    chime,
    silence_server,
)
from discord_bot.tools import ToolRegistry


def _discord_tone(seconds: float, freq: float = 440.0) -> bytes:
    t = np.arange(int(48_000 * seconds)) / 48_000
    mono = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return np.repeat(mono, 2).tobytes()


def test_input_converter_produces_server_and_wake_rates():
    converter = InputConverter()
    pcm = _discord_tone(1.0)
    server, wake = b"", []
    for i in range(0, len(pcm), DISCORD_FRAME_BYTES):
        s, w = converter.convert(pcm[i : i + DISCORD_FRAME_BYTES])
        server += s
        wake.append(w)
    wake_samples = np.concatenate(wake)
    # Streaming resamplers hold back up to ~32 ms (latency, not lost audio).
    assert 0 <= SERVER_RATE - len(server) / 2 < 0.04 * SERVER_RATE
    assert 0 <= 16_000 - len(wake_samples) < 0.04 * 16_000
    assert wake_samples.dtype == np.float32
    assert 0.3 < np.abs(wake_samples[2000:]).max() <= 1.0


def test_output_converter_upsamples_to_discord_stereo():
    converter = OutputConverter()
    mono24 = (np.sin(np.arange(24_000) / 10) * 10_000).astype(np.int16).tobytes()
    out = np.frombuffer(converter.convert(mono24), dtype=np.int16).reshape(-1, 2)
    assert 0 <= 48_000 - len(out) < 0.04 * 48_000
    assert np.array_equal(out[:, 0], out[:, 1])


def test_silence_and_chime_sizes():
    assert len(silence_server(0.05)) == int(SERVER_RATE * 0.05) * 2
    tone = chime()
    assert len(tone) % 4 == 0 and len(tone) > DISCORD_FRAME_BYTES


async def test_tool_registry_runs_tools_and_reports_errors(tmp_path, monkeypatch):
    module = tmp_path / "demo_tools.py"
    module.write_text(
        "TOOLS = [{'type': 'function', 'name': 'add', 'parameters': {'type': 'object'}}]\n"
        "class R:\n"
        "    def __init__(self, output, create_response): self.output, self.create_response = output, create_response\n"
        "async def execute_tool(name, arguments):\n"
        "    if arguments.get('boom'): raise RuntimeError('kaboom')\n"
        "    if arguments.get('quiet'): return R('done', False)\n"
        "    return {'sum': arguments['a'] + arguments['b']}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = ToolRegistry(("demo_tools",))

    assert [t["name"] for t in registry.definitions] == ["add"]
    ok = await registry.run("add", '{"a": 2, "b": 3}')
    assert json.loads(ok.output) == {"sum": 5} and ok.create_response
    quiet = await registry.run("add", '{"quiet": true}')
    assert (quiet.output, quiet.create_response) == ("done", False)
    assert "kaboom" in (await registry.run("add", '{"boom": true}')).output
    assert "Invalid arguments" in (await registry.run("add", "[1]")).output
    assert "Unknown tool" in (await registry.run("nope", "{}")).output


def test_tool_registry_rejects_duplicate_names(tmp_path, monkeypatch):
    for mod in ("a_tools", "b_tools"):
        (tmp_path / f"{mod}.py").write_text(
            "TOOLS = [{'type': 'function', 'name': 'same'}]\nasync def execute_tool(n, a): return 1\n"
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry(("a_tools", "b_tools"))
