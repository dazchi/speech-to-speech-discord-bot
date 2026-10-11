import socket
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("discord.ext.voice_recv")

from discord.ext.voice_recv import reader  # noqa: E402
from discord_bot import voice_patches  # noqa: E402


class FakeVoiceClient:
    def __init__(self, sock: socket.socket, address: tuple[str, int]) -> None:
        self._connection = SimpleNamespace(socket=sock, endpoint_ip=address[0], voice_port=address[1])
        self.connected = True

    def wait_until_connected(self, timeout: float | None = 30.0) -> bool:
        return self.connected


def run_keepalive(sock: socket.socket, address: tuple[str, int], seconds: float) -> object:
    voice_patches.apply()
    keepalive = reader.UDPKeepAlive(FakeVoiceClient(sock, address))  # type: ignore[arg-type]
    keepalive.start()
    time.sleep(seconds)
    keepalive.stop()
    keepalive.join(timeout=2)
    assert not keepalive.is_alive()
    return keepalive


@pytest.fixture
def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(1)
    yield sock
    sock.close()


def test_keepalive_works_on_a_connected_socket_without_spinning(receiver):
    # Discord's voice socket is connected; on macOS sendto(address) on it always fails.
    address = receiver.getsockname()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(address)
    start = time.process_time()
    keepalive = run_keepalive(sock, address, 0.5)
    assert receiver.recv(16) == (0).to_bytes(8, "big")  # one keep-alive, then a 5 s wait
    assert keepalive.counter == 1  # type: ignore[attr-defined]
    assert time.process_time() - start < 0.2  # the original spun a whole core here
    sock.close()


def test_keepalive_falls_back_to_sendto_on_an_unconnected_socket(receiver):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    run_keepalive(sock, receiver.getsockname(), 0.3)
    assert receiver.recv(16) == (0).to_bytes(8, "big")
    sock.close()


def test_keepalive_waits_after_an_error_instead_of_retrying_at_once():
    calls = []

    class Broken:
        def send(self, packet):
            calls.append(packet)
            raise OSError(55, "No buffer space available")

    keepalive = run_keepalive(Broken(), ("127.0.0.1", 9), 0.5)  # type: ignore[arg-type]
    assert len(calls) == 1 and keepalive.counter == 0  # type: ignore[attr-defined]


def test_apply_replaces_the_library_keepalive():
    voice_patches.apply()
    assert reader.UDPKeepAlive.run is voice_patches.keepalive_run
    assert not isinstance(threading.current_thread(), reader.UDPKeepAlive)
