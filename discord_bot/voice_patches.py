"""Workarounds for bugs in discord-ext-voice-recv (pinned to an unmerged pull request, see requirements.txt).

UDP keep-alive busy loop: ``UDPKeepAlive.run`` sends with ``socket.sendto(packet, address)``. Discord's
voice socket is a connected UDP socket, and on macOS ``sendto`` with an address on a connected socket
always fails (``[Errno 56] Socket is already connected``). The error handler then retries immediately, so
the thread spins at 100% of a CPU core for as long as the bot is in a voice channel. The original also
passes ``delay = 5000`` (milliseconds) to ``time.sleep``, which takes seconds. The replacement sends with
``send`` (falling back to ``sendto`` on an unconnected socket), every 5 s, and waits after an error.
"""

from __future__ import annotations

import errno
import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)

KEEPALIVE_INTERVAL_S = 5.0
ERROR_BACKOFF_S = 5.0
_UNCONNECTED = (errno.ENOTCONN, errno.EDESTADDRREQ)


def _send(sock: socket.socket, packet: bytes, address: tuple[str, int]) -> None:
    try:
        sock.send(packet)
    except OSError as exc:
        if exc.errno not in _UNCONNECTED:
            raise
        sock.sendto(packet, address)


def keepalive_run(self: Any) -> None:
    """Replacement for ``UDPKeepAlive.run``; ``self`` is the keep-alive thread."""
    self.voice_client.wait_until_connected()
    while not self._end_thread.is_set():
        vc = self.voice_client
        packet = (self.counter % 2**64).to_bytes(8, "big")
        try:
            _send(vc._connection.socket, packet, (vc._connection.endpoint_ip, vc._connection.voice_port))
        except Exception as exc:
            logger.debug("Error sending voice keep-alive: %s: %s", exc.__class__.__name__, exc)
            if self._end_thread.wait(ERROR_BACKOFF_S):
                break
            if not vc.wait_until_connected():
                break
            continue
        self.counter += 1
        self._end_thread.wait(KEEPALIVE_INTERVAL_S)


def apply() -> None:
    from discord.ext.voice_recv import reader

    reader.UDPKeepAlive.run = keepalive_run  # type: ignore[method-assign]
