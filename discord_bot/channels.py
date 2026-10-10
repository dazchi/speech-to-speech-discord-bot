"""Choosing which voice channel the bot should sit in."""

from __future__ import annotations


def busiest_channel(human_counts: dict[int, int], current: int | None) -> int | None:
    """Return the id of the voice channel with the most (non-bot) people.

    Ties keep the current channel, so the bot does not hop between equally busy channels.
    Returns None when every channel is empty.
    """
    if not human_counts:
        return None
    best = max(human_counts.values())
    if best == 0:
        return None
    if current is not None and human_counts.get(current, 0) == best:
        return current
    return max(human_counts, key=lambda channel_id: human_counts[channel_id])
