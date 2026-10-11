import asyncio

from discord_bot.ducking import MusicDucker


class Music:
    """Records the duck requests; ``fail`` makes them raise."""

    def __init__(self) -> None:
        self.calls: list[bool] = []
        self.fail = False

    async def send(self, down: bool) -> None:
        if self.fail:
            raise OSError("unreachable")
        self.calls.append(down)


def make(busy, **kwargs):
    music = Music()
    ducker = MusicDucker(music.send, lambda: busy[0], release_delay_s=0.05, poll_s=0.01, **kwargs)
    return ducker, music


async def test_music_goes_down_while_alex_is_busy_and_up_after():
    busy = [True]
    ducker, music = make(busy)
    ducker.speaking()
    ducker.speaking()  # more audio of the same reply: no new request
    await asyncio.sleep(0.1)
    assert music.calls == [True] and ducker.ducked
    busy[0] = False
    await asyncio.sleep(0.15)
    assert music.calls == [True, False] and not ducker.ducked


async def test_a_short_pause_between_sentences_keeps_the_music_down():
    busy = [True]
    ducker, music = make(busy)
    ducker.speaking()
    await asyncio.sleep(0.03)
    busy[0] = False
    await asyncio.sleep(0.02)  # shorter than release_delay_s
    busy[0] = True  # e.g. the answer after a web search starts
    await asyncio.sleep(0.1)
    assert music.calls == [True]
    busy[0] = False
    await asyncio.sleep(0.15)
    assert music.calls == [True, False]


async def test_a_long_reply_renews_the_duck():
    busy = [True]
    ducker, music = make(busy, hold_s=0.06)
    ducker.speaking()
    await asyncio.sleep(0.15)
    assert music.calls.count(True) >= 3 and False not in music.calls
    await ducker.close()
    assert music.calls[-1] is False


async def test_close_brings_the_music_back_up():
    ducker, music = make([True])
    ducker.speaking()
    await asyncio.sleep(0.02)
    await ducker.close()
    assert music.calls == [True, False] and not ducker.ducked


async def test_an_unreachable_music_bot_is_tried_again_next_time(caplog):
    busy = [False]
    ducker, music = make(busy)
    music.fail = True
    ducker.speaking()
    await asyncio.sleep(0.02)
    ducker.speaking()
    await asyncio.sleep(0.02)
    assert not ducker.ducked
    assert len([r for r in caplog.records if "Could not lower" in r.getMessage()]) == 1  # logged once
    music.fail = False
    ducker.speaking()
    await asyncio.sleep(0.15)
    assert music.calls == [True, False]
