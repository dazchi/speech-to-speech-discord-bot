from discord_bot.conversation import ConversationController


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make() -> tuple[ConversationController, Clock]:
    clock = Clock()
    return ConversationController(wake_timeout_s=6, followup_s=8, clock=clock), clock


def test_wake_opens_window_for_that_user_only():
    c, _ = make()
    assert c.wake(1, "alice") is True
    assert c.is_active_user(1)
    assert not c.is_active_user(2)
    assert c.wake(2, "bob") is False  # someone else has the floor
    assert c.active.name == "alice"


def test_window_closes_if_nobody_speaks_after_wake_word():
    c, clock = make()
    c.wake(1, "alice")
    clock.now = 5.9
    assert c.expire() is None
    clock.now = 6.1
    assert c.expire().name == "alice"
    assert c.is_idle


def test_window_stays_open_through_a_turn_and_closes_after_follow_up():
    c, clock = make()
    c.wake(1, "alice")
    c.speech_started()
    clock.now = 60  # long question: no expiry while speaking
    assert c.expire() is None
    c.speech_stopped()
    c.response_started()
    clock.now = 120  # slow answer: no expiry while responding
    assert c.expire() is None
    c.response_finished()
    clock.now = 127.9
    assert c.expire() is None
    clock.now = 128.1
    assert c.expire() is not None


def test_follow_up_question_keeps_the_window_open():
    c, clock = make()
    c.wake(1, "alice")
    c.speech_started()
    c.speech_stopped()
    c.response_started()
    c.response_finished()
    clock.now = 5
    c.speech_started()
    clock.now = 20
    assert c.expire() is None


def test_tool_call_keeps_window_open_until_follow_up_response():
    c, clock = make()
    c.wake(1, "alice")
    c.speech_started()
    c.speech_stopped()
    c.response_started()
    c.response_finished(tools_pending=True)
    clock.now = 100  # slow web search
    assert c.expire() is None
    c.response_started()  # follow-up response from the tool result
    c.response_finished()
    clock.now = 108.1
    assert c.expire() is not None


def test_tools_without_follow_up_restart_the_timer():
    c, clock = make()
    c.wake(1, "alice")
    c.response_started()
    c.response_finished(tools_pending=True)
    clock.now = 50
    c.tools_done_without_follow_up()
    clock.now = 58.1
    assert c.expire() is not None


def test_active_user_saying_wake_word_again_extends_window():
    c, clock = make()
    c.wake(1, "alice")
    clock.now = 5
    assert c.wake(1, "alice") is False
    clock.now = 10
    assert c.expire() is None
    clock.now = 11.1
    assert c.expire() is not None


def test_follow_up_timer_starts_when_reply_audio_finishes_playing():
    c, clock = make()
    c.wake(1, "alice")
    c.speech_started()
    c.speech_stopped()
    c.response_started()
    c.playback_started()
    c.response_finished()  # generation done, but Alex is still talking in the channel
    clock.now = 30
    assert c.expire() is None
    c.playback_finished()  # reply finished playing at t=30
    clock.now = 37.9
    assert c.expire() is None
    clock.now = 38.1
    assert c.expire() is not None


def test_zero_follow_up_closes_right_after_reply_but_not_before_it_starts():
    clock = Clock()
    c = ConversationController(wake_timeout_s=6, followup_s=0, clock=clock)
    c.wake(1, "alice")
    c.speech_started()
    c.speech_stopped()
    clock.now = 1  # the server has not started the response yet
    assert c.expire() is None
    c.response_started()
    c.playback_started()
    c.response_finished()
    clock.now = 20
    assert c.expire() is None  # still playing
    c.playback_finished()
    assert c.expire() is not None


def test_interrupting_the_reply_does_not_close_the_window_while_user_talks():
    c, clock = make()
    c.wake(1, "alice")
    c.response_started()
    c.playback_started()
    c.speech_started()  # barge-in
    c.playback_finished()
    clock.now = 100
    assert c.expire() is None


def test_chime_playback_end_is_ignored():
    clock = Clock()
    c = ConversationController(wake_timeout_s=6, followup_s=0, clock=clock)
    c.wake(1, "alice")
    c.playback_finished()  # the wake chime ended; no reply audio was playing
    clock.now = 5
    assert c.expire() is None
