from discord_bot.channels import busiest_channel


def test_picks_channel_with_most_people():
    assert busiest_channel({1: 2, 2: 5, 3: 0}, current=None) == 2


def test_moves_when_another_channel_gets_busier():
    assert busiest_channel({1: 2, 2: 3}, current=1) == 2


def test_stays_put_on_a_tie():
    assert busiest_channel({1: 3, 2: 3}, current=2) == 2
    assert busiest_channel({1: 3, 2: 3}, current=1) == 1


def test_no_one_anywhere_means_no_channel():
    assert busiest_channel({1: 0, 2: 0}, current=1) is None
    assert busiest_channel({}, current=None) is None
