"""A step result whose window decoded short keeps every bar in place, and the
novelty check counts a missing bar as a difference."""

from __future__ import annotations

import copy
from types import SimpleNamespace

from conftest import make_bar, make_note, melodic_track

from midigpt._types import Score
from midigpt.inference.session import SamplingSession


def _counts(track):
    return [len(b.notes) for b in track.bars]


def test_restore_window_puts_missing_bars_back_at_window_end():
    before = Score(tracks=[melodic_track(n_bars=32, notes_per_bar=0), melodic_track(n_bars=32)])
    after = copy.deepcopy(before)
    # window [8, 16) of track 0 decoded only 4 bars; the suffix [16, 32) follows it
    gen = [make_bar([make_note(onset=0)] * (i + 1)) for i in range(4)]
    after.tracks[0].bars = before.tracks[0].bars[:8] + gen + before.tracks[0].bars[16:]
    step = SimpleNamespace(start_bar=8, end_bar=16)

    SamplingSession._restore_window(before, after, step)

    assert len(after.tracks[0].bars) == 32
    assert _counts(after.tracks[0])[8:16] == [1, 2, 3, 4, 0, 0, 0, 0]
    assert _counts(after.tracks[1]) == _counts(before.tracks[1])


def test_is_identical_counts_a_missing_bar_as_different():
    a = Score(tracks=[melodic_track(n_bars=8)])
    b = copy.deepcopy(a)
    b.tracks[0].bars = b.tracks[0].bars[:6]
    sess = SimpleNamespace(_request=SimpleNamespace(tracks=[SimpleNamespace(id=0, bars=list(range(8)))]))

    assert SamplingSession._is_identical(sess, a, copy.deepcopy(a))
    assert not SamplingSession._is_identical(sess, a, b)
