"""The hard rhythm / remix forcers count notes from the token stream. Drum tracks emit no
NoteDuration tokens, so a forcer that counts notes on NoteDuration never sees a drum note and
keeps forcing notes at the same onset forever."""

from __future__ import annotations

import torch

from midigpt.inference.session import _RhythmForcer, _VariationForcer

# A toy vocabulary: [0, 48) TimeAbsolutePos, [48, 176) NoteOnset, [176, 208) VelocityLevel,
# [208, 336) NoteDuration, 336 BarEnd.
POS, ONSET, VEL, DUR, BAREND = (0, 48), (48, 176), (176, 208), (208, 336), 336
VOCAB = 337


def legal():
    return torch.ones(VOCAB, dtype=torch.bool)


def onset_open(mask):
    return bool(mask[ONSET[0] : ONSET[1]].any())


def test_rhythm_forcer_moves_on_after_a_drum_note():
    # one note at tick 0, one at tick 12; drums: NoteOnset with no NoteDuration
    f = _RhythmForcer([(0, 1), (12, 1)], POS, ONSET, VEL, BAREND)
    m = legal()
    f.apply(m)
    assert onset_open(m)  # the tick-0 note is required
    f.advance("NoteOnset")
    m = legal()
    f.apply(m)
    assert not onset_open(m)  # group full: no second note at tick 0
    assert bool(m[POS[0] + 12])  # move to tick 12
    f.advance("TimeAbsolutePos")
    f.advance("NoteOnset")
    m = legal()
    f.apply(m)
    assert not onset_open(m) and not bool(m[POS[0] : POS[1]].any())  # schedule done: only BarEnd
    assert bool(m[BAREND])


def test_rhythm_forcer_melodic_chord_still_fills():
    f = _RhythmForcer([(0, 2)], POS, ONSET, VEL, BAREND)
    f.advance("NoteOnset")
    f.advance("NoteDuration")
    m = legal()
    f.apply(m)
    assert onset_open(m)  # second chord note still required
    f.advance("NoteOnset")
    f.advance("NoteDuration")
    m = legal()
    f.apply(m)
    assert not onset_open(m)


def test_variation_forcer_counts_drum_notes_on_onset():
    bars = [[(0, [(36, 1)]), (12, [(38, 1)])]]
    f = _VariationForcer("pitch", 0.0, bars, POS, ONSET, DUR, VEL, BAREND, count_on_onset=True)
    m = legal()
    f.apply(m)
    assert (
        bool(m[ONSET[0] + 36]) and int(m[ONSET[0] : ONSET[1]].sum()) == 1
    )  # reference pitch forced
    f.advance("NoteOnset")
    m = legal()
    f.apply(m)
    assert not onset_open(m) and bool(m[POS[0] + 12])


# Infill generates each target bar as its own FillInStart ... FillInEnd block, with no BarEnd:
# the forcers must start the next bar's schedule on FillInEnd, or the second target bar opens
# with the first one's schedule used up and no legal token.


def test_rhythm_forcer_starts_over_after_a_fill_in():
    f = _RhythmForcer([(0, 1)], POS, ONSET, VEL, BAREND)
    f.advance("NoteOnset")
    f.advance("FillInEnd")
    f.advance("FillInStart")
    m = legal()
    f.apply(m)
    assert onset_open(m)  # the next bar's tick-0 note is required again


def test_variation_forcer_moves_to_the_next_bar_after_a_fill_in():
    bars = [[(0, [(60, 3)])], [(0, [(67, 3)])]]
    f = _VariationForcer("pitch", 0.0, bars, POS, ONSET, DUR, VEL, BAREND)
    f.advance("NoteOnset")
    f.advance("NoteDuration")
    f.advance("FillInEnd")
    f.advance("FillInStart")
    m = legal()
    f.apply(m)
    assert bool(m[ONSET[0] + 67]) and int(m[ONSET[0] : ONSET[1]].sum()) == 1  # second bar's pitch
