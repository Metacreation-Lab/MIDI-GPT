"""The hard rhythm / remix forcers count notes from the token stream. Drum tracks emit no
NoteDuration tokens, so a forcer that counts notes on NoteDuration never sees a drum note and
keeps forcing notes at the same onset forever."""

from __future__ import annotations

import torch

from midigpt.inference.session import _RhythmForcer, _VariationForcer

# A toy vocabulary: [0, 48) TimeAbsolutePos, [48, 176) NoteOnset, [176, 208) VelocityLevel,
# [208, 336) NoteDuration, 336 BarEnd, 337 FillInEnd.
POS, ONSET, VEL, DUR, BAREND, FILLEND = (0, 48), (48, 176), (176, 208), (208, 336), 336, 337
VOCAB = 338


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


def test_variation_forcer_moves_to_a_late_first_onset():
    # the bar's first note is on beat 2 (tick 12): a TimeAbsolutePos to it comes first, then that note
    bars = [[(12, [(60, 3)]), (24, [(64, 3)])]]
    f = _VariationForcer("full", 0.0, bars, POS, ONSET, DUR, VEL, BAREND)
    m = legal()
    f.apply(m)
    assert not onset_open(m) and int(m[POS[0] : POS[1]].sum()) == 1 and bool(m[POS[0] + 12])
    f.advance("TimeAbsolutePos")
    m = legal()
    f.apply(m)
    assert bool(m[ONSET[0] + 60]) and int(m[ONSET[0] : ONSET[1]].sum()) == 1  # still the first group
    f.advance("NoteOnset")
    f.advance("NoteDuration")
    m = legal()
    f.apply(m)
    assert bool(m[POS[0] + 24]) and int(m[POS[0] : POS[1]].sum()) == 1  # then on to the second


def test_forcers_keep_a_fill_in_open_until_the_schedule_is_done():
    # an infill bar closes with FillInEnd: blocked like BarEnd while notes remain, open after the last one
    r = _RhythmForcer([(0, 1), (12, 1)], POS, ONSET, VEL, BAREND, fill_end_id=FILLEND)
    v = _VariationForcer("pitch", 0.0, [[(0, [(60, 3)]), (12, [(62, 3)])]], POS, ONSET, DUR, VEL, BAREND,
                         fill_end_id=FILLEND)
    for f, count_on in ((r, "NoteOnset"), (v, "NoteDuration")):
        m = legal()
        f.apply(m)
        assert not bool(m[FILLEND]) and not bool(m[BAREND])
        f.advance("NoteOnset")
        if count_on == "NoteDuration":
            f.advance("NoteDuration")
        m = legal()
        f.apply(m)
        assert not bool(m[FILLEND])  # moving to tick 12, not closing the bar
        f.advance("TimeAbsolutePos")
        f.advance("NoteOnset")
        if count_on == "NoteDuration":
            f.advance("NoteDuration")
        m = legal()
        f.apply(m)
        assert bool(m[FILLEND])  # schedule done


def test_forcers_block_microtiming_when_no_note_may_follow():
    # a Delta is the microtiming of the note right after it: where the forcer blocks notes (moving on, schedule
    # done) it blocks the Delta too, and leaves it open before a note that is due
    DELTA, DDIR = (338, 344), (344, 346)
    vocab = 346
    for f in (_RhythmForcer([(0, 1), (12, 1)], POS, ONSET, VEL, BAREND, delta_ranges=(DELTA, DDIR)),
              _VariationForcer("pitch", 0.0, [[(0, [(36, 1)]), (12, [(38, 1)])]], POS, ONSET, DUR, VEL, BAREND,
                               count_on_onset=True, delta_ranges=(DELTA, DDIR))):
        m = torch.ones(vocab, dtype=torch.bool)
        f.apply(m)
        assert bool(m[DELTA[0]])  # a note is due: its microtiming may come first
        f.advance("NoteOnset")
        m = torch.ones(vocab, dtype=torch.bool)
        f.apply(m)
        assert not bool(m[DELTA[0] : DELTA[1]].any()) and not bool(m[DDIR[0] : DDIR[1]].any())


def test_forcers_give_way_to_a_limit():
    # a chord of 3 under a polyphony limit, then a bar full under a density limit: the forcer moves on, then lets
    # the bar end, instead of leaving no legal token
    for f in (_RhythmForcer([(0, 3), (12, 1)], POS, ONSET, VEL, BAREND, fill_end_id=FILLEND),
              _VariationForcer("pitch", 0.0, [[(0, [(60, 3), (64, 3), (67, 3)]), (12, [(62, 3)])]],
                               POS, ONSET, DUR, VEL, BAREND, fill_end_id=FILLEND)):
        f.advance("NoteOnset")
        f.advance("NoteDuration")
        f.advance("NoteOnset")
        f.advance("NoteDuration")
        m = legal()
        m[ONSET[0] : ONSET[1]] = False   # the polyphony limit: no third note
        f.apply(m)
        assert bool(m[POS[0] + 12]) and bool(m.any())   # move to the next onset
        m = legal()
        m[ONSET[0] : ONSET[1]] = False
        m[POS[0] : POS[1]] = False   # the density limit: no more notes or moves
        f.apply(m)
        assert bool(m[BAREND]) or bool(m[FILLEND])   # the bar can end


def test_variation_forcer_waits_for_the_bar_opening_before_a_late_first_onset():
    # in AR the bar opens with Bar and TimeSig before any position is legal: the late first onset still applies
    bars = [[(12, [(60, 3)])]]
    f = _VariationForcer("full", 0.0, bars, POS, ONSET, DUR, VEL, BAREND)
    m = legal()
    m[POS[0] : POS[1]] = False
    m[ONSET[0] : ONSET[1]] = False   # only the bar's opening tokens are legal here
    f.apply(m)
    m = legal()
    f.apply(m)
    assert not onset_open(m) and int(m[POS[0] : POS[1]].sum()) == 1 and bool(m[POS[0] + 12])


def test_meter_keeper_holds_a_generated_bar_to_the_piece_meter():
    from midigpt.inference.session import _MeterKeeper

    TS = (338, 374)   # 36 meters; the piece is in value 19
    vocab = 374
    k = _MeterKeeper(TS, 19, [POS, ONSET, VEL, DUR], [BAREND, FILLEND])
    k.advance("Bar")
    m = torch.ones(vocab, dtype=torch.bool)
    k.apply(m)
    assert int(m[TS[0] : TS[1]].sum()) == 1 and bool(m[TS[0] + 19])   # only the piece's meter
    assert not onset_open(m) and not bool(m[POS[0] : POS[1]].any()) and not bool(m[BAREND])   # and it comes first
    k.advance("TimeSig")
    m = torch.ones(vocab, dtype=torch.bool)
    k.apply(m)
    assert onset_open(m) and int(m[TS[0] : TS[1]].sum()) == 1   # notes once it is written


def test_variation_forcer_resamples_at_least_one_note():
    # amount is the share of reference notes resampled, at least one: a tiny amount still changes a note, 1 all
    bars = [[(0, [(60, 3)]), (12, [(62, 3)]), (24, [(64, 3)]), (36, [(65, 3)])]]
    for amount, open_notes in ((0.0001, 1), (0.5, 2), (1.0, 4)):
        f = _VariationForcer("pitch", amount, bars, POS, ONSET, DUR, VEL, BAREND)
        free = 0
        for g in range(4):
            m = legal()
            f.apply(m)
            free += int(m[ONSET[0] : ONSET[1]].sum()) > 1   # pitch left open
            f.advance("NoteOnset")
            f.advance("NoteDuration")
            f.advance("TimeAbsolutePos")
        assert free == open_notes, (amount, free)


def test_variation_forcer_leaves_an_empty_reference_bar_alone():
    # no notes to vary: nothing forced or blocked in that bar (blocking every note left no legal token)
    f = _VariationForcer("pitch", 1.0, [[], [(0, [(60, 3)])]], POS, ONSET, DUR, VEL, BAREND, fill_end_id=FILLEND)
    m = legal()
    f.apply(m)
    assert bool(m.all())
    f.advance("FillInEnd")
    m = legal()
    f.apply(m)
    assert not bool(m[POS[0] : POS[1]].any())   # the next bar's schedule applies again
