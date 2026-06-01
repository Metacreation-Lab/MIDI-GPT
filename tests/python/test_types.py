"""Tests for midigpt._types (section 3.1)."""

from __future__ import annotations

import pytest

from midigpt._types import Bar, Note, Score, Track


def test_note_defaults():
    n = Note(pitch=60, velocity=80, onset_ticks=0, duration_ticks=120)
    assert n.pitch == 60
    assert n.velocity == 80
    assert n.onset_ticks == 0
    assert n.duration_ticks == 120
    assert n.delta == 0


def test_bar_defaults():
    b = Bar()
    assert b.notes == []
    assert b.ts_numerator == 4
    assert b.ts_denominator == 4
    assert b.beat_length == 4.0
    assert b.future is False


def test_track_defaults():
    t = Track()
    assert t.bars == []
    assert t.instrument == 0
    assert t.track_type == "melodic"
    assert t.attributes == {}


def test_score_defaults():
    s = Score()
    assert s.tracks == []
    assert s.resolution == 480
    assert s.tempo == 500000


def test_two_tracks_have_independent_bars():
    t1 = Track()
    t2 = Track()
    t1.bars.append(Bar())
    assert len(t2.bars) == 0, "bars list is shared between Track instances"


def test_two_tracks_have_independent_attributes():
    t1 = Track()
    t2 = Track()
    t1.attributes["x"] = 1
    assert "x" not in t2.attributes


def test_score_to_dict_from_dict_roundtrip_structure(simple_score):
    d = simple_score.to_dict()
    r = Score.from_dict(d)
    assert r.resolution == simple_score.resolution
    assert r.tempo == simple_score.tempo
    assert len(r.tracks) == len(simple_score.tracks)
    for ot, rt in zip(simple_score.tracks, r.tracks, strict=False):
        assert rt.instrument == ot.instrument
        assert rt.track_type == ot.track_type
        assert len(rt.bars) == len(ot.bars)


def test_score_to_dict_from_dict_roundtrip_preserves_note_pitches(simple_score):
    d = simple_score.to_dict()
    r = Score.from_dict(d)
    for ot, rt in zip(simple_score.tracks, r.tracks, strict=False):
        for ob, rb in zip(ot.bars, rt.bars, strict=False):
            for on, rn in zip(ob.notes, rb.notes, strict=False):
                assert rn.pitch == on.pitch
                assert rn.velocity == on.velocity
                assert rn.onset_ticks == on.onset_ticks
                assert rn.duration_ticks == on.duration_ticks
                assert rn.delta == on.delta


def test_score_to_dict_from_dict_preserves_future_flag():
    b = Bar(future=True)
    s = Score(tracks=[Track(bars=[b])])
    r = Score.from_dict(s.to_dict())
    assert r.tracks[0].bars[0].future is True


def test_score_to_dict_has_expected_keys(simple_score):
    d = simple_score.to_dict()
    assert set(d.keys()) >= {"resolution", "tempo", "tracks"}
    assert isinstance(d["tracks"], list)
    t = d["tracks"][0]
    assert "bars" in t and "instrument" in t and "track_type" in t


def test_empty_score_roundtrip():
    s = Score(tracks=[], resolution=480, tempo=500000)
    r = Score.from_dict(s.to_dict())
    assert r.resolution == 480
    assert r.tracks == []


def test_empty_bars_roundtrip(empty_bars_score):
    r = Score.from_dict(empty_bars_score.to_dict())
    assert len(r.tracks) == 1
    assert len(r.tracks[0].bars) == 4
    for b in r.tracks[0].bars:
        assert b.notes == []


def test_from_midi_returns_nonempty_score(sample_midi_path):
    s = Score.from_midi(str(sample_midi_path))
    assert len(s.tracks) > 0
    total_notes = sum(len(b.notes) for t in s.tracks for b in t.bars)
    assert total_notes > 0


def test_midi_roundtrip_preserves_track_and_note_count(sample_midi_path, tmp_path):
    s = Score.from_midi(str(sample_midi_path))
    orig_notes = sum(len(b.notes) for t in s.tracks for b in t.bars)
    out = tmp_path / "rt.mid"
    s.to_midi(str(out))
    s2 = Score.from_midi(str(out))
    # Empty meta-only tracks may be dropped by the MIDI writer; require that
    # every track carrying notes survives the roundtrip.
    nonempty_orig = sum(1 for t in s.tracks if any(b.notes for b in t.bars))
    nonempty_rt = sum(1 for t in s2.tracks if any(b.notes for b in t.bars))
    assert nonempty_rt == nonempty_orig
    assert sum(len(b.notes) for t in s2.tracks for b in t.bars) == orig_notes


def test_two_track_score_roundtrip(two_track_score):
    r = Score.from_dict(two_track_score.to_dict())
    assert len(r.tracks) == 2
    assert {t.track_type for t in r.tracks} == {"melodic", "drum"}
