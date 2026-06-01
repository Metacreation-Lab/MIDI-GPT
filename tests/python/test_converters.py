"""Tests for midigpt._converters (section 3.2)."""
from __future__ import annotations
import pytest
import midigpt._core as _core
from midigpt._types import Note, Bar, Track, Score
from midigpt._converters import to_cpp, from_cpp


def _make_score_with_notes(pitches, res=480, instrument=0, track_type="melodic"):
    notes = [Note(pitch=p, velocity=80, onset_ticks=i * 120, duration_ticks=100, delta=0)
             for i, p in enumerate(pitches)]
    bar = Bar(notes=notes)
    track = Track(bars=[bar], instrument=instrument, track_type=track_type)
    return Score(tracks=[track], resolution=res, tempo=500000)


def test_to_cpp_returns_cpp_score():
    s = _make_score_with_notes([60, 62, 64])
    cpp = to_cpp(s)
    assert isinstance(cpp, _core.Score)


def test_roundtrip_preserves_note_pitches():
    pitches = [60, 62, 64, 67]
    s = _make_score_with_notes(pitches)
    r = from_cpp(to_cpp(s))
    result_pitches = [n.pitch for n in r.tracks[0].bars[0].notes]
    assert result_pitches == pitches


def test_roundtrip_preserves_note_velocities():
    s = _make_score_with_notes([60, 62])
    r = from_cpp(to_cpp(s))
    vels = [n.velocity for n in r.tracks[0].bars[0].notes]
    assert vels == [80, 80]


def test_roundtrip_preserves_onset_ticks():
    s = _make_score_with_notes([60, 62, 64])
    r = from_cpp(to_cpp(s))
    onsets = [n.onset_ticks for n in r.tracks[0].bars[0].notes]
    assert onsets == [0, 120, 240]


def test_roundtrip_preserves_duration_ticks():
    s = _make_score_with_notes([60, 62])
    r = from_cpp(to_cpp(s))
    durs = [n.duration_ticks for n in r.tracks[0].bars[0].notes]
    assert durs == [100, 100]


def test_roundtrip_preserves_delta():
    notes = [Note(pitch=60, velocity=80, onset_ticks=0, duration_ticks=100, delta=5)]
    s = Score(tracks=[Track(bars=[Bar(notes=notes)])], resolution=480)
    r = from_cpp(to_cpp(s))
    assert r.tracks[0].bars[0].notes[0].delta == 5


def test_roundtrip_preserves_resolution():
    s = _make_score_with_notes([60], res=12)
    r = from_cpp(to_cpp(s))
    assert r.resolution == 12


def test_roundtrip_preserves_resolution_480():
    s = _make_score_with_notes([60], res=480)
    r = from_cpp(to_cpp(s))
    assert r.resolution == 480


def test_melodic_track_type_roundtrip():
    s = _make_score_with_notes([60], track_type="melodic")
    r = from_cpp(to_cpp(s))
    assert r.tracks[0].track_type == "melodic"


def test_drum_track_type_roundtrip():
    notes = [Note(pitch=36, velocity=100, onset_ticks=0, duration_ticks=12, delta=0)]
    drum = Track(bars=[Bar(notes=notes)], instrument=0, track_type="drum")
    s = Score(tracks=[drum], resolution=480, tempo=500000)
    r = from_cpp(to_cpp(s))
    assert r.tracks[0].track_type == "drum"


def test_multiple_tracks_roundtrip():
    mel_notes = [Note(pitch=60, velocity=80, onset_ticks=0, duration_ticks=100)]
    drum_notes = [Note(pitch=36, velocity=100, onset_ticks=0, duration_ticks=50)]
    mel = Track(bars=[Bar(notes=mel_notes)], instrument=0, track_type="melodic")
    drm = Track(bars=[Bar(notes=drum_notes)], instrument=0, track_type="drum")
    s = Score(tracks=[mel, drm], resolution=480, tempo=500000)
    r = from_cpp(to_cpp(s))
    assert len(r.tracks) == 2
    assert {t.track_type for t in r.tracks} == {"melodic", "drum"}


def test_multiple_bars_roundtrip():
    bars = [Bar(notes=[Note(pitch=60 + i, velocity=80, onset_ticks=0, duration_ticks=100)])
            for i in range(4)]
    s = Score(tracks=[Track(bars=bars)], resolution=480)
    r = from_cpp(to_cpp(s))
    assert len(r.tracks[0].bars) == 4
    for i, bar in enumerate(r.tracks[0].bars):
        assert bar.notes[0].pitch == 60 + i


def test_notes_at_bar_boundaries_roundtrip():
    notes = [
        Note(pitch=60, velocity=80, onset_ticks=0, duration_ticks=100),
        Note(pitch=62, velocity=80, onset_ticks=479, duration_ticks=1),
    ]
    s = Score(tracks=[Track(bars=[Bar(notes=notes)])], resolution=480)
    r = from_cpp(to_cpp(s))
    onsets = [n.onset_ticks for n in r.tracks[0].bars[0].notes]
    assert 0 in onsets
    assert 479 in onsets


def test_to_cpp_idempotent_on_cpp_score():
    s = _make_score_with_notes([60])
    cpp = to_cpp(s)
    assert to_cpp(cpp) is cpp


def test_empty_track_roundtrip():
    s = Score(tracks=[Track(bars=[Bar()], instrument=0, track_type="melodic")], resolution=480)
    r = from_cpp(to_cpp(s))
    assert r.tracks[0].bars[0].notes == []


def test_instrument_preserved_roundtrip():
    s = Score(tracks=[Track(bars=[Bar()], instrument=42, track_type="melodic")], resolution=480)
    r = from_cpp(to_cpp(s))
    assert r.tracks[0].instrument == 42
