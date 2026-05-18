import pytest
from midigpt_refactor._types import Score, Track, Bar, Note

def test_score_dataclasses():
    n = Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)
    b = Bar(notes=[n], ts_numerator=4, ts_denominator=4)
    t = Track(bars=[b], instrument=0, track_type="melodic")
    s = Score(tracks=[t], resolution=480, tempo=500000)
    
    assert len(s.tracks) == 1
    assert len(s.tracks[0].bars) == 1
    assert s.tracks[0].bars[0].notes[0].pitch == 60
    
    # Test dictionary serialization
    d = s.to_dict()
    assert d["resolution"] == 480
    assert d["tracks"][0]["instrument"] == 0
    assert d["tracks"][0]["bars"][0]["notes"][0]["pitch"] == 60
    
    s2 = Score.from_dict(d)
    assert len(s2.tracks) == 1
    assert s2.tracks[0].bars[0].notes[0].pitch == 60
