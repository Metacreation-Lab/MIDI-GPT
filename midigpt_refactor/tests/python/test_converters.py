import pytest
from midigpt_refactor._types import Score, Track, Bar, Note
from midigpt_refactor._converters import to_cpp, from_cpp

def test_converters_roundtrip():
    n1 = Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)
    n2 = Note(pitch=62, velocity=80, onset_ticks=480, duration_ticks=480)
    b1 = Bar(notes=[n1], ts_numerator=4, ts_denominator=4)
    b2 = Bar(notes=[n2], ts_numerator=4, ts_denominator=4)
    
    t = Track(bars=[b1, b2], instrument=5, track_type="melodic")
    s = Score(tracks=[t], resolution=480, tempo=500000)
    
    cpp_score = to_cpp(s)
    
    # Check C++ note pool pooling works
    assert len(cpp_score.notes) == 2
    assert cpp_score.notes[0].pitch == 60
    assert cpp_score.notes[1].pitch == 62
    assert len(cpp_score.tracks) == 1
    assert cpp_score.tracks[0].instrument == 5
    
    py_score = from_cpp(cpp_score)
    
    # Check python roundtrip reconstructs the correct object tree
    assert py_score.resolution == 480
    assert len(py_score.tracks) == 1
    assert py_score.tracks[0].instrument == 5
    assert py_score.tracks[0].track_type == "melodic"
    assert len(py_score.tracks[0].bars) == 2
    assert py_score.tracks[0].bars[0].notes[0].pitch == 60
    assert py_score.tracks[0].bars[1].notes[0].pitch == 62
