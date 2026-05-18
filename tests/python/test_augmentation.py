import pytest
from midigpt._types import Score, Track, Bar, Note
from midigpt.augmentation import Transpose, VelocityScale, TrackPermutation, BarWindow, InstrumentSwap, AugmentationPipeline

def test_transpose():
    n1 = Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)
    b1 = Bar(notes=[n1], ts_numerator=4, ts_denominator=4)
    t1 = Track(bars=[b1], instrument=0, track_type="melodic")
    
    n2 = Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)
    b2 = Bar(notes=[n2], ts_numerator=4, ts_denominator=4)
    t2 = Track(bars=[b2], instrument=0, track_type="drum")
    
    s = Score(tracks=[t1, t2])
    
    trans = Transpose(2)
    s2 = trans(s)
    
    assert s2.tracks[0].bars[0].notes[0].pitch == 62 # Melodic track shifted
    assert s2.tracks[1].bars[0].notes[0].pitch == 60 # Drum track unchanged

def test_pipeline():
    n1 = Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)
    b1 = Bar(notes=[n1], ts_numerator=4, ts_denominator=4)
    t1 = Track(bars=[b1], instrument=0, track_type="melodic")
    s = Score(tracks=[t1])
    
    pipeline = AugmentationPipeline.default_training()
    s2 = pipeline(s)
    
    assert isinstance(s2, Score)
    assert len(s2.tracks) == 1
    assert len(s2.tracks[0].bars) == 1
