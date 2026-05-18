import pytest
from midigpt._types import Score, Track, Bar, Note
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.attributes import (
    NoteDensity, OnsetPolyphony, PitchRange, KeySignature,
    NoteDurationDistribution, SilenceProportion, BarLevelPitchClassSet,
)

def test_attribute_analyzer():
    n1 = Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)
    n2 = Note(pitch=64, velocity=100, onset_ticks=0, duration_ticks=480)
    b1 = Bar(notes=[n1, n2], ts_numerator=4, ts_denominator=4)
    t = Track(bars=[b1], instrument=0, track_type="melodic")
    s = Score(tracks=[t])

    analyzer = AttributeAnalyzer([
        NoteDensity(), OnsetPolyphony(), PitchRange(), KeySignature(),
        NoteDurationDistribution(), SilenceProportion(), BarLevelPitchClassSet(),
    ])
    
    track_tokens = {}
    for attr in analyzer._attrs.values():
        if attr.level != "track": continue
        print(f"Computing {attr.name}...")
        raw = attr.compute(s, 0)
        print(f"Done {attr.name}: {raw}")
        track_tokens[attr.name] = attr.quantize(raw)
        
    bar_tokens = {}
    for attr in analyzer._attrs.values():
        if attr.level != "bar": continue
        print(f"Computing {attr.name} (bar)...")
        raw = attr.compute(s, 0, 0)
        print(f"Done {attr.name} (bar): {raw}")
        bar_tokens[attr.name] = attr.quantize(raw)
    
    # Check Density (2 notes / 1 bar = 2.0 -> 2)
    assert track_tokens["note_density"] == 2
    # Check Polyphony (2 notes at same onset)
    assert track_tokens["onset_polyphony"] == 2
    # Check Pitch Range (64 - 60 = 4)
    assert track_tokens["pitch_range"] == 4
    # Check PitchClassSet (60 % 12, 64 % 12 -> 2 unique classes)
    assert bar_tokens["pitch_class_set"] == 2
    
    # Check NoteDurationDistribution (length 480 is index 3 in log scale)
    assert track_tokens["note_duration_dist"] == 3
    # Check SilenceProportion (1 quarter note active, 3 silent out of 4, prop = 0.75 -> 7)
    assert track_tokens["silence_proportion"] == 7
    # Check KeySignature (C Major or similar, mostly it shouldn't crash)
    assert "key_signature" in track_tokens
    
    # Check evaluate
    req = {"note_density": 2, "onset_polyphony": 2}
    eval_res = analyzer.evaluate(req, s, 0)
    assert eval_res["note_density"] == 1.0
    assert eval_res["onset_polyphony"] == 1.0
