import pytest
import midigpt_refactor._core as _core
from midigpt_refactor._converters import from_cpp, to_cpp
from .conftest import silence_stdio, midi_files
import os

def test_midi_writer_bar_alignment():
    """
    Test that MIDI writing correctly aligns notes within their respective bars.
    This verifies that bar offsets are applied correctly during export.
    """
    resolution = 12
    # Create a dummy score with 2 tracks, each with 2 bars.
    # Track 0: Bar 0 has 1 note, Bar 1 has 1 note.
    # Track 1: Bar 0 has 1 note, Bar 1 has 1 note.
    # Total bars should be 2, each with length 4 beats.
    
    from midigpt_refactor._types import Score, Track, Bar, Note
    
    track0 = Track(instrument=0, track_type="melodic", bars=[
        Bar(notes=[Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=12)], ts_numerator=4, ts_denominator=4, beat_length=4.0),
        Bar(notes=[Note(pitch=62, velocity=100, onset_ticks=0, duration_ticks=12)], ts_numerator=4, ts_denominator=4, beat_length=4.0)
    ])
    
    score = Score(tracks=[track0], resolution=resolution, tempo=500000)
    
    output_path = "test_alignment.mid"
    try:
        # Write to MIDI
        score.to_midi(output_path)
        
        # Read back
        reader = _core.MidiReader(resolution)
        read_score = from_cpp(reader.read(output_path))
        
        assert len(read_score.tracks) == 1
        assert len(read_score.tracks[0].bars) == 2
        
        # Bar 0 note should be at onset 0
        assert len(read_score.tracks[0].bars[0].notes) == 1
        assert read_score.tracks[0].bars[0].notes[0].pitch == 60
        assert read_score.tracks[0].bars[0].notes[0].onset_ticks == 0
        
        # Bar 1 note should be at onset 0 (relative to bar start)
        assert len(read_score.tracks[0].bars[1].notes) == 1
        assert read_score.tracks[0].bars[1].notes[0].pitch == 62
        assert read_score.tracks[0].bars[1].notes[0].onset_ticks == 0
        
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

@pytest.mark.parametrize("midi_path", [str(p) for p in midi_files()][:5]) # Test on first 5 files
def test_full_roundtrip_integrity(midi_path):
    """
    Verifies that Score -> MIDI -> Score preserves the note positions and bar structure.
    """
    resolution = 12
    reader = _core.MidiReader(resolution)
    
    # 1. Read original
    with silence_stdio():
        orig_cpp = reader.read(midi_path)
    orig_score = from_cpp(orig_cpp)
    
    # 2. Write to temporary MIDI
    tmp_path = f"tmp_roundtrip_{os.path.basename(midi_path)}"
    try:
        orig_score.to_midi(tmp_path)
        
        # 3. Read back
        with silence_stdio():
            roundtrip_cpp = reader.read(tmp_path)
        roundtrip_score = from_cpp(roundtrip_cpp)
        
        # 4. Compare basic structure
        assert len(roundtrip_score.tracks) == len(orig_score.tracks)
        
        for t_idx in range(len(orig_score.tracks)):
            orig_track = orig_score.tracks[t_idx]
            rt_track = roundtrip_score.tracks[t_idx]
            
            # Bar counts should match if we handle empty bars at the end consistently
            # But at least the non-empty bars should match
            assert len(rt_track.bars) >= len(orig_track.bars)
            
            for b_idx in range(len(orig_track.bars)):
                orig_bar = orig_track.bars[b_idx]
                rt_bar = rt_track.bars[b_idx]
                
                assert len(rt_bar.notes) == len(orig_bar.notes), f"Note count mismatch in track {t_idx} bar {b_idx} of {midi_path}"
                
                # Compare notes (sorted by pitch and onset)
                orig_notes = sorted([(n.pitch, n.onset_ticks, n.duration_ticks) for n in orig_bar.notes])
                rt_notes = sorted([(n.pitch, n.onset_ticks, n.duration_ticks) for n in rt_bar.notes])
                
                for i in range(len(orig_notes)):
                    assert rt_notes[i] == orig_notes[i], f"Note data mismatch in track {t_idx} bar {b_idx} note {i}"
                    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
