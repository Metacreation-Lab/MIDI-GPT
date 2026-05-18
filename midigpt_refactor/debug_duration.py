import midigpt_refactor._core as _core
from midigpt_refactor._converters import from_cpp

def debug_note_duration():
    midi_path = "tests/comparison/midi/6354774_Macabre Waltz.mid"
    resolution = 12
    reader = _core.MidiReader(resolution)
    
    orig_cpp = reader.read(midi_path)
    orig_score = from_cpp(orig_cpp)
    
    target_note_orig = None
    target_bar_orig = None
    
    # The failing assertion was: track 0 bar 53 note 4
    # assert (43, 23, 120) == (43, 23, 0)
    try:
        target_bar_orig = orig_score.tracks[0].bars[53]
        orig_notes = sorted([(n.pitch, n.onset_ticks, n.duration_ticks) for n in target_bar_orig.notes])
        print(f"Original Notes: {orig_notes}")
    except Exception as e:
        print(f"Could not find original note: {e}")
        return

    tmp_path = "tmp_macabre.mid"
    orig_score.to_midi(tmp_path)
    
    rt_cpp = reader.read(tmp_path)
    rt_score = from_cpp(rt_cpp)
    
    try:
        target_bar_rt = rt_score.tracks[0].bars[53]
        rt_notes = sorted([(n.pitch, n.onset_ticks, n.duration_ticks) for n in target_bar_rt.notes])
        print(f"Roundtrip Notes: {rt_notes}")
    except Exception as e:
        print(f"Could not find roundtrip note: {e}")

if __name__ == "__main__":
    debug_note_duration()