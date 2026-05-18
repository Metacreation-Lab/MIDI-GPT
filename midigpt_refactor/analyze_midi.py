import midigpt_refactor._core as _core

def analyze_midi(path):
    score = _core.MidiReader(12).read(path)
    print(f"Analysis of {path}:")
    print(f"Resolution: {score.resolution}")
    print(f"Tracks: {len(score.tracks)}")
    for i, track in enumerate(score.tracks):
        print(f"Track {i}: {len(track.bars)} bars")
        total_notes = 0
        for b_idx, bar in enumerate(track.bars):
            notes_in_bar = len(bar.note_indices)
            total_notes += notes_in_bar
            print(f"  Bar {b_idx}: {notes_in_bar} notes, TS={bar.ts_numerator}/{bar.ts_denominator}")
            if notes_in_bar > 0:
                onsets = []
                for idx in bar.note_indices:
                    onsets.append(score.notes[idx].onset_ticks)
                print(f"    Onsets (min, max): {min(onsets)}, {max(onsets)}")
        print(f"  Total notes in track {i}: {total_notes}")

if __name__ == "__main__":
    analyze_midi("piano_bell_fix.mid")
