"""Create a MIDI file of arbitrary length for testing.
Usage: python generate_long_midi.py <num_bars> <output_path>
"""
import mido, os, sys

def generate_midi(num_bars, out_path):
    TICKS_PER_BEAT = 480
    BPM = 100
    TEMPO = mido.bpm2tempo(BPM)
    BAR = TICKS_PER_BEAT * 4  # 4 beats per bar

    # C major progression: C - F - G - C (repeating)
    progression = [
        [60, 64, 67],  # C major
        [65, 69, 72],  # F major
        [67, 71, 74],  # G major
        [60, 64, 67],  # C major
    ]

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    track.append(mido.Message("program_change", program=0, time=0))  # piano

    for i in range(num_bars):
        chord = progression[i % len(progression)]
        # note_on all notes at time=0 (delta)
        for j, note in enumerate(chord):
            track.append(mido.Message("note_on", note=note, velocity=80, time=0))
        # note_off all notes after one bar
        for j, note in enumerate(chord):
            # The first message in the delta group carries the full bar length
            track.append(mido.Message("note_off", note=note, velocity=0,
                                      time=BAR if j == 0 else 0))

    mid.save(out_path)
    print(f"Saved {num_bars} bars to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_long_midi.py <num_bars> <output_path>")
        sys.exit(1)
    
    nb = int(sys.argv[1])
    path = sys.argv[2]
    generate_midi(nb, path)
