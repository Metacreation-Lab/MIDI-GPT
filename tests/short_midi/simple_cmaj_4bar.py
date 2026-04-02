"""Create a simple 4-bar C major chord progression MIDI for testing.

Progression: C - F - G - C (each chord held for one bar, 4/4, 100 BPM)
Voicing: 3-note chords (root, third, fifth), piano, one octave.
"""
import mido, os

out_path = os.path.join(os.path.dirname(__file__), "simple_cmaj_4bar.mid")

TICKS_PER_BEAT = 480
BPM = 100
TEMPO = mido.bpm2tempo(BPM)
BAR = TICKS_PER_BEAT * 4  # 4 beats per bar

# C=60 F=65 G=67 chords (root, 3rd, 5th)
chords = [
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

for chord in chords:
    # note_on all notes at time=0 (delta)
    for i, note in enumerate(chord):
        track.append(mido.Message("note_on", note=note, velocity=80, time=0))
    # note_off all notes after one bar
    for i, note in enumerate(chord):
        track.append(mido.Message("note_off", note=note, velocity=0,
                                  time=BAR if i == 0 else 0))

mid.save(out_path)
print(f"Saved: {out_path}")
