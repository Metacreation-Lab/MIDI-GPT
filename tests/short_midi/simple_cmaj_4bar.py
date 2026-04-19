"""Create a simple C major chord progression MIDI for testing.

Progression: C - F - G - C (repeating), each chord held for one bar, 4/4, 100 BPM.
Voicing: 3-note chords (root, third, fifth), piano, one octave.

Usage:
    python simple_cmaj_4bar.py [num_bars] [--drums]

    num_bars  Power of two (1, 2, 4, 8, 16, ...). Default: 4.
    --drums   Add a standard drum track on MIDI channel 10.

Output filename reflects options, e.g. simple_cmaj_8bar_drums.mid
"""
import argparse
import math
import os
import mido

TICKS_PER_BEAT = 480
BPM = 100
TEMPO = mido.bpm2tempo(BPM)
BAR = TICKS_PER_BEAT * 4  # ticks in one 4/4 bar
BEAT = TICKS_PER_BEAT
HALF_BEAT = TICKS_PER_BEAT // 2

# C major progression (root, 3rd, 5th)
PROGRESSION = [
    [60, 64, 67],  # C major
    [65, 69, 72],  # F major
    [67, 71, 74],  # G major
    [60, 64, 67],  # C major
]

# GM drum note numbers
KICK  = 36
SNARE = 38
HIHAT = 42


def is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def build_piano_track(num_bars: int) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))  # piano
    for i in range(num_bars):
        chord = PROGRESSION[i % len(PROGRESSION)]
        for note in chord:
            track.append(mido.Message("note_on", channel=0, note=note, velocity=80, time=0))
        for j, note in enumerate(chord):
            track.append(mido.Message("note_off", channel=0, note=note, velocity=0,
                                      time=BAR if j == 0 else 0))
    return track


def build_drum_track(num_bars: int) -> mido.MidiTrack:
    """Standard 4/4 rock drum pattern per bar:
       Beat 1: kick + hihat
       Beat 2: snare + hihat
       Beat 3: kick + hihat
       Beat 4: snare + hihat
       Off-beats (the 'and' of each beat): hihat
    """
    # Per-bar hit pattern: (tick_offset_in_bar, note, velocity)
    hits = []
    for beat in range(4):
        t = beat * BEAT
        hits.append((t, HIHAT, 70))
        if beat % 2 == 0:
            hits.append((t, KICK, 90))   # beats 1 & 3
        else:
            hits.append((t, SNARE, 85))  # beats 2 & 4
        hits.append((t + HALF_BEAT, HIHAT, 60))  # off-beat 8th
    hits.sort(key=lambda x: (x[0], x[1]))

    # Collect all events as absolute ticks first, then convert to delta
    abs_events = []  # (abs_tick, "on"/"off", note, velocity)
    for bar in range(num_bars):
        bar_start = bar * BAR
        for (offset, note, vel) in hits:
            t = bar_start + offset
            abs_events.append((t,     "on",  note, vel))
            abs_events.append((t + 1, "off", note, 0))
    abs_events.sort(key=lambda x: (x[0], x[1]))

    track = mido.MidiTrack()
    prev = 0
    for (abs_t, kind, note, vel) in abs_events:
        delta = abs_t - prev
        prev = abs_t
        msg_type = "note_on" if kind == "on" else "note_off"
        track.append(mido.Message(msg_type, channel=9, note=note,
                                  velocity=vel, time=delta))
    return track


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("num_bars", nargs="?", type=int, default=4,
                        help="Number of bars (must be a power of two, default: 4)")
    parser.add_argument("--drums", action="store_true",
                        help="Add a standard drum track")
    args = parser.parse_args()

    if not is_power_of_two(args.num_bars):
        parser.error(f"num_bars must be a power of two, got {args.num_bars}")

    suffix = f"_{args.num_bars}bar" + ("_drums" if args.drums else "")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"simple_cmaj{suffix}.mid")

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    mid.tracks.append(build_piano_track(args.num_bars))
    if args.drums:
        mid.tracks.append(build_drum_track(args.num_bars))

    mid.save(out_path)
    print(f"Saved {args.num_bars} bar(s){' + drums' if args.drums else ''}: {out_path}")


if __name__ == "__main__":
    main()
