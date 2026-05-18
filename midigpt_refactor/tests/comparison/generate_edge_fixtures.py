"""
Generate synthetic MIDI edge-case fixtures for comparison tests.

Run once to populate tests/comparison/midi/edge_*.mid:
    .venv/bin/python3 tests/comparison/generate_edge_fixtures.py
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "midigpt_refactor" / "src" / "python"))

from midigpt_refactor._types import Score, Track, Bar, Note

OUT_DIR = Path(__file__).parent / "midi"
RES = 480                   # standard MIDI resolution (ticks per quarter)
QUARTER = RES               # 480 ticks
HALF    = 2 * RES           # 960
WHOLE   = 4 * RES           # 1920 (one 4/4 bar)


def _bar44(*notes, **kw) -> Bar:
    return Bar(notes=list(notes), ts_numerator=4, ts_denominator=4,
               beat_length=4.0, **kw)

def _bar34(*notes, **kw) -> Bar:
    return Bar(notes=list(notes), ts_numerator=3, ts_denominator=4,
               beat_length=3.0, **kw)

def n(pitch, vel=80, onset=0, dur=QUARTER) -> Note:
    return Note(pitch=pitch, velocity=vel, onset_ticks=onset, duration_ticks=dur)


# ── edge_cross_bar_sustain.mid ───────────────────────────────────────────────
# Notes that start near the end of a bar and sustain into the next bar.
# Tests IterateAndConvert orphan-closure logic vs MidiReader behaviour.

def make_cross_bar_sustain(path: Path) -> None:
    score = Score(resolution=RES)
    track = Track(instrument=0, track_type="melodic")

    for bar_i in range(8):
        pitch_base = 60 + bar_i * 2
        bar = _bar44(
            n(pitch_base,     80, 0,              HALF),       # normal note
            n(pitch_base + 4, 70, WHOLE - QUARTER, HALF),      # sustains into next bar
        )
        track.bars.append(bar)

    score.tracks.append(track)
    score.to_midi(str(path))
    print(f"  wrote {path.name}  ({len(track.bars)} bars)")


# ── edge_time_sig_change.mid ─────────────────────────────────────────────────
# Piece that switches from 4/4 to 3/4 mid-way. Tests variable-length bar
# handling in both tokenizers.

def make_time_sig_change(path: Path) -> None:
    score = Score(resolution=RES)
    track = Track(instrument=0, track_type="melodic")

    # 5 bars of 4/4
    for i in range(5):
        track.bars.append(_bar44(
            n(60 + i, 80, 0,       QUARTER),
            n(64 + i, 75, QUARTER, QUARTER),
            n(67 + i, 70, HALF,    HALF),
        ))

    # 5 bars of 3/4 (beat_length=3, bar = 3 * 480 = 1440 ticks)
    bar34_len = 3 * RES
    for i in range(5):
        track.bars.append(_bar34(
            n(60 + i, 80, 0,       QUARTER),
            n(64 + i, 75, QUARTER, QUARTER),
            n(67 + i, 70, HALF,    QUARTER),
        ))

    score.tracks.append(track)
    score.to_midi(str(path))
    print(f"  wrote {path.name}  ({len(track.bars)} bars, 5×4/4 + 5×3/4)")


# ── edge_dense_polyphony.mid ─────────────────────────────────────────────────
# 8 simultaneous notes per bar — stresses polyphony attribute computation.

def make_dense_polyphony(path: Path) -> None:
    score = Score(resolution=RES)
    track = Track(instrument=0, track_type="melodic")

    CHORD = [48, 52, 55, 59, 62, 64, 67, 71]   # Cmaj9 spread across 2 octaves

    for _ in range(8):
        notes = [n(p, 80, 0, HALF) for p in CHORD]
        # Add a top melody note that sustains the whole bar
        notes.append(n(72, 90, 0, WHOLE))
        track.bars.append(_bar44(*notes))

    score.tracks.append(track)
    score.to_midi(str(path))
    print(f"  wrote {path.name}  ({len(track.bars)} bars, {len(CHORD)+1} notes/bar)")


# ── edge_very_long_note.mid ──────────────────────────────────────────────────
# A single note whose duration_ticks spans 3 bars.
# Extreme cross-bar sustain case.

def make_very_long_note(path: Path) -> None:
    score = Score(resolution=RES)
    track = Track(instrument=0, track_type="melodic")

    # Bar 0: long note (duration = 3 bars = 5760 ticks) + short filler
    track.bars.append(_bar44(
        n(60, 90, 0, 3 * WHOLE),     # sustains across bars 1 and 2
        n(67, 70, HALF, QUARTER),
    ))
    # Bars 1–2: filler (the long note's tail is here in MIDI time)
    for i in range(2):
        track.bars.append(_bar44(
            n(64 + i * 2, 70, 0, QUARTER),
            n(69 + i,     65, HALF, QUARTER),
        ))
    # Bars 3–7: normal bars so test matrix has enough bars
    for i in range(5):
        track.bars.append(_bar44(
            n(60 + i, 80, 0, QUARTER),
            n(64 + i, 75, QUARTER, QUARTER),
        ))

    score.tracks.append(track)
    score.to_midi(str(path))
    print(f"  wrote {path.name}  ({len(track.bars)} bars, 3-bar sustain)")


# ── edge_drums_and_melodic.mid ───────────────────────────────────────────────
# Drum track + melodic track with cross-bar notes.
# Tests multi-track tokenization and the drum-note-duration special case (d=1).

def make_drums_and_melodic(path: Path) -> None:
    score = Score(resolution=RES)

    # Melodic: piano (inst 0)
    mel = Track(instrument=0, track_type="melodic")
    for i in range(8):
        mel.bars.append(_bar44(
            n(60 + i, 80, 0,              HALF),
            n(64,     75, WHOLE - QUARTER, HALF),   # cross-bar
        ))

    # Drums: GM kit (inst 0, track_type="drum")
    # GM: kick=36, snare=38, hihat=42
    drm = Track(instrument=0, track_type="drum")
    for _ in range(8):
        drm.bars.append(_bar44(
            n(36, 90, 0,              QUARTER),   # kick on 1
            n(42, 60, QUARTER,        QUARTER),   # hihat on 2
            n(38, 80, HALF,           QUARTER),   # snare on 3
            n(42, 60, HALF + QUARTER, QUARTER),   # hihat on 4
        ))

    score.tracks.extend([mel, drm])
    score.to_midi(str(path))
    print(f"  wrote {path.name}  (2 tracks: piano + drums, {len(mel.bars)} bars each)")


# ── main ─────────────────────────────────────────────────────────────────────

FIXTURES = {
    "edge_cross_bar_sustain.mid": make_cross_bar_sustain,
    "edge_time_sig_change.mid":   make_time_sig_change,
    "edge_dense_polyphony.mid":   make_dense_polyphony,
    "edge_very_long_note.mid":    make_very_long_note,
    "edge_drums_and_melodic.mid": make_drums_and_melodic,
}

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Generating edge-case MIDI fixtures → {OUT_DIR}")
    for name, fn in FIXTURES.items():
        fn(OUT_DIR / name)
    print("Done.")
