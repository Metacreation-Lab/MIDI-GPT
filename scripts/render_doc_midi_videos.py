#!/usr/bin/env python3
"""Render MIDI-GPT documentation examples to MP4 piano-roll videos.

This is intentionally repo-local and independent of the old transition demo
pipeline.  It consumes generated MIDI files from docs/assets/midi, renders a
simple synthesized audio track, and encodes an MP4 with a piano-roll
visualization for each example.

Each track is rendered in its own horizontal lane. Generated tracks are
highlighted in gold; context tracks use distinct palette colors.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import symusic
from PIL import Image, ImageDraw, ImageFont


DEFAULT_EXAMPLES = [
    # ── Californication: drums ─────────────────────────────────────────────
    {
        "id": "track_sampling_drums_calif_density1",
        "task": "Track Sampling",
        "title": "Californication: sparse drums",
        "midi": "sampling_drums_calif_density1.mid",
        "prompt": "prompt_californication.mid",
        "controls": "note_density = 1",
        "track_names": ["acoustic_guitar_steel", "electric_bass_finger", "drums"],
        "generated_tracks": [2],
    },
    {
        "id": "track_sampling_drums_calif_density3",
        "task": "Track Sampling",
        "title": "Californication: medium drums",
        "midi": "sampling_drums_calif_density3.mid",
        "prompt": "prompt_californication.mid",
        "controls": "note_density = 3",
        "track_names": ["acoustic_guitar_steel", "electric_bass_finger", "drums"],
        "generated_tracks": [2],
    },
    {
        "id": "track_sampling_drums_calif_density8",
        "task": "Track Sampling",
        "title": "Californication: dense drums",
        "midi": "sampling_drums_calif_density8.mid",
        "prompt": "prompt_californication.mid",
        "controls": "note_density = 8",
        "track_names": ["acoustic_guitar_steel", "electric_bass_finger", "drums"],
        "generated_tracks": [2],
    },
    # ── Bach BWV 1052: cello ───────────────────────────────────────────────
    {
        "id": "track_sampling_cello_staccato",
        "task": "Track Sampling",
        "title": "Bach BWV 1052: staccato cello",
        "midi": "sampling_cello_staccato.mid",
        "prompt": "prompt_bach.mid",
        "controls": "max_polyphony = 0, max_note_duration = 1",
        "track_names": ["violin", "viola", "cello", "cello"],
        "generated_tracks": [3],
    },
    {
        "id": "track_sampling_cello_dur_16th",
        "task": "Track Sampling",
        "title": "Bach BWV 1052: cello 16th notes",
        "midi": "sampling_cello_dur_16th.mid",
        "prompt": "prompt_bach.mid",
        "controls": "max_polyphony = 0, max_note_duration = 1",
        "track_names": ["violin", "viola", "cello", "cello"],
        "generated_tracks": [3],
    },
    {
        "id": "track_sampling_cello_dur_8th",
        "task": "Track Sampling",
        "title": "Bach BWV 1052: cello 8th notes",
        "midi": "sampling_cello_dur_8th.mid",
        "prompt": "prompt_bach.mid",
        "controls": "max_polyphony = 0, max_note_duration = 2",
        "track_names": ["violin", "viola", "cello", "cello"],
        "generated_tracks": [3],
    },
    {
        "id": "track_sampling_cello_dur_4th",
        "task": "Track Sampling",
        "title": "Bach BWV 1052: cello quarter notes",
        "midi": "sampling_cello_dur_4th.mid",
        "prompt": "prompt_bach.mid",
        "controls": "max_polyphony = 0, max_note_duration = 3",
        "track_names": ["violin", "viola", "cello", "cello"],
        "generated_tracks": [3],
    },
    {
        "id": "track_sampling_cello_dur_half",
        "task": "Track Sampling",
        "title": "Bach BWV 1052: cello half notes",
        "midi": "sampling_cello_dur_half.mid",
        "prompt": "prompt_bach.mid",
        "controls": "max_polyphony = 0, max_note_duration = 4",
        "track_names": ["violin", "viola", "cello", "cello"],
        "generated_tracks": [3],
    },
    {
        "id": "track_sampling_cello_sustained",
        "task": "Track Sampling",
        "title": "Bach BWV 1052: sustained cello",
        "midi": "sampling_cello_sustained.mid",
        "prompt": "prompt_bach.mid",
        "controls": "max_polyphony = 0, min_note_duration = 4",
        "track_names": ["violin", "viola", "cello", "cello"],
        "generated_tracks": [3],
    },
    # ── Black Magic Woman: guitar solo ─────────────────────────────────────
    {
        "id": "bmw_solo_resample1",
        "task": "Track Sampling",
        "title": "Black Magic Woman: guitar solo (1)",
        "midi": "bmw_solo_resample1.mid",
        "prompt": "prompt_bmw.mid",
        "controls": "model-sampled controls",
        "track_names": ["electric_bass_finger", "overdriven_guitar", "drums"],
        "generated_tracks": [1],
    },
    {
        "id": "bmw_solo_resample2",
        "task": "Track Sampling",
        "title": "Black Magic Woman: guitar solo (2)",
        "midi": "bmw_solo_resample2.mid",
        "prompt": "prompt_bmw.mid",
        "controls": "model-sampled controls",
        "track_names": ["electric_bass_finger", "overdriven_guitar", "drums"],
        "generated_tracks": [1],
    },
    # ── SSBM All Star: bar infill harp ────────────────────────────────────
    {
        "id": "infill_allstar_dur_16th",
        "task": "Bar Infilling",
        "title": "SSBM All Star: harp 16th notes",
        "midi": "infill_allstar_dur_16th.mid",
        "prompt": "prompt_harp.mid",
        "controls": "max_note_duration = 1",
        "target_bars": [4, 5, 6, 7],
        "track_names": ["tubular_bells", "string_ensemble_1", "orchestral_harp"],
        "generated_tracks": [2],
    },
    {
        "id": "infill_allstar_dur_8th",
        "task": "Bar Infilling",
        "title": "SSBM All Star: harp 8th notes",
        "midi": "infill_allstar_dur_8th.mid",
        "prompt": "prompt_harp.mid",
        "controls": "max_note_duration = 2",
        "target_bars": [4, 5, 6, 7],
        "track_names": ["tubular_bells", "string_ensemble_1", "orchestral_harp"],
        "generated_tracks": [2],
    },
    # ── SSBM All Star: AR harp (original examples) ────────────────────────
    {
        "id": "harp_monophonic",
        "task": "Autoregressive Continuation",
        "title": "SSBM All Star: rapid arpeggios",
        "midi": "harp_monophonic.mid",
        "prompt": "prompt_harp.mid",
        "controls": "max_note_duration = 1",
        "target_bars": [4, 5, 6, 7],
        "track_names": ["tubular_bells", "string_ensemble_1", "orchestral_harp"],
        "generated_tracks": [2],
    },
    {
        "id": "harp_chordal",
        "task": "Autoregressive Continuation",
        "title": "SSBM All Star: slower arpeggios",
        "midi": "harp_chordal.mid",
        "prompt": "prompt_harp.mid",
        "controls": "max_polyphony = 4",
        "target_bars": [4, 5, 6, 7],
        "track_names": ["tubular_bells", "string_ensemble_1", "orchestral_harp"],
        "generated_tracks": [2],
    },
    # ── Joe Hisaishi Summer: multi-step composition ────────────────────────
    {
        "id": "summer_final",
        "task": "Multi-Step Composition",
        "title": "Summer: continuation + guitar + drums",
        "midi": "summer_final.mid",
        "prompt": "prompt_ar.mid",
        "controls": "guitar: max_polyphony=0, max_note_duration=1  |  drums: note_density=9",
        "target_bars": [4, 5, 6, 7],
        "track_names": [
            "acoustic_grand_piano",
            "electric_bass_finger",
            "synth_strings_1",
            "acoustic_guitar_nylon",
            "drums",
        ],
        "generated_tracks": [3, 4],
    },
]


# ── Colors ────────────────────────────────────────────────────────────────────

TRACK_COLORS = [
    (51, 149, 255),
    (255, 173, 51),
    (86, 205, 127),
    (237, 92, 112),
    (169, 116, 255),
    (42, 194, 198),
    (238, 217, 74),
    (244, 134, 193),
]
GEN_COLOR: tuple[int, int, int] = (255, 200, 50)

# Short display labels for common instrument names
_SHORT: dict[str, str] = {
    "acoustic_grand_piano": "Piano",
    "electric_bass_finger": "Bass",
    "acoustic_bass": "Bass",
    "acoustic_guitar_steel": "Guitar",
    "acoustic_guitar_nylon": "Guitar",
    "electric_guitar_clean": "Guitar",
    "electric_guitar_jazz": "Guitar",
    "overdriven_guitar": "Guitar OD",
    "drawbar_organ": "Organ",
    "rock_organ": "Organ",
    "drums": "Drums",
    "violin": "Violin",
    "viola": "Viola",
    "cello": "Cello",
    "contrabass": "Contrabass",
    "orchestral_harp": "Harp",
    "string_ensemble_1": "Strings",
    "string_ensemble_2": "Strings",
    "synth_strings_1": "Synth Str.",
    "tubular_bells": "Bells",
    "glockenspiel": "Bells",
    "lead_synth": "Lead Synth",
    "lead_2_sawtooth": "Sawtooth",
    "bass": "Bass",
    "pad_3_polysynth": "Synth Pad",
    "synth_bass_1": "Synth Bass",
}


def _short_name(name: str) -> str:
    return _SHORT.get(name, name.replace("_", " ").title())


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NoteEvent:
    track: int
    pitch: int
    velocity: int
    start_tick: int
    end_tick: int
    start_s: float
    end_s: float
    is_drum: bool


# ── MIDI loading helpers ───────────────────────────────────────────────────────

def _tpq(score: symusic.Score) -> int:
    return int(getattr(score, "ticks_per_quarter", getattr(score, "tpq", 480)))


def _score_end(score: symusic.Score) -> int:
    end_attr = getattr(score, "end", None)
    if callable(end_attr):
        return int(end_attr())
    return int(end_attr or 0)


def _tick_to_seconds(tick: int, tempos, tpq: int) -> float:
    elapsed = 0.0
    prev_tick = 0
    prev_qpm = 120.0
    for tempo in sorted(tempos, key=lambda t: int(t.time)):
        tempo_tick = int(tempo.time)
        if tempo_tick >= tick:
            break
        elapsed += (tempo_tick - prev_tick) / tpq * (60.0 / prev_qpm)
        prev_tick = tempo_tick
        prev_qpm = float(tempo.qpm)
    elapsed += (tick - prev_tick) / tpq * (60.0 / prev_qpm)
    return elapsed


def load_notes(midi_path: Path) -> tuple[symusic.Score, list[NoteEvent], float]:
    score = symusic.Score(str(midi_path))
    tpq = _tpq(score)
    notes: list[NoteEvent] = []
    for track_idx, track in enumerate(score.tracks):
        is_drum = bool(getattr(track, "is_drum", False))
        for note in track.notes:
            start_tick = int(note.time)
            duration = max(1, int(note.duration))
            end_tick = start_tick + duration
            notes.append(
                NoteEvent(
                    track=track_idx,
                    pitch=int(note.pitch),
                    velocity=int(note.velocity),
                    start_tick=start_tick,
                    end_tick=end_tick,
                    start_s=_tick_to_seconds(start_tick, score.tempos, tpq),
                    end_s=_tick_to_seconds(end_tick, score.tempos, tpq),
                    is_drum=is_drum,
                )
            )
    notes.sort(key=lambda n: (n.start_s, n.track, n.pitch))
    duration_s = max(
        [_tick_to_seconds(_score_end(score), score.tempos, tpq), *(n.end_s for n in notes), 1.0]
    )
    return score, notes, duration_s


# ── Audio synthesis ───────────────────────────────────────────────────────────

def write_audio(notes: list[NoteEvent], duration_s: float, wav_path: Path, sr: int = 44100) -> None:
    n_samples = int(math.ceil((duration_s + 0.35) * sr))
    audio = np.zeros(n_samples, dtype=np.float32)

    for note in notes:
        start = max(0, int(note.start_s * sr))
        end = min(n_samples, max(start + 1, int(note.end_s * sr)))
        length = end - start
        if length <= 0:
            continue

        amp = 0.07 + 0.18 * (note.velocity / 127.0)
        if note.is_drum:
            t = np.arange(length, dtype=np.float32) / sr
            if note.pitch in {35, 36}:
                tone = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 18)
            elif note.pitch in {38, 40}:
                rng = np.random.default_rng(note.pitch + note.start_tick)
                tone = rng.uniform(-1, 1, length).astype(np.float32) * np.exp(-t * 35)
            else:
                tone = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 26)
            audio[start:end] += amp * tone
            continue

        freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
        t = np.arange(length, dtype=np.float32) / sr
        wave_data = (
            0.78 * np.sin(2 * np.pi * freq * t)
            + 0.18 * np.sin(2 * np.pi * freq * 2 * t)
            + 0.04 * np.sin(2 * np.pi * freq * 3 * t)
        )
        attack = max(1, min(length, int(0.008 * sr)))
        release = max(1, min(length, int(0.04 * sr)))
        env = np.ones(length, dtype=np.float32)
        env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
        env[-release:] *= np.linspace(1, 0, release, dtype=np.float32)
        audio[start:end] += amp * wave_data.astype(np.float32) * env

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / max(peak, 1.0) * 0.88
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    stereo = np.column_stack([pcm, pcm]).ravel()

    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(stereo.tobytes())


def find_soundfont(value: str | None) -> Path | None:
    if value is None:
        return None
    if value != "auto":
        path = Path(value).expanduser()
        return path if path.exists() else None

    candidates = [
        Path("src/python/midigpt/osc/studio/static/sf2/arachno.sf2"),
        Path("src/python/midigpt/osc/studio/static/sf2/default.sf2"),
        Path.home() / "Downloads",
        Path.home() / "Downloads" / "Arachno SoundFont - Version 1.0.sf2",
        Path.home() / "Downloads" / "arachno.sf2",
        Path.home() / "Downloads" / "Arachno.sf2",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in {".sf2", ".sf3"}:
            return candidate
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        matches = sorted(
            [
                *downloads.glob("*[Aa]rachno*.sf2"),
                *downloads.glob("*[Aa]rachno*.sf3"),
                *downloads.rglob("*[Aa]rachno*.sf2"),
                *downloads.rglob("*[Aa]rachno*.sf3"),
                *downloads.glob("*.sf2"),
                *downloads.glob("*.sf3"),
            ]
        )
        if matches:
            return matches[0]
    return None


def write_audio_with_fluidsynth(
    midi_path: Path,
    wav_path: Path,
    soundfont: Path | None,
    sample_rate: int = 44100,
) -> bool:
    if soundfont is None:
        return False
    fluidsynth = shutil.which("fluidsynth")
    if fluidsynth is None:
        print("[warn] fluidsynth not found; using fallback synth")
        return False
    cmd = [
        fluidsynth,
        "-ni",
        "-F",
        str(wav_path),
        "-r",
        str(sample_rate),
        str(soundfont),
        str(midi_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"[warn] fluidsynth failed for {midi_path.name}: {exc.stderr.strip()}")
        return False
    return wav_path.exists() and wav_path.stat().st_size > 0


# ── Font helpers ──────────────────────────────────────────────────────────────

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _bar_boundaries(score: symusic.Score, duration_s: float) -> list[float]:
    tpq = _tpq(score)
    ticks_per_bar = tpq * 4
    max_tick = max(_score_end(score), ticks_per_bar)
    n_bars = max(1, math.ceil(max_tick / ticks_per_bar))
    return [_tick_to_seconds(i * ticks_per_bar, score.tempos, tpq) for i in range(n_bars + 1)]


def _pitch_range(notes: list[NoteEvent]) -> tuple[int, int]:
    melodic = [n for n in notes if not n.is_drum]
    if not melodic:
        return 24, 80
    mn = max(0, min(n.pitch for n in melodic) - 3)
    mx = min(127, max(n.pitch for n in melodic) + 3)
    if mx - mn < 12:
        mid = (mn + mx) // 2
        mn, mx = max(0, mid - 8), min(127, mid + 8)
    return mn, mx


# ── Frame renderer ────────────────────────────────────────────────────────────

def draw_frame(
    score: symusic.Score,
    notes: list[NoteEvent],
    duration_s: float,
    meta: dict,
    t_now: float,
    size: tuple[int, int],
) -> Image.Image:
    width, height = size
    img = Image.new("RGB", size, (13, 17, 23))
    draw = ImageDraw.Draw(img)

    title_font = _font(32, bold=True)
    label_font = _font(18)
    small_font = _font(14)
    tag_font = _font(12, bold=True)

    # ── Header ────────────────────────────────────────────────────────────
    header_h = 118
    footer_h = 34
    label_w = 136   # left column width for track labels
    right = width - 20

    draw.rectangle((0, 0, width, header_h), fill=(20, 27, 36))
    draw.text((34, 16), meta["title"], fill=(238, 242, 247), font=title_font)
    draw.text((36, 58), meta["task"], fill=(129, 213, 255), font=label_font)
    if meta.get("controls"):
        draw.text((36, 82), meta["controls"], fill=(185, 194, 205), font=small_font)

    # Legend
    lx, ly = width - 290, 58
    draw.rectangle((lx, ly + 3, lx + 13, ly + 15), fill=TRACK_COLORS[0])
    draw.text((lx + 17, ly), "Context", fill=(185, 194, 205), font=label_font)
    draw.rectangle((lx + 108, ly + 3, lx + 121, ly + 15), fill=GEN_COLOR)
    draw.text((lx + 125, ly), "Generated", fill=(185, 194, 205), font=label_font)

    # ── Track layout ──────────────────────────────────────────────────────
    track_names: list[str] = meta.get("track_names", [])
    generated_tracks: set[int] = set(meta.get("generated_tracks", []))
    n_tracks_in_midi = max((n.track for n in notes), default=0) + 1
    n_tracks = max(n_tracks_in_midi, len(track_names))

    # Notes grouped by track index
    by_track: list[list[NoteEvent]] = [[] for _ in range(n_tracks)]
    for note in notes:
        if note.track < n_tracks:
            by_track[note.track].append(note)

    # Per-track pitch ranges (drums get a fixed range)
    pitch_ranges: list[tuple[int, int]] = []
    for ti in range(n_tracks):
        t_notes = by_track[ti]
        if not t_notes or all(n.is_drum for n in t_notes):
            pitch_ranges.append((24, 84))
        else:
            pitch_ranges.append(_pitch_range(t_notes))

    roll_area_h = height - header_h - footer_h
    row_h = roll_area_h / max(n_tracks, 1)
    roll_w = right - label_w

    bars = _bar_boundaries(score, duration_s)
    target_bars: set[int] = set(meta.get("target_bars") or [])

    # ── Draw each track row ────────────────────────────────────────────────
    for ti in range(n_tracks):
        row_top = header_h + ti * row_h
        row_bot = row_top + row_h
        is_gen = ti in generated_tracks

        # Row backgrounds
        row_bg = (16, 12, 6) if is_gen else (10, 14, 20)
        label_bg = (22, 17, 8) if is_gen else (15, 20, 28)
        draw.rectangle((0, row_top, width, row_bot), fill=row_bg)
        draw.rectangle((0, row_top + 1, label_w - 2, row_bot - 1), fill=label_bg)
        draw.line((0, row_top, width, row_top), fill=(38, 46, 58), width=1)

        # Track label
        name = track_names[ti] if ti < len(track_names) else f"Track {ti}"
        short = _short_name(name)
        name_y = row_top + max(4, (row_h - 18) / 2 - (9 if is_gen else 0))
        draw.text((6, name_y), short, fill=(220, 226, 238), font=small_font)
        if is_gen:
            draw.text((6, name_y + 16), "GEN", fill=GEN_COLOR, font=tag_font)

        # Piano-roll area
        roll_top = int(row_top) + 2
        roll_bot = int(row_bot) - 1
        roll_rh = roll_bot - roll_top
        draw.rectangle((label_w, roll_top, right, roll_bot), fill=row_bg, outline=(42, 52, 66))

        # Pitch grid
        min_p, max_p = pitch_ranges[ti]
        pitch_span = max(1, max_p - min_p + 1)
        for pitch in range(min_p, max_p + 1):
            y = roll_bot - ((pitch - min_p + 1) / pitch_span) * roll_rh
            black_key = pitch % 12 in {1, 3, 6, 8, 10}
            if is_gen:
                gc = (22, 17, 8) if black_key else (18, 14, 6)
            else:
                gc = (18, 23, 31) if black_key else (23, 30, 41)
            draw.line((label_w, y, right, y), fill=gc)

        # Target-bar shading
        for bar_idx in target_bars:
            if bar_idx + 1 < len(bars):
                x0 = label_w + (bars[bar_idx] / duration_s) * roll_w
                x1 = label_w + (bars[bar_idx + 1] / duration_s) * roll_w
                shade = (26, 20, 10) if is_gen else (16, 24, 36)
                draw.rectangle((x0, roll_top, x1, roll_bot), fill=shade)

        # Notes
        note_color = GEN_COLOR if is_gen else TRACK_COLORS[ti % len(TRACK_COLORS)]
        for note in by_track[ti]:
            x0 = label_w + (note.start_s / duration_s) * roll_w
            x1 = label_w + (note.end_s / duration_s) * roll_w
            if x1 < label_w or x0 > right:
                continue
            if note.is_drum:
                # Drums: map pitch linearly within row
                drum_range = 60
                y0 = roll_bot - ((max(0, note.pitch - 24) + 1) / drum_range) * roll_rh
                y1 = y0 + max(4, roll_rh / drum_range * 0.8)
            else:
                y0 = roll_bot - ((note.pitch - min_p + 1) / pitch_span) * roll_rh
                y1 = y0 + max(3, roll_rh / pitch_span * 0.72)
            active = note.start_s <= t_now <= note.end_s
            color: tuple[int, ...] = tuple(min(255, c + 55) for c in note_color) if active else note_color
            draw.rounded_rectangle((x0, y0, max(x0 + 2, x1), y1), radius=2, fill=color)

    # ── Bar lines (full height across all tracks) ──────────────────────────
    for bar_s in bars:
        x = label_w + (bar_s / duration_s) * roll_w
        draw.line((x, header_h, x, height - footer_h), fill=(52, 62, 76), width=1)

    # ── Playhead ──────────────────────────────────────────────────────────
    play_x = label_w + (t_now / duration_s) * roll_w
    draw.line((play_x, header_h, play_x, height - footer_h), fill=(245, 248, 250), width=2)

    # ── Footer progress bar ───────────────────────────────────────────────
    progress_w = right - label_w
    draw.rectangle((label_w, height - 22, right, height - 16), fill=(44, 52, 63))
    draw.rectangle(
        (label_w, height - 22, label_w + (t_now / duration_s) * progress_w, height - 16),
        fill=(129, 213, 255),
    )
    draw.text(
        (right - 128, height - 31),
        f"{t_now:04.1f}s / {duration_s:04.1f}s",
        fill=(155, 166, 179),
        font=small_font,
    )
    return img


# ── Video encoding ────────────────────────────────────────────────────────────

def encode_video(
    midi_path: Path,
    out_path: Path,
    meta: dict,
    fps: int,
    size: tuple[int, int],
    keep_wav: bool,
    soundfont: Path | None,
) -> Path:
    score, notes, duration_s = load_notes(midi_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out_path.with_suffix(".wav") if keep_wav else Path(tempfile.mkstemp(suffix=".wav")[1])
    if not write_audio_with_fluidsynth(midi_path, wav_path, soundfont):
        write_audio(notes, duration_s, wav_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(wav_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    frame_count = max(1, int(math.ceil(duration_s * fps)))
    for frame_idx in range(frame_count):
        t_now = min(duration_s, frame_idx / fps)
        frame = draw_frame(score, notes, duration_s, meta, t_now, size)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}: {out_path}")
    if not keep_wav:
        wav_path.unlink(missing_ok=True)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi-dir", type=Path, default=Path("docs/assets/midi"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets/video"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--only", action="append", default=[], help="Render only matching example id(s).")
    parser.add_argument("--keep-wav", action="store_true")
    parser.add_argument(
        "--soundfont",
        default="auto",
        help="Path to .sf2/.sf3, 'auto' to search common locations, or 'none' for fallback synth.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = DEFAULT_EXAMPLES
    if args.only:
        wanted = set(args.only)
        selected = [ex for ex in DEFAULT_EXAMPLES if ex["id"] in wanted or ex["midi"] in wanted]
    if not selected:
        raise SystemExit("No examples matched --only filters.")
    soundfont = None if args.soundfont == "none" else find_soundfont(args.soundfont)
    if soundfont:
        print(f"[audio] using SoundFont: {soundfont}")
    else:
        print("[audio] no SoundFont found; using fallback synth")

    manifest = []
    for meta in selected:
        midi_path = args.midi_dir / meta["midi"]
        if not midi_path.exists():
            print(f"[skip] missing {midi_path}")
            continue
        out_path = args.out_dir / f"{meta['id']}.mp4"
        print(f"[render] {midi_path.name} -> {out_path.name}")
        encode_video(
            midi_path=midi_path,
            out_path=out_path,
            meta=meta,
            fps=args.fps,
            size=(args.width, args.height),
            keep_wav=args.keep_wav,
            soundfont=soundfont,
        )
        manifest.append({**meta, "video": str(out_path), "source_midi": str(midi_path)})

    manifest_path = args.manifest or args.out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] wrote {len(manifest)} video(s), manifest: {manifest_path}")


if __name__ == "__main__":
    main()
