#!/usr/bin/env python3
"""Render MIDI-GPT documentation examples to MP4 piano-roll videos.

This is intentionally repo-local and independent of the old transition demo
pipeline.  It consumes generated MIDI files from docs/assets/midi, renders a
simple synthesized audio track, and encodes an MP4 with a piano-roll
visualization for each example.
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
    {
        "id": "track_sampling_drums_sparse",
        "task": "Track Sampling",
        "title": "Hotel California: sparse drums",
        "midi": "sampling_drums_sparse.mid",
        "prompt": "prompt_sampling.mid",
        "controls": "note_density = 1",
    },
    {
        "id": "track_sampling_drums_dense",
        "task": "Track Sampling",
        "title": "Hotel California: dense drums",
        "midi": "sampling_drums_dense.mid",
        "prompt": "prompt_sampling.mid",
        "controls": "note_density = 8",
    },
    {
        "id": "track_sampling_cello_staccato",
        "task": "Track Sampling",
        "title": "Bach BWV 1052: staccato cello",
        "midi": "sampling_cello_staccato.mid",
        "prompt": "prompt_bach.mid",
        "controls": "max_polyphony = 0, max_note_duration = 1",
    },
    {
        "id": "track_sampling_cello_sustained",
        "task": "Track Sampling",
        "title": "Bach BWV 1052: sustained cello",
        "midi": "sampling_cello_sustained.mid",
        "prompt": "prompt_bach.mid",
        "controls": "max_polyphony = 0, min_note_duration = 4",
    },
    {
        "id": "bar_infill_sparse",
        "task": "Bar Infilling",
        "title": "Everything In Its Right Place: sparse infill",
        "midi": "infill_sparse.mid",
        "prompt": "prompt_infill.mid",
        "controls": "max_polyphony = 0, min_note_duration = 3",
        "target_bars": [3, 4],
    },
    {
        "id": "bar_infill_chordal",
        "task": "Bar Infilling",
        "title": "Everything In Its Right Place: chordal infill",
        "midi": "infill_chordal.mid",
        "prompt": "prompt_infill.mid",
        "controls": "min_polyphony = 2, max_polyphony = 6",
        "target_bars": [3, 4],
    },
    {
        "id": "ar_monophonic",
        "task": "Autoregressive Continuation",
        "title": "Summer: monophonic continuation",
        "midi": "ar_monophonic.mid",
        "prompt": "prompt_ar.mid",
        "controls": "max_polyphony = 0",
        "target_bars": [4, 5, 6, 7],
    },
    {
        "id": "ar_chordal",
        "task": "Autoregressive Continuation",
        "title": "Summer: chordal continuation",
        "midi": "ar_chordal.mid",
        "prompt": "prompt_ar.mid",
        "controls": "min_polyphony = 2, max_polyphony = 6",
        "target_bars": [4, 5, 6, 7],
    },
    {
        "id": "harp_monophonic",
        "task": "Autoregressive Continuation",
        "title": "SSBM All Star: monophonic harp continuation",
        "midi": "harp_monophonic.mid",
        "prompt": "prompt_harp.mid",
        "controls": "max_polyphony = 0",
        "target_bars": [4, 5, 6, 7],
    },
    {
        "id": "harp_chordal",
        "task": "Autoregressive Continuation",
        "title": "SSBM All Star: chordal harp continuation",
        "midi": "harp_chordal.mid",
        "prompt": "prompt_harp.mid",
        "controls": "min_polyphony = 2, max_polyphony = 6",
        "target_bars": [4, 5, 6, 7],
    },
]


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


def _bar_boundaries(score: symusic.Score, duration_s: float) -> list[float]:
    tpq = _tpq(score)
    ticks_per_bar = tpq * 4
    max_tick = max(_score_end(score), ticks_per_bar)
    n_bars = max(1, math.ceil(max_tick / ticks_per_bar))
    return [_tick_to_seconds(i * ticks_per_bar, score.tempos, tpq) for i in range(n_bars + 1)]


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
    title_font = _font(34, bold=True)
    label_font = _font(19)
    small_font = _font(15)

    header_h = 112
    footer_h = 34
    left = 58
    right = width - 34
    top = header_h + 12
    bottom = height - footer_h
    roll_w = right - left
    roll_h = bottom - top

    draw.rectangle((0, 0, width, header_h), fill=(20, 27, 36))
    draw.text((34, 22), meta["title"], fill=(238, 242, 247), font=title_font)
    draw.text((36, 67), meta["task"], fill=(129, 213, 255), font=label_font)
    draw.text((210, 68), meta.get("controls", ""), fill=(185, 194, 205), font=label_font)

    if notes:
        min_pitch = max(0, min(n.pitch for n in notes if not n.is_drum) - 3) if any(not n.is_drum for n in notes) else 24
        max_pitch = min(127, max(n.pitch for n in notes if not n.is_drum) + 3) if any(not n.is_drum for n in notes) else 84
    else:
        min_pitch, max_pitch = 24, 84
    if max_pitch - min_pitch < 18:
        mid = (min_pitch + max_pitch) // 2
        min_pitch = max(0, mid - 12)
        max_pitch = min(127, mid + 12)
    pitch_span = max(1, max_pitch - min_pitch + 1)

    draw.rectangle((left, top, right, bottom), fill=(10, 14, 20), outline=(55, 65, 78))

    for pitch in range(min_pitch, max_pitch + 1):
        y = bottom - ((pitch - min_pitch + 1) / pitch_span) * roll_h
        color = (25, 31, 40) if pitch % 12 in {1, 3, 6, 8, 10} else (31, 38, 48)
        draw.line((left, y, right, y), fill=color)
        if pitch % 12 == 0:
            draw.text((14, y - 8), f"C{pitch // 12 - 1}", fill=(107, 116, 128), font=small_font)

    for bar_s in _bar_boundaries(score, duration_s):
        x = left + (bar_s / duration_s) * roll_w
        draw.line((x, top, x, bottom), fill=(55, 65, 78), width=1)

    target_bars = set(meta.get("target_bars") or [])
    bars = _bar_boundaries(score, duration_s)
    for bar_idx in target_bars:
        if bar_idx + 1 < len(bars):
            x0 = left + (bars[bar_idx] / duration_s) * roll_w
            x1 = left + (bars[bar_idx + 1] / duration_s) * roll_w
            draw.rectangle((x0, top, x1, bottom), fill=(23, 34, 45))

    for note in notes:
        x0 = left + (note.start_s / duration_s) * roll_w
        x1 = left + (note.end_s / duration_s) * roll_w
        if note.is_drum:
            y0 = bottom - ((note.track % 10 + 1) / 11) * roll_h
            y1 = y0 + 7
        else:
            y0 = bottom - ((note.pitch - min_pitch + 1) / pitch_span) * roll_h
            y1 = y0 + max(3, roll_h / pitch_span * 0.72)
        if x1 < left or x0 > right:
            continue
        color = TRACK_COLORS[note.track % len(TRACK_COLORS)]
        if note.start_s <= t_now <= note.end_s:
            color = tuple(min(255, c + 55) for c in color)
        draw.rounded_rectangle((x0, y0, max(x0 + 2, x1), y1), radius=2, fill=color)

    play_x = left + (t_now / duration_s) * roll_w
    draw.line((play_x, top, play_x, bottom), fill=(245, 248, 250), width=3)

    progress_w = right - left
    draw.rectangle((left, height - 22, right, height - 16), fill=(44, 52, 63))
    draw.rectangle((left, height - 22, left + (t_now / duration_s) * progress_w, height - 16), fill=(129, 213, 255))
    draw.text((right - 128, height - 31), f"{t_now:04.1f}s / {duration_s:04.1f}s", fill=(155, 166, 179), font=small_font)
    return img


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
        print(f"[render] {midi_path} -> {out_path}")
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
