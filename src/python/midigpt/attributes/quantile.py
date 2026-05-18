import math
import json
import os
from typing import Optional
from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute
from midigpt._core import TrackType

def midigpt_log2(x: float) -> float:
    return math.log2(x)

def _bar_notes(bar, score):
    """Notes in a bar, agnostic to Bar shape.

    C++ `Bar` has `note_indices` (ints into `score.notes`).
    Python `Bar` has `notes` (materialized Note objects).
    """
    if hasattr(bar, "notes") and not hasattr(bar, "note_indices"):
        return bar.notes
    return [score.notes[i] for i in bar.note_indices]


def _bar_note_count(bar) -> int:
    return len(bar.notes) if hasattr(bar, "notes") and not hasattr(bar, "note_indices") else len(bar.note_indices)


def quantile(x: list, qs: list[float]) -> list:
    if not x:
        return [0] * len(qs)
    # Original C++ uses round(size * q) and nth_element
    x_sorted = sorted(x)
    size = len(x_sorted)
    results = []
    for q in qs:
        # index = min((int)round((double)x.size() * q), (int)x.size() - 1);
        idx = min(int(round(size * q)), size - 1)
        results.append(x_sorted[idx])
    return results

class PolyphonyQuantile(BaseAttribute):
    level = "track"
    track_type = "melodic"
    size = 10

    def __init__(self, mode: str):
        self.mode = mode
        self.name = f"{mode}_polyphony"
        self.token_type = "MinPolyphony" if mode == "min" else "MaxPolyphony"

    def _get_nz_polyphony(self, score: Score, track_idx: int) -> list[int]:
        track = score.tracks[track_idx]
        max_tick = 0
        notes_abs = [] # list of (start, end)

        bar_start_tick = 0
        for bar in track.bars:
            bar_len_ticks = int(round(bar.beat_length * score.resolution))
            for note in _bar_notes(bar, score):
                start = bar_start_tick + note.onset_ticks
                end = start + note.duration_ticks
                notes_abs.append((start, end))
                max_tick = max(max_tick, end)
            bar_start_tick += bar_len_ticks

        if max_tick == 0:
            return []

        flat_roll = [0] * max_tick
        for start, end in notes_abs:
            # for (int t = note.start(); t < std::min(note.end(), max_tick - 1); t++)
            end_clamped = min(end, max_tick - 1)
            for t in range(start, end_clamped):
                flat_roll[t] += 1

        return [x for x in flat_roll if x > 0]

    def compute(self, score: Score, track_idx: int, bar_idx: Optional[int] = None) -> float | int:
        nz = self._get_nz_polyphony(score, track_idx)
        qs = quantile(nz, [0.15, 0.85])
        return qs[0] if self.mode == "min" else qs[1]

    def quantize(self, value: float | int) -> int:
        # f->set_min_polyphony_q(std::max(std::min((int)std::get<2>(stat), 10), 1) - 1);
        v = int(value)
        v = max(1, min(v, 10))
        return v - 1

class NoteDurationQuantile(BaseAttribute):
    level = "track"
    track_type = "melodic"
    size = 6

    def __init__(self, mode: str):
        self.mode = mode
        self.name = f"{mode}_note_duration"
        self.token_type = "MinNoteDuration" if mode == "min" else "MaxNoteDuration"

    def compute(self, score: Score, track_idx: int, bar_idx: Optional[int] = None) -> float | int:
        track = score.tracks[track_idx]
        durations = []
        from midigpt._core import TrackType
        if hasattr(track, "track_type"):
            is_drum = track.track_type == "drum"
        else:
            is_drum = track.type == TrackType.Drum

        for bar in track.bars:
            for note in _bar_notes(bar, score):
                if is_drum:
                    d = 1.0 # Drums always have duration 1 in preprocess_tracks
                else:
                    d = float(note.duration_ticks)
                # (int)clip(midigpt_log2(max(d / 3., 1e-6)) + 1, 0., 5.)
                level = int(max(0.0, min(5.0, midigpt_log2(max(d / 3.0, 1e-6)) + 1.0)))
                durations.append(level)

        qs = quantile(durations, [0.15, 0.85])
        return qs[0] if self.mode == "min" else qs[1]

    def quantize(self, value: float | int) -> int:
        return int(value)

# Load DENSITY_QUANTILES from JSON
_qpath = os.path.join(os.path.dirname(__file__), "density_quantiles.json")
with open(_qpath, "r") as f:
    _raw_dq = json.load(f)
    DENSITY_QUANTILES = {int(k): v for k, v in _raw_dq.items()}

class NoteDensityQuantile(BaseAttribute):
    name = "note_density"
    token_type = "NoteDensity"
    level = "track"
    track_type = "drum" # Orig model: track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_DRUM
    size = 10

    def compute(self, score: Score, track_idx: int, bar_idx: Optional[int] = None) -> float | int:
        track = score.tracks[track_idx]
        num_notes = 0
        valid_bars = set()
        for i, bar in enumerate(track.bars):
            nc = _bar_note_count(bar)
            if nc:
                valid_bars.add(i)
                num_notes += nc

        num_bars = max(len(valid_bars), 1)
        av_notes_fp = num_notes / num_bars
        av_notes = int(round(av_notes_fp))

        if hasattr(track, "track_type"):
            is_melodic = track.track_type == "melodic"
        else:
            is_melodic = track.type == TrackType.Melodic
        qindex = track.instrument if is_melodic else 128
        qs = DENSITY_QUANTILES.get(qindex, DENSITY_QUANTILES[0])

        bin = 0
        while av_notes > qs[bin]:
            bin += 1
        return bin

    def quantize(self, value: float | int) -> int:
        return int(value)

