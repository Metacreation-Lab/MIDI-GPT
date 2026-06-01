import json
import math
import os

from midigpt._core import TrackType
from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute


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
    return (
        len(bar.notes)
        if hasattr(bar, "notes") and not hasattr(bar, "note_indices")
        else len(bar.note_indices)
    )


def quantile(x: list, qs: list[float]) -> list:
    if not x:
        return [0] * len(qs)
    # Original C++ uses round(size * q) and nth_element
    x_sorted = sorted(x)
    size = len(x_sorted)
    results = []
    for q in qs:
        # index = min((int)round((double)x.size() * q), (int)x.size() - 1);
        idx = min(round(size * q), size - 1)
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
        notes_abs = []  # list of (start, end)

        bar_start_tick = 0
        for bar in track.bars:
            bar_len_ticks = round(bar.beat_length * score.resolution)
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

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
        nz = self._get_nz_polyphony(score, track_idx)
        qs = quantile(nz, [0.15, 0.85])
        return qs[0] if self.mode == "min" else qs[1]

    def quantize(self, value: float | int) -> int:
        # f->set_min_polyphony_q(std::max(std::min((int)std::get<2>(stat), 10), 1) - 1);
        v = int(value)
        v = max(1, min(v, 10))
        return v - 1

    def value_labels(self) -> list[str]:
        # quantize maps polyphony N (1..10) → bin N-1. So bin i = N voices
        # sounding simultaneously (the {mode}-15/85th percentile across the
        # track's piano roll, considering only non-silent timesteps).
        return [f"{i + 1} voice" if i == 0 else f"{i + 1} voices" for i in range(9)] + [
            "10+ voices"
        ]

    def achievable_range(self, fixed_score, track_idx, generated_bars):
        # Fixed bars' polyphony is locked. Realized track-level value:
        #   max mode → max(F, G)  → override v feasible iff v ≥ F_q
        #   min mode → min(F, G)  → override v feasible iff v ≤ F_q
        # F_q is computed by re-quantizing the attribute over a score copy
        # whose `generated_bars` have been emptied (so they contribute nothing).
        if not fixed_score.tracks or track_idx >= len(fixed_score.tracks):
            return (0, self.size - 1)
        import copy

        s = copy.deepcopy(fixed_score)
        track = s.tracks[track_idx]
        gen_set = set(generated_bars)
        for b_idx, bar in enumerate(track.bars):
            if b_idx in gen_set:
                if hasattr(bar, "notes") and not hasattr(bar, "note_indices"):
                    bar.notes = []
                else:
                    bar.note_indices = []
        f_q = self.quantize(self.compute(s, track_idx))
        if self.mode == "max":
            return (f_q, self.size - 1)
        else:
            return (0, f_q)


class NoteDurationQuantile(BaseAttribute):
    level = "track"
    track_type = "melodic"
    size = 6

    def __init__(self, mode: str):
        self.mode = mode
        self.name = f"{mode}_note_duration"
        self.token_type = "MinNoteDuration" if mode == "min" else "MaxNoteDuration"

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
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
                    d = 1.0  # Drums always have duration 1 in preprocess_tracks
                else:
                    d = float(note.duration_ticks)
                # (int)clip(midigpt_log2(max(d / 3., 1e-6)) + 1, 0., 5.)
                level = int(max(0.0, min(5.0, midigpt_log2(max(d / 3.0, 1e-6)) + 1.0)))
                durations.append(level)

        qs = quantile(durations, [0.15, 0.85])
        return qs[0] if self.mode == "min" else qs[1]

    def quantize(self, value: float | int) -> int:
        return int(value)

    def value_labels(self) -> list[str]:
        # level = clip(log2(d/3) + 1, 0, 5), so each bin roughly doubles in
        # duration relative to the previous one. Approximate musical labels.
        return ["32nd", "16th", "8th", "quarter", "half", "whole"]

    def achievable_range(self, fixed_score, track_idx, generated_bars):
        # Same min/max(F, G) monotonicity as PolyphonyQuantile.
        if not fixed_score.tracks or track_idx >= len(fixed_score.tracks):
            return (0, self.size - 1)
        import copy

        s = copy.deepcopy(fixed_score)
        track = s.tracks[track_idx]
        gen_set = set(generated_bars)
        for b_idx, bar in enumerate(track.bars):
            if b_idx in gen_set:
                if hasattr(bar, "notes") and not hasattr(bar, "note_indices"):
                    bar.notes = []
                else:
                    bar.note_indices = []
        f_q = self.quantize(self.compute(s, track_idx))
        if self.mode == "max":
            return (f_q, self.size - 1)
        else:
            return (0, f_q)


# Load DENSITY_QUANTILES from JSON
_qpath = os.path.join(os.path.dirname(__file__), "density_quantiles.json")
with open(_qpath) as f:
    _raw_dq = json.load(f)
    DENSITY_QUANTILES = {int(k): v for k, v in _raw_dq.items()}


class NoteDensityQuantile(BaseAttribute):
    name = "note_density"
    token_type = "NoteDensity"
    level = "track"
    track_type = "drum"  # Orig model: track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_DRUM
    size = 10

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
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
        av_notes = round(av_notes_fp)

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

    def value_labels(self) -> list[str]:
        # 10 per-instrument density quantile bins. Bin 0 = sparsest, bin 9 =
        # densest (thresholds in density_quantiles.json vary by instrument).
        return ["sparsest (Q0)"] + [f"Q{i}" for i in range(1, 9)] + ["densest (Q9)"]

    def achievable_range(self, fixed_score, track_idx, generated_bars):
        # Density is the mean note-count over bars-with-notes, quantized via
        # per-instrument thresholds. Bound the realized value by:
        #   lower = quantize(prefix_avg) — generated bars contribute nothing
        #   upper = quantize((prefix_notes + max_per_bar * |G|) / |G_with_notes|)
        # We approximate max_per_bar with the largest threshold in the
        # instrument's quantile vector (saturates the densest bin).
        if not fixed_score.tracks or track_idx >= len(fixed_score.tracks):
            return (0, self.size - 1)
        track = fixed_score.tracks[track_idx]
        gen_set = set(generated_bars)
        prefix_notes = 0
        prefix_bars_with_notes = 0
        for b_idx, bar in enumerate(track.bars):
            if b_idx in gen_set:
                continue
            nc = _bar_note_count(bar)
            if nc:
                prefix_notes += nc
                prefix_bars_with_notes += 1
        n_gen = max(len(gen_set), 0)
        if hasattr(track, "track_type"):
            is_melodic = track.track_type == "melodic"
        else:
            is_melodic = track.type == TrackType.Melodic
        qindex = track.instrument if is_melodic else 128
        qs = DENSITY_QUANTILES.get(qindex, DENSITY_QUANTILES[0])
        max_per_bar = qs[-1] if qs else 0

        def _to_bin(av_notes):
            av_notes = round(av_notes)
            b = 0
            while b < len(qs) - 1 and av_notes > qs[b]:
                b += 1
            return b

        lower_div = max(prefix_bars_with_notes, 1)
        lower = _to_bin(prefix_notes / lower_div) if prefix_bars_with_notes else 0
        upper_div = max(prefix_bars_with_notes + n_gen, 1)
        upper = _to_bin((prefix_notes + max_per_bar * n_gen) / upper_div)
        return (lower, max(lower, upper))
