import logging
from typing import Optional, Tuple

from .piece_state import bar_ticks

log = logging.getLogger(__name__)

PARAM_DEFAULTS: dict = {
    "lookahead_bars":       2,
    "buffer_bars":          4,
    "num_anticipated_bars": 1,
    "temperature":          1.0,
    "model_dim":            4,
    "mask_top_k":           0.0,
    "sampling_seed":       -1,
    "mask_gap":             False,
    "adapt_buffer":         False,
    "gen_timeout":          0.0,
    "max_attempts":         3,
}

PARAM_RANGES: dict = {
    "lookahead_bars":       (1, 8),
    "buffer_bars":          (2, 64),
    "num_anticipated_bars": (1, 8),
    "temperature":          (0.5, 2.0),
    "model_dim":            (1, 16),
    "mask_top_k":           (0.0, 1.0),
    "sampling_seed":        (None, None),
    "mask_gap":             (None, None),
    "adapt_buffer":         (None, None),
    "gen_timeout":          (0, None),
    "max_attempts":         (1, 10),
}


def validate_param(name: str, value) -> Optional[str]:
    if name not in PARAM_RANGES:
        return f"Unknown global parameter: {name!r}"
    lo, hi = PARAM_RANGES[name]
    if lo is not None and value < lo:
        return f"Parameter {name!r} = {value} below minimum {lo}"
    if hi is not None and value > hi:
        return f"Parameter {name!r} = {value} above maximum {hi}"
    return None


def compute_target_bar(bars_completed: int, k: int, B: int,
                       adapt_buffer: bool) -> Optional[int]:
    playhead = bars_completed
    if adapt_buffer:
        if playhead + k < B:
            return None
    else:
        if playhead < B:
            return None
    return playhead + k


def compute_num_anticipation(target_bar: int, j: int, total_bars: int) -> int:
    return min(j, total_bars - target_bar)


def compute_bar_features(events: list, ts_num: int, ts_den: int,
                         resolution: int) -> Optional[dict]:
    note_ons = [e for e in events if e.get("velocity", 0) > 0]
    if not note_ons:
        return None

    ticks = bar_ticks(ts_num, ts_den, resolution)

    pitches = [e["pitch"] for e in note_ons]
    velocities = [e["velocity"] for e in note_ons]
    durations_norm = [e.get("internal_duration", 1) / ticks for e in note_ons]

    intervals = [
        (e["time"], e["time"] + e.get("internal_duration", 1))
        for e in note_ons
    ]
    max_poly = 0
    for i, (start, _) in enumerate(intervals):
        poly = sum(1 for (s, e) in intervals if s <= start < e)
        max_poly = max(max_poly, poly)

    return {
        "note_density":  len(note_ons) / max(ts_num, 1),
        "mean_pitch":    sum(pitches) / len(pitches),
        "mean_velocity": sum(velocities) / len(velocities),
        "max_polyphony": max_poly,
        "mean_duration": sum(durations_norm) / len(durations_norm),
    }


def run_inference(engine, score, request) -> Tuple[object, int]:
    """Call engine.session(score, request).run(). Returns (result_score, attempts=1)."""
    result = engine.session(score, request).run()
    return result, 1
