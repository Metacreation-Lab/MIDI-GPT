"""
midigpt.realtime_gen

Stateless helpers for the real-time generation loop.

Shared by osc_server.py and simulate_realtime_agent.py (the simulation script
uses its own equivalent logic; these functions can replace it in a future
refactor).

Generation logic follows docs/realtime_framework.md exactly.
"""

import json
import logging
import random
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global parameter defaults (§7.1 of OSC_PROTOCOL_SPEC.md)
# ---------------------------------------------------------------------------

PARAM_DEFAULTS: dict = {
    "lookahead_bars":      2,     # k: bars ahead of playhead to generate
    "buffer_bars":         4,     # B: bars of silence before agent starts
    "num_anticipated_bars": 1,    # j: bars generated per inference call
    "temperature":         1.0,   # global generation entropy
    "model_dim":           4,     # D: context window in bars
    "mask_top_k":          0.0,   # probability of masking top-k tokens
    "sampling_seed":      -1,     # RNG seed (-1 = random)
    "mask_gap":            False, # hide agent gap bars with TOKEN_MASK_BAR
    "adapt_buffer":        False, # start generating before buffer ends
    "gen_timeout":         0.0,   # seconds before inference is abandoned; 0 = disabled
}

# Valid ranges for parameter validation
PARAM_RANGES: dict = {
    "lookahead_bars":      (1, 8),
    "buffer_bars":         (2, 64),
    "num_anticipated_bars":(1, 8),
    "temperature":         (0.5, 2.0),
    "model_dim":           (1, 16),
    "mask_top_k":          (0.0, 1.0),
    "sampling_seed":       (None, None),    # any int
    "mask_gap":            (None, None),    # bool
    "adapt_buffer":        (None, None),    # bool
    "gen_timeout":         (0, None),       # seconds ≥ 0; 0 = disabled
}


def validate_param(name: str, value) -> Optional[str]:
    """Return error string if value is out of range, else None."""
    if name not in PARAM_RANGES:
        return f"Unknown global parameter: {name!r}"
    lo, hi = PARAM_RANGES[name]
    if lo is not None and value < lo:
        return f"Parameter {name!r} = {value} below minimum {lo}"
    if hi is not None and value > hi:
        return f"Parameter {name!r} = {value} above maximum {hi}"
    return None


# ---------------------------------------------------------------------------
# Target-bar computation
# ---------------------------------------------------------------------------

def compute_target_bar(bars_completed: int, k: int, B: int,
                       adapt_buffer: bool) -> Optional[int]:
    """
    Given the number of fully completed bars, compute the target generation bar.

    bars_completed  — number of bars whose /bar/end has been received
    k               — lookahead_bars
    B               — buffer_bars
    adapt_buffer    — if True, start generating when playhead + k >= B

    Returns target_bar (int) or None if generation should not fire yet.
    """
    playhead = bars_completed  # first unfinished bar

    if adapt_buffer:
        if playhead + k < B:
            return None
    else:
        if playhead < B:
            return None

    return playhead + k


def compute_num_anticipation(target_bar: int, j: int, total_bars: int) -> int:
    """Clamp j so we do not overshoot the end of the piece."""
    return min(j, total_bars - target_bar)


# ---------------------------------------------------------------------------
# Parameter dict for sample_multi_step
# ---------------------------------------------------------------------------

def build_params(ckpt: str, global_params: dict, num_anticipation: int) -> dict:
    """Build the HyperParam dict for sample_multi_step.

    Every field of midi.HyperParam that has a meaningful default is set
    explicitly: relying on protobuf defaults (e.g. polyphony_hard_limit=0)
    silently broke generation. Fields are in proto field order.
    """
    # Always set a concrete (non-negative) seed for the original implementation —
    # -1 / unset has been observed to cause the C++ sampler to emit 0 notes.
    raw_seed = int(global_params.get("sampling_seed", -1))
    if raw_seed < 0:
        raw_seed = random.randint(0, 2**31 - 1)

    return {
        "tracks_per_step":           1,
        "bars_per_step":             int(num_anticipation),
        "model_dim":                 int(global_params["model_dim"]),
        "shuffle":                   False,
        "percentage":                100,
        "temperature":               float(global_params["temperature"]),
        "batch_size":                1,
        "verbose":                   False,
        "ckpt":                      ckpt,
        "mask_top_k":                0.0,
        "sampling_seed":             raw_seed,
        "polyphony_hard_limit":      10,
        "use_per_track_temperature": False,
        "max_steps":                 0,
    }


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

def run_inference(piece_dict: dict, status_dict: dict, params_dict: dict,
                  max_attempts: int) -> Tuple[dict, int]:
    """
    Call midigpt.sample_multi_step.

    Returns (result_piece_dict, num_attempts).
    Raises RuntimeError on failure.
    """
    try:
        import midigpt  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("midigpt extension not available") from exc

    piece_json = json.dumps(piece_dict)
    status_json = json.dumps(status_dict)
    params_json = json.dumps(params_dict)

    # Attach a RecordTokenSequenceCallback so we can see exactly what orig
    # generates — useful when the resulting piece has 0 notes.
    rec = midigpt.RecordTokenSequenceCallback()
    cm = midigpt.CallbackManager()
    cm.add_callback(rec)

    res_str, attempts = midigpt.sample_multi_step(
        piece_json, status_json, params_json, max_attempts, cm
    )
    toks = list(rec.tokens)
    log.warning(
        "DEBUG captured tokens (n=%d) seed=%s: %s",
        len(toks),
        params_dict.get("sampling_seed"),
        toks[:200],
    )
    return json.loads(res_str), attempts


# ---------------------------------------------------------------------------
# Bar feature extraction (for /generated/features)
# ---------------------------------------------------------------------------

def compute_bar_features(events: list, ts_num: int, ts_den: int,
                          resolution: int) -> Optional[dict]:
    """
    Compute musical features of a generated bar for /generated/features.

    events  — list of inline event dicts {pitch, velocity, time, internal_duration}
    Returns None if there are no note-on events.
    """
    from realtime_state import bar_ticks as _bar_ticks  # local import to avoid cycles

    note_ons = [e for e in events if e.get("velocity", 0) > 0]
    if not note_ons:
        return None

    ticks = _bar_ticks(ts_num, ts_den, resolution)

    pitches = [e["pitch"] for e in note_ons]
    velocities = [e["velocity"] for e in note_ons]
    durations_norm = [e.get("internal_duration", 1) / ticks for e in note_ons]

    # Max polyphony: for each note-on, count how many other note-ons overlap it.
    intervals = [
        (e["time"], e["time"] + e.get("internal_duration", 1))
        for e in note_ons
    ]
    max_poly = 0
    for i, (start, _) in enumerate(intervals):
        poly = sum(1 for (s, e) in intervals if s <= start < e)
        max_poly = max(max_poly, poly)

    return {
        "note_density": len(note_ons) / max(ts_num, 1),
        "mean_pitch": sum(pitches) / len(pitches),
        "mean_velocity": sum(velocities) / len(velocities),
        "max_polyphony": max_poly,
        "mean_duration": sum(durations_norm) / len(durations_norm),
    }
