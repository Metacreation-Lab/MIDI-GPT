import logging

from .piece_state import bar_ticks

log = logging.getLogger(__name__)

WARMUP_POLICIES = ("a_empty", "a_masked", "b", "b_collapse")
MASK_MODES = ("token", "attention", "attention_approx", "attention_skip", "remove")

PARAM_DEFAULTS: dict = {
    "lookahead_bars": 1,
    "buffer_bars": 4,
    "num_anticipated_bars": 1,
    "temperature": 1.0,
    "model_dim": 8,
    "top_p": 1.0,
    "top_k": 0,
    "mask_p": 0.0,
    "mask_k": 0,
    "temperature_escalation": 1.0,
    "novelty_check": True,
    "silence_check": True,
    "sampling_seed": -1,
    "adapt_buffer": True,
    "gen_timeout": 0.0,
    "max_attempts": 3,
    "warmup_policy": "a_empty",
    "mask_mode": "token",
    "polyphony_hard_limit": 0,
    "density_hard_limit": 0,
}

PARAM_RANGES: dict = {
    "lookahead_bars": (1, 8),
    "buffer_bars": (2, 64),
    "num_anticipated_bars": (1, 8),
    "temperature": (0.1, 5.0),
    "model_dim": (1, 16),
    "top_p": (0.0, 1.0),
    "top_k": (0, 10000),
    "mask_p": (0.0, 1.0),
    "mask_k": (0, 10000),
    "temperature_escalation": (1.0, 3.0),
    "novelty_check": (None, None),
    "silence_check": (None, None),
    "sampling_seed": (None, None),
    "adapt_buffer": (None, None),
    "gen_timeout": (0, None),
    "max_attempts": (1, 10),
    "warmup_policy": (None, None),
    "mask_mode": (None, None),
    "polyphony_hard_limit": (0, 128),
    "density_hard_limit": (0, 1000),
}


def validate_param(name: str, value) -> str | None:
    if name not in PARAM_RANGES:
        return f"Unknown global parameter: {name!r}"
    if name == "warmup_policy":
        if value not in WARMUP_POLICIES:
            return f"warmup_policy={value!r} not in {WARMUP_POLICIES}"
        return None
    if name == "mask_mode":
        if value not in MASK_MODES:
            return f"mask_mode={value!r} not in {MASK_MODES}"
        return None
    lo, hi = PARAM_RANGES[name]
    if lo is not None and value < lo:
        return f"Parameter {name!r} = {value} below minimum {lo}"
    if hi is not None and value > hi:
        return f"Parameter {name!r} = {value} above maximum {hi}"
    return None


def compute_target_bar(bars_completed: int, k: int, B: int, adapt_buffer: bool) -> int | None:  # noqa: N803
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


def compute_bar_features(events: list, ts_num: int, ts_den: int, resolution: int) -> dict | None:
    note_ons = [e for e in events if e.get("velocity", 0) > 0]
    if not note_ons:
        return None

    ticks = bar_ticks(ts_num, ts_den, resolution)

    pitches = [e["pitch"] for e in note_ons]
    velocities = [e["velocity"] for e in note_ons]
    durations_norm = [e.get("internal_duration", 1) / ticks for e in note_ons]

    # Per-onset polyphony: simultaneous-onset count is the model's notion
    # ("OnsetPolyphony" attribute). Bar-level min/max bracket the sampled
    # track-level min_polyphony / max_polyphony tokens.
    onsets: dict[int, int] = {}
    for e in note_ons:
        onsets[e["time"]] = onsets.get(e["time"], 0) + 1
    onset_polys = list(onsets.values())

    # Raw durations (in ticks). The sampled MinNoteDuration / MaxNoteDuration
    # tokens are quantized into 6 levels — the realized values are in raw ticks
    # here; comparison is order-of-magnitude only.
    raw_durations = [e.get("internal_duration", 1) for e in note_ons]

    return {
        "note_density": len(note_ons) / max(ts_num, 1),
        "mean_pitch": sum(pitches) / len(pitches),
        "mean_velocity": sum(velocities) / len(velocities),
        "min_polyphony": min(onset_polys),
        "max_polyphony": max(onset_polys),
        "mean_duration": sum(durations_norm) / len(durations_norm),
        "min_note_duration": min(raw_durations),
        "max_note_duration": max(raw_durations),
    }


def run_inference(engine, score, request) -> tuple[object, int]:
    """Call engine.session(score, request).run(). Returns (result_score, attempts=1)."""
    result = engine.session(score, request).run()
    return result, 1
