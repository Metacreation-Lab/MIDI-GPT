"""The request space the matrix covers: one axis per independent choice a client makes, the rules that keep a
combination inside what the server is meant to accept, the risky combinations that must appear in full, and the
probes that must be refused cleanly (a RequestValidationError, never a crash).

A case is a dict {axis: value}. The first value of each axis is its default, the value a shrunk failure falls
back to.
"""

from __future__ import annotations

MODELS = ["yellow_small", "yellow_medium", "prism_medium", "expressive_medium"]
YELLOW = {"yellow_small", "yellow_medium"}
GENRE_MODELS = {"prism_medium", "expressive_medium"}
SWITCHABLE = {"expressive_medium"}

REMIX = {"remix_pitch_1", "remix_pitch_half", "remix_full_1", "remix_full_0"}
ONE_TRACK_PER_STEP = REMIX | {"rhythm_positions"}

AXES: dict[str, list] = {
    "model": MODELS,
    # what the piece looks like (build.py picks a GigaMIDI test piece of that kind)
    "piece": ["drums44", "nodrums44", "threefour", "sixeight", "sparse", "wide"],
    "length": [8, 4, 16, 32],                                   # bars in the score
    "mode": ["infill", "ar_full", "ar_partial", "mixed"],        # mixed: one infill target, one AR target
    "bars": ["contiguous", "one", "separate", "all"],            # which bars of a target are generated
    "targets": [1, 2, 3],                                        # tracks generated
    "target_kind": ["melodic", "drum", "both"],
    "ignore": ["none", "one", "half"],                           # context tracks left out of the prompt
    "mask_bars": ["none", "some"],                               # context bars masked out
    "model_dim": ["auto", 4, 8, 12, 16],
    "bars_per_step": ["auto", 1, 2, 4],
    "tracks_per_step": ["auto", 1, 2],
    "context": ["auto", "all", "listed"],
    "mask_mode": ["attention", "attention_approx", "attention_skip", "remove"],
    "prob": [1.0, 0.5, 0.1],                                     # infill_probability
    "takes": [1, 3],                                             # num_candidates
    "batch_attempts": [False, True],
    "shuffle": [False, True],
    "limits": ["none", "poly", "density", "both"],               # polyphony / density hard limits
    "sampling": ["default", "top_p", "top_k", "mask_p", "hot", "cold"],
    "checks": ["default", "off", "one_attempt"],
    "control": ["none", "remix_pitch_1", "remix_pitch_half", "remix_full_1", "remix_full_0",
                "rhythm_positions", "rhythm_grid", "pitch_scale", "pitch_classes", "pitch_list",
                "pitch_shape", "scale_and_grid"],
    "attributes": ["none", "track", "bar", "both", "extremes"],
    "piece_controls": ["none", "genre", "velocity_off", "microtiming_off", "all_off"],
    "stream": [False, True],                                     # note_sink / progress_sink attached
}

DEFAULTS = {k: v[0] for k, v in AXES.items()}


def violates(c: dict) -> str | None:
    """Why a (possibly partial) case is outside the space the server should accept, else None. Only rules whose
    axes are all present are checked, so the covering generator can test partial assignments."""
    def has(*ks):
        return all(k in c for k in ks)

    if has("model", "model_dim") and c["model"] in YELLOW and c["model_dim"] in (12, 16):
        return "yellow windows are 4 or 8 bars"
    if has("model_dim", "length") and c["model_dim"] != "auto" and c["model_dim"] > c["length"]:
        return "window longer than the score"
    if has("control", "tracks_per_step") and c["control"] in ONE_TRACK_PER_STEP and c["tracks_per_step"] not in ("auto", 1):
        return "remix / exact rhythm need one track per step"
    if has("piece_controls", "model"):
        pc, m = c["piece_controls"], c["model"]
        if pc == "genre" and m not in GENRE_MODELS:
            return "genre needs a prism-family model"
        if pc in ("velocity_off", "microtiming_off", "all_off") and m not in SWITCHABLE:
            return "velocity / microtiming switches need expressive"
    if has("attributes", "model") and c["attributes"] in ("bar", "both") and c["model"] in YELLOW:
        return "yellow has no bar-level attributes"
    if has("mode", "bars"):
        if c["mode"] == "ar_full" and c["bars"] != "all":
            return "full AR generates the whole track"
        if c["mode"] in ("ar_partial", "mixed") and c["bars"] in ("separate", "all"):
            return "AR bars are a right suffix of the track"
    if has("mode", "targets") and c["mode"] == "mixed" and c["targets"] < 2:
        return "mixed needs an infill and an AR target"
    if has("target_kind", "targets") and c["target_kind"] == "both" and c["targets"] < 2:
        return "a drum and a melodic target need two targets"
    if has("target_kind", "piece") and c["target_kind"] != "melodic" and c["piece"] not in ("drums44", "wide", "threefour"):
        return "drum targets need a piece with drums"
    if has("stream", "takes") and c["stream"] and c["takes"] > 1:
        return "streaming is one take"
    if has("bars", "length") and c["bars"] == "separate" and c["length"] < 4:
        return "separate bars need 4 bars"
    return None


# Combinations that must appear in full (every model), on top of pairwise coverage: the interactions most
# likely to break, like remix over several fill-in bars did.
def risky() -> list[dict]:
    out = []
    for ctl in [*sorted(REMIX), "rhythm_positions"]:
        for bars in ("contiguous", "separate"):
            out.append({"control": ctl, "bars": bars, "mode": "infill"})
        out.append({"control": ctl, "bars": "contiguous", "mode": "ar_partial"})
        out.append({"control": ctl, "targets": 2, "tracks_per_step": "auto"})
    for prob in (0.5, 0.1):
        out.append({"context": "auto", "model_dim": "auto", "bars_per_step": "auto", "prob": prob, "length": 32})
        out.append({"prob": prob, "mode": "mixed"})
    out.append({"targets": 3, "tracks_per_step": "auto", "length": 32, "piece": "wide", "context": "auto"})
    out.append({"context": "listed", "ignore": "half", "targets": 3})
    for mm in ("attention", "attention_approx", "attention_skip", "remove"):
        out.append({"mask_mode": mm, "mode": "ar_partial", "mask_bars": "some"})
        out.append({"mask_mode": mm, "bars": "separate", "prob": 0.5})
    out.append({"takes": 3, "batch_attempts": True, "checks": "one_attempt"})
    out.append({"takes": 3, "control": "remix_pitch_half", "bars": "separate"})
    for ctl in ("pitch_list", "pitch_classes", "scale_and_grid"):
        out.append({"control": ctl, "attributes": "extremes"})
        out.append({"control": ctl, "target_kind": "drum"})
    for piece in ("threefour", "sixeight"):
        for ctl in ("rhythm_positions", "rhythm_grid", "remix_full_1"):
            out.append({"piece": piece, "control": ctl})
    out.append({"limits": "both", "target_kind": "drum"})
    out.append({"limits": "both", "control": "rhythm_positions"})
    out.append({"checks": "off", "sampling": "cold", "control": "remix_full_0"})
    out.append({"stream": True, "mode": "mixed", "context": "auto"})
    return [dict(r, model=m) for r in out for m in MODELS]


# Requests that must be refused with a RequestValidationError (a 422), never a crash or a generation.
PROBES = [
    ("yellow_window_16", {"model": "yellow_medium", "model_dim": 16}),
    ("window_over_score", {"model_dim": 8, "length": 4}),
    ("remix_two_tracks_per_step", {"control": "remix_pitch_1", "tracks_per_step": 2, "targets": 2}),
    ("genre_on_yellow", {"model": "yellow_medium", "piece_controls": "genre"}),
    ("velocity_off_on_prism", {"model": "prism_medium", "piece_controls": "velocity_off"}),
    ("bar_attribute_on_yellow", {"probe": "bar_attribute_on_yellow", "model": "yellow_medium"}),
    ("token_mask_mode", {"mask_mode": "token"}),   # no current checkpoint has a MaskBar domain
    ("ar_separate_bars", {"mode": "ar_partial", "bars": "separate"}),
    # malformed payloads, built by build.py's probe hooks
    ("rhythm_not_from_zero", {"probe": "rhythm_not_from_zero"}),
    ("pitch_mask_empty", {"probe": "pitch_mask_empty"}),
    ("remix_and_rhythm", {"probe": "remix_and_rhythm"}),
    ("attribute_out_of_range", {"probe": "attribute_out_of_range"}),
    ("bar_attribute_off_target", {"probe": "bar_attribute_off_target", "model": "prism_medium"}),
    ("bar_out_of_range", {"probe": "bar_out_of_range"}),
    ("unknown_control", {"probe": "unknown_control"}),
    ("remix_amount_over_one", {"probe": "remix_amount_over_one"}),
    ("unknown_genre", {"probe": "unknown_genre", "model": "prism_medium"}),
]
