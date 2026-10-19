import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

# Preset scales for TrackPrompt.controls["pitch_mask"]["scale"] — semitone
# intervals from the root (pitch class 0-11), applied mod 12 across every
# octave. Root defaults to 0 (C) when unset.
PITCH_SCALE_PRESETS: dict[str, tuple[int, ...]] = {
    "chromatic": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
}


def resolve_pitch_allow_set(pitch_mask: dict[str, Any] | None) -> set[int] | None:
    """Resolve a `pitch_mask` control dict to the hard set of allowed MIDI
    pitches (0-127), or None if the dict places no hard restriction (e.g. it
    only carries a `shape`). Precedence when multiple keys are given:
    `pitches` (explicit allow-list) > `scale`+`root` (preset) > `pitch_classes`.

    Raises ValueError on an unknown scale name — callers (validation.py)
    should surface that as a RequestValidationError.
    """
    if not pitch_mask:
        return None
    if "pitches" in pitch_mask:
        return {int(p) for p in pitch_mask["pitches"] if 0 <= int(p) <= 127}
    pitch_classes: set[int] | None = None
    if "scale" in pitch_mask:
        name = pitch_mask["scale"]
        if name not in PITCH_SCALE_PRESETS:
            raise ValueError(
                f"unknown scale preset {name!r} (known: {sorted(PITCH_SCALE_PRESETS)})"
            )
        root = int(pitch_mask.get("root", 0)) % 12
        pitch_classes = {(root + iv) % 12 for iv in PITCH_SCALE_PRESETS[name]}
    elif "pitch_classes" in pitch_mask:
        pitch_classes = {int(pc) % 12 for pc in pitch_mask["pitch_classes"]}
    if pitch_classes is None:
        return None
    return {p for p in range(128) if p % 12 in pitch_classes}


def pitch_shape_weight(shape: dict[str, Any] | None, pitch: int) -> float:
    """Soft probability-shaping weight for one MIDI pitch, given a
    `pitch_mask["shape"]` dict. Multiplied into the sampling distribution
    (post hard-mask) and renormalized — never zeroes out a pitch the hard
    mask already allowed unless `type` is "uniform" and `pitch` falls
    outside [min, max].

    `{"type": "uniform", "min": lo, "max": hi}` — flat weight inside
    [lo, hi], 0.0 outside (a soft register clamp — equivalent to a hard
    min/max restriction once multiplied in).
    `{"type": "normal", "mean": m, "std": s}` — Gaussian weight centered on
    `mean`, never reaches exactly 0 (a soft register preference).
    """
    if not shape:
        return 1.0
    stype = shape.get("type")
    if stype == "uniform":
        lo = int(shape.get("min", 0))
        hi = int(shape.get("max", 127))
        return 1.0 if lo <= pitch <= hi else 0.0
    if stype == "normal":
        mean = float(shape.get("mean", 60.0))
        std = float(shape.get("std", 12.0)) or 1e-6
        return math.exp(-0.5 * ((pitch - mean) / std) ** 2)
    raise ValueError(f"unknown pitch_mask shape type {stype!r} (known: 'uniform', 'normal')")


def validate_pitch_mask(pitch_mask: dict[str, Any]) -> None:
    """Structural validation for a `pitch_mask` control dict — raises
    ValueError on anything malformed. Called from validation.py so bad
    input surfaces as a RequestValidationError instead of an opaque
    grammar-collapse deep in sampling.
    """
    known_keys = {"pitches", "scale", "root", "pitch_classes", "shape"}
    unknown = set(pitch_mask) - known_keys
    if unknown:
        raise ValueError(f"pitch_mask: unknown key(s) {sorted(unknown)} (known: {sorted(known_keys)})")
    if "pitches" in pitch_mask:
        for p in pitch_mask["pitches"]:
            if not (0 <= int(p) <= 127):
                raise ValueError(f"pitch_mask.pitches: {p} out of MIDI range [0, 127]")
    if "scale" in pitch_mask and "pitch_classes" in pitch_mask:
        raise ValueError("pitch_mask: 'scale' and 'pitch_classes' are mutually exclusive")
    allow_set = resolve_pitch_allow_set(pitch_mask)  # raises on unknown scale name
    if allow_set is not None and not allow_set:
        raise ValueError("pitch_mask: resolves to zero allowed pitches")
    shape = pitch_mask.get("shape")
    if shape is not None:
        if not isinstance(shape, dict) or "type" not in shape:
            raise ValueError("pitch_mask.shape must be a dict with a 'type' key")
        pitch_shape_weight(shape, 60)  # raises on unknown type / bad values
        if shape.get("type") == "uniform" and int(shape.get("min", 0)) > int(shape.get("max", 127)):
            raise ValueError("pitch_mask.shape: min must be <= max")


# Named grid subdivisions for TrackPrompt.controls["rhythm_mask"]["grid"]["unit"]
# — expressed as a fraction of one quarter note. Actual tick spacing is
# derived at runtime as round(encoder resolution (ticks/quarter) * fraction),
# since resolution is checkpoint-specific.
GRID_UNIT_FRACTIONS: dict[str, float] = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,
    "quarter_triplet": 1.0 / 3,
    "eighth_triplet": 1.0 / 6,
    "sixteenth_triplet": 1.0 / 12,
}


def resolve_rhythm_schedule(rhythm_mask: dict[str, Any]) -> list[tuple[int, int]]:
    """Resolve `rhythm_mask["positions"]` to a sorted [(tick, polyphony), ...]
    schedule. Raises ValueError on malformed/duplicate/non-zero-starting
    schedules — the grammar always emits an implicit note at tick 0 to open
    a bar (there is no token for "silence until tick N"), so a hard rhythm
    schedule that doesn't start at 0 can never be satisfied.
    """
    entries = rhythm_mask.get("positions")
    if not entries:
        raise ValueError("rhythm_mask.positions must be a non-empty list")
    out: list[tuple[int, int]] = []
    seen_pos: set[int] = set()
    for e in entries:
        pos = int(e["pos"])
        poly = int(e.get("polyphony", 1))
        if pos < 0:
            raise ValueError(f"rhythm_mask.positions: pos {pos} must be >= 0")
        if poly < 1:
            raise ValueError(f"rhythm_mask.positions: polyphony {poly} must be >= 1 (pos={pos})")
        if pos in seen_pos:
            raise ValueError(f"rhythm_mask.positions: duplicate pos {pos}")
        seen_pos.add(pos)
        out.append((pos, poly))
    out.sort(key=lambda t: t[0])
    if out[0][0] != 0:
        raise ValueError(
            "rhythm_mask.positions: first entry must be at pos=0 — every bar "
            "opens with an implicit note at tick 0, there is no token for "
            "leading silence within a bar"
        )
    return out


def rhythm_grid_step_ticks(grid: dict[str, Any], resolution: int) -> int:
    unit = grid.get("unit")
    if unit not in GRID_UNIT_FRACTIONS:
        raise ValueError(f"rhythm_mask.grid: unknown unit {unit!r} (known: {sorted(GRID_UNIT_FRACTIONS)})")
    step = round(resolution * GRID_UNIT_FRACTIONS[unit])
    return max(1, step)


def rhythm_grid_weight(grid: dict[str, Any], resolution: int, tick: int) -> float:
    """Soft grid-alignment weight for one TimeAbsolutePos tick value.
    strength=1.0 zeroes out every off-grid tick (a soft quantize-to-grid,
    equivalent to a hard restriction once multiplied in); strength=0.0 is a
    no-op (uniform); values in between partially suppress off-grid ticks.
    """
    strength = float(grid.get("strength", 1.0))
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"rhythm_mask.grid.strength must be in [0, 1] (got {strength})")
    step = rhythm_grid_step_ticks(grid, resolution)
    return 1.0 if tick % step == 0 else max(0.0, 1.0 - strength)


def validate_rhythm_mask(rhythm_mask: dict[str, Any]) -> None:
    """Structural validation for a `rhythm_mask` control dict — raises
    ValueError on anything malformed. Domain-size bounds (pos < the
    checkpoint's TimeAbsolutePos width) are checked in validation.py, which
    has access to the encoder config; this function only checks shape.
    """
    known_keys = {"positions", "grid"}
    unknown = set(rhythm_mask) - known_keys
    if unknown:
        raise ValueError(f"rhythm_mask: unknown key(s) {sorted(unknown)} (known: {sorted(known_keys)})")
    if "positions" in rhythm_mask and "grid" in rhythm_mask:
        raise ValueError(
            "rhythm_mask: 'positions' (hard, exact rhythm) and 'grid' (soft, "
            "grid bias) are mutually exclusive"
        )
    if "positions" in rhythm_mask:
        resolve_rhythm_schedule(rhythm_mask)  # raises on malformed schedule
    elif "grid" in rhythm_mask:
        grid = rhythm_mask["grid"]
        if not isinstance(grid, dict) or "unit" not in grid:
            raise ValueError("rhythm_mask.grid must be a dict with a 'unit' key")
        rhythm_grid_weight(grid, resolution=12, tick=0)  # raises on bad unit/strength
    else:
        raise ValueError("rhythm_mask: must set one of 'positions' or 'grid'")


def validate_remix(remix: dict[str, Any]) -> None:
    """Structural validation for a `remix` control dict — raises ValueError
    on anything malformed.
    """
    known_keys = {"amount", "mode"}
    unknown = set(remix) - known_keys
    if unknown:
        raise ValueError(f"remix: unknown key(s) {sorted(unknown)} (known: {sorted(known_keys)})")
    amount = remix.get("amount")
    if amount is None:
        raise ValueError("remix: 'amount' is required")
    if not (0.0 <= float(amount) <= 1.0):
        raise ValueError(f"remix.amount must be in [0, 1] (got {amount})")
    mode = remix.get("mode", "pitch")
    if mode not in ("pitch", "full"):
        raise ValueError(f"remix.mode must be 'pitch' or 'full' (got {mode!r})")


@dataclass
class InferenceConfig:
    temperature: float = 1.0
    seed: int = -1
    max_attempts: int = 3
    novelty_check: bool = True
    # novelty_check rejects a step whose generated bars share at least this
    # fraction of their (pitch, onset) notes with the original (1.0 = only an
    # unchanged result)
    novelty_threshold: float = 1.0
    silence_check: bool = True
    temperature_escalation: float = 1.0  # multiply temp per failed attempt (1.0 = off)
    bars_per_step: int = 1  # bars generated per step (≤ model_dim)
    tracks_per_step: int = 1  # tracks processed per step
    model_dim: int = 4  # context window size in bars (NOT vocab size)
    shuffle: bool = False  # random order for the infill steps
    # Mask mode selector: one of "token", "attention", "attention_approx",
    # "attention_skip", or "remove". Controls how future bars are represented
    # in the context and how the model attends to them.
    #   "token"            : encoder emits MaskBar token (requires vocab support)
    #   "attention"        : span masks + exact KV-buffer attention masking
    #   "attention_approx" : span masks + prefill attention masking
    #   "attention_skip"   : span masks + skip masked positions in attention
    #   "remove"           : future bars omitted entirely from token stream
    mask_mode: str = "token"
    # Global hard cap on simultaneous note onsets. 0 = no cap. Applied as an
    # extra PolyphonyConstraint on the constraint graph for every step.
    polyphony_hard_limit: int = 0
    # Global hard cap on note onsets per bar. 0 = no cap. Applied as an
    # extra DensityConstraint on the constraint graph for every step.
    density_hard_limit: int = 0
    # ── Token-sampling filters (applied to logits BEFORE multinomial draw) ──
    # Pipeline order: top_k → top_p → mask_k → mask_p. Defaults disable each
    # filter. Validation guarantees the surviving pool is non-empty.
    #   top_p ∈ (0, 1]   : keep smallest set whose cumulative descending mass
    #                      ≥ top_p (standard nucleus). 1.0 = keep all.
    #   top_k ∈ ℕ         : keep top-k highest-prob tokens. 0 = keep all.  # noqa: RUF003
    #   mask_p ∈ [0, 1)  : remove the most-likely tokens whose cumulative
    #                      descending mass ≥ mask_p ("anti-nucleus" — chops the
    #                      obvious tokens to force novelty). 0.0 = mask none.
    #   mask_k ∈ ℕ        : remove the top-k highest-prob tokens after top_*  # noqa: RUF003
    #                      filtering. 0 = mask none.
    # Combination: mask_p must be < top_p (else the surviving pool is empty);
    # mask_k must be < top_k when both are set.
    top_p: float = 1.0
    top_k: int = 0
    mask_p: float = 0.0
    mask_k: int = 0
    # When True and max_attempts > 1: run all retry attempts as ONE batched
    # forward pass per token step (batch dim = max_attempts) instead of
    # max_attempts sequential single-row passes. Same prompt/context in every
    # row, one independent seed + grammar state per row. Falls back to the
    # sequential path whenever hidden spans are present (masked attention
    # isn't batched) or max_attempts <= 1.
    # Measured (yellow_medium, this deployment): NOT a win — batching forces
    # every retry through a full-length step instead of letting cheap
    # rejections short-circuit, and bigger batched tensors make MPS's
    # allocator markedly less stable. Left available (default off) in case a
    # different model/workload profile benefits; not recommended currently.
    # For "give me several differently-seeded takes on the same prompt to
    # choose from" (the actual desired use case this was originally built
    # for), see `num_candidates` instead — no retry/accept-reject semantics,
    # just N independent generations returned for the caller to pick from.
    batch_attempts: bool = False
    # Generate N independent candidates from the SAME prompt — one seed per
    # candidate (base seed, base seed + 1, ...), no accept/reject filtering.
    # For "I generated a 5th track from these 4, didn't like it, want a few
    # alternates to pick from" workflows. 1 = default, single-candidate
    # behavior (unchanged). >1 routes through SamplingSession.run_variations()
    # instead of run() — see its docstring for the batched-vs-sequential
    # fallback rule, and http/server.py's /generate handler for the response
    # shape difference ("score" vs "candidates").
    num_candidates: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrackPrompt:
    id: int
    # Bars to GENERATE (AR targets when autoregressive=True, infill targets otherwise).
    bars: list[int]
    autoregressive: bool = False
    ignore: bool = False
    # Bars to MASK (encoded as TOKEN_MASK_BAR — "unknown / hidden"). Disjoint
    # from `bars`. Any bar in the step window not listed in either becomes
    # CONTEXT (its actual notes, including silent if empty).
    mask_bars: list[int] = field(default_factory=list)
    attributes: dict[str, int] = field(default_factory=dict)
    # keys = attribute names (e.g. "note_density"), values = quantized levels
    # First-class non-attribute controls (token-only AttributeValueConstraints
    # that don't flow through the AttributeAnalyzer). Currently:
    #   "time_signature" : int  – index into encoder_config.time_signatures  # noqa: RUF003
    #   "pitch_mask"     : dict – restrict/shape which pitches this track's
    #                      NoteOnset tokens may sample. See
    #                      `resolve_pitch_allow_set`/`pitch_shape_weight` in
    #                      this module for the full dict shape:
    #                        hard allow-set (pick one):
    #                          {"pitches": [60, 62, 64, ...]}          exact MIDI pitches
    #                          {"scale": "major", "root": 0}            preset (PITCH_SCALE_PRESETS), root=pitch class 0-11
    #                          {"pitch_classes": [0, 2, 4, 5, 7, 9, 11]}  any octave
    #                        soft reweight (optional, applied on top, multiply+renormalize):
    #                          "shape": {"type": "uniform", "min": 48, "max": 72}
    #                          "shape": {"type": "normal", "mean": 60, "std": 8}
    #   "rhythm_mask"    : dict – restrict/shape which within-bar tick this
    #                      track's TimeAbsolutePos tokens may land on. Mutually
    #                      exclusive sub-modes (pick one), see
    #                      `resolve_rhythm_schedule`/`rhythm_grid_weight` in
    #                      this module for the full dict shape:
    #                        HARD exact rhythm — forces the note-onset grid
    #                        exactly, including how many simultaneous notes
    #                        (polyphony) land on each onset point. Requires
    #                        config.tracks_per_step == 1 (see validation.py):
    #                          "positions": [{"pos": 0, "polyphony": 1},
    #                                        {"pos": 24, "polyphony": 2}, ...]
    #                          (pos = tick within the bar; first entry must be
    #                          pos=0 — every bar opens with an implicit note)
    #                        SOFT grid-granularity bias (multiply+renormalize
    #                        the TimeAbsolutePos sub-distribution):
    #                          "grid": {"unit": "eighth", "strength": 0.8}
    #                          (unit: whole/half/quarter/eighth/sixteenth/
    #                          quarter_triplet/eighth_triplet/sixteenth_triplet;
    #                          strength 1.0 = hard-quantize to that grid, 0.0 = no bias)
    #   "remix"          : dict – regenerate this track's bars as a partial
    #                      variation of the content ALREADY on `score` for
    #                      those bars, instead of generating fresh. The
    #                      onset schedule (which ticks have notes, how many
    #                      per tick) is always reproduced exactly from the
    #                      reference; only note-level VALUES are eligible
    #                      for resampling — letting the schedule itself
    #                      drift would mean the reference and generated bar
    #                      could end up with a different number of notes,
    #                      which has no well-defined "p% different" meaning.
    #                        {"amount": 0.3, "mode": "pitch"}
    #                        (amount: fraction of eligible note-attributes
    #                        resampled, in [0, 1]; mode "pitch": only pitch is
    #                        ever eligible, duration is always kept exact —
    #                        this is the "vary notes, keep the rhythm" case;
    #                        mode "full": pitch AND duration are each
    #                        independently eligible. Velocity is always left
    #                        to free sampling in both modes — this checkpoint
    #                        exposes no raw-velocity→VelocityLevel-token
    #                        quantizer to force it against reference exactly.)
    #                      Mutually exclusive with `rhythm_mask` on the same
    #                      track (remix derives its own schedule from the
    #                      reference). Requires config.tracks_per_step=1,
    #                      same reasoning as rhythm_mask.positions.
    # Kept separate from `attributes` so the analyzer plumbing stays purely
    # analyzer-derived; new controls (e.g. future "tempo_lock") plug in here.
    controls: dict[str, Any] = field(default_factory=dict)
    # Per-bar overrides, keyed by ABSOLUTE bar index. See
    # docs/updatedAttributeControl.md for full semantics.
    #   bar_attributes[bar_idx][attr_name]    = int   (analyzer attribute)
    #   bar_controls  [bar_idx][control_name] = Any   (non-attribute control)
    bar_attributes: dict[int, dict[str, int]] = field(default_factory=dict)
    bar_controls: dict[int, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "TrackPrompt":
        return cls(
            id=d["id"],
            bars=list(d["bars"]),
            autoregressive=d.get("autoregressive", False),
            ignore=d.get("ignore", False),
            mask_bars=list(d.get("mask_bars", [])),
            attributes=dict(d.get("attributes", {})),
            controls=dict(d.get("controls", {})),
            bar_attributes={int(k): v for k, v in d.get("bar_attributes", {}).items()},
            bar_controls={int(k): v for k, v in d.get("bar_controls", {}).items()},
        )


@dataclass
class GenerationRequest:
    tracks: list[TrackPrompt]
    config: InferenceConfig = field(default_factory=InferenceConfig)
    # Piece-level controls — apply uniformly to all tracks in the generation.
    # Supported keys:
    #   "velocity"    : bool  — True = use velocity tokens, False = omit them
    #                           (requires switchable_velocity=true in encoder config)
    #   "microtiming" : bool  — True = use delta tokens,    False = omit them
    #                           (requires switchable_microtiming=true in encoder config)
    #   "genre"       : str   — canonical genre label from the encoder's genre_groups
    #                           (requires genre_groups configured in encoder config)
    # Raises RequestValidationError if the control is invalid or unsupported.
    controls: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationRequest":
        return cls(
            tracks=[TrackPrompt.from_dict(t) for t in d["tracks"]],
            config=InferenceConfig.from_dict(d.get("config", {})),
            controls=dict(d.get("controls", {})),
        )
