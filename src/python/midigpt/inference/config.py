from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceConfig:
    temperature: float = 1.0
    seed: int = -1
    max_attempts: int = 3
    novelty_check: bool = True
    silence_check: bool = True
    temperature_escalation: float = 1.0  # multiply temp per failed attempt (1.0 = off)
    bars_per_step: int = 1  # bars generated per step (≤ model_dim)
    tracks_per_step: int = 1  # tracks processed per step
    model_dim: int = 4  # context window size in bars (NOT vocab size)
    shuffle: bool = False  # shuffle steps
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
    # Kept separate from `attributes` so the analyzer plumbing stays purely
    # analyzer-derived; new controls (e.g. future "tempo_lock") plug in here.
    controls: dict[str, Any] = field(default_factory=dict)
    # Per-bar overrides, keyed by ABSOLUTE bar index. See
    # docs/updatedAttributeControl.md for full semantics.
    #   bar_attributes[bar_idx][attr_name]    = int   (analyzer attribute)
    #   bar_controls  [bar_idx][control_name] = Any   (non-attribute control)
    bar_attributes: dict[int, dict[str, int]] = field(default_factory=dict)
    bar_controls: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    tracks: list[TrackPrompt]
    config: InferenceConfig = field(default_factory=InferenceConfig)
