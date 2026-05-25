from dataclasses import dataclass, field

@dataclass
class SamplingConfig:
    temperature:            float = 1.0
    seed:                   int   = -1
    max_attempts:           int   = 3
    novelty_check:          bool  = True
    silence_check:          bool  = True
    temperature_escalation: float = 1.0   # multiply temp per failed attempt (1.0 = off)
    bars_per_step:          int   = 1     # bars generated per step (≤ model_dim)
    tracks_per_step:        int   = 1     # tracks processed per step
    model_dim:              int   = 4     # context window size in bars (NOT vocab size)
    shuffle:                bool  = False # shuffle steps
    # Encoder-driven span masking: emit empty-shell bars (no MaskBar token) for
    # lookahead bars and mask their token spans out of self-attention. Lets
    # checkpoints without MaskBar in vocab (yellow.pt) drive the realtime path.
    use_span_masks:         bool  = False

@dataclass
class TrackPrompt:
    id:             int
    # Bars to GENERATE (AR targets when autoregressive=True, infill targets otherwise).
    bars:           list[int]
    autoregressive: bool          = False
    ignore:         bool          = False
    # Bars to MASK (encoded as TOKEN_MASK_BAR — "unknown / hidden"). Disjoint
    # from `bars`. Any bar in the step window not listed in either becomes
    # CONTEXT (its actual notes, including silent if empty).
    mask_bars:      list[int]     = field(default_factory=list)
    attributes:     dict[str,int] = field(default_factory=dict)
    # keys = attribute names (e.g. "note_density"), values = quantized levels

@dataclass
class GenerationRequest:
    tracks: list[TrackPrompt]
    config: SamplingConfig = field(default_factory=SamplingConfig)
