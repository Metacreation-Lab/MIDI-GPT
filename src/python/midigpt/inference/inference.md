# MIDI-GPT Inference

Reference for everything under `midigpt.inference`: the request/config schema,
the sampling pipeline, token-level filters, acceptance checks, and how
constraints flow into the C++ grammar.

This document is the source of truth for callers (e.g. MMM Studio, scripts,
notebooks). When the code disagrees with this doc, prefer the code, but please
update the doc.

---

## 1. Top-level shape

```python
from midigpt.inference import (
    GenerationRequest, InferenceConfig, TrackPrompt,
)

req = GenerationRequest(
    tracks=[TrackPrompt(id=0, bars=[4, 5, 6, 7])],
    config=InferenceConfig(temperature=1.0, top_p=0.95),
)
score_out = engine.sample(score_in, req)
```

`engine.sample()` performs three logical stages:

1. **Validation** (`validation.py::validate_request`) — sanity-checks the
   request against the encoder config + score. Fails fast with
   `RequestValidationError`. May normalize fields (e.g. clamp
   `temperature_escalation` to a max).
2. **Planning** (`_core.StepPlanner`) — decomposes the request into a sequence
   of *steps*. Each step picks a window of bars + a subset of tracks to
   generate, given `model_dim`, `bars_per_step`, `tracks_per_step`,
   `mask_mode`, and `shuffle`.
3. **Sampling** (`session.py::SamplingSession`) — for each step, runs the
   token-level loop: build constraint graph, encode prompt, repeatedly call
   the model, mask/filter logits, multinomial draw, advance the FSM. Retries
   the step up to `max_attempts` times if acceptance checks fail.

---

## 2. `InferenceConfig`

All knobs live in `midigpt/inference/config.py`. Defaults are conservative —
nothing exotic is on by default.

### Core sampling

| Field | Type | Default | Meaning |
|---|---|---|---|
| `temperature` | float | `1.0` | Softmax temperature. `<1` sharpens, `>1` flattens. Must be `> 0`. |
| `seed` | int | `-1` | Torch RNG seed. `-1` leaves the global generator alone (free-running). Bumped by attempt index on retries — see §5. |
| `max_attempts` | int | `3` | Hard retry budget per step. See §5. |

### Retry behavior

| Field | Type | Default | Meaning |
|---|---|---|---|
| `novelty_check` | bool | `True` | If enabled and the candidate is bitwise-identical (pitch + onset_ticks) to the original in every generated bar, the attempt is **rejected** and retried. When disabled, identity is just logged as a warning. See §5. |
| `silence_check` | bool | `True` | If enabled and the candidate generated 0 notes in target bars, the attempt is rejected. Disabled → warning only. |
| `temperature_escalation` | float | `1.0` | Multiplier applied to `temperature` per failed attempt. `1.0` = off. Clamped to `[1.0, 3.0]`. Useful when retries keep landing on the same tokens. |

### Step planner

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model_dim` | int | `4` | Context window length **in bars**. Must be one of the values in `encoder_config.num_bars_map`. |
| `bars_per_step` | int | `1` | Bars generated per step. Must be `≤ model_dim`. |
| `tracks_per_step` | int | `1` | Tracks generated per step. `> 1` enables interleaved multi-track AR within a single window. |
| `shuffle` | bool | `False` | Randomize step order. |

### Span-masking (lookahead)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mask_mode` | str | `"token"` | One of `"token"`, `"attention"`, `"attention_approx"`, `"attention_skip"`, `"remove"`. Controls how unfilled future bars are represented in the context window. |

Detail per mode:

- **`token`** — standard: future bars get a `MaskBar` token in the prompt. Requires the vocab to define `MaskBar` (most modern checkpoints do).
- **`attention`** — future bars are encoded as their normal token spans, but the model's self-attention is masked over those spans via a key mask carried through KV-cache. Lowest perplexity at the cost of bookkeeping.
- **`attention_approx`** — like `attention`, but the masked positions are *zeroed* in KV-cache after prefill rather than masked on every step. Cheaper, slightly noisier.
- **`attention_skip`** — the masked positions are physically removed from the input; `position_ids` are passed explicitly so the model sees the right absolute positions. Cheapest, but assumes rotary/relative positions cope.
- **`remove`** — future bars are physically removed from the prompt. Loses positional context entirely. Use only with grammars that don't depend on bar count past the window.

### Hard polyphony cap

| Field | Type | Default | Meaning |
|---|---|---|---|
| `polyphony_hard_limit` | int | `0` | Global cap on simultaneous note onsets. `0` = no cap. Implemented as a `PolyphonyConstraint(limit)` added to every step's constraint graph. Useful to prevent the model from sticking too many notes in a single onset. |

### Token-sampling filters (top / mask)

Applied to the masked, temperature-scaled probabilities **before** the
multinomial draw, in this fixed pipeline:

```
softmax(logits / T)
        │
        ▼
   top_k   (rank truncation)
        │
        ▼
   top_p   (mass truncation: nucleus)
        │
        ▼
   mask_k  (rank-based "anti-top": remove top ranks)
        │
        ▼
   mask_p  (mass-based "anti-nucleus": remove top mass)
        │
        ▼
torch.multinomial
```

| Field | Type | Default | Range | Meaning |
|---|---|---|---|---|
| `top_p`  | float | `1.0` | `(0, 1]` | Standard nucleus: keep smallest set whose cumulative *descending* mass ≥ `top_p`. `1.0` = off (keep all). |
| `top_k`  | int   | `0`   | `≥ 0`    | Keep top-k highest-prob tokens. `0` = off. |
| `mask_p` | float | `0.0` | `[0, 1)` | Remove the most-likely tokens summing (cumulatively) to ≥ `mask_p` *within the current pool*. Forces the model off its obvious picks. `0.0` = off. |
| `mask_k` | int   | `0`   | `≥ 0`    | Remove top-k highest-prob tokens *within the current pool*. `0` = off. |

**Why two axes (p, k).** They cut along different dimensions. `top_p` is
adaptive (it expands when the distribution is flat); `top_k` is fixed-rank.
Same for the mask side. Combining them gives expressivity that neither alone
can: e.g. `top_k=50, mask_k=1` = "sample from the top 50 ranks but never the
single most-likely token."

**Why pipeline order matters.** `mask_*` measures mass/rank on the
*post-top* distribution. If you set `top_p=0.9, mask_p=0.5`, the mask is
computed over the surviving nucleus tokens, not the full vocab. This makes
the controls mostly orthogonal and lets you reason about them
independently.

**Validation rules** (enforced in `validation.py`):

- `mask_p < top_p` whenever both are active (`mask_p > 0` and `top_p < 1`).
  `mask_p ≥ top_p` would empty the pool.
- `mask_k < top_k` whenever both `> 0`.
- `top_p ∈ (0, 1]`, `mask_p ∈ [0, 1)`, both `*_k ≥ 0`.

**Runtime safety net.** If a numerically-edge distribution would empty the
pool despite valid config, the filter is skipped for that token and the
unfiltered probs are used. (No exceptions thrown mid-generation.)

**Tip — using mask_* for novelty.** A small `mask_k=1` or `mask_p ≈ 0.3`
will systematically push the model off its highest-confidence choices,
which is exactly what makes retries actually explore when
`novelty_check=True`. Without `mask_*`, a highly-confident infill will
deterministically produce the same tokens on every retry.

---

## 3. `TrackPrompt`

Per-track instructions. There must be one prompt per track in the score
(use `ignore=True` for tracks you want to leave untouched).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | int | — | Track index in the score. |
| `bars` | list[int] | — | Bars to **generate**. AR targets when `autoregressive=True`, infill targets otherwise. |
| `autoregressive` | bool | `False` | Generate from scratch (no per-bar prompt for this track). For AR, `bars` must be a contiguous right-suffix (or empty = whole track). |
| `ignore` | bool | `False` | Skip this track entirely (omitted from the token sequence). Mutually exclusive with `autoregressive` and must have `bars=[]`. |
| `mask_bars` | list[int] | `[]` | Bars encoded as `TOKEN_MASK_BAR` ("unknown / hidden"). Disjoint from `bars`. Any bar in the step window not listed in either becomes **context** (its actual notes, including silent if empty). |
| `attributes` | dict[str,int] | `{}` | Analyzer-derived attribute overrides. Keys = attribute *instance* names (e.g. `"min_polyphony"`, `"note_density"`). Values = quantized levels. See §4. |
| `controls` | dict[str,Any] | `{}` | **Non-attribute** controls — first-class token-locking knobs that don't flow through the AttributeAnalyzer. Currently: `"time_signature": int` (index into `encoder_config.time_signatures`). |
| `bar_attributes` | dict[int, dict[str,int]] | `{}` | Per-bar attribute overrides, keyed by absolute bar index. Only bar-level attributes (e.g. `tension`, `bar_pitch_class_set`) may appear here. Bar index must be in `tp.bars`. See [updatedAttributeControl.md](../../../docs/updatedAttributeControl.md) for full semantics. |
| `bar_controls` | dict[int, dict[str,Any]] | `{}` | Per-bar non-attribute controls, keyed by absolute bar index. Currently: `"time_signature": int`. Cross-track coherence is enforced — the same bar index must resolve to the same TS across every generating track. |

### `attributes` vs `controls` — why two dicts

Attributes are computed by the `AttributeAnalyzer` at encode time over the
score's existing notes (per-track and per-bar tokens). User entries in
`tp.attributes` override the analyzer-computed values.

Controls are values you pin **directly to a token type** without any
analyzer participation — e.g. forcing the time signature to `4/4`. They
don't have an analyzer instance and aren't part of the attribute schema.
They live on `tp.controls` so the attribute pipeline stays purely
analyzer-driven and new controls (e.g. future `tempo_lock`) plug in
cleanly.

### Three attribute regimes

The session distinguishes three cases per track, with different attribute
handling in each:

| Regime | Detection | Attribute path |
|---|---|---|
| **Full AR** | `tp.autoregressive=True` and `bars` covers the whole track | Skip the analyzer entirely. User `attributes` become **constraints** (`AttributeValueConstraint`) so the model is forced to emit the requested tokens. The encoder emits no attribute tokens for this track. |
| **Partial AR** | `tp.autoregressive=True` and `bars` is a strict right-suffix | Analyzer runs over the prefix bars. User `attributes` override in the **prompt**. No constraint — the prompt already pins the prefix attribute tokens. |
| **Infill** | `tp.autoregressive=False` | Analyzer runs over the whole track. User `attributes` override in the **prompt**. No constraint. |

Controls (`tp.controls`) behave like AR-only constraints: they apply only
to full-AR tracks. Infill cannot lock a TS for a single masked bar in
isolation because the surrounding context already pins it.

---

## 4. Attribute controls (per-checkpoint)

Discoverable at runtime via the engine's `AttributeAnalyzer`:

```python
analyzer = engine._analyzer
analyzer.attribute_sizes()           # {"min_polyphony": 10, ...}
analyzer.attribute_value_labels()    # {"min_polyphony": ["1 voice", ...]}
analyzer.attribute_track_types()     # {"min_polyphony": "melodic", ...}
```

- `attribute_sizes()` — bin count per attribute. Values in `tp.attributes`
  must be in `[0, size)`.
- `attribute_value_labels()` — human-readable label per bin. Drives UI
  dropdowns; falls back to `str(i)` if unset.
- `attribute_track_types()` — one of `"melodic"`, `"drum"`, `"both"`.
  Validation rejects a melodic-only attribute on a drum track and vice
  versa.

The attribute → C++ TokenType mapping in `session.py::_build_constraints`
covers (subject to checkpoint vocab): `pitch_range`, `key_signature`,
`note_duration_dist`, `tension`, `silence_proportion`, `pitch_class_set`,
`min_note_duration`, `max_note_duration`, `min_polyphony`, `max_polyphony`.

Two attributes get **dedicated C++ constraints** instead of a generic
`AttributeValueConstraint` (faster, more precise FSM hookup):

- `note_density` → `DensityConstraint(value)`
- `onset_polyphony` → `PolyphonyConstraint(value)`

Setting either of these in `tp.attributes` on a **full-AR** track adds the
constraint to the per-step graph. Both are no-ops for infill / partial-AR
(the prompt does the work).

---

## 5. Per-step acceptance — `_run_step`

Each step is attempted up to `max_attempts` times. After every attempt the
session always evaluates both checks regardless of whether they're enabled:

| Check enabled? | Failure observed? | Outcome |
|---|---|---|
| Yes | No  | Accept silently. |
| Yes | Yes | **Reject (error)** — log, retry if budget remains. |
| No  | No  | Accept silently. |
| No  | Yes | **Accept (warning)** — `log.warning(...)` with the failure reason. |

If a step exhausts `max_attempts` with at least one error every time, the
session raises `RuntimeError` with a per-reason count, the note count of
the last candidate, and a hint string suggesting which knobs to relax.

### Seed bumping per attempt

When `seed >= 0`, the i-th attempt uses `seed + i` so retries actually
explore (otherwise `torch.manual_seed(seed)` would produce identical draws
on every attempt and `max_attempts > 1` would be a no-op).
`seed < 0` leaves the global RNG state untouched, so each attempt draws
from wherever the generator is.

### When checks tend to fail

- **`silence_check` on infill** — usually means the grammar accepted a
  degenerate fill (empty fill block). Try lowering temperature or relaxing
  density attributes.
- **`novelty_check` on infill** — almost always means the model is highly
  confident in the existing notes for that span. `mask_k=1` or `mask_p ≈
  0.3` is the fastest fix; raising temperature usually doesn't help past
  ~1.2.
- **`silence_check` on full-AR** — model is finishing too eagerly with
  `TrackEnd`. Reduce `max_polyphony` constraint or raise temperature.

---

## 6. Constraint graph — `_build_constraints`

Built per-step. Always includes:

- `GrammarConstraint` — the syntactic FSM.
  - For AR steps: `set_exact_bars(window_len)` and `set_autoregressive_mode(True)`.
  - Always: `set_max_tracks(n_tracks - n_ignored)` and
    `set_require_notes(True)`.
- `PolyphonyConstraint(cfg.polyphony_hard_limit)` if `> 0`.

Then per full-AR track in `step.track_indices`:

- `DensityConstraint`, `PolyphonyConstraint` for the dedicated attributes.
- `AttributeValueConstraint(TokenType.X, value)` for each generic
  attribute in `tp.attributes`.
- `AttributeValueConstraint(TokenType.X, value)` for each entry in
  `tp.controls` (currently just `time_signature` → `TokenType.TimeSig`).

`Track` / `TrackEnd` are **never** masked here — the grammar FSM and
`set_max_tracks` decide when they're syntactically legal. Hard-masking
them broke the exit transition for single-track AR; see the inline comment
in `_build_constraints` for the historical bug.

---

## 7. Step planning summary

`StepPlanner` produces `GenerationStep` objects. Each carries:

- `start_bar`, `end_bar` — absolute bar bounds of this step's window.
- `track_indices` — which tracks contribute notes to this step's
  generation.
- `bars_to_generate` — set of `(track_id, bar_abs)` tuples for the actual
  fill targets.
- `context` — `n_tracks × n_bars` boolean matrix, `True` = context bar.
  Patched in `_sample_step` so that any bar listed in `tp.mask_bars`
  becomes `False` (masked) regardless of the planner's default.
- `is_autoregressive` — True for full-AR windows.

The session may reduce `model_dim` between steps (largest-first) if the
encoded prompt overflows the model's positional context — see
`SamplingSession.run`.

---

## 8. Common usage patterns

### Full-AR generation, free attributes

```python
TrackPrompt(id=0, bars=[], autoregressive=True)
```

### Full-AR with a locked time signature and polyphony cap

```python
TrackPrompt(
    id=0,
    bars=[],
    autoregressive=True,
    controls={"time_signature": 0},        # index into encoder TS list
    attributes={"max_polyphony": 3},
)
InferenceConfig(polyphony_hard_limit=4)
```

### Infill bars 4–7 with novelty pressure

```python
req = GenerationRequest(
    tracks=[TrackPrompt(id=0, bars=[4, 5, 6, 7])],
    config=InferenceConfig(
        temperature=1.0,
        top_p=0.95,
        mask_k=1,                # never resample the model's top pick
        novelty_check=True,
        max_attempts=5,
    ),
)
```

### Multi-track AR window with leading context

```python
GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[], autoregressive=True),   # pure AR
        TrackPrompt(id=1, bars=[], autoregressive=True),   # pure AR
        TrackPrompt(id=2, bars=[], ignore=True),           # leave alone
    ],
    config=InferenceConfig(tracks_per_step=2, bars_per_step=2),
)
```

---

## 9. Diagnostics

- `engine._analyzer` — runtime introspection of attribute schema.
- `SamplingSession.enable_profiling = True` — populates `encode_time`,
  `model_forward_time`, `decode_time` after `run()`.
- `SamplingSession.prompt_state_sink = fn` — callback fired once per
  `run()` with the first step's `(track, bar) → C|M|A|T` snapshot. Useful
  for UI overlays showing context / mask / generate state per cell.
- Log channels: `INFO` for per-attempt accept/reject; `DEBUG` for prompt
  composition + per-token grammar legality; `WARNING` for clamped configs
  and disabled-check failures.

---

## 10. Versioning notes

This document corresponds to the inference surface introduced together
with:

- First-class `tp.controls` field (split from `tp.attributes` —
  `time_signature` moved here).
- `InferenceConfig.top_p / top_k / mask_p / mask_k` sampling filters.
- `polyphony_hard_limit` global cap.
- Warn-instead-of-fail behavior when `novelty_check` / `silence_check` are
  disabled (previously: disabled = skip the check entirely; now: always
  check, downgrade to warning).

If you're on an older checkpoint or older encoder config:

- `MaskBar` may not be in the vocab → set `mask_mode="attention"` or
  `"attention_skip"`.
- `time_signatures` may be empty → `controls={"time_signature": ...}`
  raises `RequestValidationError`. Omit the control.
- The analyzer may expose different attribute *instance* names than newer
  encoders. Use `analyzer.attribute_sizes()` to introspect.
