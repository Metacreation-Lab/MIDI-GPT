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
    config=InferenceConfig(temperature=1.0, top_p=0.95, mask_mode="attention"),
)
score_out = engine.session(score_in, req).run()
```

`engine.session(...).run()` performs three logical stages:

1. **Validation** (`validation.py::validate_request`) — sanity-checks the
   request against the encoder config + score. Fails fast with
   `RequestValidationError`. May normalize fields (e.g. clamp
   `temperature_escalation` to a max).
2. **Planning** (`planner.py::plan`) — decomposes the request into a sequence
   of *steps*. Each step picks a window of bars + the target bars to
   generate, given `model_dim`, `bars_per_step` and `tracks_per_step`. Context
   fitting (`context_fit.py`) first leaves out context tracks a step would not
   fit with; `shuffle` then reorders the infill steps (`session.py::_order_steps`).
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
| `seed` | int | `-1` | Torch RNG seed. `-1` draws a random seed once per session (`SamplingSession.resolved_seed`, reported by the HTTP server), so the run can be replayed. Bumped by attempt index on retries — see §5. |
| `max_attempts` | int | `3` | Hard retry budget per step. See §5. |

### Retry behavior

| Field | Type | Default | Meaning |
|---|---|---|---|
| `novelty_check` | bool | `True` | If enabled and the candidate is too similar to the original in the step's generated bars (see `novelty_threshold`), the attempt is **rejected** and retried. When disabled, it is just logged as a warning. See §5. |
| `novelty_threshold` | float | `1.0` | Similarity, in `(0, 1]`, at which `novelty_check` rejects: the share of (bar, pitch, onset) notes the candidate shares with the original. `1.0` rejects only an identical result. |
| `silence_check` | bool | `True` | If enabled and the candidate generated 0 notes in target bars, the attempt is rejected. Disabled → warning only. |
| `temperature_escalation` | float | `1.0` | Multiplier applied to `temperature` per failed attempt. `1.0` = off. Clamped to `[1.0, 3.0]`. Useful when retries keep landing on the same tokens. |

### Step planner

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model_dim` | int or `"auto"` | `4` | Context window length **in bars**. Must be one of the values in `encoder_config.num_bars_map`. `"auto"`: the largest the piece fills. |
| `bars_per_step` | int or `"auto"` | `1` | Bars generated per step. Must be `≤ model_dim`. `"auto"`: AR keeps `max(min_overlap_bars, ar_overlap × model_dim)` bars of history; infill takes a whole window. |
| `tracks_per_step` | int or `"auto"` | `1` | Infill tracks generated per step (AR is always one track per step). `"auto"`: every infill track. |
| `ar_overlap` / `min_overlap_bars` | float / int | `0.5` / `2` | AR history kept per step when `bars_per_step` is `"auto"`. |
| `context` | str | `"auto"` | Context fitting when a step would not fit: `"auto"` drops `context_drop`, then the least-attended tracks; `"listed"` only `context_drop`; `"all"` never drops. `context_keep` is never dropped. |
| `context_drop` / `context_keep` / `budget_margin` | list / list / float | `[]` / `[]` / `1.25` | See `context`; `budget_margin` multiplies a step's estimated output tokens. |
| `plan_only` | bool | `False` | Return the plan (`SamplingSession.plan_summary()`) without generating. |
| `shuffle` | bool | `False` | Randomize the order of the infill steps (AR steps keep theirs); seeded by the request seed. |
| `infill_probability` | float | `1.0` | Infill only: generate `round(p × N)` of the N infill target bars (at least 1), picked with the request seed; the rest stay as context. `SamplingSession.generated_bars` lists them. |
| `num_candidates` | int | `1` | Independent generations from the same prompt, one seed each (`run_variations()`). |

### Span-masking (lookahead)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mask_mode` | str | `"token"` | One of `"token"`, `"attention"`, `"attention_approx"`, `"attention_skip"`, `"remove"`. Controls how unfilled future bars are represented in the context window. |

Detail per mode:

- **`token`** — future bars get a `MaskBar` token in the prompt. Requires the vocab to define `MaskBar`; none of the released checkpoints does, so use one of the attention modes with them.
- **`attention`** — future bars are encoded as their normal token spans, but the model's self-attention is masked over those spans via a key mask carried through KV-cache. Lowest perplexity at the cost of bookkeeping.
- **`attention_approx`** — like `attention`, but the masked positions are *zeroed* in KV-cache after prefill rather than masked on every step. Cheaper, slightly noisier.
- **`attention_skip`** — the masked positions are physically removed from the input; `position_ids` are passed explicitly so the model sees the right absolute positions. Cheapest, but assumes rotary/relative positions cope.
- **`remove`** — future bars are physically removed from the prompt. Loses positional context entirely. Use only with grammars that don't depend on bar count past the window.

### Hard polyphony cap

| Field | Type | Default | Meaning |
|---|---|---|---|
| `polyphony_hard_limit` | int | `0` | Global cap on simultaneous note onsets. `0` = no cap. Implemented as a `PolyphonyConstraint(limit)` added to every step's constraint graph. Useful to prevent the model from sticking too many notes in a single onset. |
| `density_hard_limit` | int | `0` | Global cap on note onsets per bar. `0` = no cap. Implemented as a `DensityConstraint` on every step's constraint graph. |

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
| `bar_attributes` | dict[int, dict[str,int]] | `{}` | Per-bar attribute overrides, keyed by absolute bar index. Only bar-level attributes (e.g. `tension`, `bar_pitch_class_set`) may appear here. Bar index must be in `tp.bars`. See [attribute_controls_guide.md](../../../docs/attribute_controls_guide.md) for full semantics. |
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

The i-th attempt uses `resolved_seed + i` so retries actually explore
(otherwise `torch.manual_seed(seed)` would produce identical draws on every
attempt and `max_attempts > 1` would be a no-op). `resolved_seed` is
`config.seed`, or a random seed drawn once per session when `seed < 0`, so
every run can be replayed from the seed it reports.

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

`planner.py::plan` produces `PlannedStep` objects (converted to the C++
`GenerationStep` with `to_core()`). Each carries:

- `start_bar`, `end_bar` — absolute bar bounds of this step's window.
- `track_indices` — which tracks contribute notes to this step's
  generation.
- `bars_to_generate` — set of `(track_id, bar_abs)` tuples for the actual
  fill targets.
- `context` — `n_tracks × n_bars` boolean matrix, `True` = context bar.
  Patched in `_sample_step` so that any bar listed in `tp.mask_bars`
  becomes `False` (masked) regardless of the planner's default.
- `is_autoregressive` — True for full-AR windows.

Before generating, `_fit_context` checks each step against the model's
context length and leaves out whole context tracks (`config.context`), then
makes `"auto"` `tracks_per_step`/`bars_per_step` smaller. If a step still
overflows, or an AR step is truncated, `_run` steps `model_dim` down and
plans only what is left, keeping the steps already done.

---

## 8. Common usage patterns

In all of these, `bars` lists the bars to generate (for AR, in order), and
the config needs a `mask_mode` other than `"token"` with the released
checkpoints.

### Full-AR generation, free attributes

```python
TrackPrompt(id=0, bars=list(range(8)), autoregressive=True)
```

### Full-AR with a locked time signature and polyphony cap

```python
TrackPrompt(
    id=0,
    bars=list(range(8)),
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

### Two AR tracks, one track per step

```python
GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=list(range(8)), autoregressive=True),
        TrackPrompt(id=1, bars=list(range(8)), autoregressive=True),
        TrackPrompt(id=2, bars=[], ignore=True),           # leave alone
    ],
    config=InferenceConfig(model_dim=8, bars_per_step="auto", mask_mode="attention"),
)
```

AR steps always generate one track; track 1's steps see track 0's
generated bars as context.

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
