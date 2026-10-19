# API Reference

## Score types

The Python score representation lives in `midigpt._types`. These are plain dataclasses — no C++ dependency required.

```python
from midigpt import Score, Track, Bar, Note
```

### `Note`

| Field | Type | Default | Description |
|---|---|---|---|
| `pitch` | int | 0 | MIDI pitch (0–127) |
| `velocity` | int | 64 | MIDI velocity (0–127) |
| `onset_ticks` | int | 0 | Start time in ticks, relative to bar start |
| `duration_ticks` | int | 0 | Duration in ticks |
| `delta` | int | 0 | Sub-grid microtiming offset in ticks (used by `expressive`) |

### `Bar`

| Field | Type | Default | Description |
|---|---|---|---|
| `notes` | list[Note] | `[]` | Notes in this bar |
| `ts_numerator` | int | 4 | Time signature numerator |
| `ts_denominator` | int | 4 | Time signature denominator |
| `beat_length` | float | 4.0 | Length in beats |
| `future` | bool | `False` | If `True`, the bar will be generated (informational flag) |

### `Track`

| Field | Type | Default | Description |
|---|---|---|---|
| `bars` | list[Bar] | `[]` | Bars in this track |
| `instrument` | int | 0 | General MIDI program number (0–127) |
| `track_type` | str | `"melodic"` | `"melodic"` or `"drum"` |
| `attributes` | dict[str, int] | `{}` | Quantized attribute overrides (rarely set directly) |

### `Score`

| Field | Type | Default | Description |
|---|---|---|---|
| `tracks` | list[Track] | `[]` | Tracks in this score |
| `resolution` | int | 480 | Ticks per quarter note |
| `tempo` | int | 500000 | Microseconds per quarter note (500000 = 120 BPM) |

**Class methods:**

```python
Score.from_midi(path: str) -> Score
Score.from_dict(d: dict)   -> Score
```

**Instance methods:**

```python
score.to_midi(path: str) -> None
score.to_dict()          -> dict
```

---

## `InferenceEngine`

```python
from midigpt.inference import InferenceEngine
```

Top-level entry point. Owns the model, tokenizer, and attribute analyzer.

### `InferenceEngine.from_pretrained`

```python
@classmethod
def from_pretrained(
    cls,
    name: str,
    hf_repo: str = "Metacreation/MIDI-GPT",
    analyzer: AttributeAnalyzer | None = None,
    device: str | None = None,
) -> InferenceEngine
```

`name` is the checkpoint filename prefix on the repo, e.g. `yellow_medium`, `prism_medium`, `expressive_medium`. The actual filename is resolved dynamically — prefers `<name>-final.safetensors`, falls back to the highest-step snapshot. Downloads and caches via `huggingface_hub`. `device`: `"cpu"`, `"cuda"`, `"mps"`, or `None`/`"auto"` to auto-detect.

```python
engine = InferenceEngine.from_pretrained("yellow_medium")
engine = InferenceEngine.from_pretrained("prism_medium", hf_repo="myorg/myrepo", device="cuda")
```

### `InferenceEngine.from_checkpoint`

```python
@classmethod
def from_checkpoint(
    cls,
    path: str,
    analyzer: AttributeAnalyzer | None = None,
    device: str | None = None,
) -> InferenceEngine
```

Load from a local `.safetensors` file, packed `.pt` bundle, or a legacy checkpoint directory.

### `InferenceEngine.session`

```python
def session(self, score: Score, request: GenerationRequest) -> SamplingSession
```

Validate the request against the score and return a `SamplingSession` ready to run. Does not start generation — call `.run()` on the returned session.

### `InferenceEngine.warmup`

```python
def warmup(self) -> None
```

Pre-build the empty KV cache. Called automatically by `from_pretrained` and `from_checkpoint`. Only needed if you construct `InferenceEngine` manually.

---

## `GenerationRequest`

```python
from midigpt.inference import GenerationRequest
```

Bundle of per-track generation targets and global configuration.

| Field | Type | Description |
|---|---|---|
| `tracks` | list[TrackPrompt] | One entry per track you want to control |
| `config` | InferenceConfig | Global sampling and step-planner settings |

---

## `TrackPrompt`

```python
from midigpt.inference import TrackPrompt
```

Describes what to do with one track.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | int | — | Track index in the score (0-based) |
| `bars` | list[int] | — | Absolute bar indices to generate |
| `autoregressive` | bool | `False` | Generate in AR mode (no per-bar prompt) |
| `ignore` | bool | `False` | Exclude this track from the token stream entirely |
| `mask_bars` | list[int] | `[]` | Bars to hide with MASK_BAR (disjoint from `bars`) |
| `attributes` | dict[str, int] | `{}` | Quantized attribute overrides for the whole track |
| `controls` | dict[str, Any] | `{}` | Non-attribute, first-class controls — see below |
| `bar_attributes` | dict[int, dict] | `{}` | Per-bar attribute overrides keyed by absolute bar index |
| `bar_controls` | dict[int, dict] | `{}` | Per-bar control overrides keyed by absolute bar index |

### `controls` keys

Check a checkpoint's `supports_*` capabilities (`GET /info` — see [HTTP Server](http.md)) before relying on any of these; not every model supports every control.

| Key | Type | Description |
|---|---|---|
| `time_signature` | int | Index into `encoder_config.time_signatures` |
| `pitch_mask` | dict | Restrict/shape which pitches this track's `NoteOnset` tokens may sample. Hard allow-set (pick one): `{"pitches": [60, 62, 64]}` (exact MIDI pitches), `{"scale": "major", "root": 0}` (preset name + pitch class 0–11), or `{"pitch_classes": [0, 2, 4, 5, 7, 9, 11]}` (any octave). Optional soft reweight layered on top: `"shape": {"type": "uniform", "min": 48, "max": 72}` or `"shape": {"type": "normal", "mean": 60, "std": 8}` |
| `rhythm_mask` | dict | Restrict/shape which within-bar tick this track's `TimeAbsolutePos` tokens may land on. Mutually exclusive (pick one): `"positions"` — hard exact rhythm, forces the onset grid and polyphony exactly (`[{"pos": 0, "polyphony": 1}, {"pos": 24, "polyphony": 2}]`; first entry must be `pos=0`; requires `config.mask_mode` to leave `tracks_per_step == 1`), or `"grid"` — soft grid-granularity bias (`{"unit": "eighth", "strength": 0.8}`; `unit`: whole/half/quarter/eighth/sixteenth/quarter_triplet/eighth_triplet/sixteenth_triplet; `strength` 1.0 = hard-quantize, 0.0 = no bias) |
| `remix` | dict | Regenerate this track's bars as a partial variation of content already on `score` for those bars, instead of generating fresh. The onset schedule (which ticks have notes, how many per tick) is always reproduced exactly from the reference — only note-level values are eligible for resampling. `{"amount": 0.3, "mode": "pitch"}`: `amount` is the fraction of eligible note-attributes resampled (0–1); `mode: "pitch"` — only pitch is eligible, duration stays exact ("vary notes, keep the rhythm"); `mode: "full"` — pitch and duration are each independently eligible. Velocity is always left to free sampling in both modes |

---

## `InferenceConfig`

```python
from midigpt.inference import InferenceConfig
```

Controls the step planner and sampling pipeline.

### Step planner

| Field | Type | Default | Description |
|---|---|---|---|
| `model_dim` | int or `"auto"` | 8 | Context window size in bars — must be in the checkpoint's `num_bars_map`. `"auto"`: the largest one the piece fills |
| `bars_per_step` | int or `"auto"` | 1 | AR: new bars per step (the rest of the window is the track's own history). Infill: the span of target bars one step covers. `"auto"`: AR keeps `max(min_overlap_bars, ar_overlap × model_dim)` bars of history; infill takes a whole window |
| `tracks_per_step` | int or `"auto"` | 1 | Infill tracks generated together (AR is always one track per step). `"auto"`: all of them |
| `ar_overlap` | float | 0.5 | `"auto"` `bars_per_step`, AR: the share of the window kept as history |
| `min_overlap_bars` | int | 2 | `"auto"` `bars_per_step`, AR: at least this many bars of history |
| `plan_only` | bool | false | Return the plan (windows and the bars each step writes) without generating; `/generate` answers `{"status": "planned", "model_dim", "plan": [...]}` |
| `mask_mode` | str | `"token"` | How to represent future bars: `"token"`, `"attention"`, `"attention_approx"`, `"attention_skip"`, `"remove"` |

### Sampling

| Field | Type | Default | Description |
|---|---|---|---|
| `temperature` | float | 1.0 | Softmax temperature — higher is more random |
| `top_k` | int | 0 | Keep top-k highest-probability tokens (0 = off) |
| `top_p` | float | 1.0 | Nucleus: keep the smallest set summing to ≥ `top_p` (1.0 = off) |
| `mask_k` | int | 0 | Remove the top-k most-likely tokens for novelty (0 = off) |
| `mask_p` | float | 0.0 | Anti-nucleus: remove tokens summing to ≥ `mask_p` from the top (0.0 = off) |

### Retries and quality checks

| Field | Type | Default | Description |
|---|---|---|---|
| `max_attempts` | int | 3 | Maximum sampling retries per step |
| `temperature_escalation` | float | 1.0 | Multiply temperature by this factor on each retry |
| `silence_check` | bool | `True` | Reject steps that produce zero notes |
| `novelty_check` | bool | `True` | Reject steps that reproduce the original bars (see `novelty_threshold`) |
| `novelty_threshold` | float | 1.0 | With `novelty_check`, reject a step whose generated bars share at least this fraction of their (pitch, onset) notes with the original; velocity, duration and microtiming don't count. 1.0 rejects only an unchanged result |
| `seed` | int | -1 | Fix the RNG for reproducibility (-1 = free-running) |

### Hard limits

| Field | Type | Default | Description |
|---|---|---|---|
| `polyphony_hard_limit` | int | 0 | Reject tokens that would exceed this simultaneous-note count (0 = off) |
| `density_hard_limit` | int | 0 | Reject tokens that would exceed this notes-per-bar count (0 = off) |

### Candidates

| Field | Type | Default | Description |
|---|---|---|---|
| `num_candidates` | int | 1 | Independent full generations from the same prompt (one seed each), no accept/reject filtering — see `SamplingSession.run_variations`. Not the same as `max_attempts`, which retries the *same* candidate on rejection |
| `infill_probability` | float | 1.0 | Infill only: generate `round(p × N)` of the N infill target bars (at least 1), chosen at random with the request seed; the others stay as they are and act as context. AR targets are never dropped. `SamplingSession.generated_bars` lists the bars picked |

---

## `SamplingSession`

```python
from midigpt.inference import SamplingSession
```

Returned by `InferenceEngine.session()`. Holds the model state across the full generation run.

### `SamplingSession.run`

```python
def run(self) -> Score
```

Execute all generation steps and return the completed score. The input score is not mutated.

### `SamplingSession.gen_count`

```python
@property
def gen_count(self) -> int
```

Number of bars generated so far. Useful for progress tracking when running steps manually.

### `SamplingSession.run_variations`

```python
def run_variations(self) -> list
```

`config.num_candidates` independent full generations from the same prompt — one seed per candidate, no accept/reject filtering (unlike the `max_attempts` retry loop). For "give me a few takes, I'll pick one" workflows. Returns a list of `{"score": Score | None, "seed": int, "gen_count": int, "error": str | None, "truncated": bool}` dicts, one per candidate — `score` is `None` and `error` is set for a candidate that failed.

### `SamplingSession.cancel_event`

```python
cancel_event: threading.Event | None
```

Set this (or assign an `Event` and `.set()` it from another thread) to stop an in-flight `run()`/`run_variations()` mid-decode — it raises `GenerationCancelled` carrying whatever was generated so far. Used by the HTTP server's `POST /generate/{request_id}/cancel`; see [HTTP Server — Cancellation](http.md#post-generaterequest_idcancel).

---

## Exceptions

### `RequestValidationError`

```python
from midigpt.inference import RequestValidationError
```

Raised by `InferenceEngine.session()` when the request is structurally invalid — e.g. a bar index out of range, an unknown attribute name, a `model_dim` not in the checkpoint's map, or `mask_mode="token"` on an encoder that lacks `MaskBar`.

### `GenerationCancelled`

```python
from midigpt.inference import GenerationCancelled
```

Raised by `SamplingSession.run()`/`run_variations()` when `cancel_event` is observed set mid-decode. Carries whatever was generated before the cancellation, decoded through the same path a natural completion would use: `exc.partial` is a `Score` (from `run()`) or the same list of per-candidate dicts `run_variations()` normally returns (from `run_variations()`); `exc.gen_count` is how many bars were generated before the cancel.
