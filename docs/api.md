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
    name_or_repo_id: str,
    filename: str | None = None,
    analyzer: AttributeAnalyzer | None = None,
) -> InferenceEngine
```

Load by short name (`"yellow"`, `"yellow_small"`, `"expressive"`, `"prism_medium"`) or by HuggingFace repo ID + filename. Downloads and caches via `huggingface_hub`.

```python
engine = InferenceEngine.from_pretrained("yellow")
engine = InferenceEngine.from_pretrained("Metacreation/MIDI-GPT", filename="yellow_medium-final.safetensors")
```

### `InferenceEngine.from_checkpoint`

```python
@classmethod
def from_checkpoint(
    cls,
    path: str,
    analyzer: AttributeAnalyzer | None = None,
) -> InferenceEngine
```

Load from a local packed `.pt` bundle or a legacy checkpoint directory.

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
| `controls` | dict[str, Any] | `{}` | Non-attribute token locks, e.g. `{"time_signature": 0}` |
| `bar_attributes` | dict[int, dict] | `{}` | Per-bar attribute overrides keyed by absolute bar index |
| `bar_controls` | dict[int, dict] | `{}` | Per-bar control overrides keyed by absolute bar index |

---

## `InferenceConfig`

```python
from midigpt.inference import InferenceConfig
```

Controls the step planner and sampling pipeline.

### Step planner

| Field | Type | Default | Description |
|---|---|---|---|
| `model_dim` | int | 8 | Context window size in bars — must be in the checkpoint's `num_bars_map` |
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
| `novelty_check` | bool | `False` | Reject steps that reproduce the original bars unchanged |
| `seed` | int | -1 | Fix the RNG for reproducibility (-1 = free-running) |

### Hard limits

| Field | Type | Default | Description |
|---|---|---|---|
| `polyphony_hard_limit` | int | 0 | Reject tokens that would exceed this simultaneous-note count (0 = off) |
| `density_hard_limit` | int | 0 | Reject tokens that would exceed this notes-per-bar count (0 = off) |

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

---

## Exceptions

### `RequestValidationError`

```python
from midigpt.inference import RequestValidationError
```

Raised by `InferenceEngine.session()` when the request is structurally invalid — e.g. a bar index out of range, an unknown attribute name, a `model_dim` not in the checkpoint's map, or `mask_mode="token"` on an encoder that lacks `MaskBar`.
