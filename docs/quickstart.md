# Getting Started

## Installation

### Inference only

```bash
pip install "midigpt[inference]"
```

Pre-built wheels for CPython 3.10–3.12 on Linux (x86_64), macOS (x86_64 + arm64), and Windows (AMD64). No compiler needed.

### All extras

| Extra | What it adds |
|---|---|
| `inference` | `torch>=2.0`, `tqdm`, `huggingface_hub` |
| `train` | PyTorch Lightning, HuggingFace `datasets`, `pyarrow`, `python-dotenv` |
| `realtime` | `python-osc`, Flask, Flask-SocketIO |
| `dev` | `pytest`, `ruff`, `mypy` |
| `docs` | `mkdocs-material`, `mkdocstrings` |

### Developer install

```bash
git clone https://github.com/Metacreation-Lab/MIDI-GPT.git
cd MIDI-GPT
pip install -e ".[inference,dev]"
```

Prerequisites: Python 3.10+, CMake 3.21+, a C++20 compiler.

---

## Loading a model

Load a pretrained model by name — downloaded from HuggingFace Hub and cached locally on first use:

```python
from midigpt.inference import InferenceEngine

engine = InferenceEngine.from_pretrained("yellow")   # or "ghost", "expressive"
```

Load from a local checkpoint file:

```python
engine = InferenceEngine.from_checkpoint("path/to/model.pt")
```

See [Models](models.md) for a description of each checkpoint.

---

## Working with scores

`midigpt` represents music as a `Score` — a list of `Track`s, each containing a list of `Bar`s with `Note`s.

### Build from scratch

```python
from midigpt import Score, Track, Bar, Note

score = Score(
    tracks=[
        Track(
            bars=[Bar() for _ in range(8)],
            instrument=0,       # General MIDI program number
            track_type="melodic",
        )
    ],
    resolution=480,    # ticks per quarter note
    tempo=500000,      # microseconds per quarter note (= 120 BPM)
)
```

### Load from MIDI

```python
score = Score.from_midi("my_song.mid")
```

### Save to MIDI

```python
result.to_midi("output.mid")
```

---

## Infill

Fill in specific bars while keeping the rest of the arrangement intact:

```python
from midigpt.inference import GenerationRequest, InferenceConfig, TrackPrompt

score = Score.from_midi("my_song.mid")

request = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[4, 5, 6, 7]),   # regenerate bars 4–7 on track 0
        TrackPrompt(id=1, bars=[], ignore=True), # leave track 1 unchanged
    ],
    config=InferenceConfig(
        temperature=1.0,
        top_p=0.95,
        model_dim=8,    # context window in bars — must be in the model's num_bars_map
    ),
)

result = engine.session(score, request).run()
result.to_midi("output.mid")
```

---

## Autoregressive generation

Generate a new track from scratch, conditioned on musical attributes:

```python
request = GenerationRequest(
    tracks=[
        TrackPrompt(
            id=0,
            bars=[],               # empty = generate the whole track
            autoregressive=True,
            attributes={"max_polyphony": 3},       # quantized attribute level
            controls={"time_signature": 0},         # index into encoder TS list
        ),
    ],
    config=InferenceConfig(
        temperature=1.0,
        model_dim=8,
        polyphony_hard_limit=4,
    ),
)
result = engine.session(score, request).run()
```

---

## Attribute control

Introspect available attribute controls for the loaded model:

```python
engine._analyzer.attribute_sizes()
# {"note_density": 10, "min_polyphony": 10, "max_polyphony": 10, ...}

engine._analyzer.attribute_value_labels()
# {"note_density": ["very sparse", "sparse", ..., "very dense"], ...}

engine._analyzer.attribute_track_types()
# {"note_density": "melodic", "min_polyphony": "melodic", ...}
```

Pass quantized levels (integers in `[0, size)`) in `TrackPrompt.attributes`. Level 0 is the sparsest / quietest bin; `size - 1` is the densest / loudest.

Per-bar overrides let you shape the attribute trajectory across bars:

```python
TrackPrompt(
    id=0,
    bars=[0, 1, 2, 3],
    attributes={"note_density": 5},        # global default
    bar_attributes={
        0: {"note_density": 2},            # override bar 0 to be sparser
        3: {"note_density": 8},            # override bar 3 to be denser
    },
)
```

---

## Sampling filters

`InferenceConfig` exposes a four-stage logit-filtering pipeline (`top_k` → `top_p` → `mask_k` → `mask_p`):

| Field | Default | Meaning |
|---|---|---|
| `top_k` | `0` (off) | Keep top-k highest-probability tokens |
| `top_p` | `1.0` (off) | Nucleus: keep the smallest set summing to ≥ `top_p` |
| `mask_k` | `0` (off) | Remove the top-k most-likely tokens (novelty pressure) |
| `mask_p` | `0.0` (off) | Anti-nucleus: remove tokens summing to ≥ `mask_p` from the top |

A small `mask_k=1` or `mask_p=0.3` pushes the model off its most-confident picks — useful when `novelty_check=True` is enabled and you want diverse retries.

---

## Mask modes

Control how bars-to-be-generated appear in the context window:

| Mode | Behaviour |
|---|---|
| `"attention"` | Future bars zeroed in the KV cache via exact span masking |
| `"attention_approx"` | Single prefill mask + KV surgery; faster than `"attention"` |
| `"attention_skip"` | Future tokens filtered from input; `position_ids` passed explicitly |
| `"remove"` | Future bars omitted entirely from the token stream |
| `"token"` | Encoder emits a `MaskBar` token (requires vocab support) |

Set via `InferenceConfig(mask_mode="attention")`. The `"attention"` family works on all models; `"token"` requires the encoder vocab to include a `MaskBar` domain (only `ghost`).
