[![Metacreation Lab](https://drive.google.com/uc?export=view&id=1nzeq0DmD7hAYteRs5PA42150HIzO3Sz7)](https://metacreation.net/category/projects/)

# midigpt

[![PyPI](https://img.shields.io/pypi/v/midigpt)](https://pypi.org/project/midigpt/)
[![Python](https://img.shields.io/pypi/pyversions/midigpt)](https://pypi.org/project/midigpt/)
[![CI](https://img.shields.io/github/actions/workflow/status/Metacreation-Lab/MIDI-GPT/wheels.yml?branch=main&label=CI)](https://github.com/Metacreation-Lab/MIDI-GPT/actions/workflows/wheels.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://metacreation-lab.github.io/MIDI-GPT/)
[![License: MIT](https://img.shields.io/github/license/Metacreation-Lab/MIDI-GPT)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2501.17011-b31b1b)](https://arxiv.org/abs/2501.17011)

A transformer model for **computer-assisted multitrack music composition**.

- **Fill in missing bars** while preserving your existing arrangement
- **Generate new tracks** from scratch, conditioned on musical attributes
- **Steer the output** by controlling note density, polyphony, and note duration — globally or per bar
- **Integrate with your DAW** via a real-time OSC server
- **One-line setup** — load pretrained models from HuggingFace Hub, no compiler needed

---

## Installation

```bash
pip install "midigpt[inference]"
```

Pre-built wheels for CPython 3.10–3.12 on Linux (x86_64), macOS (x86_64 + arm64), and Windows (AMD64). No compiler needed.

| Extra | What it adds |
|---|---|
| `inference` | `torch>=2.0`, `tqdm`, `huggingface_hub` |
| `train` | PyTorch Lightning, HuggingFace `datasets`, `pyarrow`, `python-dotenv` |
| `realtime` | `python-osc`, Flask, Flask-SocketIO |
| `dev` | `pytest`, `ruff`, `mypy` |
| `all` | `realtime` + `train` |

---

## Quick start

Load a pretrained model from [HuggingFace Hub](https://huggingface.co/Metacreation/MIDI-GPT) and generate music in four lines:

```python
from midigpt import Score, Track, Bar
from midigpt.inference import InferenceEngine, GenerationRequest, InferenceConfig, TrackPrompt

engine = InferenceEngine.from_pretrained("yellow")

# 4-bar score with one empty melodic track
score = Score(tracks=[Track(bars=[Bar() for _ in range(4)])])

result = engine.session(
    score,
    GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[0, 1, 2, 3])],
        config=InferenceConfig(model_dim=4, mask_mode="attention"),
    ),
).run()

total = sum(len(b.notes) for t in result.tracks for b in t.bars)
print(f"Generated {total} notes")
result.to_midi("output.mid")
```

The model is downloaded once and cached by `huggingface_hub` in `~/.cache/huggingface/hub/`.

---

## Models

| Name | `num_bars_map` | Infill | Attributes | Download |
|---|---|---|---|---|
| `yellow` | 4, 8 | yes | note density, polyphony (min/max), note duration (min/max) | [yellow.pt](https://huggingface.co/Metacreation/MIDI-GPT/resolve/main/yellow.pt) |
| `ghost` | 4, 8, 12, 16 | yes | note density, polyphony (min/max), note duration (min/max) | coming soon |
| `expressive` | 4, 8 | yes | note density, polyphony (min/max), note duration (min/max) | coming soon |

`model_dim` in `InferenceConfig` is the context window in bars, not a vocabulary dimension — pass a value from the model's `num_bars_map`. `expressive` additionally encodes sub-grid timing via delta tokens.

---

## Inference API

### Load a model

```python
# By name (downloads from Metacreation/MIDI-GPT on HuggingFace Hub)
engine = InferenceEngine.from_pretrained("yellow")   # or "ghost", "expressive"

# From a local .pt bundle
engine = InferenceEngine.from_checkpoint("path/to/model.pt")
```

### Infill existing bars

```python
score = Score.from_midi("my_song.mid")

request = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[4, 5, 6, 7]),   # bars to regenerate
        TrackPrompt(id=1, bars=[], ignore=True), # leave track 1 unchanged
    ],
    config=InferenceConfig(temperature=1.0, top_p=0.95, model_dim=8),
)

result = engine.session(score, request).run()
result.to_midi("output.mid")
```

### Autoregressive generation from scratch

```python
request = GenerationRequest(
    tracks=[
        TrackPrompt(
            id=0,
            bars=[],
            autoregressive=True,
            attributes={"max_polyphony": 3},      # quantized attribute level
            controls={"time_signature": 0},        # index into encoder TS list
        ),
    ],
    config=InferenceConfig(temperature=1.0, model_dim=8, polyphony_hard_limit=4),
)
result = engine.session(score, request).run()
```

### Key types

| Class | Module | Purpose |
|---|---|---|
| `InferenceEngine` | `midigpt.inference` | Top-level loader and session factory |
| `GenerationRequest` | `midigpt.inference` | Bundle of per-track prompts and config |
| `TrackPrompt` | `midigpt.inference` | Per-track bars, mode, attributes, controls |
| `InferenceConfig` | `midigpt.inference` | Temperature, sampling filters, step planner |
| `SamplingSession` | `midigpt.inference` | Token-level sampling loop (returned by `session()`) |

### `TrackPrompt` fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | int | — | Track index in the score |
| `bars` | list[int] | — | Bars to generate |
| `autoregressive` | bool | `False` | Generate from scratch (no per-bar prompt) |
| `ignore` | bool | `False` | Omit this track from the token stream |
| `mask_bars` | list[int] | `[]` | Bars hidden with `MASK_BAR` (disjoint from `bars`) |
| `attributes` | dict[str,int] | `{}` | Quantized attribute overrides |
| `controls` | dict[str,Any] | `{}` | Token locks e.g. `{"time_signature": 0}` |
| `bar_attributes` | dict[int,dict] | `{}` | Per-bar attribute overrides (absolute bar index) |
| `bar_controls` | dict[int,dict] | `{}` | Per-bar control overrides (absolute bar index) |

### Sampling filters

`InferenceConfig` exposes a four-stage logit-filtering pipeline (`top_k` → `top_p` → `mask_k` → `mask_p`):

| Field | Default | Meaning |
|---|---|---|
| `top_k` | `0` (off) | Keep top-k highest-probability tokens |
| `top_p` | `1.0` (off) | Nucleus: keep the smallest set summing to ≥ `top_p` |
| `mask_k` | `0` (off) | Remove the top-k most-likely tokens (novelty pressure) |
| `mask_p` | `0.0` (off) | Anti-nucleus: remove tokens summing to ≥ `mask_p` from the top |

A small `mask_k=1` or `mask_p=0.3` pushes the model off its most-confident picks — useful for getting diverse outputs when `novelty_check=True`.

### Mask modes

Control how future bars appear in the context window:

| Mode | Behaviour |
|---|---|
| `"token"` | Encoder emits a `MaskBar` token (requires vocab support) |
| `"attention"` | Future bars zeroed in the KV cache via exact span masking |
| `"attention_approx"` | Single prefill mask + KV surgery; cheaper than `"attention"` |
| `"attention_skip"` | Future tokens filtered from input; `position_ids` passed explicitly |
| `"remove"` | Future bars omitted entirely from the token stream |

Set via `InferenceConfig(mask_mode="attention")`. `"attention"` works on all encoders; `"token"` requires the encoder vocab to include a `MaskBar` domain.

### Attribute controls

Introspect available controls at runtime:

```python
engine._analyzer.attribute_sizes()         # {"note_density": 10, "min_polyphony": 10, ...}
engine._analyzer.attribute_value_labels()  # {"note_density": ["very sparse", ...], ...}
engine._analyzer.attribute_track_types()   # {"note_density": "melodic", ...}
```

Pass quantized levels (integers in `[0, size)`) in `TrackPrompt.attributes`.

---

## Training

### 1. Preprocess parquet shards

```bash
python -m midigpt.training.preprocess \
    --parquet /data/train/*.parquet \
    --checkpoint models/yellow.pt
```

Builds a valid-index cache so dataset initialization is instant on subsequent runs. Cached in `~/.midigpt/` (override with `MIDIGPT_CACHE`).

### 2. Launch training

```bash
python -m midigpt.training.trainer \
    --config     models/train_config.json \
    --train-data /data/train/*.parquet \
    --eval-data  /data/valid/*.parquet \
    --output-dir checkpoints/run_001
```

### 3. Python API

```python
from midigpt.training.trainer import TrainConfig, train

config = TrainConfig.from_file("models/train_config.json")
train(config, train_path="/data/train/00000.parquet", eval_path="/data/valid/00000.parquet")
```

`train()` uses PyTorch Lightning and writes a packed `.pt` bundle at the end of training containing weights, architecture config, and encoder config.

### Key `TrainConfig` fields

| Field | Default | Notes |
|---|---|---|
| `n_embd` / `n_layer` / `n_head` | `512 / 6 / 8` | Model architecture |
| `max_seq_len` | `2048` | Token sequence cap |
| `infill_probability` | `0.75` | Fraction of samples trained with FillIn tokens |
| `mask_apply_probability` | `0.5` | Fraction of samples with `MASK_BAR` applied |
| `precision` | `"fp16"` | `"fp16"`, `"bf16"`, or `"fp32"` |
| `logger` | `"none"` | `"tensorboard"`, `"wandb"`, or `"none"` |
| `num_workers` | `0` | Must be 0 — the C++ MIDI parser is not fork-safe |

---

## Real-time OSC server

```bash
pip install "midigpt[realtime]"
midigpt-server --ckpt models/yellow.pt --port 7400
```

Listens for OSC messages on a UDP port and streams generated notes back in real time. Generation is triggered bar-by-bar via `/midigpt/bar/end` on a background thread.

Selected OSC addresses:

| Address | Direction | Description |
|---|---|---|
| `/midigpt/session/init` | in | Start a new session |
| `/midigpt/track/create` | in | Register a track |
| `/midigpt/note` | in | Push an incoming note |
| `/midigpt/bar/end` | in | Signal bar end (triggers generation) |
| `/midigpt/param/set` | in | Adjust sampling parameters at runtime |
| `/midigpt/attr/set` | in | Set attribute overrides |
| `/midigpt/generated/note` | out | Emit a generated note |
| `/midigpt/generated/features` | out | Per-bar statistics |
| `/midigpt/capabilities` | out | Attribute support for the loaded checkpoint |

---

## Development

### Setup

```bash
git clone https://github.com/Metacreation-Lab/MIDI-GPT.git
cd MIDI-GPT
pip install -e ".[inference,dev]"   # compiles the C++ extension in-place
```

Prerequisites: Python 3.10+, CMake 3.21+, a C++20 compiler.

### Tests

```bash
# Python
pytest tests/python/
pytest tests/python -m "not slow and not inference"   # CI subset (no model needed)

# C++
cmake -S . -B build_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpp -j
ctest --test-dir build_cpp --output-on-failure
```

### Linting

```bash
ruff check src/ tests/    # lint
ruff format src/ tests/   # format
```

[pre-commit](https://pre-commit.com/) runs both automatically on commit:

```bash
pip install pre-commit && pre-commit install
```

### Release

Tag a commit `vX.Y.Z` → `.github/workflows/wheels.yml` builds wheels on Linux / macOS / Windows × Python 3.10–3.12, drafts a GitHub Release, and publishes to PyPI via OIDC Trusted Publishing.

### Logging

Set `MIDIGPT_LOG_LEVEL=DEBUG` (or a numeric level) before importing. Accepts both string names (`DEBUG`, `INFO`, `WARNING`) and integers.

---

## Citation

```bibtex
@misc{pasquier2025midigptcontrollablegenerativemodel,
      title={MIDI-GPT: A Controllable Generative Model for Computer-Assisted Multitrack Music Composition},
      author={Philippe Pasquier and Jeff Ens and Nathan Fradet and Paul Triana and Davide Rizzotti and Jean-Baptiste Rolland and Maryam Safi},
      year={2025},
      eprint={2501.17011},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2501.17011},
}
```

---

## License

MIT License — Copyright (c) 2026 Metacreation Lab. See [LICENSE](LICENSE).
