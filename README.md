[![Metacreation Lab](https://drive.google.com/uc?export=view&id=1aCMgI91K3ik2rp17pM7cOOUi6wUbw-GZ)](https://metacreation.net/category/projects/)

# MIDI-GPT

**MIDI-GPT** is a GPT-2 transformer for symbolic music generation. It ships a
C++ tokenizer, encoder, and decoder (exposed to Python via pybind11 as
`midigpt._core`) alongside a pure-PyTorch GPT-2 implementation with SDPA
attention and a KV cache. The library supports bar-level infill (filling in
masked bars given surrounding context), autoregressive track generation from
scratch, and attribute-conditioned generation (note density, polyphony, note
duration). A real-time OSC server integrates with DAWs and live-performance
environments via the `midigpt-studio` entry point. The Python package (`midigpt`)
is distributed on PyPI and built with `scikit-build-core` for CPython 3.10,
3.11, and 3.12 on Linux, macOS, and Windows.

Paper: <https://arxiv.org/abs/2501.17011>  
Repository: <https://github.com/Metacreation-Lab/MIDI-GPT>

---

## Installation

### Users (inference only)

```bash
pip install "midigpt[inference]"
```

Pre-built wheels are available for CPython 3.10–3.12 on Linux (x86_64),
macOS (x86_64, arm64), and Windows (AMD64). No compiler is required.

### Training dependencies

```bash
pip install "midigpt[train]"
```

Adds `lightning>=2.2`, `datasets>=2.18`, `pyarrow>=15.0`, and `python-dotenv`.

### Real-time OSC server

```bash
pip install "midigpt[realtime]"
```

Adds `python-osc>=1.8`, `flask>=3.0`, and `flask-socketio>=5.3`.

### All extras

```bash
pip install "midigpt[all]"
```

### Developer install (editable + C++ extension)

Prerequisites: Python 3.10+, CMake 3.21+, a C++20 compiler.

```bash
git clone https://github.com/Metacreation-Lab/MIDI-GPT.git
cd MIDI-GPT
pip install -e ".[inference,dev]"
```

`scikit-build-core` compiles the C++ extension and copies `_core*.so` next to
`src/python/midigpt/__init__.py` so in-tree `pytest` works without
reinstallation.

| Extra | What it adds |
|---|---|
| `inference` | `torch>=2.0`, `tqdm>=4.65` |
| `train` | PyTorch Lightning, HuggingFace `datasets`, `pyarrow`, `python-dotenv` |
| `realtime` | `python-osc`, Flask, Flask-SocketIO |
| `dev` | `pytest`, `ruff`, `mypy` |
| `all` | `realtime` + `train` |

---

## Quickstart: Inference

### Load a checkpoint and run a generation session

`InferenceEngine.from_checkpoint` is the single entry point for loading any
packed `.pt` bundle. It reads the weights, builds the tokenizer from the
embedded encoder config, and runs a warmup pass to prime the KV cache.

```python
from midigpt import Score
from midigpt.inference.engine import InferenceEngine
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt

# Load a packed .pt bundle (weights + encoder config in one file).
engine = InferenceEngine.from_checkpoint("models/ghost_500_bundle.pt")

# Read an input MIDI file.
score = Score.from_midi("my_song.mid")

# Infill bars 4–7 on track 0; leave track 1 untouched.
request = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[4, 5, 6, 7]),
        TrackPrompt(id=1, bars=[], ignore=True),
    ],
    config=InferenceConfig(
        temperature=1.0,
        top_p=0.95,
        model_dim=8,      # context window in bars — must be in num_bars_map
        max_attempts=3,
    ),
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
            bars=[],               # empty = generate the whole track
            autoregressive=True,
            attributes={"max_polyphony": 3},
            controls={"time_signature": 0},   # index into encoder TS list
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

### Key inference types

| Class | Module | Purpose |
|---|---|---|
| `InferenceEngine` | `midigpt.inference.engine` | Top-level loader and session factory |
| `SamplingSession` | `midigpt.inference.session` | Token-level sampling loop |
| `GenerationRequest` | `midigpt.inference.config` | Bundle of per-track prompts and config |
| `TrackPrompt` | `midigpt.inference.config` | Per-track bars, mode, attributes, controls |
| `InferenceConfig` | `midigpt.inference.config` | Temperature, sampling filters, step planner |

### `TrackPrompt` fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | int | — | Track index in the score |
| `bars` | list[int] | — | Bars to generate (infill targets or AR suffix) |
| `autoregressive` | bool | `False` | Generate from scratch (no per-bar prompt) |
| `ignore` | bool | `False` | Omit this track from the token stream entirely |
| `mask_bars` | list[int] | `[]` | Bars hidden with `MASK_BAR` (disjoint from `bars`) |
| `attributes` | dict[str,int] | `{}` | Quantized attribute overrides (density, polyphony, duration) |
| `controls` | dict[str,Any] | `{}` | Non-attribute token locks, e.g. `{"time_signature": 0}` |
| `bar_attributes` | dict[int,dict] | `{}` | Per-bar attribute overrides keyed by absolute bar index |
| `bar_controls` | dict[int,dict] | `{}` | Per-bar non-attribute overrides keyed by absolute bar index |

### Sampling filters

`InferenceConfig` exposes a four-stage logit-filtering pipeline applied after
the grammar mask and before `torch.multinomial`. Pipeline order: `top_k` ->
`top_p` -> `mask_k` -> `mask_p`.

| Field | Default | Meaning |
|---|---|---|
| `top_k` | `0` (off) | Keep top-k highest-probability tokens |
| `top_p` | `1.0` (off) | Nucleus: keep the smallest descending-prob set summing to >= `top_p` |
| `mask_k` | `0` (off) | Remove the top-k most-likely tokens (novelty pressure) |
| `mask_p` | `0.0` (off) | Remove tokens summing to >= `mask_p` from the top (anti-nucleus) |

A small `mask_k=1` or `mask_p=0.3` pushes the model off its highest-confidence
picks, which is the most reliable way to get diverse retries when
`novelty_check=True`.

### Attribute controls

The attribute controls available depend on the checkpoint. Introspect at
runtime via the engine's analyzer:

```python
analyzer = engine._analyzer
analyzer.attribute_sizes()         # {"note_density": 10, "min_polyphony": 10, ...}
analyzer.attribute_value_labels()  # {"note_density": ["very sparse", ...], ...}
analyzer.attribute_track_types()   # {"note_density": "melodic", ...}
```

Pass quantized levels (integers in `[0, size)`) in `TrackPrompt.attributes`.

### Mask modes

Control how future (not-yet-generated) bars appear in the context window:

| Mode | Behaviour |
|---|---|
| `"token"` | Encoder emits a `MaskBar` token (requires vocab support) |
| `"attention"` | Future bar positions zeroed in the KV cache via exact span masking |
| `"attention_approx"` | Single prefill mask + KV surgery after prefill; cheaper than `"attention"` |
| `"attention_skip"` | Future tokens filtered from input; `position_ids` passed explicitly |
| `"remove"` | Future bars omitted entirely from the token stream |

Set via `InferenceConfig.mask_mode`.

---

## Quickstart: Training

### 1. Preprocess parquet shards (run once per encoder/dataset combination)

Builds a valid-index cache so dataset initialization is instant on subsequent
runs. The filter runs a fast metadata check (pure PyArrow, no MIDI parsing),
then validates each row via an isolated subprocess that bisects on crash.

```bash
python -m midigpt.training.preprocess \
    --parquet /data/train/00000.parquet /data/train/00001.parquet \
    --checkpoint models/yellow.pt
```

Alternatively, supply a raw encoder config JSON:

```bash
python -m midigpt.training.preprocess \
    --parquet /data/train/*.parquet \
    --encoder-config models/yellow_encoder.json \
    --min-bars 4 --min-tracks 1
```

Index files are cached in `~/.midigpt/` (override with `MIDIGPT_CACHE`).

### 2. Launch training

```bash
python -m midigpt.training.trainer \
    --config      models/train_config.json \
    --train-data  /data/train/00000.parquet \
    --eval-data   /data/valid/00000.parquet \
    --output-dir  checkpoints/run_001
```

### 3. Python API

```python
from midigpt.training.trainer import TrainConfig, train

config = TrainConfig.from_file("models/train_config.json")
config.output_dir = "checkpoints/run_001"

train(config,
      train_path="/data/train/00000.parquet",
      eval_path="/data/valid/00000.parquet")
```

`train()` uses PyTorch Lightning internally. At the end of training it writes a
packed `.pt` bundle (`model_final.pt`) containing weights, architecture config,
and encoder config. Intermediate checkpoints are saved every `save_steps` steps.

### Key `TrainConfig` fields

| Field | Default | Notes |
|---|---|---|
| `encoder_config_path` | `""` | Path to an encoder `.json` or a packed `.pt` bundle |
| `n_embd` / `n_layer` / `n_head` | `512 / 6 / 8` | Model architecture |
| `max_seq_len` | `2048` | Token sequence cap; must not exceed model `n_positions` |
| `infill_probability` | `0.75` | Fraction of samples trained with FillIn tokens |
| `infill_bar_fraction` | `0.5` | Max per-cell infill density (drawn from Uniform(0, this)) |
| `mask_apply_probability` | `0.5` | Fraction of samples with `MASK_BAR` applied |
| `mask_mode` | `2` | `MaskMode`: 0=RANDOM, 1=STRUCTURED, 2=MIXED |
| `precision` | `"fp16"` | `"fp16"`, `"bf16"`, or `"fp32"` |
| `logger` | `"none"` | `"tensorboard"`, `"wandb"`, or `"none"` |
| `num_workers` | `0` | Must be 0 — the C++ MIDI parser is not fork-safe |

The reference config is at `models/train_config.json`.

---

## Model Zoo

The `models/` directory contains encoder configs for the checkpoint families
shipped in this repository. Packed `.pt` bundles embed the encoder config
alongside the model weights; the configs below describe the tokenizer and
capability set.

| Model | `num_bars_map` | Infill | `MaskBar` | Microtiming | Velocity bins | Attributes | Download |
|---|---|---|---|---|---|---|---|
| Yellow | 4, 8 | yes | no | no | 32 | note density, min/max polyphony, min/max note duration | coming soon |
| Ghost | 4, 8, 12, 16 | yes | yes | yes | 32 | note density, min/max polyphony, min/max note duration | coming soon |
| Expressive | 4, 8 | yes | no | yes | 128 | note density, min/max polyphony, min/max note duration | coming soon |

**`model_dim`** in `InferenceConfig` is the context window length in bars, not a
vocabulary dimension. Pass a value from the checkpoint's `num_bars_map`. The
session automatically falls back to the next smaller window when the encoded
prompt would overflow the model's positional budget (`n_positions`).

**Microtiming** (`use_microtiming: true`) means the encoder emits `delta`
offset tokens that capture sub-grid note placement. The `expressive` config
additionally uses `emit_delta_tokens: true` for a dedicated delta token domain.

---

## Architecture

### Two-language layout

```
src/
  cpp/                     C++ static library (midigpt_core) + pybind11 module (_core)
    io/                    MIDI reader / writer (symusic)
    tokenizer/             EncoderConfig, Vocabulary, Encoder, Decoder
    masking/               ConstraintGraph, GrammarConstraint,
                           PolyphonyConstraint, DensityConstraint,
                           AttributeValueConstraint
    sampling/              StepPlanner, SessionState
    bindings/lib.cpp       pybind11 entry point

  python/midigpt/
    _core*.so              compiled extension (copied here post-build)
    _types.py              Score, Track, Bar, Note dataclasses
    inference/             InferenceEngine, SamplingSession, GPT2LMHeadModel,
                           GenerationRequest, TrackPrompt, InferenceConfig
    tokenizer/             Tokenizer, load_checkpoint, CheckpointBundle
    training/              TrainConfig, MidiGPTDataset, train()
    augmentation/          MaskBar, Transpose, VelocityScale
    attributes/            AttributeAnalyzer, BaseAttribute, ATTRIBUTE_REGISTRY
    osc/                   MidiGPTServer, studio app
```

Token IDs, vocabularies, constraint graphs, and the step planner all live
exclusively in C++. `EncoderConfig.from_json(str)` is the entry point for
everything that depends on vocab sizes or token domains. `Tokenizer.vocab_size()`
is authoritative — do not recompute it from `sum(token_domains[*].domain_size)`.

### Inference data flow

```
MIDI file ──► Score.from_midi()
                      |
                      v
           _core.Encoder.encode()       (C++ — token IDs)
                      |
                      v
       InferenceEngine.session(score, request)
                      |
                      v
       SamplingSession.run()
         for each GenerationStep from _core.StepPlanner:
           1. build ConstraintGraph    (_core C++)
           2. encode prompt            (_core.SessionState)
           3. GPT2LMHeadModel.forward  (PyTorch — logits, past_kv)
           4. apply grammar mask + top_k/top_p/mask_k/mask_p filters
           5. torch.multinomial        (sample one token)
           6. _core.SessionState.advance(token)
           7. repeat until state.complete()
                      |
                      v
           _core.Decoder.decode()      (C++ — Score)
                      |
                      v
               Score ──► to_midi()
```

`InferenceEngine` accepts any callable with signature
`(input_ids, past_kv) -> (logits, present_kv)`. `GPT2LMHeadModel` is the
production implementation; `StubModel` in `tests/python/test_inference.py` is
the test double.

### Packed checkpoint format (`format_version: 1`)

A single `.pt` file holds:

```python
{
    "format_version": 1,
    "arch":           "gpt2",
    "config":         {"vocab_size": ..., "n_positions": 2048,
                       "n_embd": 512, "n_layer": 6, "n_head": 8},
    "encoder_config": {...},   # full encoder JSON
    "state_dict":     {...},   # HuggingFace GPT-2 key layout
}
```

`GPT2LMHeadModel.from_pretrained(path)` and `load_checkpoint(path)` both
auto-detect this format. `load_checkpoint` also accepts a legacy directory
containing `config.json` + `model.pt`.

---

## OSC Studio

The `midigpt[realtime]` extra adds a real-time OSC server and a browser-based
studio for DAW integration.

```bash
# Start the low-latency OSC server (receives notes, sends generated bars)
midigpt-server --ckpt models/ghost_500_bundle.pt --port 7400

# Start the full browser-based studio (wraps the server with a web UI)
midigpt-studio --ckpt models/ghost_500_bundle.pt
```

`midigpt-server` runs `MidiGPTServer`, which listens for OSC messages on a UDP
port and sends generated notes back over the same connection. Generation is
triggered bar-by-bar via `/midigpt/bar/end` and runs on a dedicated background
thread so the OSC listener never blocks.

Selected OSC address map:

| Address | Direction | Description |
|---|---|---|
| `/midigpt/session/init` | in | Start a new session; server replies with `/midigpt/capabilities` |
| `/midigpt/session/start` | in | Begin real-time generation |
| `/midigpt/track/create` | in | Register a track (human or agent) |
| `/midigpt/note` | in | Push an incoming note event |
| `/midigpt/bar/end` | in | Signal end of a bar (triggers generation if scheduled) |
| `/midigpt/param/set` | in | Adjust sampling parameters at runtime |
| `/midigpt/attr/set` | in | Set agent attribute overrides (quantized levels) |
| `/midigpt/generated/note` | out | Emit a generated note |
| `/midigpt/generated/features` | out | Per-bar statistics (density, polyphony, etc.) |
| `/midigpt/capabilities` | out | Attribute support for the loaded checkpoint |
| `/midigpt/prompt/state` | out | Per-bar context/mask/generate state snapshot |

Runtime sampling parameters (`/midigpt/param/set`) include `temperature`,
`top_p`, `mask_mode`, `model_dim`, `buffer_bars`, `lookahead_bars`, and
`polyphony_hard_limit`, among others.

---

## Development

### Python tests

```bash
pytest tests/python/                                            # all tests
pytest tests/python/test_inference.py::test_inference_session  # single test
pytest tests/python -m "not slow and not inference"            # CI subset
```

Test markers:

- `slow` — requires real model bundles on disk
- `inference` — requires `torch` and a real model

### C++ tests

```bash
cmake -S . -B build_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpp -j
ctest --test-dir build_cpp --output-on-failure
```

C++ test targets: `test_score`, `test_io`, `test_vocabulary`, `test_tokenizer`,
`test_constraints`, `test_step_planner`, `test_session_state`,
`test_domain_transforms`.

### Linting and type checking

```bash
ruff check src/ tests/
mypy src/python/midigpt/
```

### Building wheels

```bash
pipx run cibuildwheel --platform linux    # or macos / windows
```

Wheels are built for CPython 3.10, 3.11, and 3.12 on Linux (manylinux_2_28
x86_64), macOS (x86_64 and arm64), and Windows (AMD64). musllinux and 32-bit
targets are skipped.

### CI / release

Tagging a commit as `vX.Y.Z` triggers `.github/workflows/wheels.yml`, which
builds and tests wheels on all platforms, creates a draft GitHub Release, and
publishes to PyPI via OIDC Trusted Publishing (gated by the `pypi`
environment).

### Logging verbosity

Set `MIDIGPT_LOG_LEVEL=DEBUG` (or a numeric level) before importing the
package. The Python side accepts both string names (`DEBUG`, `INFO`,
`WARNING`) and integer levels. The C++ core uses the same environment variable
via `midigpt._core.set_verbosity`.

---

## License

MIT License. Copyright (c) 2025 Metacreation Lab. See [LICENSE](LICENSE).
