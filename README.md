# MIDI-GPT

A GPT-2 transformer for symbolic music generation — tokenizes MIDI, generates new bars or in-fills missing ones, and decodes back to MIDI.

Paper: [AAAI 2025 — *MIDI-GPT: A Controllable Generative Model for Computer-Assisted Multitrack Music Composition*](https://arxiv.org/abs/2501.17011)

This release ships the **inference stack** as a clean, self-contained Python package:

- C++ tokenizer / encoder / decoder (built via `pybind11` + `scikit-build-core`)
- Pure-PyTorch GPT-2 with `F.scaled_dot_product_attention` and KV cache
- Self-contained packed checkpoint format (weights + arch config + encoder config in one file)
- CPU, MPS (Apple Silicon), and CUDA devices

> The real-time OSC server and Max MSP integration ship in a separate release.

---

## Install

Requires Python ≥ 3.10, CMake ≥ 3.21, a C++20 compiler, and PyTorch ≥ 2.0.

```bash
git clone https://github.com/Metacreation-Lab/MIDI-GPT.git
cd MIDI-GPT
python -m venv .venv
source .venv/bin/activate
pip install -e ".[inference]"
```

macOS dependencies:

```bash
brew install cmake
```

Extras:

| extra | adds |
|---|---|
| `inference` | `torch>=2.0` |
| `train` | `transformers`, `datasets`, `accelerate`, `pyarrow` |
| `dev` | `pytest`, `ruff`, `mypy` |
| `all` | everything |

---

## Download a model

Pretrained weights are distributed as packed `.pt` files (arch config + encoder config + state dict in one archive).

| Model | Encoder | Size | Download |
|---|---|---|---|
| Yellow | `EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER` | ~80 MB | **[yellow.pt](https://example.com/midigpt/yellow.pt)** *(placeholder)* |

Drop it in `models/`:

```bash
mkdir -p models
curl -L https://example.com/midigpt/yellow.pt -o models/yellow.pt
```

---

## Quick start

```python
from midigpt.inference.model import GPT2LMHeadModel
from midigpt.inference.engine import InferenceEngine
from midigpt.inference.config import GenerationRequest, TrackPrompt, SamplingConfig
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.attributes import AttributeAnalyzer
from midigpt import _core
import json

# Load the packed checkpoint — auto-detects device (cuda > mps > cpu)
model = GPT2LMHeadModel.from_pretrained("models/yellow.pt", device="auto")

# The encoder config travels inside the checkpoint
cfg       = _core.EncoderConfig.from_json(json.dumps(model.encoder_config))
analyzer  = AttributeAnalyzer.from_config(cfg)
tokenizer = Tokenizer(cfg, analyzer)
engine    = InferenceEngine(model=model, tokenizer=tokenizer, analyzer=analyzer)

# Build a request: tracks to keep as prompt, bars to fill, sampling controls
request = GenerationRequest(
    midi_path="path/to/input.mid",
    tracks=[TrackPrompt(index=0)],
    bars_to_generate=[3, 4],
    sampling=SamplingConfig(temperature=1.0, max_attempts=1),
)

result = engine.generate(request)
result.save("output.mid")
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  midigpt (Python package)                                    │
│                                                              │
│   tokenizer/    →  encode MIDI ↔ token IDs   (C++ ext)       │
│   attributes/   →  density / polyphony / duration controls   │
│   inference/                                                 │
│     model.py    →  GPT2LMHeadModel (SDPA + KV cache)         │
│     engine.py   →  high-level generate()                     │
│     session.py  →  multi-step planning + sampling            │
│   augmentation/ →  transpose / velocity / track swap …       │
│   training/     →  optional HF-trainer integration           │
└──────────────────────────────────────────────────────────────┘
```

### Packed checkpoint format (`format_version: 1`)

```python
{
    "format_version": 1,
    "config":         {"vocab_size": ..., "n_positions": ..., "n_embd": ..., "n_layer": ..., "n_head": ...},
    "encoder_config": {...},   # full encoder JSON
    "state_dict":     {...},   # HF GPT-2 layout
}
```

`GPT2LMHeadModel.from_pretrained(path)` auto-detects this format. Everything needed to tokenize and run is inside the file — no sidecar JSON.

---

## Tests

```bash
pytest tests/python/                       # Python unit tests
ctest --test-dir build_cpp                 # C++ unit tests (after a build)
```

---

## License

See `LICENSE`.
