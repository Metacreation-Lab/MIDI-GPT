# Contributing

## Development setup

### Prerequisites

- Python 3.10+
- CMake 3.21+
- A C++20 compiler (GCC 12+, Clang 14+, or MSVC 2022+)

### Clone and install

```bash
git clone https://github.com/Metacreation-Lab/MIDI-GPT.git
cd MIDI-GPT
pip install -e ".[inference,dev]"
```

`scikit-build-core` compiles the C++ extension and copies `_core*.so` next to `src/python/midigpt/__init__.py`. In-tree `pytest` works without reinstallation.

### Install pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

This runs `ruff` (lint + format) automatically on every commit.

---

## Running tests

### Python tests

```bash
# Full suite
pytest tests/python/

# CI subset — no model files or torch required
pytest tests/python -m "not slow and not inference"

# Single test
pytest tests/python/session/test_ar.py::test_ar_session_run_returns_score
```

Test markers:

| Marker | Meaning |
|---|---|
| `slow` | Requires real model bundles on disk |
| `inference` | Requires `torch` and a real model checkpoint |

### C++ tests

```bash
cmake -S . -B build_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpp -j
ctest --test-dir build_cpp --output-on-failure
```

C++ test targets: `test_score`, `test_io`, `test_vocabulary`, `test_tokenizer`, `test_constraints`, `test_step_planner`, `test_session_state`, `test_domain_transforms`.

---

## Code style

### Python — Ruff

```bash
ruff check src/ tests/    # lint
ruff format src/ tests/   # format
```

The Ruff configuration lives in `pyproject.toml` under `[tool.ruff]`. Enabled rule sets: E, W, F, I, UP, B, RUF, N. Docstring rules (D) and type annotation rules (ANN) are not enforced yet.

### C++

Follow the style of the surrounding code. All translation units are listed explicitly in `CMakeLists.txt` — add new files there when you create them.

---

## Project layout

```
src/
  cpp/                     C++ static library + pybind11 module (_core)
    io/                    MIDI reader/writer (symusic)
    tokenizer/             EncoderConfig, Vocabulary, Encoder, Decoder
    masking/               ConstraintGraph, grammar and attribute constraints
    sampling/              StepPlanner, SessionState
    bindings/lib.cpp       pybind11 entry point

  python/midigpt/
    _types.py              Score, Track, Bar, Note dataclasses
    _converters.py         to_cpp() / from_cpp() — convert between Python and C++ score types
    inference/             InferenceEngine, SamplingSession, GPT2LMHeadModel
    tokenizer/             Tokenizer, load_checkpoint
    training/              TrainConfig, MidiGPTDataset, train()
    augmentation/          MaskBar, Transpose, VelocityScale, ...
    attributes/            AttributeAnalyzer, BaseAttribute, ATTRIBUTE_REGISTRY
    osc/                   MidiGPTServer (studio excluded from PyPI wheel)

tests/
  cpp/                     C++ tests (CMake/ctest)
  python/                  Python tests (pytest)
    conftest.py            Shared fixtures
    model/                 GPT-2 model and engine tests
    session/               Sampling session and constraint tests
```

---

## Architecture notes

A few things worth knowing before modifying the code:

- **Guides for extending tokenization/conditioning:**
  - For implementing a new attribute control end-to-end, follow the [Attribute Controls Guide](attribute_controls_guide.md).
  - For designing and validating encoder configurations, follow the [Encoder Config Guide](encoder_guide.md).
- **Token IDs and vocabularies live exclusively in C++.** `EncoderConfig.from_json(str)` is the entry point. `Tokenizer.vocab_size()` is authoritative — do not recompute it from `sum(token_domains[*].domain_size)`.
- **`_types.Score` vs. `_core.Score`:** The Python side uses `_types.Score` (dataclasses). The C++ bindings use `_core.Score`. `_converters.to_cpp()` and `from_cpp()` convert between them. `SamplingSession._run_step()` normalises to `_types.Score` on entry.
- **`model_dim` is a generation-window parameter,** not a score bar count. Tests that use small scores must pad to at least `model_dim` bars.
- **cibuildwheel on Windows:** `cmd.exe` does not strip single quotes. The `test-command` in `pyproject.toml` must use TOML literal-string outer with double quotes inside.

---

## Pull request checklist

- [ ] `ruff check src/ tests/` passes with no errors
- [ ] `ruff format --check src/ tests/` passes
- [ ] `pytest tests/python -m "not slow and not inference"` passes
- [ ] C++ tests pass (`ctest --test-dir build_cpp`)
- [ ] New public API is documented in `docs/api.md`
- [ ] `pyproject.toml` and `src/python/midigpt/__init__.py` versions match (on release PRs)

---

## Release process

1. Bump `version` in `pyproject.toml` and `__version__` in `src/python/midigpt/__init__.py`.
2. Commit and push to `dev`, open a PR into `main`.
3. After merge, tag the commit: `git tag vX.Y.Z && git push upstream vX.Y.Z`.
4. The `wheels.yml` workflow builds wheels on Linux / macOS / Windows × Python 3.10–3.12, creates a draft GitHub Release, and publishes to PyPI via OIDC Trusted Publishing (gated by the `PyPI` environment).
