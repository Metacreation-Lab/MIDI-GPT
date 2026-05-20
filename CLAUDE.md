# CLAUDE.md

## Project

`midigpt` — symbolic music generation. A C++ tokenizer/encoder/decoder (exposed via pybind11 as `midigpt._core`) plus a pure-PyTorch GPT-2 (SDPA + KV cache) in `src/python/midigpt/inference/`. Distributed on PyPI as `midigpt`; built with `scikit-build-core`.

## Common commands

```bash
# Dev install (rebuilds C++ extension into src/python/midigpt/_core*.so)
pip install -e ".[inference,dev]"

# Python tests
pytest tests/python/                                  # all
pytest tests/python/test_inference.py::test_inference_session   # single
pytest tests/python -m "not slow and not inference"   # what CI runs in cibuildwheel

# C++ tests — need a separate CMake build tree
cmake -S . -B build_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpp -j
ctest --test-dir build_cpp --output-on-failure

# Build wheels locally the way CI does
pipx run cibuildwheel --platform macos    # or linux / windows
```

Release: tag `vX.Y.Z` → `.github/workflows/wheels.yml` builds wheels on linux/macos/windows × cp310/311/312, drafts a GitHub Release, and publishes to PyPI via Trusted Publishing (gated by the `pypi` environment).

## Architecture

### Two-language layout

- **`src/cpp/`** — the only place token IDs, vocabularies, constraint graphs, and the step planner live. Built into a static `midigpt_core` lib and a `_core` pybind11 module. `CMakeLists.txt` lists every translation unit explicitly (no globs).
- **`src/python/midigpt/`** — thin Python wrappers around `_core`, plus the GPT-2 model and the sampling driver. `_core.so` is copied next to `__init__.py` post-build so in-tree `pytest` works without reinstall.

### Inference data flow

```
MIDI file ──► _core encoder ──► token IDs
                                   │
                                   ▼
                      InferenceEngine.session(score, request)
                                   │
                                   ▼
              SamplingSession  (src/python/midigpt/inference/session.py)
              loops: step_planner (C++) → model.forward → constraint mask → sample
                                   │
                                   ▼
                            _core decoder ──► Score / MIDI
```

`InferenceEngine` accepts any callable with signature `(input_ids, past_kv) -> (logits, present_kv)` — `GPT2LMHeadModel` is the production one; `StubModel` in `test_inference.py` is the test double.

### Packed checkpoint (`format_version: 1`)

Single `.pt` file holds `{format_version, config, encoder_config, state_dict}`. Produced by `GPT2LMHeadModel.save_pretrained()`; loaded by `GPT2LMHeadModel.from_pretrained()`, which auto-falls-back to TorchScript if the dict layout is missing.

`tokenizer/checkpoint.py::load_checkpoint(path)` is the high-level loader and accepts **either**:
- a directory with `config.json` + `model.pt` (legacy), or
- a single packed `.pt` bundle (current).

When extending the loader, return a `CheckpointBundle` and let `InferenceEngine.from_checkpoint` pick `bundle.model` over `bundle.model_path`. Do not duplicate `GPT2LMHeadModel` — there is exactly one in `inference/model.py`.

### `_core` (C++ bindings)

`EncoderConfig.from_json(str)` is the entry point for everything that needs vocab sizes or token domains. The encoder config is the source of truth — `Tokenizer(cfg).vocab_size()` is authoritative; do not recompute vocab from `sum(token_domains[*].domain_size)` (token IDs include offsets and special tokens).

## Things that have bitten us

- **cibuildwheel test-command quoting on Windows**: cmd.exe doesn't strip single quotes. Use TOML literal-string outer with double quotes inside: `test-command = 'pytest ... -m "not slow and not inference"'`.
- **symusic v0.6.0** API uses nested `shared<vec<shared<Track<T>>>>` and `pyvec<Note<T>>`; the v0.4.x API is gone. The MSVC build needs `midigpt::TokenType` qualified inside pybind11 enum/template instantiations.
- **`model_dim` is a generation-window parameter**, not the score's bar count. Tests using small scores must pad bars to at least the model's window (see `test_inference.py` — 4-bar fixtures).
- **Stale `__version__`** in `src/python/midigpt/__init__.py` is not auto-synced with `pyproject.toml`; bump both on release.

## Memory system

Persistent notes live in `/Users/paultriana/.claude/projects/-Users-paultriana-creative-labs-MIDI-GPT/memory/`. `MEMORY.md` is the index — current entries cover the OSC-only setup, `model_dim` semantics, and which test markers to skip when token IDs diverge across encoder versions.
