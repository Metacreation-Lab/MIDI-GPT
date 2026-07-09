# Changelog

All notable changes to `midigpt` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The section header for a tagged release (e.g. `## [0.2.4]`) is extracted by
`.github/workflows/wheels.yml` and used as the GitHub Release body. Add a new
section above `[Unreleased]` before tagging.

## [Unreleased]

## [0.3.2] - 2026-07-09

### Added
- **Prism Documentation:** Full `docs/models.md` section for the `prism_medium` checkpoint — same wide attribute/genre set as `expressive`, standard 32-level velocity, `num_bars_map` up to 16 bars.
- **Real-Checkpoint Smoke Tests:** New `test_pretrained_checkpoints.py` downloads and exercises `prism_medium`/`expressive_medium` end-to-end (AR + infill generation, attribute/genre conditioning, `MaskBar`-mode rejection).

### Changed
- **Dynamic Checkpoint Resolution:** `InferenceEngine.from_pretrained` now resolves checkpoint filenames dynamically against the HuggingFace repo's file listing (preferring `-final`, falling back to the highest `-step<N>`) instead of a hardcoded filename map — re-uploading a checkpoint no longer requires a code change.

### Fixed
- **Attribute Token Names:** Corrected `TOKEN_TYPE_TO_ATTRIBUTE` and individual attribute classes' `token_type` strings to match the C++ `TokenType::name()` runtime values (`TrackLevelPitchRangeMax`, `TrackLevelSilenceProportionMax`, `BarLevelPitchClassSet`), fixing silent attribute auto-inference and BPE classification failures.

### Removed
- **Ghost Model References:** `"ghost"` removed from `InferenceEngine.from_pretrained`'s known model names (no checkpoint has ever been trained for it); docs now explicitly mark it as an unreleased, planned architecture instead of implying it's usable today.

## [0.3.1] - 2026-07-06

### Added
- **Token Diagnostics:** Opt-in `TokenLogger` for `SamplingSession` — per-token JSONL logs capturing token id/type/value, the grammar-masked T=1 distribution (log-prob, entropy, top-5 alternatives), grammar-collapse events, per-step summaries (context size, token budget, velocity mode) and per-bar summaries of the decoded result. No behavior change when no logger is attached. Also warns when the context window is nearly full and late tracks would be silently truncated.
- **Replay Scoring:** `SamplingSession.score_from_tokens()` — teacher-forced log-probability of a previously generated token sequence under the session's context, applying the same grammar legality masks used at sampling time. Enables cross-conditional scoring (score a generation made under conditioning A against conditioning B) without re-sampling. Sequences longer than the model's positional window are scored with a sliding window; generation paths are untouched.

## [0.3.0] - 2026-07-06

### Added
- **Expressive Encoder & Controls:** Support for sub-grid microtiming (`Delta`) tokens, 128 velocity levels (vs 32), and piece-level switchable microtiming and velocity controls.
- **NOMML Attribute:** Added `TrackLevelNomml` (median metric depth) attribute control end-to-end to steer rhythmic quantization.
- **Genre Grouping:** Piece-level genre conditioning via `GenreGrouping` for model training and inference.
- **HTTP Server Timeout:** New `--idle-timeout` flag to support automatic shutdown on inactivity.
- **Developer Documentation:** Added comprehensive implementation and design guides for both [Attribute Controls](docs/attribute_controls_guide.md) and [Encoder Configs](docs/encoder_guide.md).
- **Preprocessing Acceleration:** Added a `--workers` flag to parallelize preprocessing across multiple parquet shards.
- **Training Tests:** Integrated training pipeline end-to-end tests using MIDI-fixture-based parquet data.

### Changed
- **Checkpoint Format:** Switched to `.safetensors` as the default serialization format (`format_version: 2`), embedding weights and configuration metadata in the SafeTensors header.
- **Tension Attribute:** Refactored Tension extraction into a separate module structure, and updated tension quantization to decile bins.

### Fixed
- **Grammar Constraints:** Allowed `NoteOnset` immediately following `FillInStart` (at onset = 0).
- **Quantization Agnosticism:** Refactored `NoteDurationQuantile` to be TPQ- and time-signature-agnostic.
- **DDP & Lightning Training:** Corrected training configs (`num_epochs` -> `max_steps`), fixed DDP learning rate scaling, and removed unnecessary deepcopy operations.
- **Worker Safety:** Made dataloader worker spawning fork-safe for C++ pybind11 tokenizer.
- **Genre Validation:** Improved error validation to list canonical genres upon validation failures.

## [0.2.4] - 2026-06-03

### Added
- Stateless HTTP server (`midigpt[http]` install extra) with end-to-end tests.
- `CITATION.cff` and root `CONTRIBUTING.md`.
- MkDocs documentation site, dark slate theme, brand styling, docs badge in README.
- Ruff + pre-commit linting setup.
- Curated `CHANGELOG.md` — release workflow now uses per-version sections as the GitHub Release body.
- Auto-publish GitHub Release on tag push (previously created a draft only).

### Changed
- Checkpoint format migrated to safetensors (`format_version: 2`). The `.pt` packed-bundle loader still reads `format_version: 1`.

### Fixed
- Session device mismatch on GPU inference.
- PyPI publish environment name.

## [0.2.3] - 2026-06-01

### Fixed
- Inference session bugs found in end-to-end testing.
- MSVC build: `TokenType` alias collision (C2365) replaced with non-conflicting `TT`.
- MSVC build: structured-binding failure over `std::pair<TokenType, int>`.
- GCC 14 / pybind11 build failure on `manylinux_2_28`.

### Changed
- CI: upgraded `cibuildwheel` v2.21.3 → v3.4.1 to fix stale manylinux image.

## [0.2.2] - 2026-05-31

### Added
- HuggingFace Hub model loading via `InferenceEngine.from_pretrained`.
- Correct citation block in README.

### Fixed
- Ghost microtiming: `use_microtiming=false` in config and README.
- HuggingFace repo ID: `Metacreation-Lab` → `Metacreation`.

### Changed
- Studio exclusion, soundfont packaging, tension API tidy-up, Windows wheel fix.

## [0.2.1] - 2026-05-30

### Added
- `load_checkpoint` accepts a single-file `.pt` bundle (`format_version: 1`).
- Auto-publish to PyPI on tag via Trusted Publishing.

### Changed
- Org rename: README, URLs, and version metadata aligned with `Metacreation-Lab`.
- Tension API removed from public surface.

## [0.2.0] - 2026-05-29

### Added
- First release of `midigpt` as a shipping Python package (refactor from research codebase).
- Tag-triggered wheel build workflow (`linux`/`macos`/`windows` × `cp310`/`cp311`/`cp312`).
- MIT LICENSE.
- `tqdm` in inference extras; training-test guards.

### Fixed
- Wheel build: install `_core` extension; bump macOS deployment target to 10.15.
