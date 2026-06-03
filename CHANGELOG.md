# Changelog

All notable changes to `midigpt` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The section header for a tagged release (e.g. `## [0.2.4]`) is extracted by
`.github/workflows/wheels.yml` and used as the GitHub Release body. Add a new
section above `[Unreleased]` before tagging.

## [Unreleased]

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
