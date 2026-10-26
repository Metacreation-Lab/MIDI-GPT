# Changelog

All notable changes to `midigpt` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The section header for a tagged release (e.g. `## [0.2.4]`) is extracted by
`.github/workflows/wheels.yml` and used as the GitHub Release body. Add a new
section above `[Unreleased]` before tagging.

## [Unreleased]

## [0.4.2] - 2026-10-26

### Fixed
- **Remix in pitch mode kept the onsets but not the durations** (since 0.4.1): every note's duration was sampled freely. Pitch mode now keeps the reference durations again, and the request matrix checks them. ([#32](https://github.com/Metacreation-Lab/MIDI-GPT/pull/32))

## [0.4.1] - 2026-10-26

### Changed
- **Faster generation:** each attention layer's KV cache grows in place instead of being copied with `torch.cat` on every token, so decoding keeps its speed as the window fills. `expressive_medium` on CPU, 1000 → 3000 tokens: 9.9 → 35.5 tok/s for one row, 4.0 → 21.8 tok/s for three (takes, batched attempts). ([#29](https://github.com/Metacreation-Lab/MIDI-GPT/issues/29), [#30](https://github.com/Metacreation-Lab/MIDI-GPT/pull/30))
- **Streaming** decodes the step at most every 250 ms instead of after every note, and GPT-2's GELU is one fused op: a streamed 32-bar request runs 23.1 → 30.4 tok/s. The fused GELU rounds slightly differently, so a fixed seed can give different notes than 0.4.0. ([#30](https://github.com/Metacreation-Lab/MIDI-GPT/pull/30))
- **`mask_mode` defaults to `"auto"`:** `"token"` when the checkpoint has `MaskBar`, else `"attention"`. The old default, `"token"`, is supported by no released checkpoint, so a request that didn't set `mask_mode` was refused. ([#28](https://github.com/Metacreation-Lab/MIDI-GPT/pull/28))
- **Remix amount** is the share of the reference notes resampled: `round(amount × notes)`, at least one. An amount of 0 is refused, like a request with no bars. ([#28](https://github.com/Metacreation-Lab/MIDI-GPT/pull/28))
- **Polyphony and density limits win over remix and exact rhythm:** under a limit the forcers play fewer notes than the reference, never other ones. ([#28](https://github.com/Metacreation-Lab/MIDI-GPT/pull/28))

### Added
- **Request matrix** (`scripts/request_matrix/`): runs every pairwise combination of the generation request options, plus the risky combinations in full, on the released checkpoints, checks each reply and shrinks failures to a replayable request. The fixes below were found with it: 312 of 689 requests succeeded on 0.4.0; on 0.4.1 every request succeeds or is correctly refused, apart from novelty-check rejections. ([#31](https://github.com/Metacreation-Lab/MIDI-GPT/pull/31))

### Fixed
All in [#28](https://github.com/Metacreation-Lab/MIDI-GPT/pull/28) unless noted.
- **Remix and exact rhythm over several infill bars:** the forcers only moved to the next bar on `BarEnd`, which infill bars don't have, so the second bar started with the first bar's used-up schedule (no legal token, or runs to the context limit).
- **A late first onset:** a reference bar starting after the downbeat had its first notes moved to beat 1, in infill and AR bars.
- **Infill bars closed early** while reference notes remained, dropping the rest of the schedule.
- **Microtiming loops:** where a forcer blocked notes it left `Delta` open, and the model could write thousands of them. The grammar now only lets a `Delta` lead to its note.
- **Remix ignored the seed:** the same seed gave a different remix. Each attempt (and each batched row) now uses its own seed.
- **Hard limits left no legal token:** at the cap, the polyphony and density limits blocked notes but not velocity or microtiming, and their counts carried from one infill bar into the next.
- **Meters with half beats** (5/8, 7/8, 9/8, 13/16) were cut to whole beats by the grammar and the decoder.
- **Generated bars changed meter:** a generated bar could write another time signature than the piece's (6/4 in a 6/8 piece). It now keeps the piece's meter.
- **`mask_mode: "remove"`** dropped a masked bar entirely, shifting the rest of the track one bar earlier; it is now an empty bar.
- **`mask_k` / `mask_p`** could remove `BarEnd`, `FillInEnd`, `TrackEnd` or `TimeSig`; they now only remove note content.
- **A cancelled batch** (`batch_attempts`) crashed with `'list' object has no attribute 'notes'`.
- **An AR track without bars** is the whole track; before, validation accepted it but nothing was generated.
- **The error when every attempt failed** hid the real error behind `'NoneType' object has no attribute 'tracks'`.
- **Streaming:** drum-only steps and requests with `batch_attempts` streamed no notes. ([#30](https://github.com/Metacreation-Lab/MIDI-GPT/pull/30))

## [0.4.0] - 2026-10-19

### Changed
- **Breaking — single-model server id:** `midigpt-http --pretrained NAME` now serves the model as `NAME`, and `--ckpt PATH` as the checkpoint's file name, instead of `"default"`, so `/models` and `/info` say which model is loaded. Clients that send `"model": "default"` get a 400; omit `model` to use the server's default model. ([#9](https://github.com/Metacreation-Lab/MIDI-GPT/issues/9))
- **Step planner:** a new Python planner replaces the C++ grid planner. Windows are placed on the target bars instead of a grid from bar 0: an autoregressive step keeps up to `model_dim − bars_per_step` bars of the track's own history and slides only when generation reaches the window's edge, and infill bars are taken in chunks with a window centred on each. `model_dim`, `bars_per_step` and `tracks_per_step` also take `"auto"`. `config.plan_only` returns the plan without generating. ([#26](https://github.com/Metacreation-Lab/MIDI-GPT/pull/26))
- **Ignored tracks are left out of the prompt** (`TrackPrompt.ignore=True`): they no longer cost a track header and a masked bar per window, and come back unchanged in every result. ([#22](https://github.com/Metacreation-Lab/MIDI-GPT/pull/22))
- **Python 3.13 and 3.14 wheels**, so `pip` no longer falls back to a C++ source build on them. On Intel Macs, `midigpt[inference]` still needs Python ≤ 3.12 (no newer PyTorch wheels). ([#13](https://github.com/Metacreation-Lab/MIDI-GPT/issues/13))
- **Model weights license:** the pretrained weights move to the Metacreation Non-Commercial Research License, with gated access on Hugging Face; the code stays MIT. The README and docs say so and list the final checkpoints.
- **Freely licensed example and test music:** the examples page and `tests/midi/` no longer use commercial songs, MAESTRO (CC BY-NC-SA) or POP909. They use CC0 pieces by Ivan Stanton (OpenGameArt), public-domain Mozart, Bach and Beethoven scores (Mutopia Project) and Groove MIDI Dataset drums (CC BY 4.0); `tests/midi/SOURCES.md` lists each file's source and license. `scripts/gen_doc_examples.py` regenerates the examples page with fixed seeds.
- **Documentation:** the README, API reference, HTTP server, concepts, quickstart, training, OSC and models pages now match the code; their examples run against the released checkpoints (`mask_mode="attention"`, which every released checkpoint needs).
- **Version:** `midigpt.__version__`, the HTTP API docs, `CITATION.cff` and `CMakeLists.txt` all report the package version (they said 0.3.2, 0.2.3 and 0.1.0).

### Added
- **Context fitting:** when a step would not fit the model, whole context tracks are left out (`config.context` = `"auto"`, `"listed"` or `"all"`, with `context_drop` / `context_keep` / `budget_margin`), and auto steps are split when that isn't enough. Responses list them as `dropped_context`. ([#26](https://github.com/Metacreation-Lab/MIDI-GPT/pull/26))
- **`config.infill_probability`:** infill a random, seeded subset of the target bars; the rest stay as context. `SamplingSession.generated_bars` and the `generated_bars` response field say which bars were generated. ([#25](https://github.com/Metacreation-Lab/MIDI-GPT/pull/25))
- **`config.novelty_threshold`:** with `novelty_check`, reject a step whose generated bars share at least this share of their (pitch, onset) notes with the original, not only an identical one. Default 1.0 keeps the old behaviour. ([#16](https://github.com/Metacreation-Lab/MIDI-GPT/pull/16))
- **`config.shuffle`** now works: infill steps run in random order, seeded by the request seed. ([#18](https://github.com/Metacreation-Lab/MIDI-GPT/pull/18))
- **Streamed progress:** every streamed frame carries `progress` (`generated_tokens`, `step_generated_tokens`, `context_tokens`, `max_context_tokens`, `model_forward_s`, `attempt`), a `progress` frame comes every 32 tokens, and responses report `attempts` and `tokens.peak_context_tokens`. ([#14](https://github.com/Metacreation-Lab/MIDI-GPT/pull/14))
- **`GET /info`** reports `model_dim_options`, `max_context_tokens`, `switchable_velocity`, `switchable_microtiming` and `genres` in `capabilities`, and `attribute_track_types` (which tracks each attribute applies to). ([#19](https://github.com/Metacreation-Lab/MIDI-GPT/pull/19), [#10](https://github.com/Metacreation-Lab/MIDI-GPT/issues/10))
- **`--max-generation-seconds`** (or `max_generation_seconds` in the config file): cancel any generation that has run longer than this. Off by default.
- **Hugging Face tokens:** `InferenceEngine.from_pretrained(..., token=...)` and `hf_token` in the HTTP server's config file, for the gated model repo. The default is still `HF_TOKEN` or `hf auth login`. A missing license acceptance or token now raises a `PermissionError` that says what to do, and `midigpt-http` stops with that message.
- **Offline loading:** when the Hub can't be reached or refuses the download, `from_pretrained` uses the cached checkpoint.

### Fixed
- **Expressive velocities:** 128-level velocities decoded to 0 (a note-off), so most generated notes from `expressive_medium` were silent. ([#20](https://github.com/Metacreation-Lab/MIDI-GPT/pull/20))
- **Prism / expressive autoregressive drift:** the grammar masked the attribute tokens these models write in track and bar headers, so generated bars lost their headers and the part drifted (8-bar AR note F1 against the original was 0.17 on prism, against 0.46 on yellow). ([#21](https://github.com/Metacreation-Lab/MIDI-GPT/pull/21))
- **Unsorted notes:** a bar whose notes weren't listed in time order could move its downbeat note to the bar's last onset, changing context bars. ([#11](https://github.com/Metacreation-Lab/MIDI-GPT/pull/11))
- **Drum rhythm masks and remix:** drum tracks emit no `NoteDuration`, so the rhythm and remix forcers never counted a drum note and piled hundreds of notes on one onset. ([#12](https://github.com/Metacreation-Lab/MIDI-GPT/pull/12))
- **Long autoregressive generations:** the token budget was counted over the whole session instead of per attempt, so later steps came back empty and the request failed; a step window that decodes short keeps its bar positions. ([#15](https://github.com/Metacreation-Lab/MIDI-GPT/pull/15))
- **AR prompts after the first window** ended inside the last context bar, so the model kept editing it; several AR tracks in one step only generated the first. ([#17](https://github.com/Metacreation-Lab/MIDI-GPT/pull/17))
- **Step planning:** `tracks_per_step` grouped raw track indices instead of the selected tracks, and a step-down to a smaller `model_dim` regenerated every finished step instead of only the rest. ([#18](https://github.com/Metacreation-Lab/MIDI-GPT/pull/18))
- **`run_variations`** (`num_candidates > 1`) now steps `model_dim` down on a context overflow instead of failing. ([#22](https://github.com/Metacreation-Lab/MIDI-GPT/pull/22))
- **HTTP queue slots:** a streaming client that disconnected never gave its queue slot back, so after `max_queue` dropped streams every request got 503 ([#23](https://github.com/Metacreation-Lab/MIDI-GPT/pull/23)); a non-streaming request whose client went away kept generating to the end, holding the model and its queue slot. Both are now cancelled.
- **Tempo:** generated scores came back at the default 120 BPM (500000 µs per beat) instead of the input's tempo.
- **`InferenceConfig(bars_per_step="auto")`** (and `model_dim` / `tracks_per_step`) raised a `TypeError`; `"auto"` only worked through `from_dict` (JSON, HTTP).
- **Model docs** link the current checkpoints and list context lengths. ([#24](https://github.com/Metacreation-Lab/MIDI-GPT/pull/24))

## [0.3.4] - 2026-08-19

### Added
- **Multi-model HTTP serving:** `midigpt-http` now serves multiple checkpoints from one process, selectable per-request, configurable via a config file, with request backpressure (503 retry passthrough) under load.
- **Pitch masks:** `TrackPrompt.controls["pitch_mask"]` restricts/shapes which pitches a track's `NoteOnset` tokens may sample — hard allow-set (exact pitches, scale+root preset, or pitch classes) plus an optional soft reweight (`shape: uniform` or `normal`).
- **Rhythm masks:** `TrackPrompt.controls["rhythm_mask"]` restricts/shapes which within-bar ticks a track's `TimeAbsolutePos` tokens may land on — hard exact rhythm via `positions` (onset ticks + polyphony), or soft grid-granularity bias via `grid` (`{"unit": "eighth", "strength": 0.8}`).
- **Remix schedules:** `TrackPrompt.controls["remix"]` regenerates a track's bars as a partial variation of content already on the score — onset schedule reproduced exactly, only note-level values (`mode: "pitch"` or `"full"`) are eligible for resampling.
- **Cancellable sessions:** `GenerationCancelled` is now raised (and exported) so an in-flight `SamplingSession` can be stopped and return its partial result instead of erroring out.
- **Batched multi-candidate decoding:** `_sample_step_batch` supports `num_candidates > 1`.

## [0.3.3] - 2026-07-28

### Fixed
- **Piece-level controls propagation:** `validate_request` was returning a new `GenerationRequest` without the `controls` field, silently dropping `microtiming`/`velocity` controls before they reached the sampling session. Controls are now forwarded correctly.
- **Hard vocab block (`microtiming=False`, `velocity=False`):** When piece-level controls disable microtiming or velocity, `Delta`/`DeltaDirection`/`VelocityLevel` token IDs are now force-zeroed in the grammar mask buffer at every sampling step, unconditionally excluding them from the distribution. Previously, the controls only added conditioning tokens with no effect on sampling.
- **`UnboundLocalError` in `_sample_step`:** Inline `import midigpt._core as _core` inside the function shadowed the module-level import, causing `UnboundLocalError` on Python 3.12. Removed the redundant local import.

### Changed
- **Simplified `from_pretrained` API:** `InferenceEngine.from_pretrained(name, hf_repo=...)` now takes the exact checkpoint filename prefix (e.g. `yellow_medium`, `prism_medium`, `expressive_medium`) and an optional `hf_repo` argument (default: `Metacreation/MIDI-GPT`). The `_MODEL_PREFIXES` alias table is removed — unknown names are passed directly to the HuggingFace repo without error.
- **`midigpt-http` CLI:** `--hf-filename` replaced by `--hf-repo` (default: `Metacreation/MIDI-GPT`). `--pretrained` now documents that it expects the exact checkpoint prefix, not a short alias. `--ckpt` help updated to reflect `.safetensors` as the primary format.

### Removed
- **Pretrained name aliases:** `"yellow"` and `"expressive"` short aliases removed. Use `yellow_medium` and `expressive_medium` directly.
- **Ghost model documentation:** Removed ghost from the models table, `docs/models.md` overview, compatibility matrices, and code examples throughout. Ghost was never released and the planned architecture is no longer tracked here.

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
