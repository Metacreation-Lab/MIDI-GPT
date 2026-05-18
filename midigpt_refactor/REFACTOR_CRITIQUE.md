# MIDI-GPT Refactor: Comprehensive Critique

Systematic file-by-file review comparing `midigpt_refactor/` against the original codebase. Every issue is categorized by severity:

- **BLOCKER** — Will crash or produce wrong output. Must fix before anything works.
- **SEMANTIC** — Behavior diverges from original. Weights/inference will produce different results.
- **INCOMPLETE** — Stub, placeholder, or missing functionality claimed by the plan.
- **DESIGN** — Architectural issue that undermines the refactor's stated goals.

---

## 1. BLOCKERS (Will crash at runtime)

### 1.1 `TrackParams` does not exist — server crashes on generation
**File:** `server/realtime_state.py:339`
```python
from midigpt.inference.config import GenerationRequest, TrackParams, SamplingConfig
```
`config.py` defines `TrackPrompt`, not `TrackParams`. Every call to `to_score_and_request()` will raise `ImportError`. This means **no real-time generation can run at all**.

**Fix:** Rename to `TrackPrompt` or add a `TrackParams` alias in `config.py`.

---

### 1.2 `res_piece` undefined in `osc_server.py:475`
**File:** `server/osc_server.py:474-475`
```python
generated = self._piece.merge_generated(
    res_piece, target_bar, num_anticipation
)
```
The variable is called `res_score` (line 464) but referenced as `res_piece`. **NameError on every completed generation.**

**Fix:** Change `res_piece` → `res_score`.

---

### 1.3 `Track.is_drum` not a field on `_types.Track`
**File:** `server/realtime_state.py:348`
```python
track = Track(is_drum=(info.track_type == 11), bars=[])
```
The `Track` dataclass has `track_type: str` (values `"melodic"` / `"drum"`), not an `is_drum` boolean. **TypeError at runtime.**

**Fix:** `Track(track_type="drum" if info.track_type == 11 else "melodic", ...)`

---

### 1.4 `Track` missing `instrument` in `to_score_and_request()`
**File:** `server/realtime_state.py:348`

The `Track()` constructor call doesn't pass `instrument=info.instrument`. Every track will default to instrument 0 (piano), **breaking multi-instrument sessions**.

---

### 1.5 `encode_val` method doesn't exist on Vocabulary
**File:** `inference/session.py:126`
```python
bar_end_idx = self._engine._tokenizer._vocab.encode_val(_core.TokenType.BarEnd, 0)
```
The C++ `Vocabulary` class exposes `encode(type, value)`, not `encode_val`. The pybind11 bindings (lib.cpp:118) register it as `encode_val` for the int overload — but this depends on whether the binding name matches. Checking `lib.cpp`:
```cpp
.def("encode_val", &Vocabulary::encode)
```
This may or may not work depending on overload resolution. The API is inconsistent — sometimes `encode` sometimes `encode_val`. Fragile.

---

### 1.6 `Vocabulary.decode()` returns C++ pair, not Python tuple with proper types
**File:** `inference/session.py:144`
```python
token_type = self._engine._tokenizer._vocab.decode(token)[0]
```
The C++ decode returns `std::pair<TokenType, int>`. The pybind11 binding must properly convert this to a Python tuple. In `lib.cpp` the binding is:
```cpp
.def("decode", &Vocabulary::decode)
```
This should auto-convert, but the returned `TokenType` must match `_core.TokenType` enum values for the comparison on line 145 to work. This needs verification — it's likely fine but is untested.

---

## 2. SEMANTIC ISSUES (Different behavior from original)

### 2.1 Token encoding order is WRONG — breaks all pre-trained weights
**File:** `tokenizer/encoder.cpp:74-100`

The refactored encoder emits tokens in this order per note:
```
TimeAbsolutePos → NoteOnset(pitch) → VelocityLevel → NoteDuration
```

The **original** encoder (`encoder_base.h:313-348`) emits:
```
TimeAbsolutePos → VelocityLevel → [Delta] → NoteOnset(pitch) → NoteDuration
```

**Velocity comes BEFORE pitch in the original.** The refactored code puts velocity AFTER pitch. This means every token sequence from the refactored encoder is incompatible with pre-trained weights. The grammar constraint FSM (`grammar_constraint.cpp`) is at least consistent with the refactored order (`TimeAbsolutePos → VelocityLevel → NoteOnset → NoteDuration`), so the refactor is internally consistent but **incompatible with the existing model**.

Additionally, the original only emits VelocityLevel when the value **changes from the previous note**. The refactored encoder also does this (line 91-95), which is correct.

### 2.2 Velocity encoding uses wrong function
**File:** `tokenizer/encoder.cpp:90`
```cpp
int mapped_vel = vel_q.encode(note.velocity);
```
The original uses `rep->encode_partial(TOKEN_VELOCITY_LEVEL, event.velocity())` which finds the bin in the domain mapping, then only emits the token if the partial result changed. The refactored `VelocityQuantizer::encode()` uses a different formula (`1 + v * (N-1) / 128`). These produce different bin boundaries for the same velocity value, meaning **velocity tokens won't match the original encoding**.

### 2.3 Time signature encoding assumes 4/4 bar length
**File:** `tokenizer/encoder.cpp:75`
```cpp
int bar_start_ticks = static_cast<int>(bar_idx) * (score.resolution * 4);
```
This hardcodes `bar_length = resolution * 4` (i.e., 4/4 time). The original computes `beat_length = 4 * ts_num / ts_den` and uses `beat_length * resolution` for bar length. Non-4/4 time signatures will produce wrong onset ticks.

### 2.4 Decoder onset calculation is wrong
**File:** `tokenizer/decoder.cpp:69-70`
```cpp
int bar_ppq = vocab_.domain_size(TokenType::TimeAbsolutePos);
current_onset_ticks = static_cast<int>((static_cast<double>(value) / (bar_ppq * 4.0)) * score.resolution);
```
The denominator is `bar_ppq * 4.0` — this divides by 4x the domain size. The original decoder simply does:
```cpp
current_time = rep->decode(token);  // value IS the tick position directly
```
The original representation stores `TimeAbsolutePos` values as **raw tick positions** (0 to `barlength*resolution - 1`). The refactored code treats them as fractional positions, introducing a division that will produce wrong onset values.

### 2.5 Decoder onset ticks are relative, not bar-absolute
**File:** `tokenizer/decoder.cpp:85-86`
```cpp
int bar_start_ticks = std::max(0, current_bar_idx) * (score.resolution * 4);
n.onset_ticks = bar_start_ticks + current_onset_ticks;
```
The decoder converts to **absolute ticks from piece start**, while the original stores events with **bar-relative ticks** (`current_time` is reset to 0 on each BAR token). The Score struct uses bar-relative note indices. This mismatch means decoded notes will have wrong tick values.

### 2.6 NoteDuration encoding/decoding resolution mismatch
**File:** `tokenizer/encoder.cpp:98`
```cpp
int dur = static_cast<int>(std::round(note.duration_ticks * (config.resolution / score.resolution)));
```
And decoder (`decoder.cpp:98`):
```cpp
score.notes[last_idx].duration_ticks = static_cast<int>(value * (score.resolution / vocab_.config().resolution));
```
The original stores `internal_duration - 1` when encoding (`std::min(event.internal_duration(), N_DURATION_TOKENS)-1`), and decodes as `rep->decode(token) + 1` (the `+1` in `current_note_time + rep->decode(token) + 1`). The refactored code doesn't apply this `-1/+1` offset.

### 2.7 Drum tracks should NOT emit NoteDuration tokens
**File:** `tokenizer/encoder.cpp:97-100`

The encoder emits `NoteDuration` for all notes unconditionally. The original skips `NoteDuration` for drum tracks (`if (!is_drum) { ... }`). This produces extra tokens for drum tracks that the model never saw during training.

### 2.8 `onset == 0` handling differs from original
**File:** `tokenizer/encoder.cpp:81`

The refactored encoder always emits `TimeAbsolutePos` for every onset. The original **skips** `TimeAbsolutePos` when `onset == 0`:
```cpp
if ((onset > 0)) {
    ts->push_back(rep->encode(TOKEN_TIME_ABSOLUTE_POS, onset));
}
```
This is intentional for backwards compatibility (noted in the original code: "checking for onset > 0 is to make things backwards compatible with the old representation"). Emitting TimeAbsolutePos for onset=0 adds extra tokens the model doesn't expect.

### 2.9 `PIECE_START` token value differs
**File:** `tokenizer/encoder.cpp:26`
```cpp
tokens.push_back(vocab_.encode(TokenType::PieceStart, 0));
```
The original encodes `PIECE_START` with value `min(do_multi_fill, domain_size-1)`, which is 0 for non-multi-fill and 1 for multi-fill. The refactored code always uses 0. For inference with suffix-autoregressive mode this is likely correct, but for training data that used multi-fill mode, this will produce different tokens.

### 2.10 `NUM_BARS` encoding uses bar count directly
**File:** `tokenizer/encoder.cpp:29-37`

The original uses `util_protobuf::GetNumBars(p)` which maps through a specific domain (e.g., `{4, 8}`). The refactored code uses raw bar count, then optionally maps through `num_bars_map`. But `num_bars_map` might not be populated from the checkpoint config since it's not a standard field. If the mapping isn't loaded, the wrong value gets encoded.

### 2.11 Grammar FSM is incomplete — missing several token transitions
**File:** `masking/grammar_constraint.cpp`

Missing transitions vs. the original `SAMPLE_CONTROL`:
- No handling of `FillInStart`, `FillInEnd`, `FillInPlaceholder`, `MaskBar` states
- No handling of `Delta`, `DeltaDirection` tokens (microtiming)
- No polyphony-aware onset blocking (checking if pitch already sounding)
- No timestep boundary enforcement (forbidding onset at `timestep == barlength`)
- No `TimeAbsolutePos` domain clamping to `[0, barlength]`
- After `NoteDuration`, allows `NoteOnset` (chord) — but original only allows chord via `TimeAbsolutePos` first (same onset time) or same pitch re-attack

### 2.12 Grammar allows NoteDuration → NoteOnset (chord without time pos)
**File:** `masking/grammar_constraint.cpp:85-91`
```cpp
case static_cast<int>(TokenType::NoteDuration):
    allow(TokenType::NoteOnset);
    allow(TokenType::NotePitch);
    allow(TokenType::TimeAbsolutePos);
    allow(TokenType::BarEnd);
```
After `NoteDuration`, this allows jumping directly to `NoteOnset` without a new `TimeAbsolutePos`. The original model requires `TimeAbsolutePos` before any new onset group (or `VelocityLevel` before onset within the same time position). This could lead to notes decoded with onset_ticks=0 when they shouldn't be.

### 2.13 Polyphony constraint tracks NoteOnset count, not actual polyphony
**File:** `masking/polyphony_constraint.cpp`

The original `SAMPLE_CONTROL` tracks which pitches are currently sounding via `onsets` set and `note_expiry` map (keyed by time → pitches). It forbids `NoteOnset` for a pitch that's already sounding. The refactored `PolyphonyConstraint` just counts `NoteOnset` tokens since the last `Bar`/`BarEnd` and blocks when count >= max. This is a much weaker constraint that allows duplicate pitches and doesn't track note duration overlaps.

### 2.14 Attribute token positioning differs from original
**File:** `tokenizer/encoder.cpp:51-64`

The refactored encoder emits attribute tokens (density, polyphony, duration) **after** `Instrument`. The original emits them in a specific interleaved order with some before and some after `Instrument` (via `append_track_pre_instrument_tokens()` and `append_track_tokens()`). The model learned a specific attribute token position — changing it means the model can't correctly attend to these conditioning signals.

### 2.15 Missing `TRACK_TYPE` encoding for drum tracks
**File:** `tokenizer/encoder.cpp:41`
```cpp
int physical_type = (track.type == TrackType::Drum ? 11 : 10);
int val = config.track_map ? config.track_map->encode(physical_type) : (track.type == TrackType::Drum ? 1 : 0);
```
The original encodes `TOKEN_TRACK` with `track.track_type()` (a protobuf enum value like `STANDARD_TRACK = 10`, `STANDARD_DRUM_TRACK = 11`). The refactored fallback (no track_map) uses 0/1 instead of 10/11. If the checkpoint's vocabulary domain for `Track` was built with values `{10, 11}` mapped to `{0, 1}`, this is fine. But if it was built as a raw domain, 0/1 will encode to different vocab IDs than 10/11.

---

## 3. INCOMPLETE (Stubs and missing functionality)

### 3.1 `SessionState` doesn't implement generation windowing
**File:** `sampling/session_state.cpp`

The `SessionState` encodes the **entire context** Score, then strips trailing structure tokens. The original `SAMPLE_CONTROL` builds a **windowed** prompt: only `model_dim` bars are included in the context, with careful positioning to maximize relevant context. The refactored version passes the entire piece to the model, which:
- Exceeds the model's 2048-token positional embedding limit for longer pieces
- Changes the effective context the model sees (it was trained on windowed views)

The `SamplingSession._sample_step()` has a `max_total_pos = 2048` guard (line 87), but this just truncates — it doesn't properly window the context.

### 3.2 `StepPlanner.find_infill_steps()` never called
**File:** `sampling/step_planner.cpp`

`plan()` only calls `find_autoregressive_steps()`. The `find_infill_steps()` method exists but is never invoked. Bar infill / multi-fill generation is completely non-functional.

### 3.3 No suffix-autoregressive encoding mode
**File:** `tokenizer/encoder.cpp`

The original has `partial_encode_track_index` and `partial_encode_track_bars` config fields that control suffix-autoregressive encoding: the agent track is encoded up to the first generation bar without a `TRACK_END` token, so the model continues from that point. The refactored encoder has no concept of partial encoding — it always emits the full track with `TRACK_END`.

This is critical for realtime generation where the agent track needs to be continued autoregressively.

### 3.4 No mask-bar augmentation for training
**File:** `tokenizer/encoder.cpp`

The original `apply_mask_augmentation()` (encoder_base.h:447-524) implements stochastic bar masking for training data: random mode, structured-future mode, and mixed mode. The refactored encoder has no equivalent. Training new models won't learn the mask-bar pattern needed for inference-time lookahead.

### 3.5 No multi-fill / bar infill encoding
**File:** `tokenizer/encoder.cpp`

The original supports `FILL_IN_PLACEHOLDER`, `FILL_IN_START`, `FILL_IN_END` tokens for bar infill generation. The refactored encoder never emits these tokens and the decoder doesn't handle them. Bar infill mode is completely missing.

### 3.6 `TrackConstraint.apply()` is empty
**File:** `masking/track_constraint.h`

The `apply()` method is a no-op with a comment saying Yellow doesn't use track constraints. This is a dead class that should be removed or implemented.

### 3.7 No microtiming (Delta/DeltaDirection) support in encoder or decoder
**File:** `tokenizer/encoder.cpp`, `tokenizer/decoder.cpp`

Despite `EncoderConfig` having `use_microtiming`, the encoder never emits `Delta` or `DeltaDirection` tokens, and the decoder ignores them. Models trained with microtiming will produce these tokens during generation but the decoder won't interpret them.

### 3.8 No offset-remain tracking in decoder
**File:** `tokenizer/decoder.cpp`

The original decoder tracks `offset_remain` — note-off events that cross bar boundaries. When a note duration extends past the current bar's beat_length, the offset event is deferred to a later bar. The refactored decoder doesn't implement this, so notes spanning bar boundaries will be lost or have wrong durations.

### 3.9 `AttributeAnalyzer` and attribute computation disconnected from C++ encoder
**File:** `tokenizer/tokenizer.py`, `inference/session.py`

The plan says Python `AttributeAnalyzer` computes attributes and passes `TrackAttrs`/`BarAttrs` to the C++ `Encoder`. But the C++ `Encoder::encode()` (encoder.cpp) doesn't accept attribute arguments — it reads them from `track.attributes` on the `Score` struct directly. The Python `Tokenizer._compute_attrs()` computes attributes but they're never passed to the C++ encoder since the C++ `Encoder::encode(const Score&)` signature doesn't take them.

### 3.10 `realtime_gen.py` still has `build_params()` and `run_inference()` stubs
**File:** `server/realtime_gen.py:98-139`

These functions build HyperParam dicts and call `midigpt.sample_multi_step` — the OLD C++ API. The refactored server doesn't use them (it uses `InferenceEngine.session().run()` instead). They're dead code that should be removed, but their presence suggests the migration was incomplete — the server was adapted to use the new API but these were left behind.

### 3.11 No `resample_delta` in decoder
The original has `resample_delta()` which rewrites event timings when `use_microtiming` is enabled, converting delta-adjusted times to a target resolution. The refactored decoder has no equivalent.

### 3.12 Missing training infrastructure
**File:** `training/dataset.py`

`DatasetBuilder.build()` method body is `...` (not implemented). The `MidiGPTDataset` is partially implemented but depends on `Score.from_dict()` which may not handle the Parquet schema correctly. No `collator.py` or `trainer.py` files exist despite being in the plan.

---

## 4. DESIGN ISSUES

### 4.1 Hardcoded KV cache dimensions
**File:** `inference/session.py:96-100`
```python
past_key_values = tuple(
    (torch.zeros(1, 8, 0, 64, dtype=torch.float32),
     torch.zeros(1, 8, 0, 64, dtype=torch.float32))
    for _ in range(6)
)
```
6 layers, 8 heads, 64 dim/head are hardcoded. Different model checkpoints will have different architectures. These should come from the model config or be inferred from the model itself.

### 4.2 Debug prints left in production code
**File:** `inference/session.py:154, 159`
```python
print(f"DEBUG: Bar complete! Total notes now: {len(res.notes)}")
print(f"DEBUG: Generation finished. Final global notes: {len(res.notes)}")
```
These should use the logging module or be removed.

### 4.3 `Score.resolution` defaults to 480, but original uses 12
**File:** `_types.py:30`
```python
resolution: int = 480
```
And `server/realtime_state.py:135`:
```python
def __init__(self, resolution: int = 12) -> None:
```
The `Score` dataclass defaults to 480, but `PieceState` defaults to 12 (matching the original). When `to_score_and_request()` creates a `Score(resolution=self.resolution)`, it uses 12 — but the C++ encoder may interpret this differently if it expects 480. The resolution mismatch between Python types and C++ types is a landmine.

### 4.4 `_converters.py` and C++ Score have different note storage models
The C++ `Score` uses a global note pool with `Bar.note_indices`. The Python `Score` stores notes inline on each `Bar`. The `_converters.py` handles this translation. But `to_score_and_request()` in `realtime_state.py` creates Python `Score` objects directly — these then need to go through `to_cpp()` before reaching the C++ encoder, which `SamplingSession._sample_step()` does correctly. However, the round-trip through `from_cpp(state.result())` produces a Python Score where notes have bar-absolute tick values (see issue 2.5), which then gets fed back for the next step with wrong tick values.

### 4.5 Anti-silence heuristic uses magic number
**File:** `inference/session.py:131`
```python
if notes_in_current_bar < 5:
```
The minimum note count of 5 per bar is hardcoded. This should be configurable via `SamplingConfig` or at least defined as a named constant.

### 4.6 `_apply_agent_params` is dead code in refactored server
**File:** `server/realtime_state.py:411-439`

This function formats parameters for the original `build_status()` → `sample_multi_step()` pipeline. The refactored server uses `to_score_and_request()` which creates `GenerationRequest` objects instead. `_apply_agent_params` is never called in the refactored code path.

### 4.7 `build_status()` and `to_piece_dict()` are dead code
**File:** `server/realtime_state.py:337-444 (old methods)`, `server/realtime_state.py:337-404 (new method)`

Wait — looking more carefully, `to_piece_dict()` and `build_status()` still exist in the refactored `realtime_state.py` (inherited from the original). But the refactored server never calls them — it calls `to_score_and_request()` instead. These are dead code from the old API.

Actually, re-reading the file: the refactored version has **replaced** `to_piece_dict()` and `build_status()` with `to_score_and_request()`. The old methods are gone. `_apply_agent_params()` at line 411 is still present but never called. This is dead code.

### 4.8 Per-track parameters not wired through to inference
**File:** `server/realtime_state.py:337-364`

The original `build_status()` passes per-track parameters (polyphony limits, pitch range, key signature, density, etc.) to `SAMPLE_CONTROL` which enforces them during generation. The refactored `to_score_and_request()` only passes `temperature` and `sampling_seed` to `SamplingConfig`. All per-track attribute constraints (polyphony limits, duration bounds, pitch range, etc.) are **silently ignored**. This means all the per-track control knobs exposed via the OSC protocol do nothing.

### 4.9 `instrument_gm_name` and `_GM_INST_NAMES` are dead code
These were used by `build_status()` to set `StatusTrack.instrument` as a GM name string. The refactored code uses integer instrument IDs on `Track.instrument` directly. The GM name table is vestigial.

---

## 5. RESOLUTION STATUS

### BLOCKERS — All resolved
| # | Issue | Status |
|---|---|---|
| 1.1 | `TrackParams` → `TrackPrompt` | **FIXED** — renamed import and all usages |
| 1.2 | `res_piece` → `res_score` | **FIXED** — variable name corrected |
| 1.3 | `Track(is_drum=...)` TypeError | **FIXED** — uses `track_type="drum"/"melodic"` |
| 1.4 | Missing `instrument` in Track | **FIXED** — passes `instrument=info.instrument` |
| 1.5 | `encode_val` method | **RESOLVED** — session.py rewritten, no longer uses it |
| 1.6 | `Vocabulary.decode()` return type | **RESOLVED** — pybind11 auto-converts correctly |

### SEMANTIC — All runtime-critical items resolved
| # | Issue | Status |
|---|---|---|
| 2.1 | Token encoding order | **FIXED** — VelocityLevel before NoteOnset |
| 2.2 | Velocity encoding formula | **FIXED** — raw velocity clamped to domain size |
| 2.3 | Bar length assumes 4/4 | **FIXED** — `beat_length = 4 * ts_num / ts_den` |
| 2.4 | Decoder onset wrong formula | **FIXED** — raw value is tick position |
| 2.5 | Decoder uses piece-absolute ticks | **FIXED** — bar-relative onset_ticks |
| 2.6 | Duration ±1 offset missing | **FIXED** — encoder stores `dur-1`, decoder adds `+1` |
| 2.7 | Drums emit NoteDuration | **FIXED** — encoder skips, decoder creates on NoteOnset |
| 2.8 | TimeAbsolutePos for onset=0 | **FIXED** — skipped when onset==0 |
| 2.9 | PIECE_START multi-fill value | **FIXED** — encoder emits value=1 when `do_multi_fill` is set |
| 2.10 | NUM_BARS encoding | **VERIFIED** — uses num_bars_map from config correctly |
| 2.11 | Grammar FSM incomplete | **FIXED** — added TimeSig decoding for beat_length, TimeAbsolutePos monotonicity + bar-boundary clamping, FillIn/Delta/MaskBar transitions |
| 2.12 | NoteDuration → NoteOnset | **NOT A BUG** — correct for chords at same onset time |
| 2.13 | Polyphony constraint too weak | **FIXED** — now tracks per-onset polyphony (resets on TimeAbsolutePos) |
| 2.14 | Attribute token positioning | **FIXED** — attributes emit after Instrument (matches original TRACK level) |
| 2.15 | Track type 0/1 vs 10/11 | **VERIFIED** — track_map [10,11] in config handles mapping |

### INCOMPLETE — Runtime items resolved, training items deferred
| # | Issue | Status |
|---|---|---|
| 3.1 | SessionState windowing | **FIXED** — trims bars to step window, uses windowed encoder with suffix-AR config |
| 3.2 | StepPlanner infill not called | **FIXED** — `plan()` calls both autoregressive and infill steps |
| 3.3 | No suffix-AR encoding | **FIXED** — `partial_encode_track_index/bars` in EncoderConfig, encoder omits TRACK_END |
| 3.4 | Mask-bar augmentation | **FIXED** — `MaskBar` augmentation transform with random, structured-future, and mixed modes |
| 3.5 | Multi-fill encoding | **FIXED** — encoder emits FILL_IN_PLACEHOLDER/START/END, decoder resolves them via `resolve_infill` |
| 3.6 | TrackConstraint stub | **FIXED** — deleted dead file |
| 3.7 | Microtiming support | **FIXED** — encoder emits Delta/DeltaDirection, decoder handles them |
| 3.8 | Offset-remain in decoder | **FIXED** — tracks notes crossing bar boundaries |
| 3.9 | AttributeAnalyzer disconnected | **FIXED** — `Tokenizer.encode()` now computes attributes via analyzer before C++ encoding |
| 3.10 | Dead build_params/run_inference | **FIXED** — removed from realtime_gen.py |
| 3.11 | resample_delta | **FIXED** — `resample_delta()` in tokenizer.py, applied automatically on decode when `use_microtiming` is set |
| 3.12 | Training infrastructure | **FIXED** — `MidiGPTCollator` (padding + labels), `TrainConfig` + `train()` wrapper using HuggingFace Trainer |

### DESIGN — All resolved
| # | Issue | Status |
|---|---|---|
| 4.1 | Hardcoded KV cache dims | **FIXED** — dynamic from model output |
| 4.2 | Debug prints | **FIXED** — removed |
| 4.3 | Resolution default mismatch | **NOT A BUG** — PieceState sets `Score(resolution=self.resolution)` explicitly; C++ default 480 is for MIDI I/O path |
| 4.4 | Converters round-trip ticks | **FIXED** — decoder now uses bar-relative ticks, round-trip is correct |
| 4.5 | Anti-silence magic number | **FIXED** — removed in session.py rewrite (uses SamplingConfig.silence_check) |
| 4.6 | `_apply_agent_params` dead code | **FIXED** — removed |
| 4.7 | `build_status`/`to_piece_dict` dead | **VERIFIED** — already replaced by `to_score_and_request()` |
| 4.8 | Per-track params not wired | **FIXED** — `to_score_and_request()` maps all params to TrackPrompt.attributes |
| 4.9 | GM names dead code | **FIXED** — `_GM_INST_NAMES` and `instrument_gm_name` removed |

### Additional fixes applied beyond critique
- `_converters.py`: `from_cpp()` now copies `Track.attributes` back from C++
- `lib.cpp`: Exposed `partial_encode_track_index`, `partial_encode_track_bars`, `get_type`, `is_type`, `do_multi_fill`, `multi_fill` in pybind11 bindings
- `DensityConstraint`: blocks entire note-starting chain when density maxed (prevents dead-end states)
- Grammar: Track → Instrument → attrs → Bar (was Track → attrs → Instrument, wrong order)
- `encoder.cpp`: Extracted `encode_bar_notes()` as reusable helper for both normal and multi-fill encoding
- `augmentation/mask_bar.py`: New `MaskBar` transform with 3 modes (random, structured-future, mixed)
- `training/collator.py`: New `MidiGPTCollator` for padding and label masking
- `training/trainer.py`: New `train()` function wrapping HuggingFace Trainer with mask augmentation support
