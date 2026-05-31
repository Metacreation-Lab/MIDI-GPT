# Python test suite — implementation brief

This document is a **handoff for a new Claude Code session**. It explains
what midigpt is, how the test suite is organized, what fixtures already
exist, and what each test file must cover. The previous session built
`conftest.py` and laid out the directory structure; the test files
themselves still need to be written.

---

## 0. Project surface (read this first)

**midigpt** is a MIDI generation library built around a C++ core (exposed
through `midigpt._core` pybind bindings) and Python wrappers around it.

Inference pipeline at a glance:

```
Score (python dataclass)
  → Tokenizer.encode  (uses C++ Encoder)
  → list[int] tokens
  → ModelBase.forward (e.g. GPT2LMHeadModel)
  → sampled tokens via SamplingSession.step
  → Tokenizer.decode
  → Score
```

**Key files** (everything tested lives here):

```
src/python/midigpt/
  _types.py                  Score / Track / Bar / Note dataclasses
  _converters.py             Python ↔ C++ score conversion
  attributes/                Attribute analyzers (density, polyphony, …)
  augmentation/              Data augmentations
  tokenizer/
    tokenizer.py             Tokenizer (encode / decode / resample_delta)
    checkpoint.py            load_checkpoint (packed .pt or directory form)
  inference/
    base.py                  ModelBase Protocol
    config.py                InferenceConfig / TrackPrompt / GenerationRequest
    engine.py                InferenceEngine (.from_checkpoint, .warmup, .session)
    session.py               SamplingSession (the actual generation loop)
    validation.py            validate_request — input validator
    model/
      registry.py            @register / get_model_class / REGISTRY
      transformer_lm_base.py TransformerLMBase + packed I/O + resolve_device
      gpt2.py                GPT2LMHeadModel (HF-compat layout)
      torchscript_adapter.py Adapter wrapping torch.jit.ScriptModule as ModelBase
  osc/                       SKIP — not tested
```

**Key concepts the tests must respect:**

- `Score.resolution` is ticks per quarter note. `Bar.beat_length` is **beats
  per bar** (not ticks).
- A `Track` is `track_type="melodic"` or `"drum"`. Drum tracks are NOT
  transposed by augmentations and are excluded from melodic-only attributes.
- Generation request has:
  - `tp.bars` — bars to GENERATE
  - `tp.mask_bars` — bars to MASK (hidden but listed in token stream)
  - `tp.autoregressive` — AR mode vs infill
  - `tp.ignore` — track is invisible to the model
  - `tp.attributes` — track-level attribute pins (quantized bin indices)
  - `tp.controls` — non-attribute pins (e.g. `time_signature`)
  - `tp.bar_attributes[bar_idx][attr_name]` — per-bar attribute pins
  - `tp.bar_controls[bar_idx][control_name]` — per-bar control pins
- `InferenceConfig` has 5 `mask_mode`s: `"token"` (MaskBar token, requires
  `supports_mask_bar_token`), `"attention"` / `"attention_approx"` /
  `"attention_skip"` (span masks + attention masking variants), `"remove"`
  (future bars omitted from token stream).
- `polyphony_hard_limit` and `density_hard_limit` are global hard caps wired
  into the constraint graph (NOT the same as `tp.attributes["onset_polyphony"]`
  which is a token-pin via AttributeValueConstraint).

---

## 1. Hard rules for every test (no exceptions)

1. **Never modify source under `src/python/midigpt/`.** If a test reveals a
   bug, write the test asserting correct behavior with `pytest.xfail` and a
   `# BUG: <explanation>` comment. The previous session deleted the old
   tests partly because they were tweaked to mask source bugs.
2. **Strong assertions.** Never `assert isinstance(x, Score)` as the only
   check. Assert specific bar counts, pitch values, dict contents, byte
   equality after roundtrip, tensor shapes, etc.
3. **No silent skips.** No `try/except: pytest.skip(...)` to hide
   API-shape uncertainty. Either the test passes with strong assertions or
   it fails with a clear message. The only legitimate skip is a genuine
   optional-dep (the existing conftest handles that).
4. **No `MidiGPTServer.__new__` / `object.__new__` resurrection** — the old
   test_realtime.py did this; don't repeat.
5. **No snapshot literals.** Don't `assert vocab_size == 650` — that's a
   spec snapshot, not a behavior check. Assert the invariant
   (`vocab_size == sum(domain_sizes)`) instead.
6. **Test names describe the property:**
   `test_score_roundtrip_preserves_note_pitches`, not `test_roundtrip`.
7. **Strong determinism for stochastic code.** Use `torch.manual_seed(0)`
   and `InferenceConfig.seed=0` at the top of any sampling test.
8. **Use `FakeModel` (canned logits) for fine-grained sampling-filter
   tests; use `tiny_gpt2` (real GPT2 with random weights) for end-to-end
   "does the pipeline produce a valid Score" tests.** Both fixtures live in
   conftest.

---

## 2. Fixtures already in conftest.py (READ IT FIRST)

Located at `tests/python/conftest.py`. Provides:

| Fixture | Type | Purpose |
|---|---|---|
| `ghost_config_json` | str | Raw JSON of `models/ghost_config.json` |
| `ghost_config` | `_core.EncoderConfig` | Fresh config per test |
| `ghost_analyzer` | `AttributeAnalyzer` | from_config(ghost_config) |
| `ghost_tokenizer` | `Tokenizer` | Real tokenizer w/ real vocab |
| `simple_score` | `Score` | 1 melodic track, 4 bars, 4 notes/bar |
| `two_track_score` | `Score` | Melodic + drum, 4 bars each |
| `empty_bars_score` | `Score` | 1 track, 4 empty bars |
| `tiny_gpt2_config` | `GPT2Config` | vocab=real, n_layer=2, n_head=2, n_embd=16, n_pos=512 |
| `tiny_gpt2` | `GPT2LMHeadModel` | Synthetic small GPT2 (seed=0) |
| `fake_model_factory` | callable | `factory(logit_fn=...)` → `FakeModel` |
| `packed_bundle_path` | `Path` | tmp_path packed .pt of tiny_gpt2 |
| `sample_midi_path` | `Path` | tests/midi/Aicha.mid |

Module-level helpers (importable as `from conftest import ...`):
`make_bar`, `make_note`, `melodic_track`, `drum_track`.

`FakeModel` (defined in conftest) implements the full `ModelBase` Protocol
including `kv_length` / `kv_null_positions` / `max_context`. It records
calls in `model.calls` so you can assert what the session passed in
(input_ids shape, past_len, kwargs).

---

## 3. Directory layout to fill

```
tests/python/
  conftest.py                 ✅ done
  TEST_IMPLEMENTATION_PLAN.md ✅ this file
  test_types.py               (3.1)
  test_converters.py          (3.2)
  test_config.py              (3.3)
  test_tokenizer.py           (3.4)
  test_checkpoint.py          (3.5)
  test_validation.py          (3.6)
  test_attributes.py          (3.7)
  test_augmentation.py        (3.8)
  model/
    __init__.py               (empty)
    test_registry.py          (3.9)
    test_transformer_lm_base.py (3.10)
    test_gpt2.py              (3.11)
    test_torchscript_adapter.py (3.12)
    test_engine.py            (3.13)
  session/
    __init__.py               (empty)
    test_kv_runner.py         (3.14)
    test_ar.py                (3.15)
    test_infill.py            (3.16)
    test_modes.py             (3.17)
    test_constraints.py       (3.18)
```

---

## 3.1 test_types.py — `src/python/midigpt/_types.py`

Cover: dataclass defaults, default-factory independence, `to_dict` /
`from_dict` roundtrip preserving structure, `from_midi` / `to_midi`
roundtrip on a real MIDI file.

**Must-have cases:**
- `Note`/`Bar`/`Track`/`Score` constructors with default args.
- Two default `Track()`s have independent `bars` lists (no shared mutable
  state).
- `Score.from_dict(score.to_dict())` is structurally equal (same pitches,
  velocities, onsets, durations, deltas; same ts_num/ts_den/future; same
  resolution/tempo).
- `from_midi(sample_midi_path)` returns a non-empty Score, then
  `to_midi(tmp)` → `from_midi(tmp)` preserves track count and total note
  count.
- Edge cases: empty score (0 tracks), score with empty bars, multiple time
  signatures (4/4, 3/4, 6/8), low res (12) and high res (480).

---

## 3.2 test_converters.py — `src/python/midigpt/_converters.py`

Cover: `to_cpp(score)` → `from_cpp` roundtrip is identity for note-level
data. Make sure C++↔Python `track_type` enum mapping is correct
(melodic / drum).

- Single track, multiple tracks, drum track.
- Notes at bar boundaries, notes with `delta != 0`.
- Resolution 12 and 480.
- Verify pitch / velocity / onset_ticks / duration_ticks preserved
  exactly through one roundtrip.

---

## 3.3 test_config.py — `src/python/midigpt/inference/config.py`

Cover dataclass defaults and field independence:

- `InferenceConfig()` defaults: `temperature=1.0`, `seed=-1`,
  `mask_mode="token"`, `polyphony_hard_limit=0`, `density_hard_limit=0`,
  `top_p=1.0`, `top_k=0`, `mask_p=0.0`, `mask_k=0`, `bars_per_step=1`,
  `tracks_per_step=1`, `model_dim=4`, `shuffle=False`,
  `novelty_check=True`, `silence_check=True`, `max_attempts=3`,
  `temperature_escalation=1.0`.
- `TrackPrompt(id=0, bars=[0])` defaults: `autoregressive=False`,
  `ignore=False`, `mask_bars=[]`, `attributes={}`, `controls={}`,
  `bar_attributes={}`, `bar_controls={}`.
- Default factories produce independent dicts/lists (mutating one
  instance's `attributes` doesn't leak into another's).
- `GenerationRequest(tracks=[...])` with default `config` gets an
  `InferenceConfig`.

---

## 3.4 test_tokenizer.py — `src/python/midigpt/tokenizer/tokenizer.py`

- `vocab_size() > 0` and equals `sum(domain.size for domain in
  ghost_config.token_domains)` (compute this independently from the JSON).
- `encode(score)` returns `list[int]`, all values in `[0, vocab_size)`.
- `encode(score, compute_attributes=False)` does NOT mutate
  `track.attributes`. Compare `track.attributes` dict before/after.
- `encode → decode` roundtrip on `simple_score`: returned Score has the
  same number of tracks and same note pitches (subject to encoder
  quantization — assert pitch equality, but velocity may be quantized into
  ghost_config.velocity_levels bins).
- `resample_delta`:
  - same source/target res, all deltas zero → unchanged (no-op fast path).
  - 12 → 480: onset_ticks scaled by 40, duration scaled by 40.
  - 480 → 12: onset_ticks scaled by 1/40 (rounded down).
  - non-zero delta: applied and clamped at 0.

---

## 3.5 test_checkpoint.py — `src/python/midigpt/tokenizer/checkpoint.py`

- `load_checkpoint(packed_bundle_path)` returns a `CheckpointBundle` with
  `.model is not None`, `.model_path is None`, and `.encoder_config` is an
  `EncoderConfig` instance.
- Build a directory-form bundle in `tmp_path`: write
  `ghost_config_json` as `config.json` and a TorchScript-saved tiny model
  as `model.pt`. (If `torch.jit.script(tiny_gpt2)` fails on the dynamic
  features in GPT2LMHeadModel, fall back to `torch.jit.trace(tiny_gpt2,
  example_inputs)` or build a minimal `nn.Module` whose `forward(ids,
  past_kv)` shape matches and script that.) Then call `load_checkpoint(dir)`
  and verify `.model_path` is the model.pt path and `.encoder_config`
  parses.
- Error cases:
  - missing config.json → FileNotFoundError
  - missing model.pt → FileNotFoundError
  - `.pt` file without `format_version` → ValueError with message
    mentioning "packed bundle"
  - `.pt` bundle with `encoder_config=None` → ValueError mentioning
    "encoder_config"
  - random path that's neither a file nor a dir → ValueError

---

## 3.6 test_validation.py — `src/python/midigpt/inference/validation.py`

Read the source carefully — it raises `RequestValidationError`. Cover:

- A valid `GenerationRequest` (1 track, AR over bars [0,1]) passes through
  unchanged.
- Out-of-range `tp.id` (≥ `len(score.tracks)`) raises.
- Negative bar index in `tp.bars` raises.
- Overlap between `tp.bars` and `tp.mask_bars` raises (disjoint required).
- `tp.attributes` key not in `analyzer.attribute_sizes()` raises with
  "unknown attribute".
- `tp.attributes["note_density"] = analyzer_size + 1` raises (out of
  range).
- `tp.controls["time_signature"]` larger than
  `len(encoder_config.time_signatures)` raises.
- Track-level attribute placed in `tp.bar_attributes` raises ("bar-level
  required").
- `mask_mode="token"` with a config where `supports_mask_bar_token=False`
  raises. Build the second config by reading `ghost_config_json` →
  `json.loads` → set the flag false → `EncoderConfig.from_json`.
- `mask_mode="token"` with `supports_mask_bar_token=True` passes even when
  no bars are masked (validation gates on mask_mode, not on whether masking
  is in use — this is by design).

---

## 3.7 test_attributes.py — `src/python/midigpt/attributes/*`

For each attribute class (`NoteDensity`, `OnsetPolyphony`,
`NoteDensityQuantile`, `PolyphonyQuantile`, `NoteDurationDist`,
`PitchRange`, `PitchClassSet`, `KeySignature`, `SilenceProportion`,
`Tension`, `TensionDrum`):

- `.size > 0`, `.name` is a non-empty string, `.token_type` is a non-empty
  string, `.level` is "track" or "bar".
- `compute(score, track_idx)` (or with `bar_idx` for bar-level) returns a
  numeric value on a hand-crafted score with predictable structure.
  - Density: 4 notes in 4-beat bar → raw = 1.0 notes/beat.
  - Polyphony: chord of 3 simultaneous onsets → raw = 3.
  - PitchRange: notes [60..72] → raw ≈ 12.
  - SilenceProportion: empty bar → raw = 1.0.
- `quantize(raw)` maps known raw values to known bins, clamps at extremes.
- `achievable_range(score, track_idx, generated_bars)` returns full domain
  for non-monotone attributes; tighter ranges for min_*/max_* (the source
  has the formula — read first).

For `AttributeAnalyzer`:
- `from_config(ghost_config)` returns an analyzer with N attributes where
  N matches the count of `token_domains` mapped via
  `TOKEN_TYPE_TO_ATTRIBUTE`.
- `compute_track_tokens` excludes bar-level attrs and filters by
  `track_type` (melodic-only excluded on drum track, drum-only excluded on
  melodic).
- `compute_bar_tokens` only includes bar-level attrs.
- `evaluate(requested, realized, idx)` returns 1.0 for matching bins, 0.0
  for non-matching.
- `token_domain_specs()` returns list of `(token_type, size)` for every
  attr with size > 0.

---

## 3.8 test_augmentation.py — `src/python/midigpt/augmentation/*`

For each augmentation, read the source first (APIs vary in
constructor signature and whether they take seed vs rng):

- `Transpose(semitones=N)`: every melodic note pitch shifted by N
  (clamped 0..127); drum notes unchanged; bar/track count unchanged.
- `Velocity`: every velocity in [1, 127], structure unchanged.
- `BarWindow`: output bar count ≤ max_bars; bars are contiguous.
- `ScoreWindow`: all tracks sliced to the same bar range.
- `MaskBar`: some bars get `.future=True` (or per-source semantics);
  determinism with fixed seed; statistical check: probability ≈ requested
  over many runs (use loose tolerance).
- `InstrumentSwap`: instrument numbers change per the policy in source;
  drum tracks not swapped.
- `TrackPermutation`: track order changes; track contents byte-identical
  post-permute; deterministic with seed.
- `AugmentationPipeline`: empty pipeline = identity; composing two
  transforms applies them in order.

---

## 3.9 model/test_registry.py — `inference/model/registry.py`

- `@register("foo")` decorator adds the class to `REGISTRY["foo"]` and
  returns the class.
- `get_model_class("foo")` returns the registered class.
- `get_model_class("nonexistent_arch")` raises KeyError.
- `register("foo")` twice with different classes overwrites (or raises —
  check the source first and assert observed behavior).

---

## 3.10 model/test_transformer_lm_base.py — `transformer_lm_base.py`

- `resolve_device("cpu")` returns `torch.device("cpu")`.
- `resolve_device("cuda")` raises `RuntimeError` when CUDA unavailable
  (skip the test only when CUDA IS available — flip the polarity).
- `resolve_device(None)` chooses cpu when no accelerator.
- Roundtrip: `tiny_gpt2.save_pretrained(tmp_path / "x.pt", encoder_config=
  {...})` → `GPT2LMHeadModel.from_pretrained(tmp_path / "x.pt")` produces
  a model whose `state_dict()` matches the original under
  `torch.allclose` per tensor, same `cfg`, same `encoder_config`.
- Loading a packed bundle with `arch != cls.arch`: pack a bundle with
  `arch="other"` (just edit the dict before `torch.save`) → assert
  `from_pretrained` raises ValueError mentioning arch.
- Loading a non-bundle (`torch.save({"foo":1}, p)`) raises ValueError
  mentioning "packed bundle".

---

## 3.11 model/test_gpt2.py — `inference/model/gpt2.py`

- `GPT2Config()` defaults; `head_dim = n_embd / n_head`.
- `tiny_gpt2.forward(ids)`: returns `(logits, present_kv)`. Assert
  `logits.shape == (B, T, vocab_size)` and `len(present_kv) == n_layer`
  and each `present_kv[i][0].shape == (1, n_head, T, head_dim)`.
- Second forward chains: pass `past_kv` from first → `kv_length` of new
  KV grows by T.
- `kv_null_positions(kv, [(0, 3)])` zeros V and writes -1e4 to K at those
  positions. Assert tensor values directly.
- `make_empty_kv()` returns zero-length KVs (`shape[2] == 0`) for every
  layer.
- `max_context() == cfg.n_positions`.
- `forward_with_hooks(ids, kv, {"attn": fn, "hidden": fn, "logits": fn})`:
  collects `n_layer` attn outputs, `n_layer` hidden outputs, and 1 logits
  output. Assert shapes:
  - attn: `(B, n_head, T, T)`
  - hidden: `(B, T, n_embd)`
  - logits: `(B, T, vocab_size)`

---

## 3.12 model/test_torchscript_adapter.py — `torchscript_adapter.py`

Build a minimal scripted module shaped like a probe target:

```python
class FakeScripted(nn.Module):
    def __init__(self, n_head=2, n_layer=2, n_embd=16, n_pos=128):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab, n_embd)
        self.transformer.wpe = nn.Embedding(n_pos, n_embd)
        self.transformer.h = nn.ModuleList([nn.Identity() for _ in range(n_layer)])
    def forward(self, ids, past_kv=None):
        # accept the probe call — must succeed for n_head where n_embd % n_head == 0
        return torch.zeros(ids.shape[0], ids.shape[1], vocab), past_kv or ()
```

Then:
- `TorchScriptAdapter(scripted)` (no `ts_config`) probes successfully;
  `max_context`, `kv_length`, `make_empty_kv` all behave.
- `TorchScriptAdapter(scripted, ts_config={...})` skips probe and uses
  config directly.
- Probe failure case: a module whose `n_embd` is not divisible by any of
  (8, 16, 12, 4) raises `RuntimeError` with the expected message.

---

## 3.13 model/test_engine.py — `inference/engine.py`

- `InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)` constructs
  cleanly, `_initial_kv is None` before warmup.
- `.warmup()` calls model forward once and caches `_initial_kv` as a tuple
  of (K, V) tensors with `kv_length == 0`.
- `.from_checkpoint(packed_bundle_path)` returns an engine with an
  initialized model and tokenizer; `engine._initial_kv` is populated.
- `.session(score, request)` returns a `SamplingSession` (no exception on
  a valid request).

---

## 3.14 session/test_kv_runner.py — `_KVRunner` in `session.py`

`_KVRunner` is the helper class defined at the top of session.py. Use
`FakeModel` (which records all calls):

- `kv = _KVRunner(fake_model, initial_kv)`; `kv.is_prefill is True`
  before first forward.
- `kv.forward(ids, key_mask=km, position_ids=pos)` calls the model with
  the right kwargs (inspect `fake_model.calls[-1]["kwargs"]`).
- After one forward, `kv.is_prefill is False` and a second forward passes
  `past_kv` from the first call (assert `fake_model.calls[1]["past_len"]
  == ids.shape[1]` of the first call).
- `kv.null_positions([(0, 2)])` invokes `fake_model.kv_null_positions` —
  verify the FakeModel's KV tensors are zeroed at those positions.
- On model exception with full signature, `_KVRunner.forward` falls back
  to positional-only call. (Trigger by making FakeModel raise on `key_mask=`
  kwarg.)

---

## 3.15 session/test_ar.py — autoregressive sampling

Use `tiny_gpt2` end-to-end. Use small bar counts (≤ 4) and
`InferenceConfig(seed=0, temperature=1.0, max_attempts=1,
silence_check=False, novelty_check=False)` for speed.

- Single-track AR over 2 bars: returned Score has the same track count,
  same resolution, the AR track has the expected bar count, the AR track's
  bars now contain notes (probably — assert at least the structure is
  intact even if empty).
- Multi-track AR (2 tracks, `tracks_per_step=1`): both tracks have bars
  generated.
- Multi-track AR with `tracks_per_step=2`: same result, different code
  path.
- `tp.ignore=True` on track 1 with `tp.autoregressive=True` on track 0:
  track 1's bars are byte-identical to input.
- Vary `bars_per_step` ∈ {1, 2, 4}: all complete successfully.
- Vary `temperature` ∈ {0.5, 1.0, 1.5}: all complete.
- `shuffle=True` and `shuffle=False` produce different but valid outputs.

---

## 3.16 session/test_infill.py — infill sampling

- Single-bar infill (one bar in `tp.bars`, rest are context): only the
  target bar has new notes; context bars are byte-identical to input.
- Multi-bar infill: same, multiple targets.
- Multi-track infill: only the targeted (track, bar) pairs are modified.
- `tp.mask_bars` separate from `tp.bars`: the mask_bars are hidden during
  generation but reappear with original content in the output (NOT
  regenerated — verify byte equality with input).
- Mixed AR + infill: track 0 has `autoregressive=True`, track 1 has
  `bars=[2,3]` (infill). Both code paths execute in one request.

---

## 3.17 session/test_modes.py — mask_mode behavior

Five modes: `"token"`, `"attention"`, `"attention_approx"`,
`"attention_skip"`, `"remove"`. For each:

- A request with `tp.mask_bars=[1, 2]` completes and returns a valid
  Score.
- Use `FakeModel` to inspect what was actually passed:
  - `"token"`: encoded context contains MASK_BAR tokens at the right
    positions (decode the input via the vocab to verify, OR just assert
    `len(context_tokens)` matches the unmasked count + N MASK_BAR tokens).
  - `"remove"`: the FakeModel's first call's `input_ids.shape[1]` is
    strictly shorter than in `"token"` mode (masked bars are omitted from
    the stream entirely).
  - `"attention"` / `"attention_approx"`: same token stream as "token"
    mode minus the MASK_BAR tokens — context length matches "remove"
    + however the source builds it; assert the kwargs include the right
    `key_mask` tensor.
  - `"attention_approx"`: after prefill, the model's KV has -1e4 at the
    masked positions (assert via `fake_model.kv_null_positions` call
    record).
- `"token"` mode with a config where `supports_mask_bar_token=False`:
  validation should reject the request BEFORE entering the session loop
  (this is covered in test_validation.py too — keep both).

---

## 3.18 session/test_constraints.py — constraint plumbing

- `polyphony_hard_limit=1`: decode the result and verify no timestep has
  > 1 simultaneous onset (within the generated bars).
- `density_hard_limit=2`: verify each generated bar has ≤ 2 onsets.
- `tp.attributes["note_density"] = K` (track-level pin): for a
  deterministic test, use FakeModel where the NoteDensity token K has the
  highest logit. Then the realized score should quantize to bin K. Skip if
  too flaky — fall back to "the AttributeValueConstraint is added to the
  graph" assertion (inspect the constraint graph if accessible, else
  rely on FakeModel.calls to verify the sampled token was constrained).
- `tp.controls["time_signature"] = idx`: similar — pin via FakeModel
  logit and assert the generated bar's ts_numerator/ts_denominator match
  the index in `ghost_config.time_signatures`.
- `tp.bar_attributes[bar_idx] = {"tension": V}`: bar-level
  BarAttributeValueConstraint is wired; assert via FakeModel.calls or via
  realized output if deterministic.
- Sampling filters:
  - `top_k=1` with FakeModel where token 7 has the only finite logit:
    every sampled token (modulo grammar masking) is 7.
  - `top_p=0.5` keeps a small set; assert via repeated runs that low-prob
    tokens are never sampled.
  - `mask_k=2` removes the top-2; assert top-2 tokens never appear.
  - `mask_p=0.5` removes most-likely set; assert top-1 token never appears.
  - Combination guard: `mask_k >= top_k` is rejected at validation time —
    test that too.

---

## 4. Suggested dispatch plan (parallel agents)

Once in the right cwd (`/Users/paultriana/creative_labs/MIDI-GPT`),
dispatch six parallel agents. Each gets a self-contained brief that
references this document by file path.

Suggested grouping (independent file sets, no overlap):

1. **Agent A**: 3.1 + 3.2 + 3.3 (types, converters, config)
2. **Agent B**: 3.4 + 3.5 + 3.6 (tokenizer, checkpoint, validation)
3. **Agent C**: 3.7 (attributes — single big file)
4. **Agent D**: 3.8 (augmentation — single big file)
5. **Agent E**: 3.9 + 3.10 + 3.11 + 3.12 + 3.13 (model/ folder)
6. **Agent F**: 3.14 + 3.15 + 3.16 + 3.17 + 3.18 (session/ folder)

Each brief should include:
- Repo path (`/Users/paultriana/creative_labs/MIDI-GPT`)
- This document path
  (`tests/python/TEST_IMPLEMENTATION_PLAN.md`) — instruct agent to read
  the relevant section
- The list of section numbers it owns
- The hard rules (section 1)
- A verification command: `source .venv/bin/activate && python -m pytest
  tests/python/<their files> --collect-only` then `python -m pytest
  tests/python/<their files> -x`

---

## 5. Verification after all agents finish

```bash
source .venv/bin/activate
python -m pytest tests/python -x -v
```

Expect: every test passes OR is `xfail` with a `# BUG:` comment pointing
to a real source-code defect. No silent skips.

Then run with timing:

```bash
python -m pytest tests/python --durations=20
```

Anything > 5s on tiny_gpt2 is a smell — investigate.

---

## 6. Known follow-ups (do not address in this round)

- `_KVRunner.forward` has a bare `except Exception` fallback to
  positional-only call. Tests should assert the fallback works (3.14), but
  tightening the guard to only catch `TypeError`/`RuntimeError` is a
  follow-up for after the test suite is green.
- OSC layer (`midigpt.osc.*`) is intentionally not tested in this round.
- Comparison tests at `tests/comparison/` are intentionally out of scope.

---

## 7. Backup of old tests

The pre-rewrite tests are preserved at:

```
/tmp/midigpt_python_tests_backup_20260528_185604/
```

Reference them if you need to see *what kinds of cases* the old suite
covered (especially `test_step_planner.py` parity tests against a JS
reference — those are valuable and should be reintroduced for the
C++ `StepPlanner`/`SelectionMask` bindings).
