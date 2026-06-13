# Encoder Config — Design Guide

This document explains every field in an encoder config JSON, how to design a new encoder from scratch, and the relationships between the JSON, the C++ tokenizer, and the Python attribute pipeline.

---

## What is an encoder config?

An encoder config defines a MIDI-GPT model's **vocabulary and conditioning signals**. It controls:

- Which token types exist and how many discrete values each can take
- Which musical attributes the Python side computes (density, polyphony, pitch class, etc.) and at what granularity (bar vs. track)
- Genre conditioning (canonical genre labels and their aliases)
- Structural tokenization parameters (resolution, time signatures, bar lengths)

A trained model is permanently bound to the encoder it was trained with. Changing any field that affects vocabulary size or token ID assignment invalidates a checkpoint.

---

## File location and naming

Encoder configs live in `models/`. Name them `<name>_encoder.json`. The `<name>` is used as a handle everywhere (training configs, CLI flags, checkpoint metadata).

Existing encoders:
- `models/yellow_encoder.json` — 5 track-level attributes, no genre
- `models/prism_encoder.json` — 10 attributes (bar + track level), 18-genre control

---

## Full field reference

### `token_domains`

**Required. Not inferred.** Defines the vocabulary — every token type the encoder emits, and the number of distinct values it can take. Order matters: token IDs are assigned sequentially across domains.

```json
"token_domains": [
  { "type": "PieceStart",    "domain_size": 2   },
  { "type": "Genre",         "domain_size": 18  },
  { "type": "Track",         "domain_size": 2   },
  { "type": "TrackEnd",      "domain_size": 1   },
  { "type": "Bar",           "domain_size": 1   },
  { "type": "BarEnd",        "domain_size": 1   },
  { "type": "TimeSig",       "domain_size": 36  },
  { "type": "Instrument",    "domain_size": 109 },
  { "type": "NumBars",       "domain_size": 2   },
  { "type": "NoteOnset",     "domain_size": 128 },
  { "type": "NoteDuration",  "domain_size": 96  },
  { "type": "TimeAbsolutePos","domain_size": 192 },
  { "type": "VelocityLevel", "domain_size": 32  },
  { "type": "FillInPlaceholder","domain_size": 1 },
  { "type": "FillInStart",   "domain_size": 1   },
  { "type": "FillInEnd",     "domain_size": 1   }
]
```

The first 16 entries above are required for all encoders. Attribute tokens are appended after them.

**Critical rules:**
- `domain_size` for an attribute token must exactly match the Python attribute class's `size` field — there is no automatic check.
- `Genre` must appear before `Track` (it is a piece-level token). If `genre_groups` is omitted, omit `Genre` from `token_domains` too.
- Adding or reordering entries changes all subsequent token IDs. Only add to the end, or accept that the checkpoint is incompatible.
- `"type"` strings must match the C++ `TokenType` enum exactly (see `attribute_controls_guide.md` for the full list of registered names).

**Attribute token domain sizes** (from Python `BaseAttribute.size`):

| Token type | Domain size | Level | Python class |
|---|---|---|---|
| `NoteDensity` | 10 | track | `NoteDensityQuantile` |
| `MinPolyphony` | 10 | track | `PolyphonyQuantile(mode="min")` |
| `MaxPolyphony` | 10 | track | `PolyphonyQuantile(mode="max")` |
| `MinNoteDuration` | 6 | track | `NoteDurationQuantile(mode="min")` |
| `MaxNoteDuration` | 6 | track | `NoteDurationQuantile(mode="max")` |
| `SilenceProportion` | 10 | track | `SilenceProportion` |
| `PitchRange` | 128 | track | `PitchRange` |
| `KeySignature` | 25 | track | `KeySignature` |
| `OnsetPolyphony` | 10 | track | `OnsetPolyphony` |
| `PitchClassSetTrack` | 13 | track | `PitchClassSet(level="track")` |
| `BarLevelOnsetDensity` | 10 | bar | `NoteDensityQuantile(level="bar")` |
| `BarLevelOnsetPolyphonyMin` | 10 | bar | `PolyphonyQuantile(mode="min", level="bar")` |
| `BarLevelOnsetPolyphonyMax` | 10 | bar | `PolyphonyQuantile(mode="max", level="bar")` |
| `PitchClassSet` | 13 | bar | `PitchClassSet` |
| `Tension` | 10 | bar | `Tension` |
| `TensionDrum` | 10 | bar | `TensionDrum` |
| `Genre` | N | piece | `genre_grouping.num_genres()` |

Note: `PitchClassSetTrack` has no C++ entry yet — it cannot be used in `token_domains` until one is added. See `attribute_controls_guide.md` § "Token types with no C++ entry yet".

---

### `attribute_controls`

Tells the Python `AttributeAnalyzer` which attributes to compute during training and encoding. Each entry must have a corresponding `token_domains` entry with the correct domain size.

```json
"attribute_controls": [
  { "name": "key_signature" },
  { "name": "silence_proportion" },
  { "name": "pitch_range" },
  { "name": "note_duration_quantile", "params": { "mode": "min" } },
  { "name": "note_duration_quantile", "params": { "mode": "max" } },
  { "name": "onset_polyphony" },
  { "name": "note_density_quantile",  "params": { "level": "bar" } },
  { "name": "polyphony_quantile",     "params": { "mode": "min", "level": "bar" } },
  { "name": "polyphony_quantile",     "params": { "mode": "max", "level": "bar" } },
  { "name": "pitch_class_set" }
]
```

`name` is a key in `ATTRIBUTE_REGISTRY` (`src/python/midigpt/attributes/__init__.py`). `params` are passed as constructor kwargs to the Python attribute class.

**Bar vs. track level:**

| Level | Specify | Token type used | Token position in sequence |
|---|---|---|---|
| track (default) | no `level` param | e.g. `NoteDensity` | Once, before first bar of each track |
| bar | `"level": "bar"` | e.g. `BarLevelOnsetDensity` | Once per bar, after `Bar` token |

Bar-level attributes increase sequence length by `num_bars × num_tracks` tokens. They give the model fine-grained per-bar control at the cost of longer contexts. Prefer bar-level for attributes that vary meaningfully within a piece (density, polyphony, harmony); prefer track-level for global characteristics (key, range, silence).

**The token_domains entry must match the level.** A `note_density_quantile` without `level` param expects `NoteDensity` in `token_domains`; with `"level": "bar"` it expects `BarLevelOnsetDensity`. There is no runtime error if they mismatch — the C++ will silently emit nothing.

---

### `genre_groups`

Optional. Enables piece-level genre conditioning. If present, add `Genre` to `token_domains` with `domain_size` equal to the number of canonical genre keys.

```json
"genre_groups": {
  "rock":  ["rock"],
  "metal": ["metal", "heavy metal", "grunge", "hardcore"],
  "pop":   ["pop", "k pop", "dance", "disco"]
}
```

**Format rules:**
- Keys are **canonical genre names** — these are what the model learns and what inference accepts.
- Values are **alias lists** — raw strings (from the dataset) that map to that canonical name. Lookup is **case-insensitive**. An alias may appear in only one group.
- **Canonical names self-register automatically.** The C++ `GenreGrouping` always adds the canonical key to its lookup table regardless of whether it appears in the alias list. So `encode("rock")` works even if `"rock"` is not listed under `"rock"`. You do not need to repeat the key in its own alias list.
- **Unknown genres are silently skipped during training.** `_genre_token()` calls `contains()` before `encode()` — dataset rows with genres not in any alias list simply emit no genre token (genre = -1). The C++ `encode()` would throw for unknown genres, but training never reaches it.
- Dense IDs are assigned in **alphabetical key order** — `metal` = 0, `pop` = 1, `rock` = 2 in the example above. Token ID assignment is fixed once training begins.
- `domain_size` in `token_domains` must equal the number of keys.

**GigaMIDI specifics:**

The genre field in GigaMIDI v2.0.0 is `music_style_scraped`. It is a **URL-encoded, comma-separated** string:

```
"alternative-indie%2cpop%2crock"  →  ["alternative-indie", "pop", "rock"]
```

The Python dataset reads it as:
```python
from urllib.parse import unquote
genres = [g.strip() for g in unquote(raw).split(",") if g.strip()]
known = [g for g in genres if genre_grouping.contains(g)]
# randomly sample one known genre per training example
```

Coverage for the `prism_encoder.json` mapping (18 canonical genres, 18M genre-labeled rows in GigaMIDI v2.0.0 train): **99.9%** of labeled rows map to a known canonical genre. Only ~6% of all rows have any genre label.

---

### `resolution`

Ticks per quarter note for the **tokenization grid**. Controls the finest time granularity the model can represent.

```json
"resolution": 12
```

Common values: 12 (default), 24 (finer grid, larger vocabulary). Higher resolution increases `TimeAbsolutePos` domain size and sequence length.

---

### `decode_resolution`

Ticks per quarter note for the **output MIDI**. Usually much higher than `resolution` (e.g. 1920) to produce standard-compliant MIDI files. The decoder up-scales token positions by `decode_resolution / resolution`.

```json
"decode_resolution": 1920
```

---

### `note_duration_max_beats`

Maximum note duration in beats. Notes longer than this are clamped. Determines `NoteDuration` domain size indirectly.

```json
"note_duration_max_beats": 8
```

---

### `num_bars_map`

List of valid bar-count window sizes the model is trained on. Used to set the `NumBars` token domain size and to configure the training sampler.

```json
"num_bars_map": [4, 8]
```

This means the model sees 4-bar and 8-bar windows; `domain_size` for `NumBars` should equal the length of this list.

---

### `time_signatures`

Allowable time signatures as `"numerator/denominator"` strings. Filters out rows during preprocessing (Phase 2). Must match `domain_size` for `TimeSig` in `token_domains`.

```json
"time_signatures": ["4/4", "3/4", "6/8", ...]
```

`TimeSig` domain size = `len(time_signatures)`.

---

### `pitch_range`

MIDI pitch range `[min, max]` (inclusive). Notes outside this range are filtered during encoding.

```json
"pitch_range": [0, 127]
```

---

### `velocity_levels` and `velocity_sticky`

Number of discrete velocity bins. `velocity_sticky` means the velocity token is only emitted when it changes (reduces sequence length).

```json
"velocity_levels": 32,
"velocity_sticky": true
```

---

### `instrument_merge_groups`

Groups of MIDI program numbers (0-indexed) that are treated as the same instrument for tokenization. Reduces instrument vocabulary fragmentation.

```json
"instrument_merge_groups": [[0, 1, 2], [4, 5], ...]
```

---

### `supports_infill` and `supports_mask_bar_token`

Feature flags. `supports_infill: true` enables fill-in-the-blank generation (requires `FillInPlaceholder`, `FillInStart`, `FillInEnd` in `token_domains`). `supports_mask_bar_token` enables the `MaskBar` token for bar-level masking.

---

### `emit_delta_tokens`

If `true`, emit time-delta tokens instead of absolute position tokens. Not used in current production encoders.

---

## Designing a new encoder: decisions checklist

**1. What attributes to include?**

Start from the available attribute types (table above). Ask:
- Is the attribute musically meaningful at bar level or only globally? → bar vs. track
- How many bins does it need? More bins = finer control but larger vocab and slower convergence.
- Will the training data have good coverage? (e.g. genre is sparse in GigaMIDI — only 6% of rows are labeled)

**2. Bar-level or track-level for density/polyphony?**

Bar-level is recommended if you want the model to follow per-bar musical shape. Track-level is cheaper and sufficient for style-level control. Do not include both versions of the same attribute — pick one.

**3. Genre groups?**

Only add genre if your training data has meaningful genre coverage and you want inference-time genre control. Build the alias list from an audit of the actual dataset values (see the analysis that produced `prism_encoder.json`).

**4. Vocabulary size budget**

A rough guide:
- Base tokens (structural, no attributes, no genre): ~650 tokens
- Each track-level attribute: +domain_size tokens
- Each bar-level attribute: +domain_size tokens (but adds tokens per bar in sequence)
- Genre: +num_genres tokens

Larger vocabularies increase model size (embedding table) and may slow convergence for rare tokens.

---

## Validating a new encoder config

After writing the JSON, always validate before starting training:

```python
import midigpt._core as _core

with open("models/my_encoder.json") as f:
    cfg = _core.EncoderConfig.from_json(f.read())

vocab = _core.Vocabulary(cfg)
print(f"vocab_size: {vocab.size()}")

# If genre_groups is present:
g = cfg.genre_grouping
print(f"num_genres: {g.num_genres()}")
print(g.encode("rock"), g.decode(0))
```

Errors at this stage:
- `std::runtime_error: Unknown token type 'Foo'` → `Foo` is not a registered C++ TokenType name
- `AttributeError: 'NoneType' ...` on `cfg.genre_grouping` → `genre_groups` is missing from JSON but `Genre` is in `token_domains`, or vice versa

Also verify that `domain_size` in `token_domains` matches Python attribute `size`:

```python
from midigpt.attributes import ATTRIBUTE_REGISTRY

for ctrl in config["attribute_controls"]:
    cls = ATTRIBUTE_REGISTRY[ctrl["name"]]
    inst = cls(**ctrl.get("params", {}))
    print(inst.token_type, inst.size)
# Cross-check these sizes against token_domains entries
```

---

## Common pitfalls

**`attribute_controls` and `token_domains` must both be updated.** Adding an attribute to `attribute_controls` without a matching `token_domains` entry means Python computes the value but the C++ tokenizer emits nothing — silent data loss, no error.

**Genre domain_size must equal the number of keys** in `genre_groups`. The C++ reads this as a fixed-size domain at vocab-build time; a mismatch causes an out-of-range panic during encoding.

**Aliases must be unique across groups.** `"swing"` appearing in both `"jazz"` and `"folk"` causes the first-seen mapping to win silently. Audit with a deduplication check before finalizing.

**`PitchClassSetTrack` has no C++ entry.** Do not add it to `token_domains` — it will fail. See `attribute_controls_guide.md` § "Token types with no C++ entry yet".

**Alphabetical key order determines genre IDs.** If you rename or reorder keys after training begins, all genre tokens in the dataset's cached indices become wrong. Freeze the genre key set before running `preprocess.sh`.
