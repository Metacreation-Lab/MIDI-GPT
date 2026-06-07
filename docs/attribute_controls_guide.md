# Attribute Controls — Implementation Guide

This document explains every step required to add a new attribute control to MIDI-GPT, from the C++ token type through the Python class to model configuration and inference.

---

## What is an attribute control?

An attribute control is a single token prepended to a track or bar in the token stream that steers the model toward a target musical property (density, polyphony, key, etc.). The model learns to condition on these tokens during training; at inference time you can set them to any value in their domain to guide generation.

There are two levels:

| Level | Token position | One value per |
|-------|---------------|---------------|
| `"track"` | Before the first `Bar` token of a track | Entire track (generation window) |
| `"bar"` | After each `Bar` token | Individual bar |

---

## Overview of the pipeline

```
types.h enum           → integer ID for this token type
encoder_config.cpp     → from_string() resolves "Name" → enum value
add_attribute_token_domains() → extends vocab with (name, size) pairs
Python BaseAttribute   → compute() + quantize() + metadata
ATTRIBUTE_REGISTRY     → maps "registry_key" → class
TOKEN_TYPE_TO_ATTRIBUTE→ maps C++ name → (registry_key, params)
AttributeAnalyzer      → orchestrates compute/quantize, builds constraints
BarAttributeValueConstraint / AttributeValueConstraint → enforce at inference
decoder.cpp            → reads emitted tokens back into track.attributes
```

---

## Step-by-step

### 1. Add to the C++ `TokenType` enum

File: `src/cpp/core/types.h`

Pick the next available integer and add an enumerator:

```cpp
MyNewAttribute = 85,
```

Rules:
- The integer is permanent once a model is trained on it — do not reuse or reorder.
- Check `from_string()` in `encoder_config.cpp` to see which names are already registered; add the new name there if it is missing (see step 2).
- Avoid creating aliases to existing integers unless you intend to reuse the same vocabulary slot.

### 2. Register the name in `from_string()` / `to_string()`

File: `src/cpp/tokenizer/encoder_config.cpp`

Find the `from_string` switch and add:

```cpp
if (s == "MyNewAttribute") return TokenType::MyNewAttribute;
```

And the reverse in `to_string`:

```cpp
case TokenType::MyNewAttribute: return "MyNewAttribute";
```

This is the exact string Python must use as `token_type`. A mismatch causes a `std::runtime_error` at vocab-build time.

### 3. Wire the token into `encoder.cpp`

File: `src/cpp/tokenizer/encoder.cpp`

The encoder pulls quantized values out of `track.attributes` and emits them as tokens. You must register your attribute in one of two places depending on its level. **Skipping this step is silent**: Python will compute the value, the decoder will know how to read it back, but no token will ever reach the model.

**Track-level**: extend the `post_inst_attrs` table (in the per-track loop, after the `Instrument` token):

```cpp
const std::vector<std::pair<std::string, TokenType>> post_inst_attrs = {
    {"min_polyphony",     TokenType::MinPolyphony},
    ...
    {"my_new_attribute",  TokenType::MyNewAttribute},  // ← add here
};
```

The string key MUST equal the Python class's `name` field. The `vocab_.has(type)` and `track.attributes.count(key)` guards mean: nothing is emitted when the model's vocab doesn't include the type, or when Python didn't compute a value for this track.

**Bar-level**: extend the `bar_attrs` table inside the per-bar loop:

```cpp
const std::vector<std::pair<std::string, TokenType>> bar_attrs = {
    {"bar_Tension_" + idx_str,       TokenType::Tension},
    ...
    {"bar_MyNewAttribute_" + idx_str, TokenType::MyNewAttribute},  // ← add here
};
```

The bar-level key convention is `"bar_<TokenType_string>_<bar_idx>"` — using the exact string from `to_string(TokenType)`, NOT the Python `name`. This must match what the decoder writes (step 4) so that encode → decode → encode is a fixed point.

If your attribute differentiates per track type (e.g. `Tension` for melodic, `TensionDrum` for drums), add BOTH bar-key/TokenType pairs — the encoder simply emits whichever key the Python side populated, and Python's track-type filter ensures only one is populated per track.

### 4. Handle the token in `decoder.cpp`

File: `src/cpp/tokenizer/decoder.cpp`

Add a `case` in the main switch. For track-level attributes:

```cpp
case TokenType::MyNewAttribute:
    if (current_track) current_track->attributes["my_new_attribute"] = value;
    break;
```

For bar-level attributes:

```cpp
case TokenType::MyNewAttribute:
    if (current_track && current_bar_idx >= 0)
        current_track->attributes[
            "bar_MyNewAttribute_" + std::to_string(current_bar_idx)
        ] = value;
    break;
```

The string key must match the Python attribute's `name` field (track-level) or follow the `"bar_{token_type}_{bar_idx}"` convention (bar-level). This is what enables the encode → decode → encode fixed-point and powers `AttributeAnalyzer.report()`.

### 5. Write the Python attribute class

File: create or extend a file in `src/python/midigpt/attributes/`

Inherit from `BaseAttribute`:

```python
from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute


class MyNewAttribute(BaseAttribute):
    size = 8  # number of discrete bins (vocab domain size)

    def __init__(self, level: str = "track", track_type: str = "melodic"):
        self.level = level
        self.track_type = track_type  # "melodic" | "drum" | "both"
        self.name = ("bar_" if level == "bar" else "") + "my_new_attribute"
        self.token_type = "MyNewAttribute"  # must match C++ from_string() exactly

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
        # Return a raw (un-quantized) value.
        # When level == "bar" and bar_idx is not None, scope to that bar only.
        track = score.tracks[track_idx]
        ...
        return raw_value

    def quantize(self, value: float | int) -> int:
        # Map raw value into [0, size-1].
        return max(0, min(self.size - 1, int(value)))

    def value_labels(self) -> list[str]:
        # Optional: human-readable label for each bin — used by UIs.
        return [f"level {i}" for i in range(self.size)]
```

Key contracts:
- `compute()` must work for both track-level (no `bar_idx`) and bar-level (with `bar_idx`) if your class supports both levels.
- `quantize()` must return a valid index in `[0, size - 1]`.
- `token_type` must be the exact string returned by C++ `to_string()`.
- `name` is the dict key used in `track.attributes` (decoder → Python).

### 6. Register in `__init__.py`

File: `src/python/midigpt/attributes/__init__.py`

**`ATTRIBUTE_REGISTRY`** — maps a short key (used in `attribute_controls_json` configs) to the class:

```python
from midigpt.attributes.my_new_attribute import MyNewAttribute

ATTRIBUTE_REGISTRY = {
    ...
    "my_new_attribute": MyNewAttribute,
}
```

**`TOKEN_TYPE_TO_ATTRIBUTE`** — maps the C++ token type name to `(registry_key, constructor_params)`. This is used by `AttributeAnalyzer.from_config()` to auto-wire attributes when loading a checkpoint whose `token_domains` contains this token type:

```python
TOKEN_TYPE_TO_ATTRIBUTE = {
    ...
    "MyNewAttribute": ("my_new_attribute", {}),
    # Or with params:  ("my_new_attribute", {"level": "bar"}),
}
```

Add the class to `__all__` as well.

### 7. Configure a model to use it

In an encoder config JSON, add the attribute to `attribute_controls_json`:

```json
{
  "attribute_controls_json": [
    {"name": "my_new_attribute", "params": {"level": "track", "track_type": "melodic"}}
  ]
}
```

Or, if the model's `token_domains` already includes `"MyNewAttribute"`, `from_config()` auto-instantiates it with default params via `TOKEN_TYPE_TO_ATTRIBUTE` — no explicit `attribute_controls_json` needed.

### 8. Use it at inference

```python
from midigpt import generate, TrackPrompt

tp = TrackPrompt(
    instrument=0,
    attributes={"my_new_attribute": 5},  # track-level: bin index 0..size-1
    # bar_attributes={0: {"my_new_attribute": 3}},  # bar-level
)
result = generate(score, track_prompts=[tp], ...)
```

The validation layer checks:
- Value is in range `[0, size-1]`
- Track-level attributes go in `tp.attributes`, bar-level in `tp.bar_attributes` (wrong placement raises `ValueError`)
- `achievable_range()` is called to warn (not reject) when a target is physically infeasible given the fixed bars

### 9. Inspect results with `report()`

After generation, compare what the model sampled against what the notes actually encode:

```python
analyzer = AttributeAnalyzer.from_config(encoder_config)
report = analyzer.report(result_score, track_idx=0, requested=tp.attributes)

for attr_name, entry in report["track"].items():
    print(attr_name, entry)
# {"sampled": 5, "realized": 4, "requested": 5, "consistent": False, "achieved": False}

for bar_idx, bar_entry in enumerate(report["bars"]):
    print(bar_idx, bar_entry)
```

- `sampled`: token value the model emitted (written by decoder into `track.attributes`)
- `realized`: value computed from the actual generated notes
- `requested`: the value you asked for (if you passed `requested=`)
- `consistent`: `sampled == realized` — True means the model's token accurately describes its own output
- `achieved`: `realized == requested` — True means generation hit the target

---

## Checklist

- [ ] Add `MyNewAttribute = N` to `TokenType` enum in `types.h`
- [ ] Add to `from_string()` and `to_string()` in `encoder_config.cpp`
- [ ] **Add the encoder lookup in `encoder.cpp`** (track-level → `post_inst_attrs`; bar-level → `bar_attrs`). Skipping this is silent: Python computes, decoder reads, but the model never sees a token.
- [ ] Add `case TokenType::MyNewAttribute:` in `decoder.cpp`
- [ ] Write `MyNewAttribute(BaseAttribute)` class with `compute()`, `quantize()`, `size`, `name`, `token_type`
- [ ] Add to `ATTRIBUTE_REGISTRY` and `TOKEN_TYPE_TO_ATTRIBUTE` in `__init__.py`
- [ ] Add to `__all__` in `__init__.py`
- [ ] Test: `pytest tests/python/test_attributes.py`
- [ ] Add a config entry and verify `from_config()` instantiates the attribute correctly
- [ ] Run `report()` on a generated sample and inspect `consistent` / `achieved`
- [ ] **Round-trip test**: encode a score with your attribute populated, decode, confirm `decoded.tracks[t].attributes[<key>]` matches the input quantized value

---

## Notes on aliases and disambiguation

Several token types share an integer with a shorter alias defined in `types.h`. These are intentional — the alias is the canonical Python-facing name, and the longer name describes the architectural role:

| Canonical alias | Full name | Integer |
|----------------|-----------|---------|
| `OnsetPolyphony` | `BarLevelOnsetPolyphonyMax` | 42 |
| `PitchClassSet` | `BarLevelPitchClassSet` | 51 |
| `SilenceProportion` | `TrackLevelSilenceProportionMax` | 53 |
| `PitchRange` | `TrackLevelPitchRangeMax` | 49 |
| `NoteDurationDist` | `NoteDuration` | 26 |

Because `OnsetPolyphony = 42` is the same integer as `BarLevelOnsetPolyphonyMax`, the decoder uses token position to disambiguate: a token 42 before the first `Bar` in a track writes `"onset_polyphony"` (track-level); after a `Bar` token it writes `"bar_BarLevelOnsetPolyphonyMax_N"` (bar-level).

Do not introduce new aliases. Bar-level and track-level variants of the same concept need distinct integers (e.g. `BarLevelOnsetDensity = 40` vs `TrackLevelOnsetDensity = 43`).

---

## Token types with no C++ entry yet

These Python classes exist and work for compute/quantize but cannot be injected into the model vocab until a C++ token type is assigned:

- `SilenceProportion(level="bar")` — token_type `"SilenceProportionBar"` (no C++ enum entry)
- `NoteDurationQuantile(level="bar")` — token_type `"MinNoteDurationBar"` / `"MaxNoteDurationBar"` (no C++ enum entry)
- `PitchClassSet(level="track")` — token_type `"PitchClassSetTrack"` (no C++ enum entry)

They are excluded from `TOKEN_TYPE_TO_ATTRIBUTE` and will not be auto-inferred from checkpoints until corresponding C++ types are added.
