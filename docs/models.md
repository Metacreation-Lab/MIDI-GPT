# Models

midigpt ships three pretrained checkpoints, each trained on a different encoder configuration. Choosing the right model depends on the kind of music you're working with and the generation mode you need.

## Overview

| Model | `num_bars_map` | MaskBar | Microtiming | Velocity bins | Download |
|---|---|---|---|---|---|
| `yellow` | 4, 8 | no | no | 32 | [yellow.pt](https://huggingface.co/Metacreation/MIDI-GPT/resolve/main/yellow.pt) |
| `ghost` | 4, 8, 12, 16 | yes | no | 32 | coming soon |
| `expressive` | 4, 8 | no | yes | 128 | coming soon |

All three models condition on the same attribute set: **note density**, **min/max polyphony**, and **min/max note duration**.

---

## Yellow

The baseline checkpoint. Clean encoder, broad context window support (4 or 8 bars), works with any attention-based mask mode. A good default for most composition tasks.

```python
engine = InferenceEngine.from_pretrained("yellow")
# or load locally:
engine = InferenceEngine.from_checkpoint("yellow.pt")
```

**When to use:** General-purpose melody and accompaniment generation. If you are not sure which model to pick, start here.

**Mask mode:** `yellow` does not have a `MaskBar` token — use `mask_mode="attention"` (or `"attention_approx"`, `"attention_skip"`, `"remove"`).

---

## Ghost

An extended checkpoint trained with larger context windows (up to 16 bars) and a `MaskBar` vocabulary entry. The wider windows allow Ghost to model longer-range phrasing and repeating structures.

**When to use:** When you need to infill bars inside a long section (8–16 bars of context), or when you want to use the explicit `"token"` mask mode.

**Mask mode:** `ghost` supports all five mask modes including `"token"`.

---

## Expressive

A microtiming-aware checkpoint. The encoder emits `delta` offset tokens that capture note placement at sub-grid resolution, and uses 128 velocity bins (vs. 32 for the other models). This produces output that feels more "human" and less quantized.

**When to use:** When timbral and rhythmic nuance matters more than structural control — e.g. jazz, solo piano, or any music where expressive timing is essential.

**Mask mode:** `expressive` does not have a `MaskBar` token — use `mask_mode="attention"` or similar.

---

## Compatibility matrix

Which `mask_mode` values work with each model:

| `mask_mode` | yellow | ghost | expressive |
|---|---|---|---|
| `"attention"` | yes | yes | yes |
| `"attention_approx"` | yes | yes | yes |
| `"attention_skip"` | yes | yes | yes |
| `"remove"` | yes | yes | yes |
| `"token"` | **no** | yes | **no** |

Which `model_dim` values are valid:

| `model_dim` | yellow | ghost | expressive |
|---|---|---|---|
| 4 | yes | yes | yes |
| 8 | yes | yes | yes |
| 12 | no | yes | no |
| 16 | no | yes | no |

---

## `model_dim` and context

`model_dim` is the number of bars in the model's context window — it is **not** a vocabulary or architecture dimension. See [Concepts — The context window](concepts.md#the-context-window) for a full explanation.

Pass a value from the model's `num_bars_map`. The session will raise a `RequestValidationError` if you pass a value that does not appear in the checkpoint's map.

```python
# Valid for yellow and ghost
InferenceConfig(model_dim=4, mask_mode="attention")
InferenceConfig(model_dim=8, mask_mode="attention")

# Valid for ghost only
InferenceConfig(model_dim=12, mask_mode="attention")
InferenceConfig(model_dim=16, mask_mode="token")   # ghost supports "token"
```

---

## Checkpoint format

Packed `.pt` bundles embed the encoder config alongside the model weights:

```python
{
    "format_version": 1,
    "arch":           "gpt2",
    "config":         {"vocab_size": ..., "n_positions": 2048, "n_embd": 512, ...},
    "encoder_config": { ... },   # full encoder JSON — source of truth for vocab sizes
    "state_dict":     { ... },   # HuggingFace GPT-2 key layout
}
```

`load_checkpoint(path)` also accepts a legacy directory containing `config.json` + `model.pt`.
