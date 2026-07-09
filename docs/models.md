# Models

midigpt ships pretrained checkpoints for three encoder families — `yellow`, `prism`, and `expressive` — each trained on a different encoder configuration and loadable via `InferenceEngine.from_pretrained`. A fourth encoder, `ghost`, is a planned architecture documented below for reference; no checkpoint has been trained or released for it, and `from_pretrained("ghost")` raises `ValueError`. Choosing the right model depends on the kind of music you're working with and the generation mode you need.

## Overview

| Model | `num_bars_map` | MaskBar | Microtiming | Velocity bins | Download |
|---|---|---|---|---|---|
| `yellow` (medium) | 4, 8 | no | no | 32 | [yellow_medium-final.safetensors](https://huggingface.co/Metacreation/MIDI-GPT/resolve/main/yellow_medium-final.safetensors) |
| `yellow_small` | 4, 8 | no | no | 32 | [yellow_small-final.safetensors](https://huggingface.co/Metacreation/MIDI-GPT/resolve/main/yellow_small-final.safetensors) |
| `prism_medium` | 4, 8, 12, 16 | no | no | 32 | [prism_medium-step58000.safetensors](https://huggingface.co/Metacreation/MIDI-GPT/resolve/main/prism_medium-step58000.safetensors) — training in progress, not a final checkpoint |
| `expressive` (medium) | 4, 8, 12, 16 | no | yes | 128 | [expressive_medium-step56000.safetensors](https://huggingface.co/Metacreation/MIDI-GPT/resolve/main/expressive_medium-step56000.safetensors) — training in progress, not a final checkpoint |
| `ghost` | 4, 8, 12, 16 | yes | no | 32 | not released — `from_pretrained("ghost")` raises `ValueError` (see [Ghost](#ghost)) |

The `yellow` model conditions on: **note density**, **min/max polyphony**, and **min/max note duration**. `prism` and `expressive` support a wider set of attribute controls: **key signature**, **pitch range**, **silence proportion**, **min/max note duration**, **note density (bar-level)**, **min/max polyphony (bar-level)**, **pitch class set (bar-level)**, and 18 **genre** groups. `expressive` additionally has **NOMML** (median metric depth) and piece-level switchable microtiming/velocity controls; `prism` does not.

`InferenceEngine.from_pretrained(name)` always resolves to the newest checkpoint for that name — a `-final` snapshot if training has completed, otherwise the highest-step in-progress snapshot — so the exact filenames in the table above may lag what `from_pretrained` fetches.

---

## Yellow

The baseline checkpoint. Clean encoder, broad context window support (4 or 8 bars), works with any attention-based mask mode. A good default for most composition tasks.

```python
engine = InferenceEngine.from_pretrained("yellow")
# or load locally:
engine = InferenceEngine.from_checkpoint("yellow_medium-final.safetensors")
```

**When to use:** General-purpose melody and accompaniment generation. If you are not sure which model to pick, start here.

**Mask mode:** `yellow` does not have a `MaskBar` token — use `mask_mode="attention"` (or `"attention_approx"`, `"attention_skip"`, `"remove"`).

---

## Prism

A general-purpose checkpoint trained on the same wide attribute set as `expressive` — key signature, pitch range, silence proportion, note duration, note density (bar-level), polyphony (bar-level), pitch class set (bar-level), and 18 genre groups — across the full range of context windows (4, 8, 12, or 16 bars). Unlike `expressive`, it uses standard 32-level velocity, does not emit microtiming/delta tokens, and has no switchable velocity/microtiming controls.

```python
engine = InferenceEngine.from_pretrained("prism_medium")
```

**When to use:** General-purpose composition and infill when you want genre- or attribute-conditioned control and/or a longer context window (up to 16 bars), without the microtiming overhead of `expressive`.

**Mask mode:** `prism` does not have a `MaskBar` token — use `mask_mode="attention"` (or `"attention_approx"`, `"attention_skip"`, `"remove"`).

---

## Ghost

> **Not currently available.** No `ghost` checkpoint has been trained or uploaded, and `"ghost"` was removed from `InferenceEngine.from_pretrained`'s known model names — `InferenceEngine.from_pretrained("ghost")` raises `ValueError`. The rest of this section documents the *planned* architecture for future reference; nothing below is usable today.

An extended checkpoint design with larger context windows (up to 16 bars) and a `MaskBar` vocabulary entry — as opposed to representing masked bars purely via attention. The wider windows are intended to let Ghost model longer-range phrasing and repeating structures.

**Intended use:** Infilling bars inside a long section (8–16 bars of context), or using the explicit `"token"` mask mode.

**Mask mode:** would support all five mask modes including `"token"`, once trained.

---

## Expressive

A microtiming-aware checkpoint. The encoder emits `delta` offset tokens that capture note placement at sub-grid resolution, and uses 128 velocity bins (vs. 32 for the other models). It includes a `nomml` attribute control (median metric depth) to govern the degree of expressive timing vs. quantization. It also supports piece-level switchable controls for velocity and microtiming. This produces output that feels more "human" and less quantized.

**When to use:** When timbral and rhythmic nuance matters more than structural control — e.g. jazz, solo piano, or any music where expressive timing is essential.

**Mask mode:** `expressive` does not have a `MaskBar` token — use `mask_mode="attention"` or similar.

---

## Compatibility matrix

Which `mask_mode` values work with each model:

| `mask_mode` | yellow | prism | expressive | ghost* |
|---|---|---|---|---|
| `"attention"` | yes | yes | yes | yes |
| `"attention_approx"` | yes | yes | yes | yes |
| `"attention_skip"` | yes | yes | yes | yes |
| `"remove"` | yes | yes | yes | yes |
| `"token"` | **no** | **no** | **no** | yes |

Which `model_dim` values are valid:

| `model_dim` | yellow | prism | expressive | ghost* |
|---|---|---|---|---|
| 4 | yes | yes | yes | yes |
| 8 | yes | yes | yes | yes |
| 12 | no | yes | yes | yes |
| 16 | no | yes | yes | yes |

\* `ghost` has no released checkpoint — its columns describe the planned architecture only; `from_pretrained("ghost")` is not available. See [Ghost](#ghost).

---

## `model_dim` and context

`model_dim` is the number of bars in the model's context window — it is **not** a vocabulary or architecture dimension. See [Concepts — The context window](concepts.md#the-context-window) for a full explanation.

Pass a value from the model's `num_bars_map`. The session will raise a `RequestValidationError` if you pass a value that does not appear in the checkpoint's map.

```python
# Valid for yellow, prism, and expressive (4 or 8 bars)
InferenceConfig(model_dim=4, mask_mode="attention")
InferenceConfig(model_dim=8, mask_mode="attention")

# Valid for prism and expressive only (4, 8, 12, or 16 bars)
InferenceConfig(model_dim=12, mask_mode="attention")
InferenceConfig(model_dim=16, mask_mode="attention")

# "token" mask mode is part of ghost's planned architecture only — no
# released checkpoint currently supports it.
```

---

## Checkpoint format

By default, checkpoints are packaged as `.safetensors` files (`format_version: 2`) embedding the weights and metadata:

* **Weights:** Stored natively in SafeTensors format.
* **Metadata:** Stored inside the SafeTensors file header with the following keys:
  * `format_version`: `"2"`
  * `arch`: `"gpt2"`
  * `config`: A JSON string representing the model architecture configuration (e.g., `n_embd`, `n_layer`, `n_head`).
  * `encoder_config`: A JSON string representing the full encoder configuration (vocabulary domains, resolution, etc.).

`load_checkpoint(path)` is backwards-compatible and also accepts:
* Legacy `.pt` packed-bundle files (`format_version: 1`) containing a pickled dict of weights, architecture, and encoder configuration.
* A directory containing `config.json` + `model.pt` (legacy TorchScript representation).
