# HTTP Server

The `midigpt[http]` extra adds a stateless REST API for generation. Every request carries the full score and generation parameters — the server holds no session state between calls. The only persistent state is the loaded model and device.

## Setup

```bash
pip install "midigpt[http]"
```

### Local checkpoint

```bash
midigpt-http --ckpt models/yellow.pt --port 8000
```

### Pretrained from HuggingFace Hub

```bash
# By short name (downloads once, cached in ~/.cache/huggingface/)
midigpt-http --pretrained yellow --port 8000

# By repo ID + filename
midigpt-http --pretrained Metacreation/MIDI-GPT --hf-filename yellow.pt --port 8000
```

### Device selection

```bash
midigpt-http --pretrained yellow --device cuda   # explicit GPU
midigpt-http --pretrained yellow --device mps    # Apple Silicon
midigpt-http --pretrained yellow --device auto   # auto-detect (default)
midigpt-http --pretrained yellow --device cpu    # force CPU
```

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--ckpt PATH` | — | Local `.pt` bundle or checkpoint directory (mutually exclusive with `--pretrained`) |
| `--pretrained NAME_OR_REPO` | — | Short name (`yellow`) or HuggingFace repo ID |
| `--hf-filename FILE` | — | Filename within the HF repo (required when `--pretrained` is a full repo ID) |
| `--device DEVICE` | auto | `cpu`, `cuda`, `mps`, or `auto` |
| `--host HOST` | `0.0.0.0` | Bind address |
| `--port PORT` | `8000` | TCP port |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Endpoints

Interactive docs are available at `http://localhost:8000/docs` once the server is running.

### `GET /health`

Liveness probe.

```json
{"status": "ok"}
```

### `GET /info`

Returns the loaded checkpoint label, model capabilities, and available attribute controls.

```json
{
  "checkpoint": "yellow",
  "capabilities": {
    "note_density": true,
    "min_polyphony": true,
    "max_polyphony": true,
    "min_note_duration": true,
    "max_note_duration": true,
    "tension": false,
    "supports_token_mask": true,
    "supports_attention_mask": true,
    "supports_attention_approx": true,
    "supports_attention_skip": true,
    "supports_remove": true
  },
  "attributes": {
    "note_density": 10,
    "min_polyphony": 10,
    "max_polyphony": 10,
    "min_note_duration": 10,
    "max_note_duration": 10
  }
}
```

### `POST /generate`

Generate or infill music. Pass the full score and generation parameters; receive the result score back.

**Request body**

```json
{
  "score": { ... },
  "request": { ... }
}
```

- `score` — a `Score` serialised with `Score.to_dict()` (see [Inference API](api.md))
- `request` — a `GenerationRequest` dict (see below)

**Response**

```json
{
  "score": { ... },
  "timing": {
    "model_forward_s": 0.42,
    "encode_s": 0.01,
    "decode_s": 0.01,
    "gen_count": 4
  }
}
```

**Error codes**

| Status | Meaning |
|---|---|
| `400` | Malformed score or request dict |
| `422` | `RequestValidationError` — structurally invalid generation request |
| `500` | Inference failure |

---

## Client example

The server accepts plain JSON — no `midigpt` dependency needed on the client side.

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "score": {
      "resolution": 480, "tempo": 500000,
      "tracks": [{
        "instrument": 0, "track_type": "melodic",
        "bars": [
          {"ts_numerator": 4, "ts_denominator": 4, "notes": []},
          {"ts_numerator": 4, "ts_denominator": 4, "notes": []},
          {"ts_numerator": 4, "ts_denominator": 4, "notes": []},
          {"ts_numerator": 4, "ts_denominator": 4, "notes": []}
        ]
      }]
    },
    "request": {
      "tracks": [{"id": 0, "bars": [0, 1, 2, 3]}],
      "config": {"model_dim": 4}
    }
  }'
```

The response `score` field contains the filled-in notes in the same JSON shape.

---

## Concurrency

Only one inference call runs at a time (GPU cannot parallelise). Additional `/generate` requests queue behind the active one; `/health` and `/info` remain responsive throughout.
