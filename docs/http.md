# HTTP Server

The `midigpt[http]` extra adds a stateless REST API for generation. Every request carries the full score and generation parameters — the server holds no per-generation session state between calls. The persistent state is the loaded model(s) and device.

## Setup

```bash
pip install "midigpt[http]"
```

### Local checkpoint

```bash
midigpt-http --ckpt checkpoints/run_001/model_final.safetensors --port 8000
```

### Pretrained from HuggingFace Hub

```bash
# Checkpoint filename prefix on the hub (downloads once, cached in ~/.cache/huggingface/)
midigpt-http --pretrained yellow_medium --port 8000
midigpt-http --pretrained prism_medium --port 8000
midigpt-http --pretrained expressive_medium --port 8000

# Custom repo
midigpt-http --pretrained my_model --hf-repo myorg/myrepo --port 8000
```

### Device selection

```bash
midigpt-http --pretrained yellow_medium --device cuda   # explicit GPU
midigpt-http --pretrained yellow_medium --device mps    # Apple Silicon
midigpt-http --pretrained yellow_medium --device auto   # auto-detect (default)
midigpt-http --pretrained yellow_medium --device cpu    # force CPU
```

### Multiple models in one server

Pass `--config` with a YAML file instead of `--ckpt`/`--pretrained` to serve several checkpoints from one process, selectable per-request via the `model` id:

```yaml
# models.yaml
host: 0.0.0.0
port: 8000
idle_timeout: 0
max_queue: 64            # server-wide across every model
hf_repo: Metacreation/MIDI-GPT   # default for pretrained entries
device: cpu               # default device for every entry
max_parallel: 1           # default per-model concurrency
default_model: yellow_medium     # defaults to the first entry's id if omitted

models:
  - id: yellow_medium            # the "model" id clients pass to /generate
    pretrained: yellow_medium
  - id: prism_medium
    pretrained: prism_medium
    device: cpu                 # per-entry override
    max_parallel: 2              # per-entry override — benchmark before raising, see Concurrency below
  - id: my_custom
    ckpt: /path/to/custom-final.safetensors
    hf_repo: myorg/myrepo       # per-entry override (pretrained entries only)
```

```bash
midigpt-http --config models.yaml
```

`host`/`port`/`idle_timeout`/`log_level`/`max_queue` in the file override the equivalent CLI flags if both are given.

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | — | YAML file describing one or more models (see above). Mutually exclusive with `--ckpt`/`--pretrained` |
| `--ckpt PATH` | — | Local `.safetensors`, `.pt` bundle, or checkpoint directory (mutually exclusive with `--config`/`--pretrained`) |
| `--pretrained NAME` | — | Checkpoint filename prefix on HuggingFace (`yellow_medium`, `prism_medium`, `expressive_medium`, ...) |
| `--hf-repo REPO` | `Metacreation/MIDI-GPT` | HuggingFace repo ID to download `--pretrained` from |
| `--device DEVICE` | auto | `cpu`, `cuda`, `mps`, or `auto` |
| `--host HOST` | `0.0.0.0` | Bind address |
| `--port PORT` | `8000` | TCP port |
| `--idle-timeout SECONDS` | `0` (off) | Shut down automatically after this many seconds of inactivity |
| `--max-parallel N` | `1` | Max concurrent `/generate` calls against this model (single-model mode only — use per-entry `max_parallel` in `--config` for multi-model). Each is still a full sequential decode sharing CPU/GPU with the others, not a throughput multiplier — benchmark before raising above 1 |
| `--max-queue N` | `64` | Total requests admitted (queued + running), across all models, before returning `503` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

One of `--config`, `--ckpt`, or `--pretrained` is required.

---

## Endpoints

Interactive docs are available at `http://localhost:8000/docs` once the server is running.

### `GET /health`

Liveness probe (also resets the idle-shutdown timer).

```json
{"status": "ok", "inflight": 0, "max_queue": 64}
```

`inflight` is the number of requests currently admitted (queued behind a model's concurrency limit or actively running) server-wide, across every model.

### `GET /info`

Returns capabilities/attributes/resolution for one loaded model.

```
GET /info?model=yellow_medium
```

Omit `model` to get the server's default model.

```json
{
  "model": "yellow_medium",
  "default_model": "yellow_medium",
  "checkpoint": "Metacreation/MIDI-GPT/yellow_medium",
  "capabilities": {
    "tension": false,
    "note_density": true,
    "min_polyphony": true,
    "max_polyphony": true,
    "min_note_duration": true,
    "max_note_duration": true,
    "supports_token_mask": true,
    "supports_attention_mask": true,
    "supports_attention_approx": true,
    "supports_attention_skip": true,
    "supports_remove": true,
    "supports_pitch_mask": true,
    "pitch_mask_scale_presets": ["major", "minor", "..."],
    "supports_rhythm_mask": true,
    "rhythm_mask_grid_units": ["eighth", "eighth_triplet", "..."],
    "supports_remix": true,
    "supports_streaming": true
  },
  "attributes": {
    "note_density": 10,
    "min_polyphony": 10,
    "max_polyphony": 10,
    "min_note_duration": 10,
    "max_note_duration": 10
  },
  "resolution": 480
}
```

`supports_pitch_mask`/`supports_rhythm_mask`/`supports_remix` gate whether this model's checkpoint can accept the `pitch_mask`, `rhythm_mask`, and `remix` controls documented in [API Reference — `TrackPrompt`](api.md#trackprompt) — always check per model rather than assuming any two checkpoints agree.

### `GET /models`

Ids + labels of every model this server has loaded.

```json
{
  "default_model": "yellow_medium",
  "models": [
    {"id": "yellow_medium", "checkpoint": "Metacreation/MIDI-GPT/yellow_medium"},
    {"id": "prism_medium", "checkpoint": "Metacreation/MIDI-GPT/prism_medium"}
  ]
}
```

### `POST /generate`

Generate or infill music. Pass the full score and generation parameters; receive the result score back.

**Request body**

```json
{
  "score": { ... },
  "request": { ... },
  "request_id": "optional-caller-supplied-id",
  "stream": false,
  "model": "yellow_medium"
}
```

- `score` — a `Score` serialised with `Score.to_dict()` (see [Inference API](api.md))
- `request` — a `GenerationRequest` dict (see below)
- `request_id` — optional idempotency/correlation id. If omitted, the server generates one (uuid4 hex) and returns it in the response body and the `X-Request-Id` header. Needed to target this specific generation with `POST /generate/{request_id}/cancel`
- `stream` — `true` for a `text/event-stream` of newly-completed notes as they're generated (see [Streaming](#streaming-stream-true) below). Not yet supported together with `config.num_candidates > 1`
- `model` — which loaded model to run this request against (an id from `GET /models`). Omit to use the server's default model; single-model deployments never need to set this

**Response** (`config.num_candidates == 1`, the default)

```json
{
  "request_id": "a1b2c3d4...",
  "model": "yellow_medium",
  "status": "completed",
  "score": { ... },
  "seed": 12345,
  "timing": {
    "model_forward_s": 0.42,
    "encode_s": 0.01,
    "decode_s": 0.01,
    "gen_count": 4
  },
  "tokens": {
    "context_tokens": 512,
    "generated_tokens": 128,
    "peak_context_tokens": 640,
    "max_context_tokens": 1024,
    "context_utilization": 0.5,
    "tokens_per_second": 304.76,
    "truncated": false
  }
}
```

`status` is `"completed"` or `"cancelled"` (see [`POST /generate/{request_id}/cancel`](#post-generaterequest_idcancel) below) — a cancelled response still carries whatever partial `score` was generated before the cancel was observed. `context_tokens` is the last step's prompt and `generated_tokens` the whole request's output; `peak_context_tokens` is the fullest the model's window got in any step (prompt plus that step's tokens). `truncated: true` means generation stopped because it hit the context budget, not because it reached a natural end — it may be cut off mid-bar/mid-track.

**Response** (`config.num_candidates > 1`)

`num_candidates` independent full generations from the same prompt (one seed each, no accept/reject filtering) — for "give me a few takes, I'll pick one" workflows.

```json
{
  "request_id": "a1b2c3d4...",
  "model": "yellow_medium",
  "status": "completed",
  "base_seed": 12345,
  "candidates": [
    {"score": { ... }, "seed": 12345, "gen_count": 4, "error": null, "truncated": false},
    {"score": null, "seed": 12346, "gen_count": 0, "error": "...", "truncated": false}
  ],
  "summary": {"requested": 2, "succeeded": 1, "failed": 1, "failures": [{"seed": 12346, "reason": "..."}]},
  "timing": { ... },
  "tokens": { ... }
}
```

**Error codes**

| Status | Meaning |
|---|---|
| `400` | Malformed score/request dict, or unknown `model` id |
| `422` | `RequestValidationError` — structurally invalid generation request |
| `500` | Inference failure |
| `503` | Server busy — `max_queue` requests already admitted; retry after the `Retry-After` header (seconds) |

Failed requests (`500`) are appended to a JSONL log for offline analysis — see `MIDIGPT_FAILED_REQUESTS_LOG` (default `failed_requests.jsonl`).

### `POST /generate/{request_id}/cancel`

Stop an in-flight generation. Returns the partial result via the original `/generate` call's response (streaming) or its eventual HTTP response (non-streaming), with `status: "cancelled"`, instead of the caller having to wait for a natural finish or drop the connection.

```json
{"request_id": "a1b2c3d4...", "status": "cancelling"}
```

`404` if `request_id` has no matching in-flight request (already finished, already cancelled, or never existed).

---

## Streaming (`stream: true`)

Instead of one JSON response at the end, `POST /generate` with `"stream": true` returns a `text/event-stream` of newly-completed notes as they're generated, ending with exactly one terminal event carrying the same response shape a non-streaming call would have returned:

```
data: {"type": "notes", "notes": [...]}

data: {"type": "notes", "notes": [...]}

data: {"type": "done", "response": { ...same shape as the non-streaming response... }}
```

`type` is one of `"notes"` (a batch of newly-generated notes), `"done"` (completed normally), `"cancelled"` (stopped via the cancel endpoint or a dropped connection), or `"error"` (`{"type": "error", "request_id": ..., "error": "..."}`). Scoped to the single-candidate path — `stream: true` with `config.num_candidates > 1` returns `400`.

`"notes"` frames, and `"progress"` frames sent every 32 generated tokens between them, carry a `progress` object:

| Field | Meaning |
|---|---|
| `generated_tokens` | tokens generated so far in the whole request (every step and attempt) |
| `step_generated_tokens` | tokens generated in the current step |
| `context_tokens` | the current step's prompt |
| `max_context_tokens` | the model's context length |
| `model_forward_s` | time in the model so far |
| `attempt` | 1-based retry attempt of the current step |

`context_tokens + step_generated_tokens` is the model's window in use; it never exceeds `max_context_tokens`. A request longer than the window runs in several steps, each with a new prompt, so `generated_tokens` keeps growing past it.

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

Each model gets its own semaphore, sized by that model's `max_parallel` (default 1 — fully serialized). Two requests against **different** models never wait on each other at all; within one model, calls overlap up to `max_parallel` — each overlapping call is still a full sequential decode sharing the box's CPU/GPU via the OS scheduler and torch's own intra-op thread pool, not a throughput multiplier, so raise it only after benchmarking on the target machine.

On top of per-model concurrency, `max_queue` bounds how many requests may be admitted (queued + running) across **all** models at once — once that many are admitted, further `POST /generate` calls get an immediate `503` (with `Retry-After`) instead of hanging on an unbounded wait, mirroring Ollama's `OLLAMA_MAX_QUEUE` backpressure. `GET /health` and `GET /info` remain responsive throughout regardless of queue depth.
