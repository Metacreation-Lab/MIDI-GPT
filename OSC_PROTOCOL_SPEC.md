# MIDI-GPT Max MSP ↔ Server OSC Communication Protocol

**Version:** 0.1-draft
**Transport:** OSC over UDP (bidirectional)
**Default ports:** Max → Server `7400`, Server → Max `7401`

---

## 0. System Context

### What this system does

MIDI-GPT is a deep learning Transformer trained on MIDI music. It generates music token by token, conditioned on a surrounding musical context (other tracks, surrounding bars). This protocol connects it to a **live performance system** in Max MSP.

The scenario is: a musician (or multiple musicians) is performing live. Their notes are captured in Max MSP. A generative AI agent — powered by MIDI-GPT — listens to the performance and continuously generates one additional musical voice (the *agent track*), always staying a configurable number of bars ahead of playback. The result is a human-AI co-performance where the AI agent harmonically and rhythmically responds to the human input in real time.

### Roles

| Component | Role |
|-----------|------|
| **Max MSP** | Performance environment. Captures live MIDI notes, drives the bar clock, hosts the user interface (knobs, sliders), and plays back the generated agent output. |
| **MIDI-GPT Server** | Stateful generation backend. Maintains the full `midi::Piece` (all tracks, all bars), manages the rolling context window, runs the model, and owns the agent track. |
| **OSC (this protocol)** | Thin communication layer between the two. Max pushes notes and parameters; the server pushes generated bars back. |

### Key constraints

- The model sees a **rolling window** of up to `model_dim` bars (typically 4–8). Older bars are discarded from the active context, but the full piece history is retained on the server for record-keeping.
- The model is **tempo-agnostic**: it does not see absolute timestamps or BPM. Musical time is quantized into subdivisions per beat (default: 12 per quarter note = 48 steps in 4/4).
- Generation is **always one bar at a time**, for the agent track only, conditioned on all other tracks within the context window.
- The server decides *when* to generate based on the lookahead setting. Max decides *what* is played by the humans and *when* bars end.
- All musical state (piece, parameters, track layout) lives on the server. Max is stateless with respect to the model — it only needs to know what notes to play back.

---

## 1. Overview

The server maintains the complete musical state: the current `midi::Piece`, all track definitions, generation parameters, and the rolling context window. Max sends notes as they happen, advances the bar clock, and updates parameters via OSC. The server generates the agent track one bar ahead of playback and pushes the result back to Max.

```
Max MSP                              MIDI-GPT Server
  │                                       │
  │── /session/init ──────────────────────▶│  reset state
  │── /track/create ─────────────────────▶│  define tracks
  │── /param/set ──────────────────────── ▶│  set defaults
  │── /session/start ─────────────────────▶│  begin
  │                                       │
  │ (music plays)                         │
  │── /note ────────────────────────────── ▶│  live notes land here
  │── /bar/end ──────────────────────────▶│  bar boundary
  │◀── /generated/bar ─────────────────── │  server sends generated bar
  │                                       │
  │── /track/remove ───────────────────── ▶│  mid-session track removal
  │── /param/set ───────────────────────▶│  knob moved → param update
  │── /session/stop ─────────────────────▶│  end session
```

---

## 2. Transport & Connection

| Property | Value |
|----------|-------|
| Protocol | OSC 1.0 over UDP |
| Max → Server port | `7400` (default, configurable in server config) |
| Server → Max | Replies to the **UDP source address and port** of incoming packets |
| Message encoding | Standard OSC type tags |

The server replies to the UDP source address (`ip:port`) of each incoming packet. Max does not need to declare a return address — it just needs to bind a consistent UDP port before sending `/session/init`. This means Max can run on any IP without server reconfiguration. All server→Max messages go to the address of the most recently received packet from Max.

---

## 3. Conventions

### 3.1 Address Namespace

All messages are prefixed with `/midigpt/`.

### 3.2 Types

| OSC type | Used for |
|----------|---------|
| `i` (int32) | IDs, MIDI values, enum levels, bar/track indices |
| `f` (float32) | Normalized positions [0.0–1.0], temperature, ratios |
| `s` (string) | Names, status codes, error messages |
| `T`/`F` | Boolean flags |

### 3.3 Normalized Time

Note positions and durations within a bar are sent as **float32 normalized values** in `[0.0, 1.0]`:
- `0.0` = bar start
- `1.0` = bar end (exclusive)

The server maps these to the model's quantized steps using its configured resolution (default: `resolution=12`, i.e. 48 steps per 4/4 bar). This decouples Max's timing from the model's internal quantization.

### 3.4 Track IDs

`track_id` is an **integer assigned by Max** at creation time. It persists for the session. The server uses this as the index into the `midi::Piece.tracks` array — so IDs should be assigned sequentially starting from `0`, or the server must maintain a `track_id → piece_index` mapping (server-side implementation detail).

### 3.5 Bar Index

`bar_index` is a **monotonically increasing integer** starting at `0`, incremented by Max with each `/bar/end` message. The server uses this to grow the piece and manage the rolling context window.

---

## 4. Session Lifecycle

```
UNINITIALIZED
     │
     ▼ /session/init
INITIALIZING
     │ /track/create (one or more)
     │ /param/set    (zero or more)
     │
     ▼ /session/start
RUNNING
     │ /note, /bar/end, /param/set, /track/create, /track/remove
     │ ◀── /generated/bar (server → Max)
     │
     ▼ /session/stop
STOPPED
```

During `INITIALIZING`, the server accepts only session/track/param messages. `/note` and `/bar/end` are rejected (server sends `/midigpt/error`).

During `RUNNING`, `/track/create` and `/track/remove` are allowed between bar boundaries (i.e., after `/bar/end` and before the next `/note`).

---

## 5. Messages: Max → Server

### 5.1 Session Management

---

#### `/midigpt/session/init`

```
/midigpt/session/init  s:session_name
```

Resets all server state. Begins initialization phase. The server captures the UDP source address of this packet and uses it as the return address for all subsequent server→Max messages.

| Argument | Type | Description |
|----------|------|-------------|
| `session_name` | `s` | Human-readable label (for logging) |

Server responds with `/midigpt/session/ready` on success.

---

#### `/midigpt/session/start`

```
/midigpt/session/start
```

Transitions from INITIALIZING → RUNNING. Server begins waiting for notes and bar advances. At least one non-agent track and exactly one agent track must have been created.

Server responds with `/midigpt/session/started` or `/midigpt/error`.

---

#### `/midigpt/session/stop`

```
/midigpt/session/stop
```

Gracefully terminates the session. Server flushes any in-progress generation, transitions to STOPPED.

---

### 5.2 Track Management

---

#### `/midigpt/track/create`

```
/midigpt/track/create  i:track_id  i:instrument  i:track_type  i:is_agent
```

Defines a track. Can be sent during INITIALIZING or RUNNING (between bar boundaries).

| Argument | Type | Range | Description |
|----------|------|-------|-------------|
| `track_id` | `i` | `≥ 0` | Unique track identifier assigned by Max |
| `instrument` | `i` | `0–139` | GM instrument number (see note below) |
| `track_type` | `i` | `10`=STANDARD, `11`=DRUM | Mirrors `midi::TRACK_TYPE` |
| `is_agent` | `i` | `0` or `1` | `1` = this is the generated track; only one agent track allowed |

**Instrument values:** 0–127 = General MIDI melodic instruments; 128–139 = individual drum instruments.

When called during RUNNING, the server inserts an empty track with `bar_index` empty bars to align with the current piece length.

---

#### `/midigpt/track/remove`

```
/midigpt/track/remove  i:track_id
```

Removes a track from the piece. Cannot remove the agent track while session is RUNNING (server returns error). Can be called during RUNNING between bar boundaries.

---

#### `/midigpt/track/set_ignore`

```
/midigpt/track/set_ignore  i:track_id  i:ignored
```

Marks a conditioning track as ignored (`1`) or active (`0`) for future generation calls. Ignored tracks remain in the piece but are excluded from the model's conditioning context (`midi::StatusTrack.ignore = true`).

---

### 5.3 Note Input

---

#### `/midigpt/note`

```
/midigpt/note  i:track_id  i:pitch  i:velocity  f:onset  f:duration  i:bar_index
```

Sends a complete note event to the server. Notes for the agent track (`is_agent=1`) are ignored — the server owns that track's content.

| Argument | Type | Range | Description |
|----------|------|-------|-------------|
| `track_id` | `i` | `≥ 0` | Which track this note belongs to |
| `pitch` | `i` | `0–127` | MIDI pitch |
| `velocity` | `i` | `1–127` | MIDI velocity (note-on only; note-offs are implicit via `duration`) |
| `onset` | `f` | `[0.0, 1.0)` | Normalized position within bar |
| `duration` | `f` | `(0.0, 1.0]` | Normalized duration; clamped to bar end if `onset + duration > 1.0` |
| `bar_index` | `i` | `≥ 0` | Which bar this note belongs to |

Notes can arrive out of order (Max may buffer). The server accumulates notes per `(track_id, bar_index)` until that bar is finalized by `/bar/end`.

---

### 5.4 Bar Control

---

#### `/midigpt/bar/end`

```
/midigpt/bar/end  i:bar_index  i:ts_numerator  i:ts_denominator
```

Signals that bar `bar_index` has ended and all notes for that bar have been sent. This is the bar clock. The server finalizes the bar, extends the piece, and triggers generation if the lookahead threshold is reached.

| Argument | Type | Description |
|----------|------|-------------|
| `bar_index` | `i` | The bar that just ended |
| `ts_numerator` | `i` | Time signature numerator (e.g. `4`) |
| `ts_denominator` | `i` | Time signature denominator (e.g. `4`) |

Time signature applies to all tracks for this bar (model constraint: time sig is per-bar, not per-track).

---

### 5.5 Parameter Updates

All parameter updates take effect on the **next generation call**. Parameters are persistent unless sent via the one-shot variant.

---

#### `/midigpt/param/set`

```
/midigpt/param/set  s:param_name  <value>
```

Persistent parameter update. Value type depends on parameter (see §7).

---

#### `/midigpt/param/set_once`

```
/midigpt/param/set_once  s:param_name  <value>
```

One-shot parameter update. Applied to the next generation call only, then reverted to the current persistent value. Use this for volatile parameters like `temperature` where forgetting to reset could corrupt generation.

---

#### `/midigpt/param/reset`

```
/midigpt/param/reset  s:param_name
```

Resets a parameter to its default value.

---

#### `/midigpt/param/reset_all`

```
/midigpt/param/reset_all
```

Resets all parameters to defaults.

---

#### `/midigpt/track/param/set`

```
/midigpt/track/param/set  i:track_id  s:param_name  <value>
```

Sets a per-track parameter on a specific track. Agent-track parameters are the primary generation controls. Parameters on conditioning tracks affect how much the server weighs them (currently: `ignore` only; future: per-conditioning-track temperature).

---

#### `/midigpt/track/param/set_once`

```
/midigpt/track/param/set_once  i:track_id  s:param_name  <value>
```

One-shot version for per-track parameters.

---

## 6. Messages: Server → Max

### 6.1 Session Responses

| Message | Arguments | Sent when |
|---------|-----------|-----------|
| `/midigpt/session/ready` | — | `/session/init` accepted |
| `/midigpt/session/started` | — | `/session/start` accepted |
| `/midigpt/session/stopped` | — | `/session/stop` complete |

---

### 6.2 Generated Notes

The server pushes notes back to Max as a **flat list**, framed by open/close delimiters. The batch is agnostic to granularity — it may contain a single note, a handful of notes, or a full bar's worth. Max treats them all the same: each note carries enough information (`track_id`, `bar_index`, `onset`) to be placed in the right position without any additional context.

All messages within a batch for the same `(track_id, bar_index)` are sent contiguously without interleaving.

---

#### `/midigpt/generated/open`

```
/midigpt/generated/open  i:track_id  i:bar_index  i:note_count
```

Signals the start of a note batch. `note_count` tells Max how many `/generated/note` messages follow, so it can pre-allocate if needed. A batch with `note_count=0` is valid (silent bar — no notes to play).

---

#### `/midigpt/generated/note`

```
/midigpt/generated/note  i:track_id  i:bar_index  i:pitch  i:velocity  f:onset  f:duration
```

A single generated note. Same field semantics as the incoming `/midigpt/note` message — symmetric by design.

| Argument | Type | Range | Description |
|----------|------|-------|-------------|
| `track_id` | `i` | `≥ 0` | Agent track ID |
| `bar_index` | `i` | `≥ 0` | Bar this note belongs to |
| `pitch` | `i` | `0–127` | MIDI pitch |
| `velocity` | `i` | `1–127` | MIDI velocity |
| `onset` | `f` | `[0.0, 1.0)` | Normalized onset within bar |
| `duration` | `f` | `(0.0, 1.0]` | Normalized duration |

---

#### `/midigpt/generated/close`

```
/midigpt/generated/close  i:track_id  i:bar_index
```

Signals end of batch. Max can safely schedule or commit all notes from this batch once it receives this message.

---

#### `/midigpt/generated/features`

```
/midigpt/generated/features  i:track_id  i:bar_index
    f:note_density  f:mean_pitch  f:mean_velocity  i:max_polyphony  f:mean_duration
```

Musical features of the generated batch, sent immediately after `/generated/close`. Useful for driving visualizations or reactive UI in Max. Omitted if the batch is empty (silent bar).

| Feature | Type | Description |
|---------|------|-------------|
| `note_density` | `f` | Notes per beat |
| `mean_pitch` | `f` | Average MIDI pitch |
| `mean_velocity` | `f` | Average MIDI velocity |
| `max_polyphony` | `i` | Maximum simultaneous notes |
| `mean_duration` | `f` | Mean note duration (normalized) |

---

### 6.3 Status & Errors

---

#### `/midigpt/status`

```
/midigpt/status  s:state
```

Periodic server heartbeat. `state` is one of: `"idle"`, `"generating"`, `"ready"`.

---

#### `/midigpt/error`

```
/midigpt/error  i:code  s:message
```

| Code | Meaning |
|------|---------|
| `1` | Invalid session state (e.g. `/note` before `/session/start`) |
| `2` | Unknown track ID |
| `3` | Duplicate track ID |
| `4` | Invalid parameter name or value |
| `5` | Generation failed (model error) |
| `6` | Agent track missing |
| `7` | Multiple agent tracks defined |
| `8` | Note sent for agent track (ignored with warning) |

---

## 7. Parameter Reference

### 7.1 Global Parameters (`/midigpt/param/set`)

These correspond to `midi::HyperParam` fields.

| Name | Type | Default | Range | One-shot recommended | Description |
|------|------|---------|-------|----------------------|-------------|
| `lookahead_bars` | `i` | `2` | `1–8` | No | How many bars ahead the server generates |
| `temperature` | `f` | `1.0` | `0.5–2.0` | **Yes** | Global generation entropy |
| `model_dim` | `i` | `4` | `1–8` | No | Model context window in bars |
| `mask_top_k` | `f` | `0.0` | `0.0–1.0` | No | Probability of masking top-k tokens |
| `sampling_seed` | `i` | `-1` | any | **Yes** | RNG seed (-1 = random) |

### 7.2 Per-Track Parameters (`/midigpt/track/param/set`)

These correspond to `midi::StatusTrack` fields. Most are only meaningful on the agent track; `ignore` applies to conditioning tracks.

#### Agent track (melodic, `track_type=10`)

| Name | Type | Default | Range | One-shot recommended | Description |
|------|------|---------|-------|----------------------|-------------|
| `temperature` | `f` | `1.0` | `0.5–2.0` | **Yes** | Per-track entropy override |
| `polyphony_hard_limit` | `i` | `0` | `0–6` | No | Max simultaneous notes (0 = no limit) |
| `min_polyphony_q` | `i` | `0` | `0–6` | No | Min polyphony quantile (0=ANY, 1–6 = levels) |
| `max_polyphony_q` | `i` | `0` | `0–6` | No | Max polyphony quantile |
| `min_note_duration_q` | `i` | `0` | `0–6` | No | Min note duration quantile (0=ANY, 1=32nd…6=whole) |
| `max_note_duration_q` | `i` | `0` | `0–6` | No | Max note duration quantile |
| `min_pitch` | `i` | `0` | `0–127` | No | Minimum MIDI pitch |
| `max_pitch` | `i` | `127` | `0–127` | No | Maximum MIDI pitch |
| `key_signature` | `i` | `0` | `0–24` | No | 0=ANY, 1=C maj … 24=B min (see proto enum) |
| `onset_density` | `i` | `0` | `0–16` | No | Bar-level onset density (0=ANY) |
| `onset_polyphony_min` | `i` | `0` | `0–6` | No | Bar-level min onset polyphony |
| `onset_polyphony_max` | `i` | `0` | `0–6` | No | Bar-level max onset polyphony |

#### Agent track (drum, `track_type=11`)

| Name | Type | Default | Range | One-shot recommended | Description |
|------|------|---------|-------|----------------------|-------------|
| `temperature` | `f` | `1.0` | `0.5–2.0` | **Yes** | Per-track entropy override |
| `density` | `i` | `0` | `0–10` | No | Drum density level (0=ANY, 1–10) |
| `polyphony_hard_limit` | `i` | `0` | `0–6` | No | Max simultaneous hits |

#### Conditioning tracks (any)

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ignore` | `i` | `0` | `1` = exclude track from model conditioning |

---

## 8. Generation Triggering Logic (Server-Side Reference)

*This section describes server behavior for protocol completeness — not an OSC specification.*

The server generates bar `B` for the agent track when:
- Bar `B - lookahead_bars` has been finalized (its `/bar/end` has arrived)
- All conditioning tracks have notes for bars `max(0, B - model_dim + 1)` through `B - 1` (or earlier bars if some are empty/silent)
- No generation for bar `B` is already in progress

The server constructs:
- `midi::Piece`: rolling window of the last `model_dim` bars across all tracks
- `midi::Status`: agent track's `selected_bars = [False, ..., False, True]` (only bar `B` selected); conditioning tracks have all `False`
- `midi::HyperParam`: current global parameters

On completion, the generated bar `B` is appended to the agent track in the piece, and the `/generated/bar/*` sequence is sent to Max.

---

## 9. Example Flow

### Session initialization

```
Max → Server:  /midigpt/session/init "live-performance" 7401
Server → Max:  /midigpt/session/ready

Max → Server:  /midigpt/track/create 0 0 10 0     # track 0: piano, melodic, conditioning
Max → Server:  /midigpt/track/create 1 25 10 0    # track 1: acoustic guitar, conditioning
Max → Server:  /midigpt/track/create 2 0 10 1     # track 2: piano, AGENT

Max → Server:  /midigpt/param/set "lookahead_bars" 2
Max → Server:  /midigpt/param/set "model_dim" 4
Max → Server:  /midigpt/param/set "temperature" 1.0
Max → Server:  /midigpt/track/param/set 2 "polyphony_hard_limit" 4
Max → Server:  /midigpt/track/param/set 2 "min_pitch" 48
Max → Server:  /midigpt/track/param/set 2 "max_pitch" 84

Max → Server:  /midigpt/session/start
Server → Max:  /midigpt/session/started
```

### Running — bar 0

```
# Notes arrive in real-time on tracks 0 and 1
Max → Server:  /midigpt/note 0 60 80 0.0 0.25 0
Max → Server:  /midigpt/note 0 64 75 0.25 0.25 0
Max → Server:  /midigpt/note 1 55 90 0.0 1.0 0
...
Max → Server:  /midigpt/bar/end 0 4 4
# Bar 0 finalized. lookahead=2, so generation of bar 2 starts when bar 0 ends.
# Server cannot generate yet — needs bar 1 first for bar 2 generation.
```

### Running — bar 1

```
Max → Server:  /midigpt/note 0 62 70 0.0 0.5 1
...
Max → Server:  /midigpt/bar/end 1 4 4
# Bar 1 finalized. Server now has bars 0+1 → triggers generation of bar 2 (agent track).
```

### Generated notes response

```
Server → Max:  /midigpt/generated/open 2 2 5
Server → Max:  /midigpt/generated/note 2 2 65 82 0.0 0.25
Server → Max:  /midigpt/generated/note 2 2 69 78 0.25 0.25
Server → Max:  /midigpt/generated/note 2 2 72 90 0.0 0.5
Server → Max:  /midigpt/generated/note 2 2 65 85 0.5 0.25
Server → Max:  /midigpt/generated/note 2 2 67 80 0.75 0.25
Server → Max:  /midigpt/generated/close 2 2
Server → Max:  /midigpt/generated/features 2 2 2.0 67.4 83.0 3 0.28
```

### Parameter update mid-session (one-shot temperature spike)

```
# User turns a knob — increase entropy just for the next bar
Max → Server:  /midigpt/param/set_once "temperature" 1.6
# Next generation uses 1.6, then reverts to 1.0
```

### Track removal mid-session

```
# Guitar player leaves — remove track 1
Max → Server:  /midigpt/bar/end 7 4 4     # bar boundary
Max → Server:  /midigpt/track/remove 1
# Server removes track from piece; future generation ignores it
```

---

## 10. Open Questions / Future Extensions

| Topic | Note |
|-------|------|
| **Pre-recorded loops** | A `/track/load_bar` message could inject pre-computed bars into a conditioning track instead of live notes — to be specified. |
| **Multiple agent tracks** | Currently one agent track only. Multi-agent (e.g. co-generate drums + melody) would require coordination of bar scheduling. |
| **MIDI channel binding** | Max could optionally announce a MIDI channel per track (`/track/bind_channel`) for automatic note routing — useful if Max uses hardware MIDI. |
| **Piece persistence** | A `/session/save` / `/session/load` message pair could dump/restore the full `midi::Piece` JSON for session continuity. |
| **Time signature changes** | Currently per-bar in `/bar/end`; bar-level `StatusBar` fields (onset_density, onset_polyphony) could be exposed as per-bar param messages. |
| **Velocity normalization** | Max sends raw MIDI velocity (1–127); server quantizes to 32 levels per the encoder — no action needed but worth documenting. |
