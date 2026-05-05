# MIDI-GPT Max MSP Developer Guide

**For:** Max MSP developer building the performance interface  
**Server version:** 0.1  
**Protocol:** OSC over UDP — see `OSC_PROTOCOL_SPEC.md` for the full reference

---

## 1. What You Are Building

A Max patch that does three things:

1. **Captures live MIDI** from one or more performers and forwards every note to the MIDI-GPT server.
2. **Drives a bar clock** — tells the server when each bar ends.
3. **Receives generated notes** from the server and schedules them for MIDI playback.

Everything musical lives on the server. Max is stateless with respect to the AI — it just pushes notes in and plays notes out.

---

## 2. Network Setup

```
Max MSP  ──UDP 7400──▶  MIDI-GPT Server  (your machine, default port 7400)
Max MSP  ◀──UDP 7401──  MIDI-GPT Server  (server replies to source IP:port)
```

The server replies to **whatever source address Max sends from**. Max must bind a fixed UDP port (e.g. 7401) and use it for all outgoing packets, so the server knows where to reply.

**Max objects:**
- `udpsend 127.0.0.1 7400` — send to server
- `udpreceive 7401` — receive from server (or use `OSC-route` on top of `udpreceive`)

If the server is on a different machine, replace `127.0.0.1` with its IP.

**OSC libraries in Max:**
- CNMAT `OpenSoundControl` externals (recommended — `OSC-route`)
- Or the built-in `udpsend`/`udpreceive` with manual byte packing (more work)

---

## 3. Session Lifecycle — What Max Must Do

Walk through this sequence every time you start a performance.

### Step 1: Init

```
send:  /midigpt/session/init  s:"live-set-1"
wait:  /midigpt/session/ready
```

This resets the server completely. Safe to call at any time — it wipes all tracks and state.

### Step 2: Create Tracks

Send one message per track. Do this **before** `/session/start`.

```
send:  /midigpt/track/create  i:0  i:0  i:10  i:0    ← piano, conditioning
send:  /midigpt/track/create  i:1  i:32  i:10  i:0   ← bass, conditioning
send:  /midigpt/track/create  i:2  i:0  i:10  i:1    ← piano, AGENT (is_agent=1)
```

Arguments: `track_id  instrument  track_type  is_agent`

- `track_type` 10 = melodic instrument, 11 = drums
- `instrument` = GM program number (0–127). See §7 of the protocol spec for the full table.
- Exactly **one** track must have `is_agent=1`. This is the track the AI generates.
- All other tracks are conditioning (the humans).
- `track_id` is an integer you assign. Keep them in a `coll` or `dict`.

### Step 3: Set Parameters (optional)

```
send:  /midigpt/param/set  s:"lookahead_bars"  i:2
send:  /midigpt/param/set  s:"model_dim"  i:4
send:  /midigpt/param/set  s:"temperature"  f:1.0
send:  /midigpt/param/set  s:"buffer_bars"  i:4
```

Sensible defaults are already set on the server. Only override if you need different values. See §7 of the protocol spec for the full parameter list.

### Step 4: Start

```
send:  /midigpt/session/start
wait:  /midigpt/session/started
```

After this the server is RUNNING. Bar clock and notes can now flow.

### Step 5: Stop

```
send:  /midigpt/session/stop
wait:  /midigpt/session/stopped
```

---

## 4. The Bar Clock — Critical

The bar clock is the backbone of the system. **The server generates based on bar boundaries, not on time.** Max must send `/midigpt/bar/end` at the exact moment each bar ends.

```
send:  /midigpt/bar/end  i:<bar_index>  i:<ts_num>  i:<ts_den>
```

- `bar_index` starts at 0, increments by 1 for every bar, forever.
- `ts_num` / `ts_den` = time signature for that bar (e.g. 4 and 4 for 4/4).

**How to drive this in Max:**

Use `transport` to detect bar boundaries:

```
[transport]  →  [sel 1]  →  [counter]  → bar_index
                          →  /midigpt/bar/end  bar_index 4 4
```

Or use `metro` locked to the transport's bar period. Either way:

- Fire `bar/end` **after** all notes for that bar have been sent.
- Do not fire `bar/end` for a bar until all notes that landed in it have been forwarded.
- Notes can arrive slightly after the bar end (network jitter) — the server accumulates notes per `(track_id, bar_index)` so late arrivals within a bar are fine as long as `bar/end` hasn't been sent yet.

**Practical note:** If you use `transport` with `[timepoint 1 0 0]` to fire at bar boundaries, make sure you flush all MIDI input buffers *before* sending `bar/end`.

---

## 5. Forwarding Live MIDI Notes

For every note-on/note-off pair from a human performer, send one `/midigpt/note` message to the server:

```
send:  /midigpt/note  i:track_id  i:pitch  i:velocity  f:onset  f:duration  i:bar_index
```

- `onset` = normalized position within the bar: `beat_offset_in_bar / beats_per_bar`, range `[0.0, 1.0)`.
- `duration` = normalized duration: `note_duration_in_beats / beats_per_bar`, range `(0.0, 1.0]`.
- `bar_index` = which bar this note belongs to (same counter as your `bar/end` messages).
- `velocity` = raw MIDI velocity, 1–127.

**Do NOT send notes for the agent track** (`is_agent=1`). The server owns that track. If you accidentally forward notes to it, the server ignores them with error code 8.

**Max patch sketch:**

```
[midiin] → [midiparse]
               │pitch  velocity  (note-on only)
               │
           [capture onset time vs. bar start]
           [capture note-off to compute duration]
               │
           [prepend /midigpt/note]
               │
           [udpsend 127.0.0.1 7400]
```

### Computing onset and duration

The server is **tempo-agnostic** — it only sees normalized values in `[0.0, 1.0]` relative to the bar. Your patch must convert from ticks/ms to normalized beat position:

```
onset    = (note_on_tick  - bar_start_tick) / this_bar_ticks
duration = (note_off_tick - note_on_tick)   / this_bar_ticks
```

where `this_bar_ticks` is the length of the bar the note belongs to (depends on *that bar's* time signature, not a global constant). Since Max drives the bar clock, it knows exactly when each bar started and how long it is.

Cap `onset + duration` at 1.0 (notes cannot extend past the bar end in this message).

---

## 6. Receiving Generated Notes

The server sends notes back as a bracketed batch. **Batch is per `(track_id, bar_index)`.**

```
receive:  /midigpt/generated/open   i:track_id  i:bar_index  i:note_count
receive:  /midigpt/generated/note   i:track_id  i:bar_index  i:pitch  i:velocity  f:onset  f:duration
... (note_count times)
receive:  /midigpt/generated/close  i:track_id  i:bar_index
receive:  /midigpt/generated/features  i:track_id  i:bar_index  f:note_density  f:mean_pitch  f:mean_velocity  i:max_polyphony  f:mean_duration
```

### What you must do with these messages

1. **On `open`**: Allocate a buffer for this `(track_id, bar_index)` pair. `note_count` tells you how many notes are coming (0 = silent bar, still need to handle the close).

2. **On `note`**: Accumulate into the buffer. Do NOT play immediately — the notes may be for a bar that's 2+ bars in the future (because of lookahead).

3. **On `close`**: The batch is complete. Schedule the notes for playback at the correct future time.

4. **On `features`**: Optional. Use `note_density`, `mean_pitch`, etc. to drive UI elements if you want.

### Scheduling playback

**Do not compute bar start time from `bar_index` alone.** Bar start ticks are not simply `bar_index * ticks_per_bar` — that formula only holds if the time signature never changes. Max drives the bar clock, so Max already knows when each bar started. Use that directly.

**Recommended approach — flush-on-bar-end:**

1. Store each received batch in a `dict` (or `coll`) keyed by `bar_index`.
2. When your bar-end logic fires for bar `N` (i.e., bar `N` has just ended and bar `N+1` is starting *right now*):
   - Look up the buffered notes for bar `N+1`.
   - For each note, fire a `delay` of `onset * bar_duration_ms` milliseconds from *this moment*.
   - `bar_duration_ms` = the duration of bar `N+1`, computed from *its* time signature and current BPM.

This works because `lookahead_bars ≥ 1`, so the server always delivers notes for bar `N+1` before bar `N` ends.

```
bar_duration_ms = (ts_num / ts_den) * 4.0 * (60000.0 / bpm)
```

Note that `ts_num` and `ts_den` come from the `/bar/end` message Max sent for that bar, so Max already has them. BPM is whatever Max's `transport` is running at.

**MIDI playback:**

```
[note buffer] → [delay onset_ms] → [makenote pitch velocity duration_ms] → [noteout channel]
```

`duration_ms = duration * bar_duration_ms`

---

## 7. Parameter Controls (Knobs / Sliders)

Map knobs/sliders to `/midigpt/param/set` messages. These take effect on the **next generation call**.

| UI control | OSC message |
|-----------|-------------|
| Temperature knob (0.5–2.0) | `/midigpt/param/set "temperature" f:value` |
| Lookahead (1–8 bars) | `/midigpt/param/set "lookahead_bars" i:value` |
| Context window (1–8 bars) | `/midigpt/param/set "model_dim" i:value` |
| Buffer (2–64 bars) | `/midigpt/param/set "buffer_bars" i:value` |

For volatile controls (like a "randomize" button), use `set_once` so the change only applies once:

```
send:  /midigpt/param/set_once  s:"temperature"  f:1.8
```

### Per-agent-track controls

```
send:  /midigpt/track/param/set  i:<agent_track_id>  s:"polyphony_hard_limit"  i:3
send:  /midigpt/track/param/set  i:<agent_track_id>  s:"min_pitch"  i:48
send:  /midigpt/track/param/set  i:<agent_track_id>  s:"max_pitch"  i:84
```

---

## 8. Status and Error Handling

### Status updates

```
receive:  /midigpt/status  s:state
```

`state` is `"generating"` (model running), `"ready"` (idle, waiting for bars), or `"idle"`.  
Use this to drive a simple connection indicator (green/yellow light).

### Error messages

```
receive:  /midigpt/error  i:code  s:message
```

| Code | What it means | Action |
|------|--------------|--------|
| 1 | Message sent in wrong state | Check your session flow |
| 2 | Unknown track_id | You referenced a track that wasn't created |
| 3 | Duplicate track_id | You called `/track/create` with a taken ID |
| 4 | Invalid parameter | Check name spelling and value range |
| 5 | Generation failed | Model error — retry or check server logs |
| 6 | No agent track | Call `/track/create` with `is_agent=1` first |
| 7 | Multiple agent tracks | Only one `is_agent=1` track allowed |
| 8 | Note for agent track | Don't forward notes to the AI track |

Print all errors to the Max console (`[print midigpt-error]`).

---

## 9. Mid-Session Track Changes

You can add or remove conditioning tracks while the session is RUNNING, but only **between bar boundaries** — i.e., after `/bar/end N` and before the first `/note` for bar `N+1`.

```
# Guitarist leaves
send:  /midigpt/track/remove  i:1

# New player joins
send:  /midigpt/track/create  i:3  i:41  i:10  i:0   ← violin, conditioning
```

To temporarily mute a track without removing it:

```
send:  /midigpt/track/set_ignore  i:1  i:1    ← ignore track 1
send:  /midigpt/track/set_ignore  i:1  i:0    ← re-enable track 1
```

---

## 10. Minimal Patch Checklist

Use this as a build order:

- [ ] UDP send/receive objects configured (ports 7400 out, 7401 in)
- [ ] OSC routing for all `/midigpt/...` receive addresses
- [ ] Session init/start/stop buttons wired
- [ ] Track creation messages (hardcode for the first version)
- [ ] Bar counter driving `bar/end` at every bar boundary
- [ ] MIDI input capture + normalize onset/duration + forward `/note`
- [ ] Generated note accumulator per `(track_id, bar_index)`
- [ ] Playback scheduler (flush at bar boundary)
- [ ] MIDI output for agent track notes
- [ ] Error printout
- [ ] Status LED

---

## 11. Complete Example Message Flow

```
── Session init ─────────────────────────────────────────────────────────────
Max → /midigpt/session/init "live-set"
Srv → /midigpt/session/ready

Max → /midigpt/track/create 0 0 10 0          piano (conditioning)
Max → /midigpt/track/create 1 0 10 1          piano (AGENT)
Max → /midigpt/param/set "buffer_bars" 4
Max → /midigpt/param/set "lookahead_bars" 2
Max → /midigpt/session/start
Srv → /midigpt/session/started

── Bar 0 ─────────────────────────────────────────────────────────────────────
Max → /midigpt/note 0 60 80 0.0 0.25 0
Max → /midigpt/note 0 64 70 0.25 0.25 0
Max → /midigpt/note 0 67 75 0.5 0.5 0
Max → /midigpt/bar/end 0 4 4
  (server buffering — not generating yet, buffer_bars=4)

── Bars 1, 2, 3 — same pattern, bar clock advances ──────────────────────────
Max → /midigpt/bar/end 1 4 4
Max → /midigpt/bar/end 2 4 4
Max → /midigpt/bar/end 3 4 4
  (buffer fills up — server begins generating at bar/end 3 for target bar 5)

Srv → /midigpt/status "generating"

── Bar 4 ─────────────────────────────────────────────────────────────────────
Max → /midigpt/note 0 62 72 0.0 0.5 4
...
  (while bar 4 plays, server finishes generating bar 5)

Srv → /midigpt/generated/open    1 5 4
Srv → /midigpt/generated/note    1 5 65 82 0.0  0.25
Srv → /midigpt/generated/note    1 5 69 78 0.25 0.25
Srv → /midigpt/generated/note    1 5 72 90 0.5  0.25
Srv → /midigpt/generated/note    1 5 64 80 0.75 0.25
Srv → /midigpt/generated/close   1 5
Srv → /midigpt/generated/features 1 5 1.0 67.5 82.5 2 0.25
Srv → /midigpt/status "ready"

  Max buffers bar 5 notes — schedules them to play when bar 5 starts

Max → /midigpt/bar/end 4 4 4
  (bar 4 ends → bar 5 starts playing → Max fires buffered notes for bar 5)

── Temperature one-shot (user turns a knob) ────────────────────────────────
Max → /midigpt/param/set_once "temperature" 1.7
  (next generation only uses 1.7, then reverts)

── Stop ─────────────────────────────────────────────────────────────────────
Max → /midigpt/session/stop
Srv → /midigpt/session/stopped
```

---

## 12. Starting the Server

The developer running the server starts it from the terminal:

```bash
# Install (once)
pip install -e ".[osc]"

# Start
midigpt-server --ckpt /path/to/model.pt --port 7400
```

Optional flags:
```
--host 0.0.0.0       bind all interfaces (default)
--buffer 4           override buffer_bars at startup
--lookahead 2        override lookahead_bars at startup
--model_dim 4        override model_dim at startup
--log_level DEBUG    verbose logging
```

The server prints each incoming OSC message and each generation call to stdout. Check there if something seems wrong.
