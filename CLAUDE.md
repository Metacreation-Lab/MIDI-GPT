# MIDI-GPT — Claude Code Guide

## Project Overview

MIDI-GPT is a GPT-2 Transformer for real-time MIDI music generation. It has two main surfaces:

1. **C++/Python core** — tokenizer, model inference, dataset pipeline (scikit-build-core + pybind11)
2. **Real-time OSC server** — Python server (`midigpt-server`) that receives live MIDI from Max MSP via OSC, runs the model, and streams generated notes back

Max MSP is the performance interface. An AI agent (Claude) can build/edit Max patches using **maxpylang** (offline `.maxpat` generation) and **MaxMSP-MCP** (live patch manipulation via Socket.IO).

---

## Repository Layout

```
MIDI-GPT/
├── src/
│   ├── python/midigpt/          # Python package (importable)
│   │   ├── osc_server.py        # Real-time OSC server (MidiGPTServer)
│   │   ├── realtime_gen.py      # Inference helpers (run_inference, build_params)
│   │   └── realtime_state.py    # PieceState — bar/note/track state machine
│   └── cpp/                     # C++ tokenizer + model bindings
├── MaxMSP-MCP-Server/           # MCP server for live Max patch manipulation
│   ├── server.py                # FastMCP server (Socket.IO → Max JS API)
│   └── MaxMSP_Agent/demo.maxpat # Demo Max patch with JS agent
├── python_scripts/              # Standalone scripts (train, eval, dataset)
├── python_scripts_for_testing/  # OSC simulation / stress tests
├── docs/
│   ├── OSC_PROTOCOL_SPEC.md     # Full OSC message reference
│   └── maxmsp_developer_guide.md # Max developer guide (bar clock, note format)
├── tests/                       # pytest test suite
├── .venv/                       # Project virtualenv (Python 3.12)
│   └── bin/python3              # Use this for all Python commands
├── MaxMSP-MCP-Server/.venv/     # Separate venv for MCP server (Python 3.10)
├── pyproject.toml               # Build config, extras: [train] [osc] [dataset]
├── .mcp.json                    # MaxMSP-MCP server config for Claude Code CLI
└── CLAUDE.md                    # This file
```

---

## Python Environment

The project uses **Python 3.12** in `.venv/`.

```bash
# Always use the project venv
.venv/bin/python3 script.py
.venv/bin/pytest tests/

# Install / update deps
.venv/bin/pip install -e ".[osc]"          # core + OSC server
.venv/bin/pip install -e ".[train]"        # + training stack
.venv/bin/pip install -e ".[osc,train]"    # everything
```

**Key installed packages:**
- `midigpt` — this project (editable install)
- `torch`, `transformers`, `accelerate` — ML stack
- `python-osc` (pythonosc) — OSC UDP server
- `maxpylang 0.1.1` — offline Max patch generation
- `pytest` — test runner

The **MaxMSP-MCP-Server** has its own venv at `MaxMSP-MCP-Server/.venv/` (Python 3.10) with `mcp`, `python-socketio`, `fastmcp`.

---

## Build System

The C++ extension is built with CMake via scikit-build-core.

```bash
# First-time build (requires cmake, abseil, protobuf)
pip install -e ".[osc]"

# macOS deps:
brew install cmake abseil protobuf

# Rebuild after C++ changes:
pip install -e ".[osc]" --no-build-isolation

# Skip torch (C++ only, dataset work):
CMAKE_ARGS="-DMIDIGPT_NO_TORCH=ON" pip install -e .

# CMake 4.x compatibility flag (already handled in project):
# CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
```

---

## Running the OSC Server

```bash
# Start server (requires a TorchScript checkpoint)
.venv/bin/midigpt-server --ckpt /path/to/model.pt --port 7400

# Full options:
.venv/bin/midigpt-server \
  --ckpt model.pt \
  --port 7400 \
  --host 0.0.0.0 \
  --buffer 4 \
  --lookahead 2 \
  --model_dim 4 \
  --log_level DEBUG
```

The server listens on UDP 7400 and replies to the source IP/port of each packet. Max MSP binds port 7401 and uses `udpsend 127.0.0.1 7400` / `udpreceive 7401`.

### State machine
`UNINITIALIZED → INITIALIZING → RUNNING → STOPPED`

Session must follow: `init → track/create (×N) → start → [bar/end + notes loop] → stop`

---

## MaxMSP-MCP Server

The MCP server at `MaxMSP-MCP-Server/server.py` lets Claude manipulate a live open Max patch via Socket.IO.

**Starting Max side:**
1. Open `MaxMSP-MCP-Server/MaxMSP_Agent/demo.maxpat` in Max 9+
2. Tab 1: click `script npm install` (first time only)
3. Tab 2: click `script start` → connects to Python Socket.IO server

**MCP server is registered in `.mcp.json`** and starts automatically when Claude Code loads the project.

**Available MCP tools (Claude can call these):**
- `add_max_object(position, obj_type, varname, args)` — add a box to the open patch
- `remove_max_object(varname)` — delete a box
- `connect_max_objects(src_varname, outlet_idx, dst_varname, inlet_idx)` — connect boxes
- `disconnect_max_objects(...)` — remove patchcord
- `set_object_attribute(varname, attr_name, attr_value)` — set attribute
- `set_message_text(varname, text_list)` — set message box content
- `send_bang_to_object(varname)` — send a bang
- `send_messages_to_object(varname, message)` — send arbitrary message
- `set_number(varname, num)` — set number/slider/dial value
- `list_all_objects()` — list all known Max object names
- `get_object_doc(object_name)` — fetch official Max docs for an object
- `get_objects_in_patch()` — current patch state (objects + patchcords)
- `get_objects_in_selected()` — selected objects only
- `get_object_attributes(varname)` — get all attributes of an object
- `get_avoid_rect_position()` — bounding rect of existing objects (for placement)

**Workflow:** Always call `get_avoid_rect_position()` first to know where existing objects are, then place new objects outside that rect.

---

## maxpylang — Offline Patch Generation

`maxpylang` generates `.maxpat` files from Python without Max running. Use it for scaffolding complete patches, then open in Max.

### Core API

```python
from maxpylang import MaxPatch

# Create patch
patch = MaxPatch()
patch = MaxPatch(load_file="existing.maxpat")

# Place objects — returns list[MaxObject]
objs = patch.place("cycle~ 440")
objs = patch.place("cycle~ 440", "ezdac~", spacing_type="grid", spacing=[80, 60])
objs = patch.place("button", spacing_type="vertical", spacing=70, starting_pos=[100, 50])
objs = patch.place("gain~", num_objs=4)

# Connect (Outlet, Inlet) tuples — 0-indexed
osc, dac = patch.place("cycle~ 440", "ezdac~")
patch.connect((osc.outs[0], dac.ins[0]), (osc.outs[0], dac.ins[1]))

# Edit / move
osc.edit(text="cycle~ 880")
osc.move(200, 150)

# Replace by obj-id
patch.replace("obj-2", "saw~ 440")

# Delete
patch.delete(objs=["obj-1"], cords=[(osc.outs[0], dac.ins[0])])

# Check and save
patch.check()                          # logs unknowns / unlinked js
patch.save("my_patch.maxpat")
```

### Key object families
- **Audio DSP (end in `~`):** `cycle~`, `saw~`, `rect~`, `tri~`, `noise~`, `adc~`, `dac~`, `ezdac~`, `ezadc~`, `gain~`, `*~`, `+~`, `-~`, `/~`, `delay~`, `tapin~`, `tapout~`, `groove~`, `buffer~`, `sfplay~`, `sfrecord~`, `resonators~`, `biquad~`, `svf~`, `reson~`
- **MIDI:** `notein`, `noteout`, `makenote`, `midiout`, `midiin`, `ctlin`, `ctlout`, `pgmin`, `pgmout`
- **Control:** `metro`, `delay`, `timer`, `clocker`, `transport`, `sel`, `route`, `gate`, `switch`, `trigger`, `if`, `expr`, `coll`, `dict`, `table`
- **OSC networking:** `udpsend`, `udpreceive`, `pack`, `prepend`, `OSC-route`
- **UI:** `button`, `toggle`, `number`, `flonum`, `slider`, `dial`, `comment`, `message`, `panel`
- **Jitter:** `jit.matrix`, `jit.movie`, `jit.pix`, `jit.window`, `jit.gl.render`

### Spacing types
| `spacing_type` | `spacing` format | Use for |
|---|---|---|
| `"grid"` (default) | `[x, y]` | Multiple objects in rows |
| `"vertical"` | `float` (height) | Column layout |
| `"custom"` | `[[x,y], ...]` one per object | Precise positioning |
| `"random"` | N/A | Generative/artistic patches |

### Run scripts with:
```bash
cd /Users/paultriana/creative_labs/MIDI-GPT
.venv/bin/python3 my_patch_script.py
```

---

## OSC Protocol — MIDI-GPT

Full spec: `docs/OSC_PROTOCOL_SPEC.md` | Developer guide: `docs/maxmsp_developer_guide.md`

### Session flow
```
/midigpt/session/init    s:"name"          → /midigpt/session/ready
/midigpt/track/create    i:id i:inst i:type i:is_agent
/midigpt/param/set       s:"param" value
/midigpt/session/start                     → /midigpt/session/started
  [bar clock + notes loop]
/midigpt/session/stop                      → /midigpt/session/stopped
```

### Per-bar inputs (Max → Server)
```
/midigpt/note    i:track_id i:pitch i:velocity f:onset f:duration i:bar_index
/midigpt/bar/end i:bar_index i:ts_num i:ts_den
```

- `onset` = `note_on_tick / bar_ticks` ∈ `[0.0, 1.0)`
- `duration` = `(note_off_tick - note_on_tick) / bar_ticks` ∈ `(0.0, 1.0]`
- **Send `bar/end` AFTER all notes for that bar**

### Generated notes (Server → Max)
```
/midigpt/generated/open     i:track_id i:bar_index i:note_count
/midigpt/generated/note     i:track_id i:bar_index i:pitch i:velocity f:onset f:duration
/midigpt/generated/close    i:track_id i:bar_index
/midigpt/generated/features i:track_id i:bar_index f:density f:mean_pitch f:mean_vel i:polyphony f:mean_dur
```

### Parameters
| Parameter | Type | Default | Range |
|---|---|---|---|
| `temperature` | float | 1.0 | 0.5–2.0 |
| `lookahead_bars` | int | 2 | 1–8 |
| `buffer_bars` | int | 4 | 2–64 |
| `model_dim` | int | 4 | 1–8 |
| `mask_gap` | bool | False | — |
| `adapt_buffer` | bool | False | — |

```
/midigpt/param/set       s:"temperature" f:1.0   # persistent
/midigpt/param/set_once  s:"temperature" f:1.8   # one generation only
/midigpt/param/reset     s:"temperature"          # reset to default
/midigpt/param/reset_all                          # reset all params
```

### Track management
```
/midigpt/track/create      i:id i:instrument i:track_type i:is_agent
/midigpt/track/remove      i:id
/midigpt/track/set_ignore  i:id i:0_or_1
/midigpt/track/param/set   i:id s:"param" value
```
- `track_type`: 10 = melodic, 11 = drums
- `instrument`: GM program number 0–127
- Exactly **one** track must have `is_agent=1`

### Error codes
| Code | Meaning |
|---|---|
| 1 | Message in wrong state |
| 2 | Unknown track_id |
| 3 | Duplicate track_id |
| 4 | Invalid parameter |
| 5 | Generation failed |
| 6 | No agent track |
| 7 | Multiple agent tracks |
| 8 | Note sent to agent track |

---

## Testing

```bash
# Run full suite
.venv/bin/pytest tests/

# Skip slow tests
.venv/bin/pytest tests/ -m "not benchmark and not inference"

# Run realtime tests only
.venv/bin/pytest tests/test_realtime.py

# Run with a model checkpoint
MIDIGPT_CKPT=/path/to/model.pt .venv/bin/pytest tests/ -m inference

# Simulate a full OSC session (no model needed)
bash python_scripts_for_testing/run_realtime_simulation.sh

# Stress test OSC server
.venv/bin/python3 python_scripts_for_testing/stress_test_osc.py
```

---

## Claude Skills Available (Slash Commands)

| Command | Scope | Purpose |
|---|---|---|
| `/generate-maxpatch` | global | Generate a `.maxpat` from description using maxpylang |
| `/explain-maxpatch` | global | Load and explain an existing `.maxpat` file |
| `/midigpt-patch` | project | Generate a MIDI-GPT OSC client patch for Max |
| `/maxpylang-ref` | project | Inline maxpylang API quick reference |

---

## Common Workflows

### 1. Build a new Max patch offline
```bash
# Write a Python script using maxpylang, then:
.venv/bin/python3 scripts/my_patch.py
# → opens resulting .maxpat in Max
```

### 2. Live-edit a Max patch with Claude
- Open `MaxMSP-MCP-Server/MaxMSP_Agent/demo.maxpat` in Max
- Click `script start` in tab 2
- Ask Claude: "Add a gain~ object at position [200, 300] with varname gain1"
- Claude will use the `add_max_object` MCP tool

### 3. Start the MIDI-GPT OSC server
```bash
.venv/bin/midigpt-server --ckpt model.pt --port 7400 --log_level DEBUG
```

### 4. Run OSC simulation test (no model needed)
```bash
.venv/bin/python3 python_scripts_for_testing/simulate_osc_session.py
```

---

## Architecture Notes

- **C++ core** builds a shared library (`midigpt.so`) bound to Python via pybind11. Functions: `tokenize`, `detokenize`, `sample_multi_step`, `get_notes`.
- **`realtime_state.py`** holds `PieceState` — a pure-Python musical state machine (tracks, bars, notes). Thread-safe reads for the gen worker.
- **`realtime_gen.py`** wraps the C++ inference: `run_inference(piece_dict, status_dict, params_dict, max_attempts)` returns `(result_piece, attempts)`.
- **`osc_server.py`** runs `MidiGPTServer` which owns the OSC dispatcher (main thread) and a `gen_worker` background thread. Generation requests queue with `maxsize=1` — missed generations are logged, never block the OSC thread.
- The gen worker sends results back to Max via `_send_generated_bar()` using the same UDP client.

---

## Important Constraints

- **Never send notes to the agent track** (is_agent=1) — server drops them with error 8
- **Bar clock drives everything** — server generates on bar boundaries only
- **`onset` and `duration` are normalized** — always `[0, 1)` fractions of bar length
- **buffer_bars ≥ 2 required** — server rejects start with buffer < 2; recommend ≥ 4
- **One agent track only** — multiple is_agent=1 tracks → error 7
- **maxpylang requires Max 9+** for generated patches (V8 JavaScript engine for js objects)
- **MCP server uses Python 3.10** (MaxMSP-MCP-Server/.venv) — do not mix with project .venv
- **The project .venv uses Python 3.12** — use `.venv/bin/python3` for all project scripts
