# Real-Time Framework: Technical Implementation Plan

This document is the working implementation plan for the real-time co-generation system. It covers what to build, how, and how to test it. Read `docs/realtime_framework.md` first for the conceptual design.

---

## Current State Assessment

The design is complete and correct (`realtime_framework.md`). The infrastructure that already works:

- `midigpt.sample_multi_step()` — the stateless inference call that drives every generation step
- `midi::Piece` / `midi::Status` JSON protocol — how piece state and generation targets are passed to C++
- `TOKEN_MASK_BAR` + `Bar.future` — the masking mechanism for future bars
- `suffix_autoregressive` + `selected_bars` — how the model is told to generate a suffix

What does **not** yet work or is broken:

1. `simulate_realtime_agent.py` — the simulation loop — has **5 bugs** (see Phase 1)
2. `src/inference/realtime/realtime_session.h` and `realtime_kv_cache.h` — **dead code** from an abandoned stateful KV-cache approach; they are misleading
3. `src/inference/realtime/incremental_encoder.h` — only used by `realtime_session.h`; also dead
4. `tests/test_realtime.py` — tests the dead `RealtimeSession` bindings; must be replaced
5. The OSC server (`OSC_PROTOCOL_SPEC.md`) — not yet implemented

---

## Phase 0: Remove Dead Code

**Files to delete** (confirm nothing in CMakeLists or tests depends on them first):

```
src/inference/realtime/realtime_session.h
src/inference/realtime/realtime_kv_cache.h
src/inference/realtime/incremental_encoder.h
```

**How to confirm no dependencies:**

```bash
grep -r "realtime_session\|realtime_kv_cache\|incremental_encoder\|RealtimeSession\|RealtimeKVCache\|IncrementalEncoder" \
  src/ tests/ python_scripts/ CMakeLists.txt
```

The grep should only return hits inside those three header files themselves. If `lib.cpp` or CMakeLists.txt has any `#include` or `add_library` for them, remove those lines first.

**Also**: `tests/test_realtime.py` tests `midigpt.RealtimeSession` and `midigpt.RealtimeConfig` — bindings from the dead code. Delete or gut this file and replace it with the new test suite described in Phase 3.

---

## Phase 1: Fix `simulate_realtime_agent.py`

This is the core simulation loop. Five bugs need fixing before end-to-end testing is possible.

### Bug 1 — Bar selection selects too many bars (documented)

**File**: `python_scripts_for_testing/simulate_realtime_agent.py` ~line 279

**Current (wrong)**:
```python
for b_idx in range(target_bar, total_bars):
    sel[b_idx] = True
```

**Fix**:
```python
for b_idx in range(target_bar, min(target_bar + num_anticipation, total_bars)):
    sel[b_idx] = True
```

**Why it matters**: selecting all bars from `target_bar` to end causes `find_steps` to create one STEP per bar from `target_bar` to end, generating the entire remainder of the piece in a single call. This breaks the single-step generation contract.

---

### Bug 2 — Agent track `future` flags wrong for multi-bar anticipation

**File**: ~line 312

**Current (wrong)**:
```python
for b, b_data in enumerate(agent_bars):
    if b < target_bar:
        b_data['future'] = False
    elif b == target_bar:
        b_data['future'] = False   # only marks the first target bar
    else:
        b_data['future'] = True    # wrongly masks bars [target_bar+1, target_bar+j)
```

**Fix**:
```python
for b, b_data in enumerate(agent_bars):
    if b < target_bar + num_anticipation:
        b_data['future'] = False   # context bars + all j target bars are visible
    else:
        b_data['future'] = True    # bars beyond the generation window
```

**Why it matters**: bars `[target_bar+1, target_bar+j)` are the remaining target slots when `j > 1`. They must be `future=False` so the model can write into them.

---

### Bug 3 — `mask_lookahead_gap` not applied to agent track gap bars

**File**: ~line 315 (same future-flag block)

When `mask_gap=True` (i.e., `mask_lookahead_gap=True`), the agent's gap bars `[playhead, target_bar)` must be encoded as `TOKEN_MASK_BAR` by setting `future=True`.

**Add this** inside the future-flag loop, after setting context bars to `future=False`:
```python
    if args.mask_gap and playhead <= b < target_bar:
        b_data['future'] = True    # hide previously generated gap bars
```

The full corrected loop:
```python
for b, b_data in enumerate(agent_bars):
    if b >= target_bar + num_anticipation:
        b_data['future'] = True
    elif args.mask_gap and playhead <= b < target_bar:
        b_data['future'] = True    # mask gap when mask_lookahead_gap=True
    else:
        b_data['future'] = False
```

---

### Bug 4 — Result extraction only writes back `target_bar`, not all j bars

**File**: ~line 374

**Current (wrong)**:
```python
agent_bars[target_bar]['events'] = res_agent_track['bars'][target_bar].get('events', [])
```

**Fix** — write back all j generated bars and their events:
```python
res_events = res_piece.get('events', [])
for b_offset in range(num_anticipation):
    b_global = target_bar + b_offset
    if b_global >= total_bars:
        break
    res_bar = res_agent_track['bars'][b_global]
    # res_bar['events'] contains indices into res_piece['events']
    # Copy the actual event objects into agent_bars
    bar_event_objects = [res_events[i] for i in res_bar.get('events', [])]
    agent_bars[b_global]['events'] = bar_event_objects
```

Note: `sim_piece['events']` is a shared event pool. The cleaner approach is to copy event objects directly rather than index-remapping into the global pool. Adjust `write_midi` call accordingly when saving output (serialize `agent_bars` events inline, not as indices).

---

### Bug 5 — Playhead advances by 1 every iteration, should advance by `j`

**File**: ~line 223

**Current (wrong)**:
```python
for playhead in range(total_bars):
    ...
    if target_bar is not None:
        # fires inference every bar
```

With `num_anticipation = j > 1`, inference should only fire every j bars. The current code fires every bar, overwrites bars that haven't finished their lookahead window, and is inconsistent with the framework spec.

**Fix**: restructure the loop to advance by j between generation calls:

```python
playhead = 0
while playhead < total_bars:
    # ... compute target_bar, build piece/status, run inference ...
    
    # Advance playhead by j (or 1 if still in buffer/pre-generation phase)
    if should_gen:
        playhead += num_anticipation
    else:
        playhead += 1
```

The visualization grid can still be rendered for every step — just keep the grid render before the conditional `+= num_anticipation`. If you want to render every bar tick even during j-bar advance phases, keep a separate `display_playhead` that advances by 1 and only run inference when `display_playhead == playhead`.

---

## Phase 2: End-to-End Simulation Validation

Once bugs 1–5 are fixed, validate correctness before building the OSC server.

### 2.1 Dry-run grid check

Run without inference to verify masking logic across all parameter combinations:

```bash
python python_scripts_for_testing/simulate_realtime_agent.py \
  --midi tests/short_midi/test.mid \
  --buffer 4 --lookahead 1 --num_anticipated_bars 1 \
  --model_dim 8 --dry_run --delay 0
```

Cross-check the printed masking summary against the worked examples in `realtime_framework.md`:
- Example 1: `buffer=4, k=1, j=1, model_dim=8` — window starts sliding at step 4
- Example 2: `buffer=4, k=2, j=1, model_dim=8` — gap bar visible vs masked
- Example 3: `buffer=4, k=2, j=2, model_dim=8` — j=2, no copy-paste after step 1
- Example 4: `buffer=4, k=1, j=2, model_dim=10` — copy-paste bars in early phase

Write a small Python helper `python_scripts_for_testing/check_masking.py` that reconstructs the expected masking state for each step from the spec formulas and diffs against what the simulation actually produces. This is fully deterministic (no model needed).

### 2.2 Live inference run

```bash
python python_scripts_for_testing/simulate_realtime_agent.py \
  --midi tests/short_midi/test.mid \
  --ckpt /scratch/triana24/MIDI-GPT/outputs/<checkpoint>.pt \
  --buffer 4 --lookahead 1 --num_anticipated_bars 1 \
  --model_dim 8 --delay 0.2 \
  --output /scratch/triana24/MIDI-GPT/outputs/realtime_test.mid
```

Verify:
- Each step generates exactly j bars (check print output: "Generated N notes" for each step)
- Generated MIDI is audible and musically coherent
- No C++ exceptions or JSON parse errors

### 2.3 Edge cases to test manually

| Scenario | Parameters | What to verify |
|----------|-----------|----------------|
| First step, k>1, gap bars empty | `k=2, j=1, buffer=4` | Gap bar 5 is silent (no prior generation) |
| mask_gap behavior | `k=2, j=1, --mask_gap` | Gap bars show `future=True` in masking summary |
| adapt_buffer | `k=2, buffer=4, --adapt_buffer` | Generation starts at playhead=2, first note at bar 4 |
| Window slide | `k=1, j=1, model_dim=8, buffer=4` | Window starts sliding when `t+j > 8` (at step 4) |
| j=2 multi-bar | `k=1, j=2, model_dim=8` | Playhead advances by 2; each call produces 2 bars |

---

## Phase 3: New Test Suite

Replace `tests/test_realtime.py` (which tested dead code) with tests that cover the actual implementation.

### 3.1 Unit tests — pure Python, no model

**File**: `tests/test_realtime_loop.py`

These tests import only Python stdlib + `midigpt` and exercise the simulation loop logic directly. They do not call `sample_multi_step`.

```
TestMaskingLogic
  test_human_tracks_masked_from_playhead
      - Build a piece with 2 human tracks, playhead=4
      - Verify all bars >= 4 have future=True on human tracks

  test_agent_context_bars_not_masked
      - k=1, j=1, playhead=4, target=5
      - Verify agent bars [0,4] have future=False

  test_agent_target_bars_not_masked
      - k=1, j=2, playhead=4, target=5
      - Verify agent bars 5 AND 6 have future=False

  test_mask_lookahead_gap_true
      - k=2, j=1, playhead=4, target=6, mask_gap=True
      - Verify agent bar 5 (gap) has future=True

  test_mask_lookahead_gap_false
      - k=2, j=1, playhead=4, target=6, mask_gap=False
      - Verify agent bar 5 (gap) has future=False

TestBarSelection
  test_selected_bars_exactly_j
      - j=1: exactly 1 bar selected
      - j=2: exactly 2 bars selected; no bars outside [t, t+j) selected

  test_suffix_autoregressive_set
      - Agent track status always has suffix_autoregressive=True

TestWindowAlignment
  test_early_phase_window_anchored_at_zero
      - t+j <= D: window_start=0, window_end=D

  test_sliding_phase_window_follows_target
      - t+j > D: window_start = t+j-D, window_end = t+j

TestPlayheadAdvance
  test_playhead_advances_by_j
      - Simulate 3 steps with j=2; verify playhead goes 0 → 2 → 4 → 6

  test_adapt_buffer_early_start
      - k=2, buffer=4, adapt_buffer=True: first generation at playhead=2
```

### 3.2 Integration tests — require model checkpoint

**File**: `tests/test_realtime_integration.py`

```
Env var: REALTIME_MODEL_PATH (skip if not set)
Env var: REALTIME_MIDI_PATH  (skip if not found)

TestSingleStep
  test_single_step_j1_returns_nonempty
      - One call: buffer=4, k=1, j=1, playhead=4
      - Verify result piece has ≥ 1 event in agent track bar 5

  test_single_step_j2_returns_two_bars
      - j=2: verify bars 5 AND 6 both have content written back

  test_result_does_not_overwrite_context
      - After step 1, bars [0,4] on agent track unchanged (still empty)

TestMultiStep
  test_three_steps_j1_accumulates
      - Run 3 steps with j=1; after step 3, bars 5,6,7 all have events

  test_mask_gap_does_not_corrupt_prior_generation
      - k=2, j=1, mask_gap=True; after 3 steps verify gap bars retain events
        (mask_gap only hides them from the model, doesn't erase them)

TestOutputMidi
  test_full_simulation_produces_midi
      - Run full simulation on test MIDI, save output, verify file exists
        and has nonzero size
```

### 3.3 Running the test suite

```bash
# Unit tests only (no model)
module load StdEnv/2023 python/3.11.5 abseil/20230125.3 protobuf/24.4
source /scratch/triana24/.venvs/midigpt/bin/activate
python3 -m pytest tests/test_realtime_loop.py -v

# Integration tests (need checkpoint)
export REALTIME_MODEL_PATH=/scratch/triana24/MIDI-GPT/outputs/<ckpt>.pt
export REALTIME_MIDI_PATH=tests/short_midi/test.mid
python3 -m pytest tests/test_realtime_integration.py -v
```

---

## Phase 4: OSC Server

The OSC server translates the Max MSP message stream into `sample_multi_step` calls, implementing the protocol in `OSC_PROTOCOL_SPEC.md`. This is a pure Python layer; C++ is only touched through `midigpt`.

### 4.1 Architecture

```
Max MSP
  │ UDP port 7400
  ▼
osc_server.py
  ├── MessageRouter  — dispatches incoming OSC to handlers
  ├── PieceState     — owns the global midi::Piece dict; updated by /note and /bar/end
  ├── GenerationLoop — runs in a background thread; calls sample_multi_step
  └── OSCSender      — pushes /generated/bar back to Max on UDP port 7401
```

**Library**: use `python-osc` (`pip install python-osc`). Already installable on login node.

### 4.2 State machine

```
IDLE ──/session/init──▶ CONFIGURED ──/session/start──▶ RUNNING
  ▲                                                         │
  └──────────────────/session/stop─────────────────────────┘
```

In `RUNNING`:
- `/note pitch vel time track_id` — appends event to current bar buffer for `track_id`
- `/bar/end bar_id track_id` — flushes bar buffer into `PieceState`, checks if generation should fire
- `/param/set key value` — updates generation parameters (temperature, lookahead, etc.)
- Generation fires when: `should_gen` condition met (same logic as simulation script)
- Result sent as `/generated/bar bar_id pitch vel time ...` (see OSC spec)

### 4.3 Thread model

- **Main thread**: OSC listener (blocking `server.serve_forever()`)
- **Generation thread**: one dedicated thread; waits on a `threading.Event`; when signaled by `/bar/end`, runs `sample_multi_step` and sends result back via OSCSender
- **Handoff**: `/bar/end` handler sets a `threading.Event`; generation thread picks up a snapshot of `PieceState` (deep copy), calls inference, merges result back into `PieceState` under a lock

Inference is CPU-only (C++ LibTorch), so no GIL issues.

### 4.4 File layout

```
python_scripts/osc_server.py      — entry point; argparse for ports, ckpt, params
python_scripts/realtime_state.py  — PieceState class (piece dict management)
python_scripts/realtime_gen.py    — generation loop (wraps simulate logic, minus visualization)
```

`realtime_gen.py` should share core logic with `simulate_realtime_agent.py` — factor the masking/selection/window logic into helper functions in `realtime_state.py` so both the simulation script and the OSC server call the same code.

### 4.5 OSC server test strategy

- Unit test `PieceState` directly: push notes, end bar, verify piece JSON structure
- Integration test: mock OSC client that sends `/session/init` → `/track/create` → `/session/start` → series of `/note` + `/bar/end` messages, then assert received `/generated/bar` messages

---

## Design Decisions (resolved)

### 1. `status_rehighlight` and early-phase trailing generation — accepted behavior

**Finding**: `status_rehighlight` (`multi_step_sample.h:57-85`) operates on the per-step status (window `[t_start, t_start+D)`). With `suffix_autoregressive=True`, it sets all bars from `first_selected` to the end of the window to `True`.

In the **sliding phase** (`t+j = t_start+D`), the window ends at `t+j-1` — `status_rehighlight` selects exactly the j target bars. Correct and tight.

In the **early phase** (`t+j < D`, `t_start=0`), bars `[t+j, D-1]` are inside the window but after the last generated bar. `status_rehighlight` selects them, the model generates tokens for them, but `piece_insert` only writes back `bar_mapping` bars (`[t, t+j)`). The trailing bars retain their prior content (empty in buffer phase, or previously generated content in later early steps).

**Decision**: accept this as-is. `model_dim` is a trained hyperparameter fixed per-checkpoint (the ghost encoder vocabulary includes `NUM_BARS` tokens for specific values: 4, 8, 12, 16) and cannot be varied dynamically. The copy-paste behavior is correct — `piece_insert` is already the "early stopping" mechanism. The compute spent generating trailing tokens in the first few steps is a small, accepted cost. The early phase is brief; once `t+j ≥ D` the system is in the sliding phase permanently with no wasted generation.

---

### 2. Event storage — inline objects, serialize on inference call

**Decision**: Do not maintain a global flat events pool in the Python simulation state. Store bar events as **inline lists of event dicts** directly in each bar: `bar['events'] = [{"pitch": 60, "velocity": 80, "time": 0}, ...]`.

When calling `sample_multi_step`:
- Serialize: convert inline events to the flat pool format (`piece['events']` + integer index arrays per bar) right before `json.dumps`.
- Deserialize: extract events from the result piece (which has its own self-consistent pool) as inline objects.

This avoids all index remapping on merge. The serialization step is O(n_events) and negligible.

The existing simulation script's approach of sharing a global `sim_piece['events']` pool and index-slicing the result is fragile and incorrect for multi-bar updates. Replace it entirely.

---

### 3. Minimum buffer — hard floor B=2, recommended default B=4

**Math**: `k ≤ B/2` with `k ≥ 1` forces `B ≥ 2`. B=0 or B=1 is impossible with the constraint.

**Generating at bar zero is not possible** in this framework by design. The earliest the agent can play is bar `B` (with `adapt_buffer`) or `B+k` (without). The framework requires at least k bars of completed human context before the first generation step, and k ≥ 1.

**Enforcement**:
- `B < 2`: **hard error** — violates the math. Exit immediately.
- `B < 4`: **warning** — only 1–3 bars of human context at first generation step. Musically thin.
- Recommended default: `B=4` (one 4-bar phrase). This is enough for the model to infer rhythm, key, and texture before generating.
- With `adapt_buffer=True` the same floors apply. Early steps still have limited context, but the buffer provides the bulk of conditioning.

---

## Implementation Order

| Step | What | Prerequisite |
|------|------|-------------|
| 0 | Delete dead C++ files + old test file | None |
| 1a | Fix bugs 1–4 in simulation script | None |
| 1b | Fix bug 5 (playhead advance) | 1a |
| 2a | Dry-run grid validation | 1a,1b |
| 2b | Write `check_masking.py` helper | 2a |
| 3a | Write `test_realtime_loop.py` unit tests | 1a,1b |
| 3b | Write `test_realtime_integration.py` | 3a |
| 2c | Live inference run | 1a,1b |
| 4a | Factor shared masking logic into `realtime_state.py` | 1a |
| 4b | OSC server skeleton | 4a |
| 4c | OSC server integration test | 4b |
