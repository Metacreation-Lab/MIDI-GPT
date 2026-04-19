# Real-Time Framework: Implementation Progress

This document tracks bugs found, decisions made, and open tasks across sessions. Update it as work progresses.

---

## Session 1 — Framework Design (2026-04-01)

### Completed
- Defined and documented the full real-time generation framework in `docs/realtime_framework.md`.
- Replaced three outdated/incorrect documents (`realtime_agent_implementation.md`, `realtime_agent_technical_design.md`, `realtime_agent_testing_plan.md`) with the single authoritative reference.

### Key Design Decisions
- `bars_per_step = num_anticipation` (j). One inference call per real-time step generates j bars causally. No benefit to splitting into j separate calls since the autoregressive context within a single call is equivalent.
- Playhead advances by j between generation steps (not by 1).
- `mask_lookahead_gap` controls whether the agent's previously generated gap bars are visible to the model or hidden with `TOKEN_MASK_BAR`. Gap bars are NOT silence — they are prior generated content (except possibly the very first step with k > 1).
- Copy-paste is implicit: `piece_insert` only writes `bar_mapping` bars. Bars after `t+j-1` in the window are untouched in the global piece.
- Window alignment: `t_start = max(0, (t + j - 1) - D + 1)`. Last generated bar is always the last bar of the window. Copy-paste situation only arises in the early phase when the window is anchored at 0.

### Bugs / Issues Found

#### BUG 1 — `simulate_realtime_agent.py`: incorrect bar selection
**File**: `python_scripts_for_testing/simulate_realtime_agent.py`
**Location**: the block building `sel` for the agent track status (~line 279)
**Problem**: Selects all bars from `target_bar` to `total_bars` as True:
```python
for b_idx in range(target_bar, total_bars):
    sel[b_idx] = True
```
This causes `find_steps` to create one step per bar from `target_bar` to end, generating all remaining bars instead of just `[t, t+j)`.
**Fix**: Only select bars `[t, t + num_anticipation)`:
```python
for b_idx in range(target_bar, min(target_bar + num_anticipation, total_bars)):
    sel[b_idx] = True
```

#### DEAD CODE — `realtime_session.h` and `realtime_kv_cache.h`
**Files**: `src/inference/realtime/realtime_session.h`, `src/inference/realtime/realtime_kv_cache.h`
**Problem**: These implement a stateful KV-cache approach that was abandoned. The current framework is stateless — each step calls `sample_multi_step()` fresh with the updated piece JSON. These files are misleading and should be removed.
**Action**: Delete both files. Confirm nothing in the build or tests depends on them first.

### Open Questions / Next Steps (from Session 1)
- ~~Verify status_rehighlight behavior~~ → resolved in Session 2
- Fix BUG 1 in `simulate_realtime_agent.py` (+ 4 more bugs found in Session 2)
- Delete dead code (`realtime_session.h`, `realtime_kv_cache.h`) after confirming no dependencies
- Run simulation end-to-end with a real model checkpoint

---

## Session 2 — Technical Plan + Bug Audit (2026-04-01)

### Completed
- Read and fully understood the framework spec, simulation script, all dead C++ files, and `multi_step_sample.h` / `status_rehighlight`.
- Wrote `docs/realtime_technical_plan.md`: full implementation plan covering Phase 0 (dead code removal) through Phase 4 (OSC server), with test suite design.
- Resolved all three open questions from Session 1 (see design decisions in `realtime_technical_plan.md`).

### Bugs Found in `simulate_realtime_agent.py` (full audit)

#### BUG 1 (documented in Session 1) — bar selection too broad
```python
# Wrong: selects target_bar to end of piece
for b_idx in range(target_bar, total_bars): sel[b_idx] = True
# Fix: only select [t, t+j)
for b_idx in range(target_bar, min(target_bar + num_anticipation, total_bars)): sel[b_idx] = True
```

#### BUG 2 — agent future flags wrong for j > 1
The code only sets `future=False` for `b == target_bar`. With j > 1, bars `[target_bar+1, target_bar+j)` are also generation targets and must be `future=False`.
```python
# Fix: future=False for all bars in [0, target_bar+num_anticipation)
for b, b_data in enumerate(agent_bars):
    if b >= target_bar + num_anticipation:
        b_data['future'] = True
    elif args.mask_gap and playhead <= b < target_bar:
        b_data['future'] = True  # hide gap bars when mask_lookahead_gap=True
    else:
        b_data['future'] = False
```

#### BUG 3 — mask_lookahead_gap not applied to agent gap bars
When `--mask_gap` is set, agent gap bars `[playhead, target_bar)` must have `future=True`. Currently they are always `future=False`. Fixed in BUG 2 fix above.

#### BUG 4 — result extraction only writes target_bar, not all j bars
```python
# Wrong: only writes one bar
agent_bars[target_bar]['events'] = res_agent_track['bars'][target_bar].get('events', [])
# Fix: loop over all j bars, extract event objects from result pool
```

#### BUG 5 — playhead advances by 1 each iteration instead of j
The loop `for playhead in range(total_bars)` fires inference every bar. With j > 1, inference should fire every j bars. Playhead must advance by j after each generation step.

### Key Design Decisions Made

1. **Copy-paste is accepted behavior**: `model_dim` is a trained hyperparameter (ghost encoder supports 4/8/12/16 only — `NUM_BARS` token vocab), cannot be set dynamically. In the early phase, `status_rehighlight` selects trailing bars beyond `t+j-1` and the model generates tokens for them, but `piece_insert` writes only `[t, t+j)`. `piece_insert` IS the early-stopping mechanism. Trailing bars retain prior content (empty or previously generated). This is correct and accepted — the compute cost is small and the early phase is brief.

2. **Event storage — global pool, index remapping on merge**: The simulation keeps `sim_piece['events']` as a flat global pool. When merging results, event objects are copied from the result pool into the global pool (new indices assigned). `agent_bars[b]['events']` stores global indices. This is correct and simple; the "inline objects" approach was considered but not needed since index remapping during merge is O(n_events) and straightforward.

3. **Minimum buffer**: B < 2 → hard error. B < 4 → warning. Default B=4. Generating at bar 0 is impossible (requires k=0, but k ≥ 1 always).

---

## Session 3 — Implementation (2026-04-01)

### Completed

#### Phase 0: Dead code removal
- Removed `#include "inference/realtime/realtime_session.h"` from `lib.cpp`
- Removed `RealtimeConfig` and `RealtimeSession` PyBind11 bindings from `lib.cpp`
- Deleted `src/inference/realtime/realtime_session.h`, `realtime_kv_cache.h`, `incremental_encoder.h`

#### `future` flag moved from piece to status (new design)
- Added `optional bool future = 8;` to `StatusBar` in `libraries/protobuf/src/midi.proto`
- Added `apply_future_flags_from_status()` to `src/inference/sampling/multi_step_sample.h`
  - Called in `sample()` right after `add_timesigs_to_status()`
  - Copies `StatusBar.future → Bar.future` for any bar where `StatusBar.has_future()`
  - No-op for callers that don't set `StatusBar.future` (backward compatible)
  - `Bar.future` in the piece still valid for training augmentation / old API

#### Phase 1: All 5 bugs fixed in `simulate_realtime_agent.py`
- BUG 1: selection now `range(target_bar, min(target_bar + num_anticipation, total_bars))`
- BUG 2: agent future=False for all j target bars `[target_bar, target_bar+num_anticipation)`
- BUG 3: mask_gap flag applies `future=True` to agent gap bars `[playhead, target_bar)`
- BUG 4: result extraction loops over all j bars, copies events with global index remapping
- BUG 5: while loop with `playhead += num_anticipation` when should_gen, else `+= 1`
- **Bonus**: `future` flags moved from piece mutation to `status['tracks'][i]['bars']`
- **Bonus**: B < 2 hard error; B < 4 warning added to validation

#### Phase 3: New test suite
- Replaced `tests/test_realtime.py` with complete new test suite:
  - `TestMaskingLogic` — 8 tests, no model needed
  - `TestBarSelection` — 5 tests, no model needed
  - `TestWindowAlignment` — 4 tests (framework examples), no model needed
  - `TestPlayheadAdvance` — 5 tests including adapt_buffer, no model needed
  - `TestIntegrationSingleStep` — 4 tests, require `REALTIME_MODEL_PATH`
  - `TestIntegrationMultiStep` — 1 test, require `REALTIME_MODEL_PATH`

### Compile script
`scripts/compile_install.sh` uses:
```bash
module load StdEnv/2023 python/3.11 gcc/12 cmake protobuf/24.4 abseil/20230125.3 cuda/12.2
source /home/triana24/scratch/.venvs/midigpt/bin/activate  # same as /scratch/triana24/.venvs/midigpt
pip install -e ".[train]"
```

### Open Tasks for Next Session
- [ ] Rebuild: run `scripts/compile_install.sh` (protobuf schema changed — must recompile)
- [ ] Phase 2a: dry-run grid check with `--dry_run` against worked examples in `realtime_framework.md`
- [ ] Phase 2b: live inference run with real checkpoint (`REALTIME_MODEL_PATH`)
- [ ] Phase 3 unit tests: run `python3 -m pytest tests/test_realtime.py -v -k "not Integration"` (no model needed)
- [ ] Phase 3 integration tests: run with model env var set
