# Real-Time Generation Framework

This document is the authoritative reference for MIDI-GPT's real-time co-generation framework. It replaces `realtime_agent_implementation.md`, `realtime_agent_technical_design.md`, and `realtime_agent_testing_plan.md`.

---

## Overview

The real-time framework enables a live human-AI performance where one or more human musicians perform live (conditioning tracks) while the AI generates a complementary track (agent track) in real time.

The central problem: the transformer was trained on complete, static pieces. In real time, the piece is incomplete and streaming. The framework solves this by:

1. Masking future human bars so the model cannot see what hasn't been played yet.
2. Generating agent bars one step ahead of the playhead (lookahead), so the output is ready before it needs to be played.
3. Using suffix auto-regressive generation so the agent continues its own musical history incrementally.
4. Sliding the context window to always give the model the maximum amount of past context.

---

## Parameters

| Parameter | Description |
|-----------|-------------|
| `model_dim` (D) | Context window size in bars. The model attends to at most D bars at once. |
| `buffer` (B) | Bars of silence at the start before the agent begins generating. Allows human context to build. |
| `lookahead` (k) | Distance ahead of the playhead where the agent generates. While playing bar N, the agent generates bar N+k. Minimum 1. |
| `num_anticipation` (j) | Bars generated per real-time step. Equals `bars_per_step` in the inference call. The playhead advances by j between steps. |
| `mask_lookahead_gap` | Whether gap bars (between playhead and target) on the agent track are shown as previously generated content or hidden with `TOKEN_MASK_BAR`. |
| `adapt_buffer_for_lookahead` | If True, start generating during the buffer period so the first generated bar is ready exactly when the buffer ends. |

### Derived values

- **Target bar**: `t = playhead + k`. The first bar generated in the current step.
- **Generated range**: bars `[t, t + j)`.
- **Playhead advance**: after each generation step, playhead advances by `j`.
- **`bars_per_step` in the inference call**: always set to `num_anticipation` (j).

### Constraints

- `k ≤ B / 2`: When the buffer ends, the model needs at least k bars of completed human context for meaningful generation at the first step.
- `k + j < D`: The context window must contain at least one bar of non-masked past context. If k+j fills the entire window, the model generates blind.

---

## Track Roles

### Conditioning tracks (humans)

Played live. At any given step, only bars before the playhead are known.

- Bars `[0, playhead)`: ground truth. `future = False`. Model sees the actual notes.
- Bar `playhead` onward: `future = True` → encoded as `TOKEN_MASK_BAR`. The content is unknown or incomplete.

### Agent track (AI)

- **Buffer bars** `[0, B)`: empty (silent). `future = False`. Not masked — treated as known silence. The agent deliberately did not play during the buffer.
- **Gap bars** `[B, t)`: bars between the end of the buffer and the current target. Content here is the agent's previously generated output from prior steps (not silence, except possibly at the very first step with k > 1). Whether this content is shown to the model depends on `mask_lookahead_gap`.
- **Target bars** `[t, t + j)`: empty bar slots where generation happens this step. `future = False`.
- **Beyond target** `[t + j, ...)`: not part of the current generation window.

### TOKEN_MASK_BAR vs infill placeholder

These are distinct tokens and must not be confused.

- **Infill placeholder** (standard infilling): "fill this bar using past and future context." Non-causal.
- **TOKEN_MASK_BAR** (real-time): "music will be here eventually, but I cannot see it yet." Causal — enforces that the model cannot condition on future human notes.

---

## The mask_lookahead_gap Parameter

When `k > 1`, there is a gap between the playhead and the target: bars `[playhead, t)` on the agent track.

These bars are **not empty**. They were generated in prior steps (bar `t-1` was generated when the playhead was at `playhead - 1`, bar `t-2` when the playhead was at `playhead - 2`, etc.). The only exception is the very first generation step with k > 1, when some gap bars may not yet have been generated.

- `mask_lookahead_gap = False`: gap bars are shown as ground truth (previously generated content). The model conditions on what it planned ahead earlier.
- `mask_lookahead_gap = True`: gap bars are hidden with `TOKEN_MASK_BAR`. The model cannot see its own prior lookahead outputs. This prevents the model from reinforcing early, lower-quality decisions made with less context.

---

## Adaptive Suffix Auto-Regressive Generation

### What suffix auto-regressive means here

The agent track is not generated from scratch each step. Instead, the model treats everything before the target bar as a prompt (the agent's established musical history plus the human context), and generates a continuation starting at bar `t`.

In the status passed to `sample_multi_step`:
- Agent track: `suffix_autoregressive = True`, `selected_bars` has `True` only for bars `[t, t + j)`.
- All other bars on the agent track are context, not generation targets.

### What the model actually does

1. `find_steps` sees j selected bars on the agent track with autoregressive mode. It creates one STEP with a window of size D, aligned so the **last generated bar** (`t + j - 1`) is the last bar of the window.
2. Inside `sample_step`, `status_rehighlight` enforces suffix-AR contiguity: within the window, all bars from `t` to the end of the window are set to `True`.
3. `generate()` runs the model. It tokenizes bars `[window_start, t)` as a prompt (no token sampling), then auto-regressively generates tokens from bar `t` to the end of the window. With `num_anticipation = j`, bars `t` through `t + j - 1` are the intended outputs; any bars the model generates beyond `t + j - 1` within the window are a byproduct of suffix-AR expansion.
4. `piece_insert` writes back **only the bars in bar_mapping**, which corresponds to the original STEP's selected bars: `[t, t + j)`. Everything after bar `t + j - 1` in the global piece is left unchanged.

### The copy-paste mechanism

The term "copy-paste" refers to the implicit preservation of bars after the last generated bar. `piece_insert` does not overwrite them; they retain whatever content was in the global piece before the inference call. In early steps this is empty bars. This is only relevant in the early phase (see below).

### Within a single call, num_anticipation bars are generated causally

When `j > 1`, bar `t` is generated first, then bar `t+1` is generated with bar `t`'s tokens already in the autoregressive context. The causal attention mask ensures bar `t+1` conditions on bar `t`. There is no quality loss vs. two separate calls, and one call is strictly more efficient. This is why `bars_per_step = num_anticipation`.

---

## Context Window Management

### Early phase: t + j ≤ D (piece fits within model_dim)

The window is anchored at bar 0: `window = [0, D)`.

The alignment in `find_steps_inner` positions the window so that bar `t + j - 1` is the last bar:

```
t_start = max(0, (t + j - 1) - D + 1)
```

When `t + j - 1 < D - 1`, `t_start = 0` and the window starts at the beginning of the piece. The generated bars `[t, t+j)` are not at the very end of the window; there are bars after them up to `D`. Those trailing bars in the window are untouched by `piece_insert` — this is the copy-paste situation. They are empty in the buffer phase, or previously generated content from prior steps.

### Sliding phase: t + j > D (piece exceeds model_dim)

The window slides so that bar `t + j - 1` is the last bar of the window:

```
t_start = (t + j - 1) - D + 1
window = [t_start, t_start + D)
```

The model sees `D - j` bars of past context (maximum possible) followed by j generation targets at the end of the window. There are no bars after the last generated bar in this window — no copy-paste situation. This is the standard operating mode for long performances.

### Summary

| Phase | Condition | t_start | Copy-paste after last generated bar? |
|-------|-----------|---------|---------------------------------------|
| Early | t + j ≤ D | 0 | Yes (empty bars or prior generated content) |
| Sliding | t + j > D | t + j - D | No (window ends at last generated bar) |

---

## The Buffer and Generation Start

During bars `[0, B)`, the agent is silent. No inference calls are made. The human tracks build up musical context.

### Without adapt_buffer_for_lookahead

Generation starts when `playhead ≥ B`. At that point, `target = playhead + k`. The first generated bar is `B + k`, so the agent is silent for k additional bars after the buffer ends before its first note is played.

**Example**: B=4, k=3 → agent first plays at bar 8 (bars 5, 6, 7 are silent gap before the first generated bar).

### With adapt_buffer_for_lookahead

Generation starts when `playhead + k ≥ B`, i.e., at `playhead = B - k`. The agent generates bar B at the very first step, so its first note plays immediately when the buffer ends.

**Example**: B=4, k=3 → generation starts at playhead=1. Targets: bar 4 (at playhead=1), bar 5 (at playhead=2), bar 6 (at playhead=3), bar 7 (at playhead=4). Agent plays from bar 4 onward with no silence gap after the buffer.

Trade-off: early generation steps have very little human context (playhead=1 means only bar 0 is ground truth for the humans). Early generated bars may be less musically coherent.

---

## The Generation Loop

### Per-step logic

Given playhead `p`, `num_anticipation` j, `lookahead` k:

1. Compute target bar: `t = p + k`.
2. Build the piece JSON:
   - Conditioning tracks: `future = True` for all bars at index ≥ p.
   - Agent track: `future = False` for bars `[0, t)` (context — buffer, previously generated, and gap). If `mask_lookahead_gap = True`, set `future = True` for gap bars `[p, t)`. `future = False` for target bars `[t, t+j)` (empty slots for generation).
3. Build the status JSON:
   - Conditioning tracks: `selected_bars = [False, ...]`, `suffix_autoregressive = False`.
   - Agent track: `selected_bars` = True for `[t, t+j)`, False elsewhere. `suffix_autoregressive = True`.
4. Build params: `model_dim = D`, `bars_per_step = j`, `temperature = ...`.
5. Call `sample_multi_step(piece_json, status_json, params_json, max_attempts, callbacks)`.
6. Extract the generated content for bars `[t, t+j)` from the returned piece and write them into the global piece state as ground truth.
7. Advance playhead by j. Repeat.

### Generation does not fire every playhead bar

With `num_anticipation = j`, each inference call covers j bars. The playhead advances by j between calls. So with j=2, a call fires at playhead=4, next at playhead=6, next at playhead=8, etc.

---

## Worked Examples

### Example 1: model_dim=8, buffer=4, k=1, j=1, mask_lookahead_gap=False

```
Step   Playhead  Target  Window    Humans            Agent context (future=False)
------+--------+-------+--------+-----------------+---------------------------------
  1  |    4   |   5   |  [0,8) | bars 4+ masked  | bars 0-4 (empty buffer)
  2  |    5   |   6   |  [0,8) | bars 5+ masked  | bars 0-4 empty, bar 5 generated
  3  |    6   |   7   |  [0,8) | bars 6+ masked  | bars 0-4 empty, bars 5-6 generated
  4  |    7   |   8   |  [1,9) | bars 7+ masked  | bars 1-4 empty, bars 5-7 generated
  ...
  8  |   11   |  12   |  [5,13)| bars 11+ masked | bars 5-11 generated
```

At step 4, the window starts sliding. Bar 8 is the last bar of window `[1,9)`. No copy-paste.

### Example 2: model_dim=8, buffer=4, k=2, j=1, mask_lookahead_gap=False

```
Step   Playhead  Target  Window    Gap bar (visible)
------+--------+-------+--------+------------------
  1  |    4   |   6   |  [0,8) | bar 5 = empty (not yet generated, first step)
  2  |    5   |   7   |  [0,8) | bar 6 = generated at step 1
  3  |    6   |   8   |  [1,9) | bar 7 = generated at step 2
```

At step 2, bar 6 was generated in step 1 and is visible as context (mask_lookahead_gap=False). With mask_lookahead_gap=True, bar 6 would be TOKEN_MASK_BAR.

### Example 3: model_dim=8, buffer=4, k=2, j=2, mask_lookahead_gap=False

Playhead advances by j=2 between steps.

```
Step   Playhead  Target  Generated    Window    Copy-paste after?
------+--------+-------+------------+--------+-----------------
  1  |    4   |   6   |  bars 6,7  |  [0,8) | No (bar 7 = window_end-1)
  2  |    6   |   8   |  bars 8,9  |  [2,10)| No (bar 9 = window_end-1)
  3  |    8   |  10   |  bars 10,11|  [4,12)| No
```

With j=2 and model_dim=8, the last generated bar (`t+j-1`) always aligns to the end of the window once the sliding phase begins. No copy-paste after step 1.

### Example 4: model_dim=10, buffer=4, k=1, j=2

Early phase (before sliding starts): `t + j - 1 < D - 1` → copy-paste applies.

```
Step   Playhead  Target  Generated    Window     Copy-paste bars
------+--------+-------+------------+---------+-----------------
  1  |    4   |   5   |  bars 5,6  |  [0,10) | bars 7-9 (empty, untouched)
  2  |    6   |   7   |  bars 7,8  |  [0,10) | bar 9 (untouched from step 1)
  3  |    8   |   9   |  bars 9,10 |  [1,11) | No (bar 10 = window_end-1)
```

---

## Differences from Standard Inference

| Aspect | Standard autoregressive | Standard infilling | Real-time |
|--------|------------------------|--------------------|-----------|
| Piece state | Static, fully known | Static, fully known | Streaming, partial |
| Conditioning | Full ground truth | Full ground truth | Masked from playhead onward with TOKEN_MASK_BAR |
| Generation target | All bars, one call | Selected bars, one call | One step (j bars) per inference call |
| Bar selection | All agent bars | Specific bars (any order) | Only [t, t+j) |
| Suffix-AR | Optional | No | Always |
| Context window | End-aligned (AR) or centered (infill) | Centered on gap | End-aligned at last generated bar |
| Agent history | N/A | N/A | Previously generated bars are ground truth |
| Future context | Used (non-causal) | Used (non-causal) | Never used (causal: TOKEN_MASK_BAR) |

---

## API Usage

```python
import midigpt
import json

# Build these at each generation step (see loop logic above)
piece_json = json.dumps(sim_piece)   # midi::Piece with future flags set
status_json = json.dumps(status)     # midi::Status with agent bars [t, t+j) selected
params_json = json.dumps({
    "model_dim": D,
    "bars_per_step": j,              # = num_anticipation
    "temperature": 1.0,
    "ckpt": "/path/to/model.pt",
    "tracks_per_step": 1,
})

res_json, attempts = midigpt.sample_multi_step(
    piece_json,
    status_json,
    params_json,
    max_attempts,   # retry limit if generation is empty
    None            # optional CallbackManager
)

res_piece = json.loads(res_json)
# Extract generated bars [t, t+j) from res_piece and merge into global piece state
```

---

## Simulation Script

`python_scripts_for_testing/simulate_realtime_agent.py` implements the full loop above using a pre-recorded MIDI file as a stand-in for live human input. It visualizes the masking state at each step and optionally writes the final piece to MIDI. Run with `--dry_run` to inspect the grid without calling inference.
