"""Step planner: which bars each generation step writes, and the window it sees.

Windows follow the targets, not a grid from bar 0:

- AR (one track per step, for each run of consecutive target bars [s, e)): a window of model_dim bars keeps
  H = model_dim - bars_per_step bars of the track's own history before the bars it writes, and generates up to its
  right edge. In a fixed window, writing bars 0-1 then 2-3 is the same as writing 0-3 at once, so a window only
  slides once the generated bars reach its edge: from bar 0 the first step writes the whole window; after that each
  step slides by bars_per_step. The rest of a run that fits in one window ending on it, with at least
  min_overlap_bars of history, is written in that window.
- Infill (tracks_per_step tracks at a time): the target bars of a group are taken in chunks that span at most
  bars_per_step bars, each with a window centred on it (shifted at the piece's edges).

"auto" (0) bars_per_step: AR keeps max(min_overlap_bars, ceil(ar_overlap * model_dim)) bars of history; infill
does a whole window at once. "auto" (0) tracks_per_step: every infill track at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import midigpt._core as _core


@dataclass
class PlannedStep:
    start_bar: int
    end_bar: int
    autoregressive: bool
    bars: set = field(default_factory=set)           # {(track, bar)} to generate

    def to_core(self, n_tracks: int, n_bars: int, ignored: set) -> "_core.GenerationStep":
        st = _core.GenerationStep()
        st.start_bar, st.end_bar, st.is_autoregressive = self.start_bar, self.end_bar, self.autoregressive
        st.bars_to_generate = set(self.bars)
        ctx = [[False] * n_bars for _ in range(n_tracks)]
        tracks = set()
        for t in range(n_tracks):
            if t in ignored:
                continue
            for b in range(self.start_bar, self.end_bar):
                if (t, b) not in self.bars:
                    ctx[t][b] = True
            tracks.add(t)
        st.context = ctx
        st.track_indices = sorted(tracks)
        local = {t: i for i, t in enumerate(st.track_indices)}
        st.bar_mapping = [(local[t], b - self.start_bar, t, b) for t, b in sorted(self.bars)]
        return st

    def summary(self) -> dict:
        by_track: dict[int, list[int]] = {}
        for t, b in sorted(self.bars):
            by_track.setdefault(t, []).append(b)
        return {"window": [self.start_bar, self.end_bar], "autoregressive": self.autoregressive,
                "generate": {str(t): bs for t, bs in by_track.items()}}


def auto_bars_per_step(model_dim: int, autoregressive: bool, ar_overlap: float = 0.5, min_overlap_bars: int = 2) -> int:
    if not autoregressive:
        return model_dim
    history = max(min_overlap_bars, math.ceil(ar_overlap * model_dim))
    return max(1, model_dim - history)


def runs(bars) -> list[tuple[int, int]]:
    """Consecutive runs of sorted bar indices, as [start, end)."""
    out = []
    for b in sorted(set(bars)):
        if out and out[-1][1] == b:
            out[-1][1] = b + 1
        else:
            out.append([b, b + 1])
    return [tuple(r) for r in out]


def plan_ar(track: int, bars, n_bars: int, D: int, B: int, min_history: int = 2) -> list[PlannedStep]:
    H = D - B
    steps = []
    for s, e in runs(bars):
        pos = s
        while pos < e:
            # the rest of the run in one window that ends on it, when that still leaves enough history
            last = min(max(0, e - D), max(0, n_bars - D))
            if e <= last + D and (pos - last >= min(H, min_history) or last == 0) and pos >= last:
                steps.append(PlannedStep(last, min(last + D, n_bars), True, {(track, b) for b in range(pos, e)}))
                break
            ws = max(0, pos - H)
            if ws + D > e:                   # would run past the run's end: end the window on it
                ws = max(0, e - D)
            ws = min(ws, max(0, n_bars - D))
            we = min(ws + D, n_bars)
            end = min(e, we)
            steps.append(PlannedStep(ws, we, True, {(track, b) for b in range(pos, end)}))
            pos = end
    return steps


def plan_infill(group: list[int], targets: dict, n_bars: int, D: int, B: int) -> list[PlannedStep]:
    steps = []
    remaining = sorted({b for t in group for b in targets[t]})
    span = max(1, min(B, D))
    while remaining:
        a = remaining[0]
        chunk = [b for b in remaining if b < a + span]
        z = chunk[-1] + 1
        ws = min(max(0, (a + z - D) // 2), max(0, n_bars - D))
        we = min(ws + D, n_bars)
        gen = {(t, b) for t in group for b in targets[t] if a <= b < z}
        steps.append(PlannedStep(ws, we, False, gen))
        remaining = remaining[len(chunk):]
    return steps


def plan(selected, autoregressive, ignore, D: int, bars_per_step: int = 0, tracks_per_step: int = 0,
         ar_overlap: float = 0.5, min_overlap_bars: int = 2) -> list[PlannedStep]:
    """selected[t][b]: bar b of track t is a target; autoregressive[t] / ignore[t] per track. AR steps first (in
    track order: a later AR track sees the earlier ones' new bars), then infill."""
    nt = len(selected)
    nb = len(selected[0]) if nt else 0
    targets = {t: [b for b in range(nb) if selected[t][b]] for t in range(nt) if not ignore[t]}
    targets = {t: bs for t, bs in targets.items() if bs}
    steps = []
    B_ar = bars_per_step if bars_per_step > 0 else auto_bars_per_step(D, True, ar_overlap, min_overlap_bars)
    for t in sorted(targets):
        if autoregressive[t]:
            steps += plan_ar(t, targets[t], nb, D, min(B_ar, D), min_overlap_bars)
    infill = [t for t in sorted(targets) if not autoregressive[t]]
    T = tracks_per_step if tracks_per_step > 0 else max(1, len(infill))
    B_in = bars_per_step if bars_per_step > 0 else D
    for i in range(0, len(infill), T):
        steps += plan_infill(infill[i:i + T], targets, nb, D, min(B_in, D))
    return steps
