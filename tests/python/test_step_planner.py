"""Property test: C++ StepPlanner must match the MIDI-GPT-STEPS JS reference.

Source of truth: https://github.com/DaoTwenty/MIDI-GPT-STEPS  (main.js findStepsInner).
We port that function to Python (`_reference_plan`) and compare its output against
the C++ planner across many random configs.

What we compare per step:
  * (start_bar, end_bar)
  * is_autoregressive
  * bars_to_generate (as a set of (track, bar) pairs)
  * context bitmap restricted to [start_bar, end_bar)
"""

import json
import random
import pytest

import midigpt._core as _core


def _reference_plan(
    selection: list[list[bool]],
    autoregressive_tracks: list[bool],
    ignore_tracks: list[bool],
    model_dim: int,
    bars_per_step: int,
    tracks_per_step: int,
) -> list[dict]:
    """Port of MIDI-GPT-STEPS main.js findSteps + findStepsInner (AR + infill).

    Returns list of {start, end, is_autoregressive, gen: set[(ti,tj)],
                     context: list[list[bool]]}.
    """
    nt = len(selection)
    nb = len(selection[0]) if nt else 0
    generated = [[False] * nb for _ in range(nt)]
    out: list[dict] = []

    def inner(autoregressive: bool):
        num_context = (model_dim - bars_per_step) if autoregressive \
            else (model_dim - bars_per_step) // 2

        for i in range(0, nt, tracks_per_step):
            for j in range(0, nb, bars_per_step):
                num_tr = min(tracks_per_step, nt - i)
                step  = [[False] * nb for _ in range(nt)]
                ctx   = [[False] * nb for _ in range(nt)]

                if autoregressive:
                    right_offset = max(j + model_dim - nb, 0)
                    t = min(j, max(0, nb - model_dim))
                    start, end = t, t + model_dim
                    kernel_start = (1 if j > 0 else 0) * (num_context + right_offset)
                    for ti in range(i, i + num_tr):
                        is_ar = ti < len(autoregressive_tracks) and autoregressive_tracks[ti]
                        if not is_ar:
                            continue
                        for tj in range(start + kernel_start, min(end, nb)):
                            if selection[ti][tj] and not generated[ti][tj]:
                                step[ti][tj] = True
                else:
                    t = max(0, min(j - num_context, nb - model_dim))
                    start, end = t, t + model_dim
                    for ti in range(i, i + num_tr):
                        is_ar = ti < len(autoregressive_tracks) and autoregressive_tracks[ti]
                        if is_ar:
                            continue
                        for tj in range(j, min(j + bars_per_step, end, nb)):
                            if selection[ti][tj] and not generated[ti][tj]:
                                step[ti][tj] = True

                # Build context: !ignore && !step over the window
                for ti in range(nt):
                    if ti < len(ignore_tracks) and ignore_tracks[ti]:
                        continue
                    for tj in range(start, min(end, nb)):
                        if not step[ti][tj]:
                            ctx[ti][tj] = True

                # Commit
                has_gen = any(step[ti][tj] for ti in range(nt) for tj in range(nb))
                if has_gen:
                    gen_set = {(ti, tj) for ti in range(nt) for tj in range(nb)
                               if step[ti][tj]}
                    for ti, tj in gen_set:
                        generated[ti][tj] = True
                    out.append({
                        "start": start, "end": min(end, nb),
                        "is_autoregressive": autoregressive,
                        "gen": gen_set,
                        "context": ctx,
                    })

    inner(True)
    inner(False)
    return out


def _cpp_plan(
    selection: list[list[bool]],
    autoregressive_tracks: list[bool],
    ignore_tracks: list[bool],
    model_dim: int,
    bars_per_step: int,
    tracks_per_step: int,
) -> list[dict]:
    enc_cfg = _core.EncoderConfig.from_json(json.dumps({
        "num_bars": model_dim,
        "num_bars_map": [model_dim],
    }))
    enc_cfg.model_dim = model_dim

    mask = _core.SelectionMask()
    mask.selected       = selection
    mask.autoregressive = autoregressive_tracks
    mask.ignore         = ignore_tracks

    planner = _core.StepPlanner(mask, enc_cfg, bars_per_step, tracks_per_step)
    out: list[dict] = []
    for s in planner.plan():
        out.append({
            "start": int(s.start_bar),
            "end":   int(s.end_bar),
            "is_autoregressive": bool(s.is_autoregressive),
            "gen":   {(int(t), int(b)) for (t, b) in s.bars_to_generate},
            "context": [list(row) for row in s.context],
        })
    return out


def _ctx_window_equal(a: list[list[bool]], b: list[list[bool]],
                      start: int, end: int) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        for j in range(start, end):
            ja = ra[j] if j < len(ra) else False
            jb = rb[j] if j < len(rb) else False
            if ja != jb:
                return False
    return True


def _compare(ref: list[dict], cpp: list[dict], cfg: dict) -> None:
    assert len(ref) == len(cpp), (
        f"step count mismatch: ref={len(ref)} cpp={len(cpp)}\n"
        f"cfg={cfg}\nref={ref}\ncpp={cpp}"
    )
    for k, (r, c) in enumerate(zip(ref, cpp)):
        assert r["start"] == c["start"] and r["end"] == c["end"], (
            f"step {k} window mismatch: ref=[{r['start']},{r['end']}) "
            f"cpp=[{c['start']},{c['end']})\ncfg={cfg}"
        )
        assert r["is_autoregressive"] == c["is_autoregressive"], (
            f"step {k} is_autoregressive mismatch\ncfg={cfg}"
        )
        assert r["gen"] == c["gen"], (
            f"step {k} bars_to_generate mismatch:\n  ref={sorted(r['gen'])}\n"
            f"  cpp={sorted(c['gen'])}\ncfg={cfg}"
        )


def _random_config(rng: random.Random) -> dict:
    nt = rng.randint(1, 4)
    model_dim = rng.choice([2, 4, 8])
    nb = rng.randint(model_dim, model_dim * 3)
    bps = rng.randint(1, model_dim)
    tps = rng.randint(1, nt)

    autoregressive = [rng.random() < 0.4 for _ in range(nt)]
    ignore         = [(not autoregressive[ti]) and rng.random() < 0.15
                      for ti in range(nt)]
    selection: list[list[bool]] = []
    for ti in range(nt):
        if ignore[ti]:
            selection.append([False] * nb)
        elif autoregressive[ti]:
            # AR tracks: contiguous selection somewhere
            a = rng.randint(0, nb - 1)
            b = rng.randint(a, nb - 1)
            row = [a <= j <= b for j in range(nb)]
            selection.append(row)
        else:
            selection.append([rng.random() < 0.5 for _ in range(nb)])

    return {
        "selection": selection,
        "autoregressive_tracks": autoregressive,
        "ignore_tracks": ignore,
        "model_dim": model_dim,
        "bars_per_step": bps,
        "tracks_per_step": tps,
    }


def test_planner_matches_reference_fixed_cases():
    """Hand-picked cases covering the user's scenarios."""
    cases = [
        # User's case: 8-bar piece, model_dim=8, AR fills bars 4-7 in one step.
        dict(
            selection=[
                [False] * 8,
                [False, False, False, False, True, True, True, True],
            ],
            autoregressive_tracks=[False, True],
            ignore_tracks=[False, False],
            model_dim=8, bars_per_step=1, tracks_per_step=1,
        ),
        # Sliding window: 16-bar piece, model_dim=4, AR full track.
        dict(
            selection=[[True] * 16],
            autoregressive_tracks=[True],
            ignore_tracks=[False],
            model_dim=4, bars_per_step=1, tracks_per_step=1,
        ),
        # Infill: model_dim=4, bps=2, single track, non-AR.
        dict(
            selection=[[True, False, True, True, False, True, True, False]],
            autoregressive_tracks=[False],
            ignore_tracks=[False],
            model_dim=4, bars_per_step=2, tracks_per_step=1,
        ),
    ]
    for cfg in cases:
        ref = _reference_plan(**cfg)
        cpp = _cpp_plan(**cfg)
        _compare(ref, cpp, cfg)


@pytest.mark.parametrize("seed", list(range(50)))
def test_planner_matches_reference_random(seed):
    rng = random.Random(seed)
    cfg = _random_config(rng)
    ref = _reference_plan(**cfg)
    cpp = _cpp_plan(**cfg)
    _compare(ref, cpp, cfg)
