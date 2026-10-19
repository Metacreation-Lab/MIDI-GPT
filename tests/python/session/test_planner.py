"""midigpt.inference.planner: windows that follow the targets; "auto" planning fields; plan_only."""

from __future__ import annotations

import copy

import pytest
from conftest import melodic_track
from fastapi.testclient import TestClient

from midigpt._types import Score
from midigpt.http.server import HttpServer, ModelEntry
from midigpt.inference import planner
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.engine import InferenceEngine


def _plan(targets: dict, ar=False, nb=16, D=8, bps=0, tps=0, nt=None):
    nt = nt or max(targets) + 1
    sel = [[False] * nb for _ in range(nt)]
    for t, bs in targets.items():
        for b in bs:
            sel[t][b] = True
    return [s.summary() for s in planner.plan(sel, [ar] * nt, [False] * nt, D, bps, tps)]


def _windows(plan):
    return [(tuple(s["window"]), [b for bs in s["generate"].values() for b in bs]) for s in plan]


def test_ar_from_bar_0_writes_the_first_window_whole_then_slides():
    # auto bars_per_step at 8 bars: 4 bars of history
    assert _windows(_plan({0: range(16)}, ar=True)) == [
        ((0, 8), list(range(8))), ((4, 12), [8, 9, 10, 11]), ((8, 16), [12, 13, 14, 15])]


def test_ar_continuation_keeps_its_history_in_the_window():
    assert _windows(_plan({0: range(13, 16)}, ar=True)) == [((8, 16), [13, 14, 15])]
    # from bar 2 the first window still starts at 0 and writes up to its edge
    assert _windows(_plan({0: range(2, 16)}, ar=True))[0] == ((0, 8), [2, 3, 4, 5, 6, 7])


def test_ar_run_ending_mid_piece_ends_its_last_window_on_the_run():
    assert _windows(_plan({0: range(5, 11)}, ar=True)) == [((3, 11), [5, 6, 7, 8, 9, 10])]


def test_ar_rest_of_the_run_in_one_window_when_the_history_allows():
    assert _windows(_plan({0: range(11, 16)}, ar=True)) == [((8, 16), [11, 12, 13, 14, 15])]   # 3 bars of history
    assert _windows(_plan({0: range(9, 16)}, ar=True)) == [((5, 13), [9, 10, 11, 12]), ((8, 16), [13, 14, 15])]


def test_explicit_bars_per_step_equal_to_window_gives_independent_windows():
    assert _windows(_plan({0: range(16)}, ar=True, bps=8)) == [((0, 8), list(range(8))), ((8, 16), list(range(8, 16)))]


def test_infill_chunk_is_one_centred_window():
    assert _windows(_plan({0: [3, 4]})) == [((0, 8), [3, 4])]          # not split at a grid line
    assert _windows(_plan({0: [13, 14]})) == [((8, 16), [13, 14])]
    assert _windows(_plan({0: [3, 11]}, bps=2)) == [((0, 8), [3]), ((7, 15), [11])]


def test_infill_auto_tracks_per_step_takes_every_track_at_once():
    plan = _plan({0: [1, 2], 1: [2, 3]}, nt=2)
    assert len(plan) == 1 and plan[0]["generate"] == {"0": [1, 2], "1": [2, 3]}
    assert len(_plan({0: [1, 2], 1: [2, 3]}, nt=2, tps=1)) == 2


def test_auto_bars_per_step():
    assert planner.auto_bars_per_step(8, True) == 4
    assert planner.auto_bars_per_step(4, True) == 2
    assert planner.auto_bars_per_step(4, True, min_overlap_bars=3) == 1
    assert planner.auto_bars_per_step(16, False) == 16


def test_to_core_step_fields():
    st = planner.PlannedStep(4, 12, True, {(1, 9), (1, 10)}).to_core(3, 16, ignored={2})
    assert (st.start_bar, st.end_bar, st.is_autoregressive) == (4, 12, True)
    assert list(st.track_indices) == [0, 1]
    assert st.context[1][9] is False and st.context[1][8] is True and st.context[2][5] is False
    assert sorted(st.bar_mapping) == [(1, 5, 1, 9), (1, 6, 1, 10)]


def test_auto_strings_in_the_request():
    cfg = InferenceConfig.from_dict({"model_dim": "auto", "bars_per_step": "auto", "tracks_per_step": "auto"})
    assert (cfg.model_dim, cfg.bars_per_step, cfg.tracks_per_step) == (0, 0, 0)


def _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    return InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)


def test_auto_model_dim_is_the_largest_window_the_piece_fills(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    score = Score(tracks=[melodic_track(n_bars=8)])
    req = GenerationRequest(tracks=[TrackPrompt(id=0, bars=[2])], config=InferenceConfig.from_dict({"model_dim": "auto"}))
    sess = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer).session(score, req)
    assert sess._request.config.model_dim == 8


def test_ar_step_keeps_the_bars_it_was_not_asked_for(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    eng = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    score = Score(tracks=[melodic_track(n_bars=8, notes_per_bar=2)])
    req = GenerationRequest(tracks=[TrackPrompt(id=0, bars=[4, 5, 6, 7], autoregressive=True)],
                            config=InferenceConfig(model_dim=4, bars_per_step=2))
    sess = eng.session(score, req)
    step = planner.PlannedStep(0, 4, True, {(0, 2)}).to_core(1, 8, set())
    before = eng._tokenizer.normalize_output(eng._tokenizer.normalize_input(copy.deepcopy(score)))
    after = copy.deepcopy(before)
    for b in range(4):
        after.tracks[0].bars[b].notes = []
    out = sess._keep_non_targets(score, after, step)
    counts = [len(b.notes) for b in out.tracks[0].bars[:4]]
    assert counts[2] == 0                                   # the target keeps what was generated
    assert counts[0] == counts[1] == counts[3] == 2          # history and the bar after the run are put back


def test_plan_only_returns_the_plan(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    score = Score(tracks=[melodic_track(n_bars=16), melodic_track(n_bars=16)])
    entry = ModelEntry(id="m", engine=_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer), label="m")
    client = TestClient(HttpServer({"m": entry}, default_model="m").app)
    body = {"score": score.to_dict(), "request": {
        "tracks": [{"id": 0, "bars": []}, {"id": 1, "bars": list(range(13, 16)), "autoregressive": True}],
        "config": {"model_dim": "auto", "bars_per_step": "auto", "plan_only": True}}}
    r = client.post("/generate", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "planned"
    assert data["model_dim"] == 16            # auto: the largest window the 16-bar piece fills
    assert data["plan"] == [{"window": [0, 16], "autoregressive": True, "generate": {"1": [13, 14, 15]}}]


@pytest.mark.parametrize("field,value", [("ar_overlap", 1.0), ("min_overlap_bars", -1), ("bars_per_step", -2)])
def test_bad_planning_values_rejected(tiny_gpt2, ghost_tokenizer, ghost_analyzer, field, value):
    from midigpt.inference.validation import RequestValidationError
    score = Score(tracks=[melodic_track(n_bars=8)])
    req = GenerationRequest(tracks=[TrackPrompt(id=0, bars=[2])], config=InferenceConfig(model_dim=4, **{field: value}))
    with pytest.raises(RequestValidationError):
        _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer).session(score, req)
