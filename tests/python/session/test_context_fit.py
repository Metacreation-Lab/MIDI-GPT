"""Context fitting: whole context tracks are left out when a step would not fit the model (context_fit)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import drum_track, make_bar, make_note, melodic_track

from midigpt._types import Score, Track
from midigpt.inference import context_fit
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.engine import InferenceEngine


def _hats(n_bars=4):
    bars = [make_bar([make_note(pitch=42, onset=i * 3, dur=3) for i in range(16)]) for _ in range(n_bars)]
    return Track(bars=bars, instrument=0, track_type="drum")


def _dense_score(n_bars=4):
    # target track 0 (melodic) and five busy context tracks: far more than the tiny model's 512 positions
    return Score(tracks=[melodic_track(n_bars=n_bars, notes_per_bar=4), melodic_track(n_bars, 16, base_pitch=48),
                         melodic_track(n_bars, 16, base_pitch=72), _hats(n_bars), drum_track(n_bars),
                         melodic_track(n_bars, 16, base_pitch=60)])


def _session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, score, **cfg):
    eng = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    tracks = [TrackPrompt(id=0, bars=[1, 2])] + [TrackPrompt(id=i, bars=[]) for i in range(1, len(score.tracks))]
    return eng.session(score, GenerationRequest(tracks=tracks, config=InferenceConfig(model_dim=4, mask_mode="attention", **cfg)))


def test_nothing_dropped_when_it_fits(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    score = Score(tracks=[melodic_track(n_bars=4, notes_per_bar=2), melodic_track(n_bars=4, notes_per_bar=2)])
    sess = _session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, score)
    assert sess.dropped_context == []


def test_auto_drops_context_tracks_never_targets(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    sess = _session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, _dense_score())
    assert sess.dropped_context and 0 not in sess.dropped_context
    assert 0 in sess._kept and len(sess._kept) == 6 - len(sess.dropped_context)


def test_context_all_never_drops(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    sess = _session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, _dense_score(), context="all")
    assert sess.dropped_context == []


def test_context_keep_is_kept_and_context_drop_goes_first(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    auto = _session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, _dense_score()).dropped_context
    keep = auto[0]
    sess = _session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, _dense_score(), context_keep=[keep])
    assert keep not in sess.dropped_context
    first = next(t for t in range(1, 6) if t not in auto)
    sess = _session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, _dense_score(), context_drop=[first])
    assert first in sess.dropped_context


def test_listed_drops_only_the_list(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    sess = _session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, _dense_score(), context="listed", context_drop=[4])
    assert sess.dropped_context == [4]


def test_bad_context_values_rejected(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    from midigpt.inference.validation import RequestValidationError
    with pytest.raises(RequestValidationError):
        _session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, _dense_score(), context="some")
    with pytest.raises(RequestValidationError):
        _session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, _dense_score(), context_drop=[9])


def test_roles_drop_hats_before_bass_for_a_melodic_target():
    score = Score(tracks=[melodic_track(4, 4, base_pitch=72), melodic_track(4, 1, base_pitch=36), _hats()])
    order = context_fit.rank_by_roles(score, [0], [1, 2], {1: 100, 2: 100})
    assert order == [2, 1]
    assert context_fit.track_role(score.tracks[1], score.resolution) == "bass"
    assert context_fit.track_role(score.tracks[2], score.resolution) == "hats"


def test_rank_by_attention_per_token():
    att = np.array([0.0, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.4])
    owners = [-1, 1, 1, 2, 2, 2, 2, 0]
    # track 1: 1.0 over 2 tokens; track 2: 0.4 over 4 tokens -> track 2 goes first
    assert context_fit.rank_by_attention(att, owners, [1, 2], {1: 2, 2: 4}) == [2, 1]


def test_attention_from_last_reads_the_model(tiny_gpt2):
    a = context_fit.attention_from_last(tiny_gpt2, list(range(1, 40)))
    assert a.shape == (39,) and a.sum() > 0


def _two_target_session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, **cfg):
    # two busy infill targets over the whole window: generating both at once cannot fit 512 positions
    score = Score(tracks=[melodic_track(4, 16, base_pitch=48), melodic_track(4, 16, base_pitch=72), melodic_track(4, 2)])
    eng = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    tracks = [TrackPrompt(id=0, bars=[0, 1, 2, 3]), TrackPrompt(id=1, bars=[0, 1, 2, 3]), TrackPrompt(id=2, bars=[])]
    return eng.session(score, GenerationRequest(tracks=tracks, config=InferenceConfig(model_dim=4, mask_mode="attention", **cfg)))


def test_auto_steps_split_when_dropping_is_not_allowed(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    sess = _two_target_session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, context="all",
                               tracks_per_step=0, bars_per_step=0)
    plan = sess.plan_summary()
    assert sess.dropped_context == []
    assert len(plan) > 1
    assert all(len(st["generate"]) == 1 for st in plan)            # one track per step first


def test_fixed_steps_are_not_split(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    sess = _two_target_session(tiny_gpt2, ghost_tokenizer, ghost_analyzer, context="all",
                               tracks_per_step=2, bars_per_step=4)
    assert len(sess.plan_summary()) == 1
