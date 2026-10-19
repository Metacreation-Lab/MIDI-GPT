"""config.infill_probability: an exact, seeded random subset of the infill target bars."""

from __future__ import annotations

import pytest

from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.engine import InferenceEngine
from midigpt.inference.session import _sample_infill_bars
from midigpt.inference.validation import RequestValidationError


def _request(p, seed=0, tracks=None):
    tracks = tracks or [TrackPrompt(id=0, bars=[0, 1, 2, 3]), TrackPrompt(id=1, bars=[0, 1, 2, 3])]
    return GenerationRequest(tracks=tracks, config=InferenceConfig(seed=seed, infill_probability=p))


def _picked(req):
    return sorted((tp.id, b) for tp in req.tracks for b in tp.bars)


@pytest.mark.parametrize("p,n", [(1.0, 8), (0.5, 4), (0.25, 2), (0.2, 2), (0.01, 1)])
def test_exact_count(p, n):
    assert len(_picked(_sample_infill_bars(_request(p), seed=3))) == n


def test_same_seed_same_bars_and_seeds_differ():
    a = _picked(_sample_infill_bars(_request(0.25), seed=1))
    assert a == _picked(_sample_infill_bars(_request(0.25), seed=1))
    assert len({tuple(_picked(_sample_infill_bars(_request(0.25), seed=s))) for s in range(20)}) > 1


def test_picks_are_a_subset_of_the_targets():
    req = _request(0.5, tracks=[TrackPrompt(id=0, bars=[1, 5, 9]), TrackPrompt(id=2, bars=[3])])
    for s in range(10):
        assert set(_picked(_sample_infill_bars(req, seed=s))) <= {(0, 1), (0, 5), (0, 9), (2, 3)}


def test_ar_and_ignored_tracks_untouched():
    tracks = [TrackPrompt(id=0, bars=[0, 1, 2, 3]), TrackPrompt(id=1, bars=[2, 3], autoregressive=True),
              TrackPrompt(id=2, bars=[], ignore=True)]
    out = _sample_infill_bars(_request(0.25, tracks=tracks), seed=0)
    assert len(out.tracks[0].bars) == 1
    assert out.tracks[1].bars == [2, 3]
    assert out.tracks[2].ignore


@pytest.mark.parametrize("p", [0.0, -0.1, 1.5])
def test_out_of_range_rejected(tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score, p):
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    with pytest.raises(RequestValidationError):
        engine.session(simple_score, GenerationRequest(
            tracks=[TrackPrompt(id=0, bars=[1, 2])], config=InferenceConfig(infill_probability=p)))


def test_session_targets_only_the_picked_bars(tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score):
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = GenerationRequest(tracks=[TrackPrompt(id=0, bars=[0, 1, 2, 3])],
                            config=InferenceConfig(seed=7, infill_probability=0.5, model_dim=4))
    sess = engine.session(simple_score, req)
    assert len(sess.generated_bars[0]) == 2
    assert sess._request.tracks[0].bars == sess.generated_bars[0]
    assert sess.generated_bars == engine.session(simple_score, req).generated_bars


def test_full_probability_keeps_every_bar(tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score):
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = GenerationRequest(tracks=[TrackPrompt(id=0, bars=[3, 1])], config=InferenceConfig(model_dim=4))
    assert engine.session(simple_score, req).generated_bars == {0: [1, 3]}
