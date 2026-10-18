"""novelty_threshold: how close a candidate may be to the original before
novelty_check rejects it."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from conftest import melodic_track

from midigpt._types import Score
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.session import SamplingSession
from midigpt.inference.validation import RequestValidationError, validate_request


STEP = SimpleNamespace(bars_to_generate={(0, 0), (0, 1)})
SESS = SimpleNamespace(_request=SimpleNamespace(tracks=[SimpleNamespace(id=0, bars=[0, 1])]))
SESS._is_identical = lambda a, b: SamplingSession._is_identical(SESS, a, b)
SESS._similarity = SamplingSession._similarity


def _too_similar(a, b, threshold):
    return SamplingSession._too_similar(SESS, a, b, STEP, InferenceConfig(novelty_threshold=threshold))


def test_velocity_duration_and_microtiming_do_not_count():
    a = Score(tracks=[melodic_track(n_bars=2, notes_per_bar=4)])
    b = copy.deepcopy(a)
    for bar in b.tracks[0].bars:
        for n in bar.notes:
            n.velocity, n.duration_ticks, n.delta = 30, n.duration_ticks * 2, 5
    assert SamplingSession._similarity(a, b, STEP) == 1.0
    assert _too_similar(a, b, 1.0)
    assert _too_similar(a, b, 0.9)


def test_threshold_rejects_near_copies_only():
    a = Score(tracks=[melodic_track(n_bars=2, notes_per_bar=4)])
    b = copy.deepcopy(a)
    b.tracks[0].bars[1].notes[0].pitch += 1          # 7 of 8 notes shared: 7 / 9
    sim = SamplingSession._similarity(a, b, STEP)
    assert sim == pytest.approx(7 / 9)
    assert not _too_similar(a, b, 1.0)                # exact match only: novel
    assert _too_similar(a, b, 0.75)
    assert not _too_similar(a, b, 0.8)


def test_empty_bars_are_identical():
    a = Score(tracks=[melodic_track(n_bars=2, notes_per_bar=0)])
    assert SamplingSession._similarity(a, copy.deepcopy(a), STEP) == 1.0


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_threshold_range(bad, simple_score, ghost_config, ghost_analyzer):
    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[0, 1, 2, 3], autoregressive=True)],
        config=InferenceConfig(model_dim=4, novelty_threshold=bad),
    )
    with pytest.raises(RequestValidationError, match="novelty_threshold"):
        validate_request(req, simple_score, ghost_config, analyzer=ghost_analyzer)
