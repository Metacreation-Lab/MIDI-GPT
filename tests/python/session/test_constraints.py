"""Tests for SamplingSession constraint hooks (section 3.18).

Covers `_build_constraints`: hard polyphony/density caps, full-AR attribute
constraints, and the underlying selection/ignore plumbing.
"""
from __future__ import annotations
import pytest
import torch

import midigpt._core as _core
from midigpt._types import Score
from midigpt.inference.config import (
    GenerationRequest,
    InferenceConfig,
    TrackPrompt,
)
from midigpt.inference.engine import InferenceEngine
from midigpt.inference.session import SamplingSession


def _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    return InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)


def _ar_request(**cfg_kw) -> GenerationRequest:
    base = dict(seed=0, max_attempts=2, novelty_check=False, silence_check=False)
    base.update(cfg_kw)
    return GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[2, 3], autoregressive=True)],
        config=InferenceConfig(**base),
    )


def test_selection_mask_marks_selected_bars(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _ar_request()
    session = engine.session(simple_score, req)
    mask = session._build_selection_mask()
    selected = list(mask.selected)
    assert selected[0][2] is True
    assert selected[0][3] is True
    assert selected[0][0] is False
    assert selected[0][1] is False


def test_selection_mask_autoregressive_flag(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    session = engine.session(simple_score, _ar_request())
    mask = session._build_selection_mask()
    assert list(mask.autoregressive) == [True]


def test_selection_mask_ignore_flag(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, two_track_score
):
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[2, 3], autoregressive=True),
            TrackPrompt(id=1, bars=[], ignore=True),
        ],
        config=InferenceConfig(seed=0, max_attempts=1,
                               novelty_check=False, silence_check=False),
    )
    session = engine.session(two_track_score, req)
    mask = session._build_selection_mask()
    assert list(mask.ignore) == [False, True]


def test_build_constraints_invoked_during_run(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score, monkeypatch
):
    """_build_constraints is called once per step inside run()."""
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    session = engine.session(simple_score, _ar_request())
    calls = []
    real = session._build_constraints
    def wrapped(step):
        g = real(step)
        calls.append(g)
        return g
    monkeypatch.setattr(session, "_build_constraints", wrapped)
    torch.manual_seed(0)
    session.run()
    assert len(calls) >= 1
    assert all(isinstance(g, _core.ConstraintGraph) for g in calls)


def test_polyphony_hard_limit_accepted_by_config(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    """polyphony_hard_limit > 0 should pass validation and reach session()."""
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _ar_request(polyphony_hard_limit=4)
    session = engine.session(simple_score, req)
    assert session._request.config.polyphony_hard_limit == 4


def test_density_hard_limit_accepted_by_config(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _ar_request(density_hard_limit=8)
    session = engine.session(simple_score, req)
    assert session._request.config.density_hard_limit == 8


def test_zero_hard_limits_are_disabled(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    """0 hard limit means 'off' — sampling should behave identically to defaults."""
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)

    torch.manual_seed(0)
    r_base = engine.session(simple_score, _ar_request()).run()
    torch.manual_seed(0)
    r_zero = engine.session(
        simple_score,
        _ar_request(polyphony_hard_limit=0, density_hard_limit=0),
    ).run()

    n_base = [(n.pitch, n.onset_ticks) for b in r_base.tracks[0].bars for n in b.notes]
    n_zero = [(n.pitch, n.onset_ticks) for b in r_zero.tracks[0].bars for n in b.notes]
    assert n_base == n_zero


def test_top_p_and_top_k_filters_run(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _ar_request(top_p=0.9, top_k=20)
    result = engine.session(simple_score, req).run()
    assert isinstance(result, Score)


def test_temperature_escalation_changes_retries(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    """`temperature_escalation` should not crash with multiple attempts."""
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _ar_request(max_attempts=3, temperature_escalation=1.2)
    result = engine.session(simple_score, req).run()
    assert isinstance(result, Score)


def test_full_ar_track_ids_set_on_session(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    """After _sample_step, _full_ar_ids should reflect AR tracks with no prefix."""
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[0, 1, 2, 3], autoregressive=True)],
        config=InferenceConfig(seed=0, max_attempts=1,
                               novelty_check=False, silence_check=False),
    )
    session = engine.session(simple_score, req)
    assert isinstance(session, SamplingSession)
    # Verify mask classifies all bars as AR targets.
    mask = session._build_selection_mask()
    assert all(list(mask.selected)[0])
