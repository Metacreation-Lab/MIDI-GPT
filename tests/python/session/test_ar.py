"""Tests for autoregressive sampling via SamplingSession (section 3.15)."""

from __future__ import annotations

import pytest
import torch

from midigpt._types import Score
from midigpt.inference.config import (
    GenerationRequest,
    InferenceConfig,
    TrackPrompt,
)
from midigpt.inference.engine import InferenceEngine


def _ar_request(track_id: int, bars: list[int], **cfg_kw) -> GenerationRequest:
    base = dict(seed=0, max_attempts=2, novelty_check=False, silence_check=False)
    base.update(cfg_kw)
    return GenerationRequest(
        tracks=[TrackPrompt(id=track_id, bars=bars, autoregressive=True)],
        config=InferenceConfig(**base),
    )


def _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    return InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)


def test_ar_session_run_returns_score(tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score):
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _ar_request(track_id=0, bars=[2, 3])
    session = engine.session(simple_score, req)
    result = session.run()
    assert isinstance(result, Score)
    assert len(result.tracks) == len(simple_score.tracks)


def test_ar_session_run_preserves_track_count(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, two_track_score
):
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[2, 3], autoregressive=True),
            TrackPrompt(id=1, bars=[], ignore=True),
        ],
        config=InferenceConfig(seed=0, max_attempts=2, novelty_check=False, silence_check=False),
    )
    session = engine.session(two_track_score, req)
    result = session.run()
    assert len(result.tracks) == 2


def test_ar_session_run_does_not_mutate_input_score(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    before = [
        [[(n.pitch, n.onset_ticks) for n in b.notes] for b in t.bars] for t in simple_score.tracks
    ]
    req = _ar_request(track_id=0, bars=[2, 3])
    engine.session(simple_score, req).run()
    after = [
        [[(n.pitch, n.onset_ticks) for n in b.notes] for b in t.bars] for t in simple_score.tracks
    ]
    assert before == after


def test_ar_session_deterministic_with_fixed_seed(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _ar_request(track_id=0, bars=[2, 3])

    torch.manual_seed(0)
    r1 = engine.session(simple_score, req).run()
    torch.manual_seed(0)
    r2 = engine.session(simple_score, req).run()

    notes1 = [(n.pitch, n.onset_ticks) for b in r1.tracks[0].bars for n in b.notes]
    notes2 = [(n.pitch, n.onset_ticks) for b in r2.tracks[0].bars for n in b.notes]
    assert notes1 == notes2


def test_ar_session_run_increments_gen_count(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _ar_request(track_id=0, bars=[2, 3])
    session = engine.session(simple_score, req)
    assert session.gen_count == 0
    session.run()
    assert session.gen_count > 0


def test_ar_session_real_score_smoke(tiny_gpt2, ghost_tokenizer, ghost_analyzer, real_score):
    """End-to-end on realistic MIDI (trimmed to fit tiny_gpt2's 512 ctx)."""
    # Take the first track only and trim to 4 bars -- real_score is 14 tracks x 16
    # bars which exceeds tiny_gpt2's n_positions=512 context.
    trimmed = Score(
        tracks=real_score.tracks[:1], resolution=real_score.resolution, tempo=real_score.tempo
    )
    trimmed.tracks[0].bars = trimmed.tracks[0].bars[:4]

    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _ar_request(track_id=0, bars=[2, 3])
    result = engine.session(trimmed, req).run()
    assert isinstance(result, Score)
    assert len(result.tracks) == 1
