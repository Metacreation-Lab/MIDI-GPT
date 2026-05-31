"""Tests for infill sampling via SamplingSession (section 3.16).

Infill = autoregressive=False. `bars` lists the generation targets (bars to
fill in); every OTHER bar in the track supplies context.
"""
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


def _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    return InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)


def _infill_request(track_id: int, target_bars: list[int],
                    **cfg_kw) -> GenerationRequest:
    """`target_bars` are the bars to regenerate; all other bars are context."""
    base = dict(seed=0, max_attempts=3, novelty_check=False, silence_check=False)
    base.update(cfg_kw)
    return GenerationRequest(
        tracks=[TrackPrompt(id=track_id, bars=target_bars,
                            autoregressive=False)],
        config=InferenceConfig(**base),
    )


def test_infill_session_run_returns_score(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _infill_request(track_id=0, target_bars=[2])
    result = engine.session(simple_score, req).run()
    assert isinstance(result, Score)
    assert len(result.tracks) == 1
    assert len(result.tracks[0].bars) == 4


def test_infill_preserves_unmasked_bars(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    """Bars NOT in target_bars should be returned unchanged."""
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    original_bar0_pitches = sorted(n.pitch for n in simple_score.tracks[0].bars[0].notes)
    original_bar3_pitches = sorted(n.pitch for n in simple_score.tracks[0].bars[3].notes)

    req = _infill_request(track_id=0, target_bars=[2])
    result = engine.session(simple_score, req).run()

    out_bar0_pitches = sorted(n.pitch for n in result.tracks[0].bars[0].notes)
    out_bar3_pitches = sorted(n.pitch for n in result.tracks[0].bars[3].notes)
    assert out_bar0_pitches == original_bar0_pitches
    assert out_bar3_pitches == original_bar3_pitches


def test_infill_deterministic_with_fixed_seed(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _infill_request(track_id=0, target_bars=[1])

    torch.manual_seed(0)
    r1 = engine.session(simple_score, req).run()
    torch.manual_seed(0)
    r2 = engine.session(simple_score, req).run()

    n1 = [(n.pitch, n.onset_ticks) for b in r1.tracks[0].bars for n in b.notes]
    n2 = [(n.pitch, n.onset_ticks) for b in r2.tracks[0].bars for n in b.notes]
    assert n1 == n2


def test_infill_does_not_mutate_input_score(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score
):
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    before = [
        [[(n.pitch, n.onset_ticks) for n in b.notes] for b in t.bars]
        for t in simple_score.tracks
    ]
    req = _infill_request(track_id=0, target_bars=[2])
    engine.session(simple_score, req).run()
    after = [
        [[(n.pitch, n.onset_ticks) for n in b.notes] for b in t.bars]
        for t in simple_score.tracks
    ]
    assert before == after


def test_infill_with_two_track_score(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, two_track_score
):
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[2], autoregressive=False),
            TrackPrompt(id=1, bars=[], ignore=True),
        ],
        config=InferenceConfig(seed=0, max_attempts=3,
                               novelty_check=False, silence_check=False),
    )
    result = engine.session(two_track_score, req).run()
    assert len(result.tracks) == 2
