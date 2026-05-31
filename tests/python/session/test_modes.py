"""Tests for SamplingSession mask_mode variants (section 3.17).

Exercises each mask_mode supported by InferenceConfig:
  token, attention, attention_approx, attention_skip, remove.
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


MODES = ("token", "attention", "attention_approx", "attention_skip", "remove")


def _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    return InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)


def _infill_request(mask_mode: str) -> GenerationRequest:
    return GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[2], autoregressive=False)],
        config=InferenceConfig(
            seed=0, max_attempts=3, mask_mode=mask_mode,
            novelty_check=False, silence_check=False,
        ),
    )


@pytest.mark.parametrize("mode", MODES)
def test_each_mask_mode_runs_to_completion(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score, mode
):
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    result = engine.session(simple_score, _infill_request(mode)).run()
    assert isinstance(result, Score)
    assert len(result.tracks) == 1
    assert len(result.tracks[0].bars) == 4


@pytest.mark.parametrize("mode", MODES)
def test_each_mask_mode_is_deterministic_with_seed(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score, mode
):
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _infill_request(mode)

    torch.manual_seed(0)
    r1 = engine.session(simple_score, req).run()
    torch.manual_seed(0)
    r2 = engine.session(simple_score, req).run()

    n1 = [(n.pitch, n.onset_ticks) for b in r1.tracks[0].bars for n in b.notes]
    n2 = [(n.pitch, n.onset_ticks) for b in r2.tracks[0].bars for n in b.notes]
    assert n1 == n2


def test_token_mode_is_default(simple_score):
    """The default mask_mode should be `token`."""
    req = GenerationRequest(tracks=[TrackPrompt(id=0, bars=[0])])
    assert req.config.mask_mode == "token"


@pytest.mark.parametrize("mode", ("attention", "attention_approx", "attention_skip"))
def test_attention_modes_produce_non_silent_output(
    tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score, mode
):
    """Attention-based masking modes should still produce valid scores
    with notes in the target bar."""
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = _infill_request(mode)
    result = engine.session(simple_score, req).run()
    total_notes = sum(len(b.notes) for t in result.tracks for b in t.bars)
    assert total_notes > 0


def test_remove_mode_runs_ar(tiny_gpt2, ghost_tokenizer, ghost_analyzer,
                              simple_score):
    """`remove` mode strips future bars; pair it with an AR request."""
    torch.manual_seed(0)
    engine = _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[2, 3], autoregressive=True)],
        config=InferenceConfig(
            seed=0, max_attempts=2, mask_mode="remove",
            novelty_check=False, silence_check=False,
        ),
    )
    result = engine.session(simple_score, req).run()
    assert isinstance(result, Score)
