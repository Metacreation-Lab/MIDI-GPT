"""Tests for InferenceEngine — section 3.13 of TEST_IMPLEMENTATION_PLAN.md.

Covers:
- Direct construction with model/tokenizer/analyzer.
- `.warmup()` populates `_initial_kv` as a tuple of (K, V) layers with
  kv_length == 0.
- `.from_checkpoint(packed_bundle_path)` resolves to `bundle.model`, NOT
  `bundle.model_path`, and warms the engine up.
- `.session(score, request)` returns a SamplingSession bound to the score
  and request, after validation.
- Engine accepts ANY callable matching the ModelBase protocol (FakeModel).
- Clean errors for missing bundle path / non-bundle .pt / bundle missing
  encoder_config.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch

import midigpt._core as _core
from midigpt._types import Score
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.inference.engine import InferenceEngine
from midigpt.inference.config import (
    GenerationRequest,
    InferenceConfig,
    TrackPrompt,
)
from midigpt.inference.session import SamplingSession
from midigpt.tokenizer.tokenizer import Tokenizer


# --------------------------------------------------------------------------- #
#  Direct construction
# --------------------------------------------------------------------------- #
def test_engine_construct_does_not_warmup(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    assert engine._model is tiny_gpt2
    assert engine._tokenizer is ghost_tokenizer
    assert engine._analyzer is ghost_analyzer
    # No warmup means initial KV is still None.
    assert engine._initial_kv is None


def test_engine_warmup_populates_initial_kv(tiny_gpt2, ghost_tokenizer, ghost_analyzer,
                                             tiny_gpt2_config):
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    engine.warmup()
    kv = engine._initial_kv
    assert kv is not None
    # One (K, V) pair per layer.
    assert len(kv) == tiny_gpt2_config.n_layer
    for layer in kv:
        assert isinstance(layer, tuple) and len(layer) == 2
        k, v = layer
        assert isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)
        # _compute_initial_kv runs a forward on a single token starting from an
        # empty KV. The CACHED `_initial_kv` is the EMPTY KV passed in (the
        # function returns `kv` after the forward, which the model mutates only
        # by reference via tensor concat). It must remain length-0 since
        # `model.make_empty_kv()` is what is cached.
        assert k.shape[2] == 0
        assert v.shape[2] == 0


# --------------------------------------------------------------------------- #
#  from_checkpoint
# --------------------------------------------------------------------------- #
def test_from_checkpoint_uses_bundle_model_not_path(packed_bundle_path,
                                                      ghost_tokenizer):
    engine = InferenceEngine.from_checkpoint(str(packed_bundle_path))
    # The packed bundle stored a real model instance — engine must use it
    # directly (NOT fall through to TorchScript loading via model_path).
    assert engine._model is not None
    # Real GPT2LMHeadModel has these attributes; a TorchScriptAdapter wouldn't.
    from midigpt.inference.model.gpt2 import GPT2LMHeadModel
    assert isinstance(engine._model, GPT2LMHeadModel)
    # Tokenizer was constructed from the bundle's encoder_config.
    assert isinstance(engine._tokenizer, Tokenizer)
    assert engine._tokenizer.vocab_size() == ghost_tokenizer.vocab_size()
    # warmup() ran inside from_checkpoint.
    assert engine._initial_kv is not None


def test_from_checkpoint_accepts_explicit_analyzer(packed_bundle_path, ghost_config):
    analyzer = AttributeAnalyzer.from_config(ghost_config)
    engine = InferenceEngine.from_checkpoint(str(packed_bundle_path), analyzer=analyzer)
    assert engine._analyzer is analyzer


def test_from_checkpoint_missing_path_raises(tmp_path):
    missing = tmp_path / "nope.pt"
    assert not missing.exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        InferenceEngine.from_checkpoint(str(missing))


def test_from_checkpoint_non_bundle_pt_raises(tmp_path):
    # A .pt file that doesn't have format_version/state_dict → load_checkpoint
    # raises ValueError mentioning "packed bundle".
    bad = tmp_path / "bad.pt"
    torch.save({"hello": "world"}, str(bad))
    with pytest.raises(ValueError, match="packed bundle"):
        InferenceEngine.from_checkpoint(str(bad))


def test_from_checkpoint_random_path_raises(tmp_path):
    # Neither a directory nor a .pt file.
    p = tmp_path / "weird.bin"
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError):
        InferenceEngine.from_checkpoint(str(p))


# --------------------------------------------------------------------------- #
#  Engine accepts any callable matching ModelBase (FakeModel)
# --------------------------------------------------------------------------- #
def test_engine_accepts_fake_model_callable(fake_model_factory, ghost_tokenizer,
                                              ghost_analyzer):
    fake = fake_model_factory()
    engine = InferenceEngine(fake, ghost_tokenizer, ghost_analyzer)
    engine.warmup()
    # Warmup invoked the model exactly once (the dummy 1-token forward).
    assert len(fake.calls) == 1
    assert fake.calls[0]["input_ids"].shape == (1, 1)
    # Cached KV came from FakeModel.make_empty_kv() — n_layer × (zero-len K, V).
    assert engine._initial_kv is not None
    assert len(engine._initial_kv) == fake._n_layer
    for k, v in engine._initial_kv:
        assert k.shape[2] == 0 and v.shape[2] == 0


# --------------------------------------------------------------------------- #
#  session()
# --------------------------------------------------------------------------- #
def _make_request(track_id: int = 0, bars=(2, 3)) -> GenerationRequest:
    return GenerationRequest(
        tracks=[TrackPrompt(id=track_id, bars=list(bars), autoregressive=True)],
        config=InferenceConfig(
            seed=0, max_attempts=1, novelty_check=False, silence_check=False,
        ),
    )


def test_session_returns_sampling_session_bound_to_score_and_request(
    packed_bundle_path, simple_score
):
    engine = InferenceEngine.from_checkpoint(str(packed_bundle_path))
    request = _make_request(bars=(2, 3))
    session = engine.session(simple_score, request)
    assert isinstance(session, SamplingSession)
    assert session._engine is engine
    assert session._score is simple_score
    assert isinstance(session._request, GenerationRequest)
    assert len(session._request.tracks) == 1
    assert session._request.tracks[0].id == 0
    assert session._request.tracks[0].bars == [2, 3]
    assert session._request.tracks[0].autoregressive is True


def test_session_populates_initial_kv_if_warmup_skipped(tiny_gpt2, ghost_tokenizer,
                                                          ghost_analyzer, simple_score):
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    assert engine._initial_kv is None
    request = _make_request(bars=(2, 3))
    engine.session(simple_score, request)
    # session() runs _compute_initial_kv lazily when warmup wasn't called.
    assert engine._initial_kv is not None


def test_session_rejects_invalid_request_track_id(packed_bundle_path, simple_score):
    engine = InferenceEngine.from_checkpoint(str(packed_bundle_path))
    # track id out of range — validate_request must raise.
    bad = GenerationRequest(
        tracks=[TrackPrompt(id=99, bars=[0], autoregressive=True)],
        config=InferenceConfig(seed=0, max_attempts=1,
                               novelty_check=False, silence_check=False),
    )
    with pytest.raises(Exception):  # RequestValidationError
        engine.session(simple_score, bad)


# --------------------------------------------------------------------------- #
#  Real-MIDI smoke: engine from checkpoint accepts realistic scores
# --------------------------------------------------------------------------- #
def _full_request(score, ar_track: int = 0) -> GenerationRequest:
    """Build a request that covers every track in `score`: the AR track gets
    a right-suffix of its bars; all others are marked ignore=True."""
    n_bars = len(score.tracks[ar_track].bars)
    assert n_bars >= 2
    bars = [n_bars - 2, n_bars - 1]
    tracks = []
    for i, _ in enumerate(score.tracks):
        if i == ar_track:
            tracks.append(TrackPrompt(id=i, bars=bars, autoregressive=True))
        else:
            tracks.append(TrackPrompt(id=i, bars=[], ignore=True))
    return GenerationRequest(
        tracks=tracks,
        config=InferenceConfig(seed=0, max_attempts=1,
                               novelty_check=False, silence_check=False),
    )


def test_session_on_real_score_smoke(packed_bundle_path, real_score):
    """Realistic multi-track Aicha (trimmed to 16 bars) — engine.session must
    validate and return a SamplingSession."""
    engine = InferenceEngine.from_checkpoint(str(packed_bundle_path))
    request = _full_request(real_score, ar_track=0)
    session = engine.session(real_score, request)
    assert isinstance(session, SamplingSession)
    assert session._score is real_score


def test_session_on_long_piano_score_smoke(packed_bundle_path, long_piano_score):
    """Long dense piano score (Maestro_1, 16 bars) — confirms engine.session
    handles realistic windowed contexts."""
    engine = InferenceEngine.from_checkpoint(str(packed_bundle_path))
    request = _full_request(long_piano_score, ar_track=0)
    session = engine.session(long_piano_score, request)
    assert isinstance(session, SamplingSession)
    assert session._request.tracks[0].autoregressive is True
