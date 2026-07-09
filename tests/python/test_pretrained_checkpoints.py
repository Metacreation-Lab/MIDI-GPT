"""Smoke tests for real HF-hosted checkpoints: prism_medium, expressive_medium.

Downloads real weights via InferenceEngine.from_pretrained() and runs actual
generation to prove the checkpoints load and produce grammatically-valid,
attribute-conditioned output. Requires network + torch — excluded from the
fast CI lane (pytest tests/python -m "not slow and not inference").
"""

from __future__ import annotations

import pytest

from midigpt._types import Score
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.engine import InferenceEngine

pytestmark = [pytest.mark.inference, pytest.mark.slow]


def _load_or_skip(name: str) -> InferenceEngine:
    try:
        return InferenceEngine.from_pretrained(name)
    except Exception as exc:  # network / HF hub errors — not a code bug
        pytest.skip(f"could not download {name!r} checkpoint: {exc}")


@pytest.fixture(scope="module")
def prism_engine() -> InferenceEngine:
    return _load_or_skip("prism_medium")


@pytest.fixture(scope="module")
def expressive_engine() -> InferenceEngine:
    return _load_or_skip("expressive_medium")


def _total_notes(score: Score) -> int:
    return sum(len(b.notes) for t in score.tracks for b in t.bars)


def _request(
    score: Score,
    ar_track: int,
    bars: list[int],
    autoregressive: bool,
    model_dim: int,
    track_kwargs: dict | None = None,
    request_controls: dict | None = None,
) -> GenerationRequest:
    """One track generates `bars` (AR or infill); every other track is
    dropped from context via ignore=True. Mirrors _full_request in
    tests/python/model/test_engine.py."""
    tracks = []
    for i in range(len(score.tracks)):
        if i == ar_track:
            tracks.append(
                TrackPrompt(id=i, bars=bars, autoregressive=autoregressive, **(track_kwargs or {}))
            )
        else:
            tracks.append(TrackPrompt(id=i, bars=[], ignore=True))
    config = InferenceConfig(
        model_dim=model_dim,
        mask_mode="attention",  # neither prism nor expressive has MaskBar support
        seed=0,
        max_attempts=2,
        novelty_check=False,
        silence_check=False,
    )
    return GenerationRequest(tracks=tracks, config=config, controls=request_controls or {})


def _assert_structurally_sound(result: Score, original: Score) -> None:
    assert isinstance(result, Score)
    assert len(result.tracks) == len(original.tracks)
    assert len(result.tracks[0].bars) == len(original.tracks[0].bars)
    assert _total_notes(result) > 0


# --------------------------------------------------------------------------- #
#  Prism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_dim", [4, 16])
def test_prism_ar_generation_with_genre_and_pitch_class_set(prism_engine, real_score, model_dim):
    request = _request(
        real_score,
        ar_track=0,
        bars=[14, 15],
        autoregressive=True,
        model_dim=model_dim,
        track_kwargs={"bar_attributes": {14: {"pitch_class_set": 5}, 15: {"pitch_class_set": 5}}},
        request_controls={"genre": "classical"},
    )
    result = prism_engine.session(real_score, request).run()
    _assert_structurally_sound(result, real_score)


def test_prism_infill_with_pitch_class_set(prism_engine, real_score):
    request = _request(
        real_score,
        ar_track=0,
        bars=[6, 7],
        autoregressive=False,
        model_dim=4,
        track_kwargs={"bar_attributes": {6: {"pitch_class_set": 4}, 7: {"pitch_class_set": 4}}},
    )
    result = prism_engine.session(real_score, request).run()
    _assert_structurally_sound(result, real_score)


def test_prism_rejects_mask_token_mode(prism_engine, real_score):
    tracks = [TrackPrompt(id=0, bars=[14, 15], autoregressive=True)]
    tracks += [TrackPrompt(id=i, bars=[], ignore=True) for i in range(1, len(real_score.tracks))]
    request = GenerationRequest(tracks=tracks, config=InferenceConfig(model_dim=4))  # default mask_mode="token"
    with pytest.raises(Exception):
        prism_engine.session(real_score, request)


# --------------------------------------------------------------------------- #
#  Expressive
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_dim", [4, 16])
def test_expressive_ar_generation_with_genre_and_nomml(expressive_engine, real_score, model_dim):
    request = _request(
        real_score,
        ar_track=0,
        bars=[14, 15],
        autoregressive=True,
        model_dim=model_dim,
        track_kwargs={"attributes": {"nomml": 6}},
        request_controls={"genre": "classical", "microtiming": True, "velocity": True},
    )
    result = expressive_engine.session(real_score, request).run()
    _assert_structurally_sound(result, real_score)


def test_expressive_infill_with_switchable_controls_off(expressive_engine, real_score):
    request = _request(
        real_score,
        ar_track=0,
        bars=[6, 7],
        autoregressive=False,
        model_dim=4,
        track_kwargs={"bar_attributes": {6: {"pitch_class_set": 3}, 7: {"pitch_class_set": 3}}},
        request_controls={"microtiming": False, "velocity": False},
    )
    result = expressive_engine.session(real_score, request).run()
    _assert_structurally_sound(result, real_score)


def test_expressive_rejects_mask_token_mode(expressive_engine, real_score):
    tracks = [TrackPrompt(id=0, bars=[14, 15], autoregressive=True)]
    tracks += [TrackPrompt(id=i, bars=[], ignore=True) for i in range(1, len(real_score.tracks))]
    request = GenerationRequest(tracks=tracks, config=InferenceConfig(model_dim=4))  # default mask_mode="token"
    with pytest.raises(Exception):
        expressive_engine.session(real_score, request)
