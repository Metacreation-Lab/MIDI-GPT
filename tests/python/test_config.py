"""Tests for midigpt.inference.config (section 3.3)."""
from __future__ import annotations
import pytest
from midigpt.inference.config import InferenceConfig, TrackPrompt, GenerationRequest


def test_inference_config_defaults():
    cfg = InferenceConfig()
    assert cfg.temperature == 1.0
    assert cfg.seed == -1
    assert cfg.mask_mode == "token"
    assert cfg.polyphony_hard_limit == 0
    assert cfg.density_hard_limit == 0
    assert cfg.top_p == 1.0
    assert cfg.top_k == 0
    assert cfg.mask_p == 0.0
    assert cfg.mask_k == 0
    assert cfg.bars_per_step == 1
    assert cfg.tracks_per_step == 1
    assert cfg.model_dim == 4
    assert cfg.shuffle is False
    assert cfg.novelty_check is True
    assert cfg.silence_check is True
    assert cfg.max_attempts == 3
    assert cfg.temperature_escalation == 1.0


def test_track_prompt_defaults():
    tp = TrackPrompt(id=0, bars=[0])
    assert tp.autoregressive is False
    assert tp.ignore is False
    assert tp.mask_bars == []
    assert tp.attributes == {}
    assert tp.controls == {}
    assert tp.bar_attributes == {}
    assert tp.bar_controls == {}


def test_track_prompt_attributes_independent():
    tp1 = TrackPrompt(id=0, bars=[0])
    tp2 = TrackPrompt(id=1, bars=[0])
    tp1.attributes["note_density"] = 5
    assert "note_density" not in tp2.attributes


def test_track_prompt_controls_independent():
    tp1 = TrackPrompt(id=0, bars=[0])
    tp2 = TrackPrompt(id=1, bars=[0])
    tp1.controls["time_signature"] = 0
    assert "time_signature" not in tp2.controls


def test_track_prompt_bar_attributes_independent():
    tp1 = TrackPrompt(id=0, bars=[0])
    tp2 = TrackPrompt(id=1, bars=[0])
    tp1.bar_attributes[0] = {"tension": 3}
    assert 0 not in tp2.bar_attributes


def test_track_prompt_mask_bars_independent():
    tp1 = TrackPrompt(id=0, bars=[0])
    tp2 = TrackPrompt(id=1, bars=[0])
    tp1.mask_bars.append(1)
    assert 1 not in tp2.mask_bars


def test_generation_request_default_config():
    req = GenerationRequest(tracks=[TrackPrompt(id=0, bars=[0])])
    assert isinstance(req.config, InferenceConfig)
    assert req.config.temperature == 1.0


def test_generation_request_custom_config():
    cfg = InferenceConfig(temperature=0.8, seed=42)
    req = GenerationRequest(tracks=[TrackPrompt(id=0, bars=[0])], config=cfg)
    assert req.config.temperature == 0.8
    assert req.config.seed == 42


def test_inference_config_independent_instances():
    c1 = InferenceConfig()
    c2 = InferenceConfig()
    # dataclass fields are value-typed; mutation of one shouldn't affect other
    # (all fields are scalars, so this is a basic sanity check)
    assert c1 is not c2
    assert c1.temperature == c2.temperature


def test_track_prompt_with_multiple_bars():
    tp = TrackPrompt(id=2, bars=[0, 1, 2, 3])
    assert tp.id == 2
    assert tp.bars == [0, 1, 2, 3]


def test_inference_config_mask_modes():
    for mode in ("token", "attention", "attention_approx", "attention_skip", "remove"):
        cfg = InferenceConfig(mask_mode=mode)
        assert cfg.mask_mode == mode


def test_inference_config_all_fields_settable():
    cfg = InferenceConfig(
        temperature=0.5,
        seed=7,
        mask_mode="remove",
        polyphony_hard_limit=3,
        density_hard_limit=8,
        top_p=0.9,
        top_k=50,
        mask_p=0.1,
        mask_k=2,
        bars_per_step=2,
        tracks_per_step=2,
        model_dim=8,
        shuffle=True,
        novelty_check=False,
        silence_check=False,
        max_attempts=5,
        temperature_escalation=1.2,
    )
    assert cfg.temperature == 0.5
    assert cfg.seed == 7
    assert cfg.mask_mode == "remove"
    assert cfg.polyphony_hard_limit == 3
    assert cfg.density_hard_limit == 8
    assert cfg.top_p == 0.9
    assert cfg.top_k == 50
    assert cfg.mask_p == 0.1
    assert cfg.mask_k == 2
    assert cfg.bars_per_step == 2
    assert cfg.tracks_per_step == 2
    assert cfg.model_dim == 8
    assert cfg.shuffle is True
    assert cfg.novelty_check is False
    assert cfg.silence_check is False
    assert cfg.max_attempts == 5
    assert cfg.temperature_escalation == 1.2
