"""Tests for `midigpt.inference.validation.validate_request`.

Covers section 3.6 of TEST_IMPLEMENTATION_PLAN.md. Validation raises
`RequestValidationError` (a `ValueError` subclass) on structural problems;
each failing test asserts both the exception type and an identifying message
fragment via `pytest.raises(..., match=...)`.
"""
from __future__ import annotations

import json

import pytest

import midigpt._core as _core
from midigpt.inference.config import (
    GenerationRequest,
    InferenceConfig,
    TrackPrompt,
)
from midigpt.inference.validation import (
    RequestValidationError,
    validate_request,
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _ar_request(track_id: int = 0, bars=None, *, mask_mode: str = "token",
                model_dim: int = 4) -> GenerationRequest:
    """Build a minimal autoregressive request for one track."""
    return GenerationRequest(
        tracks=[TrackPrompt(
            id=track_id,
            bars=list(bars) if bars is not None else [0, 1, 2, 3],
            autoregressive=True,
        )],
        config=InferenceConfig(mask_mode=mask_mode, model_dim=model_dim),
    )


def _config_with_mask_bar_support(ghost_config_json: str, supported: bool) -> _core.EncoderConfig:
    """Build an EncoderConfig with supports_mask_bar_token toggled."""
    cfg = json.loads(ghost_config_json)
    cfg["supports_mask_bar_token"] = bool(supported)
    return _core.EncoderConfig.from_json(json.dumps(cfg))


# --------------------------------------------------------------------------- #
#  Happy path
# --------------------------------------------------------------------------- #
def test_valid_ar_request_passes_through_unchanged(
    simple_score, ghost_config, ghost_analyzer,
):
    """An AR request over the full track must validate and return a request
    whose track list is identical (same TrackPrompt objects)."""
    req = _ar_request(bars=[0, 1, 2, 3])
    out = validate_request(req, simple_score, ghost_config, analyzer=ghost_analyzer)

    assert isinstance(out, GenerationRequest)
    assert out.tracks == req.tracks
    assert out.config.mask_mode == "token"
    assert out.config.model_dim == req.config.model_dim


# --------------------------------------------------------------------------- #
#  Track-id range
# --------------------------------------------------------------------------- #
def test_out_of_range_track_id_raises(simple_score, ghost_config, ghost_analyzer):
    """tp.id >= len(score.tracks) must raise. simple_score has 1 track, so
    id=5 is invalid. The validator catches it via the missing-prompts check
    (every score track requires a TrackPrompt) before the per-track range
    check, but either path raises RequestValidationError."""
    bad = GenerationRequest(
        tracks=[TrackPrompt(id=5, bars=[0, 1, 2, 3], autoregressive=True)],
        config=InferenceConfig(model_dim=4),
    )
    with pytest.raises(RequestValidationError):
        validate_request(bad, simple_score, ghost_config, analyzer=ghost_analyzer)


def test_out_of_range_track_id_with_full_prompt_coverage_raises(
    two_track_score, ghost_config, ghost_analyzer,
):
    """Build a 2-track score and supply prompts for ids 0 and 5 — the
    per-track range check fires explicitly with 'out of range'."""
    bad = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[0, 1, 2, 3], autoregressive=True),
            TrackPrompt(id=1, bars=[], ignore=True),
            TrackPrompt(id=5, bars=[0, 1, 2, 3], autoregressive=True),
        ],
        config=InferenceConfig(model_dim=4),
    )
    with pytest.raises(RequestValidationError, match=r"out of range"):
        validate_request(bad, two_track_score, ghost_config, analyzer=ghost_analyzer)


# --------------------------------------------------------------------------- #
#  Bar-index range / overlap
# --------------------------------------------------------------------------- #
def test_negative_bar_index_raises(simple_score, ghost_config, ghost_analyzer):
    bad = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[-1, 0, 1, 2, 3], autoregressive=True)],
        config=InferenceConfig(model_dim=4),
    )
    with pytest.raises(RequestValidationError, match=r"out of range"):
        validate_request(bad, simple_score, ghost_config, analyzer=ghost_analyzer)


def test_bars_and_mask_bars_overlap_raises(
    simple_score, ghost_config, ghost_analyzer,
):
    """`bars` (generation targets) and `mask_bars` must be disjoint."""
    bad = GenerationRequest(
        tracks=[TrackPrompt(
            id=0,
            bars=[2, 3],       # infill targets
            mask_bars=[2],     # overlaps target 2 → conflict
            autoregressive=False,
        )],
        config=InferenceConfig(model_dim=4, mask_mode="attention"),
    )
    with pytest.raises(RequestValidationError, match=r"mutually exclusive"):
        validate_request(bad, simple_score, ghost_config, analyzer=ghost_analyzer)


# --------------------------------------------------------------------------- #
#  Attribute name / value / level
# --------------------------------------------------------------------------- #
def test_unknown_attribute_name_raises(simple_score, ghost_config, ghost_analyzer):
    bad = GenerationRequest(
        tracks=[TrackPrompt(
            id=0,
            bars=[0, 1, 2, 3],
            autoregressive=True,
            attributes={"definitely_not_a_real_attribute": 0},
        )],
        config=InferenceConfig(model_dim=4),
    )
    with pytest.raises(RequestValidationError, match=r"unknown attribute"):
        validate_request(bad, simple_score, ghost_config, analyzer=ghost_analyzer)


def test_attribute_value_out_of_range_raises(
    simple_score, ghost_config, ghost_analyzer,
):
    """Pick a real track-level attribute and set value = size (just past max)."""
    sizes = ghost_analyzer.attribute_sizes()
    levels = ghost_analyzer.attribute_levels()
    track_types = ghost_analyzer.attribute_track_types()
    candidates = [
        (name, sz) for name, sz in sizes.items()
        if levels.get(name, "track") == "track"
        and sz > 0
        and track_types.get(name, "both") in ("both", "melodic")
    ]
    assert candidates, (
        "ghost_analyzer must expose at least one track-level attribute "
        "compatible with a melodic track"
    )
    name, sz = candidates[0]

    bad = GenerationRequest(
        tracks=[TrackPrompt(
            id=0,
            bars=[0, 1, 2, 3],
            autoregressive=True,
            attributes={name: sz},  # valid is [0, sz)
        )],
        config=InferenceConfig(model_dim=4),
    )
    with pytest.raises(RequestValidationError, match=r"out of range"):
        validate_request(bad, simple_score, ghost_config, analyzer=ghost_analyzer)


def test_bar_level_attribute_placed_on_tp_attributes_raises(
    simple_score, ghost_config, ghost_analyzer,
):
    """Bar-level attributes must go in `bar_attributes`, not `attributes`."""
    levels = ghost_analyzer.attribute_levels()
    bar_level = [n for n, lv in levels.items() if lv == "bar"]
    if not bar_level:
        pytest.skip("ghost_analyzer exposes no bar-level attributes")
    name = bar_level[0]

    bad = GenerationRequest(
        tracks=[TrackPrompt(
            id=0,
            bars=[0, 1, 2, 3],
            autoregressive=True,
            attributes={name: 0},
        )],
        config=InferenceConfig(model_dim=4),
    )
    with pytest.raises(RequestValidationError, match=r"bar-level"):
        validate_request(bad, simple_score, ghost_config, analyzer=ghost_analyzer)


# --------------------------------------------------------------------------- #
#  Controls — time_signature
# --------------------------------------------------------------------------- #
def test_time_signature_control_out_of_range_raises(
    simple_score, ghost_config, ghost_config_json, ghost_analyzer,
):
    """tp.controls['time_signature'] index must be < len(time_signatures)."""
    cfg_dict = json.loads(ghost_config_json)
    ts_count = len(cfg_dict.get("time_signatures") or [])
    assert ts_count > 0, "ghost_config must define at least one time signature"

    bad = GenerationRequest(
        tracks=[TrackPrompt(
            id=0,
            bars=[0, 1, 2, 3],
            autoregressive=True,
            controls={"time_signature": ts_count + 1},
        )],
        config=InferenceConfig(model_dim=4),
    )
    with pytest.raises(RequestValidationError, match=r"time_signature.*out of range"):
        validate_request(bad, simple_score, ghost_config, analyzer=ghost_analyzer)


def test_unknown_control_name_raises(simple_score, ghost_config, ghost_analyzer):
    bad = GenerationRequest(
        tracks=[TrackPrompt(
            id=0,
            bars=[0, 1, 2, 3],
            autoregressive=True,
            controls={"tempo_lock": 120},  # not in KNOWN_CONTROLS
        )],
        config=InferenceConfig(model_dim=4),
    )
    with pytest.raises(RequestValidationError, match=r"unknown control"):
        validate_request(bad, simple_score, ghost_config, analyzer=ghost_analyzer)


# --------------------------------------------------------------------------- #
#  mask_mode vs encoder vocab
# --------------------------------------------------------------------------- #
def test_mask_mode_token_without_vocab_support_raises(
    simple_score, ghost_config_json, ghost_analyzer,
):
    """mask_mode='token' requires a MaskBar entry in token_domains. Build a
    config with supports_mask_bar_token=False — the C++ side drops the
    MaskBar token domain, so validation must reject 'token' mode."""
    unsupported_cfg = _config_with_mask_bar_support(ghost_config_json, False)

    # Sanity: the heuristic ("MaskBar" in token_domains) must return False
    # for this config — otherwise we're not exercising the path under test.
    parsed = json.loads(unsupported_cfg.to_json())
    has_mask_bar = any(
        d.get("type") == "MaskBar" for d in (parsed.get("token_domains") or [])
    )
    assert not has_mask_bar, (
        "supports_mask_bar_token=False must drop the MaskBar token domain "
        "from the serialised config"
    )

    req = _ar_request(mask_mode="token")
    with pytest.raises(RequestValidationError, match=r"MaskBar"):
        validate_request(req, simple_score, unsupported_cfg, analyzer=ghost_analyzer)


def test_mask_mode_token_with_vocab_support_passes_without_masked_bars(
    simple_score, ghost_config_json, ghost_analyzer,
):
    """Validation gates on mask_mode, not on whether any bars are actually
    masked. A 'token' request with empty mask_bars must pass when the
    encoder vocab supports MaskBar."""
    supported_cfg = _config_with_mask_bar_support(ghost_config_json, True)

    parsed = json.loads(supported_cfg.to_json())
    has_mask_bar = any(
        d.get("type") == "MaskBar" for d in (parsed.get("token_domains") or [])
    )
    assert has_mask_bar, (
        "supports_mask_bar_token=True must include a MaskBar token domain "
        "in the serialised config"
    )

    req = _ar_request(mask_mode="token")
    out = validate_request(req, simple_score, supported_cfg, analyzer=ghost_analyzer)
    assert isinstance(out, GenerationRequest)
    assert out.config.mask_mode == "token"
    assert all(tp.mask_bars == [] for tp in out.tracks)
