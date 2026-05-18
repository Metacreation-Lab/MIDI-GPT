import json
import pytest
from dataclasses import replace

from midigpt_refactor._types import Score, Track, Bar, Note
from midigpt_refactor.attributes.base import AttributeAnalyzer
from midigpt_refactor.inference.config import (
    GenerationRequest, SamplingConfig, TrackPrompt,
)
from midigpt_refactor.inference.validation import (
    validate_request, RequestValidationError,
)
import midigpt_refactor._core as _core


CONFIG_JSON = json.dumps({
    "supports_infill": True,
    "num_bars_map": [4, 8],
    "time_signatures": ["4/4", "3/4"],
    "token_domains": [
        {"type": "PieceStart", "domain_size": 1},
        {"type": "PieceEnd", "domain_size": 1},
        {"type": "Track", "domain_size": 2},
        {"type": "TrackEnd", "domain_size": 1},
        {"type": "Bar", "domain_size": 1},
        {"type": "BarEnd", "domain_size": 1},
        {"type": "Instrument", "domain_size": 128},
        {"type": "TimeSig", "domain_size": 32},
        {"type": "NoteOnset", "domain_size": 128},
        {"type": "NoteDuration", "domain_size": 128},
        {"type": "VelocityLevel", "domain_size": 128},
        {"type": "DeltaDirection", "domain_size": 2},
        {"type": "Delta", "domain_size": 128},
        {"type": "NoteDensity", "domain_size": 128},
    ],
    "attribute_controls": [
        {"name": "note_density"},
    ],
})

CONFIG_NO_INFILL_JSON = json.dumps({
    **json.loads(CONFIG_JSON),
    "supports_infill": False,
})


def _config(json_str=CONFIG_JSON):
    return _core.EncoderConfig.from_json(json_str)


def _analyzer(cfg):
    return AttributeAnalyzer.from_config(cfg)


def _score(bars_per_track=4, n_tracks=1, ts=(4, 4)):
    return Score(tracks=[
        Track(
            bars=[Bar(notes=[Note(pitch=60, velocity=100,
                                  onset_ticks=0, duration_ticks=480)],
                      ts_numerator=ts[0], ts_denominator=ts[1])
                  for _ in range(bars_per_track)],
            instrument=0, track_type="melodic")
        for _ in range(n_tracks)
    ])


def _req(tracks=None, **cfg_kw):
    tracks = tracks if tracks is not None else [TrackPrompt(id=0, bars=[0])]
    defaults = dict(max_attempts=1, silence_check=False, novelty_check=False)
    defaults.update(cfg_kw)
    return GenerationRequest(tracks=tracks, config=SamplingConfig(**defaults))


def test_happy_path():
    cfg, an = _config(), _analyzer(_config())
    out = validate_request(_req(), _score(), cfg, an)
    assert out.config.model_dim == 4  # defaulted from num_bars_map


def test_model_dim_default_from_num_bars_map():
    cfg, an = _config(), _analyzer(_config())
    r = _req()
    r = replace(r, config=replace(r.config, model_dim=0))
    out = validate_request(r, _score(), cfg, an)
    assert out.config.model_dim == 4


def test_model_dim_not_in_domain_errors():
    cfg, an = _config(), _analyzer(_config())
    r = _req()
    r = replace(r, config=replace(r.config, model_dim=7))
    with pytest.raises(RequestValidationError, match="model_dim"):
        validate_request(r, _score(), cfg, an)


def test_bars_per_step_exceeds_model_dim():
    cfg, an = _config(), _analyzer(_config())
    r = _req(bars_per_step=5)
    with pytest.raises(RequestValidationError, match="bars_per_step"):
        validate_request(r, _score(), cfg, an)


def test_bars_per_step_zero():
    cfg, an = _config(), _analyzer(_config())
    r = _req(bars_per_step=0)
    with pytest.raises(RequestValidationError, match="bars_per_step"):
        validate_request(r, _score(), cfg, an)


def test_tracks_per_step_zero():
    cfg, an = _config(), _analyzer(_config())
    r = _req(tracks_per_step=0)
    with pytest.raises(RequestValidationError, match="tracks_per_step"):
        validate_request(r, _score(), cfg, an)


def test_max_attempts_zero():
    cfg, an = _config(), _analyzer(_config())
    r = _req(max_attempts=0)
    with pytest.raises(RequestValidationError, match="max_attempts"):
        validate_request(r, _score(), cfg, an)


def test_temperature_nonpositive():
    cfg, an = _config(), _analyzer(_config())
    r = _req(temperature=0.0)
    with pytest.raises(RequestValidationError, match="temperature"):
        validate_request(r, _score(), cfg, an)


def test_temperature_escalation_below_one_errors():
    cfg, an = _config(), _analyzer(_config())
    r = _req(temperature_escalation=0.9)
    with pytest.raises(RequestValidationError, match="temperature_escalation"):
        validate_request(r, _score(), cfg, an)


def test_temperature_escalation_clamped_above_max():
    cfg, an = _config(), _analyzer(_config())
    r = _req(temperature_escalation=10.0)
    out = validate_request(r, _score(), cfg, an)
    assert out.config.temperature_escalation == 3.0


def test_score_too_short():
    cfg, an = _config(), _analyzer(_config())
    with pytest.raises(RequestValidationError, match="bars"):
        validate_request(_req(), _score(bars_per_track=2), cfg, an)


def test_score_no_tracks():
    cfg, an = _config(), _analyzer(_config())
    with pytest.raises(RequestValidationError, match="0 tracks"):
        validate_request(_req(), Score(tracks=[]), cfg, an)


def test_tracks_with_zero_bars():
    cfg, an = _config(), _analyzer(_config())
    s = Score(tracks=[Track(bars=[], instrument=0, track_type="melodic")])
    with pytest.raises(RequestValidationError, match="at least one bar"):
        validate_request(_req(), s, cfg, an)


def test_inconsistent_bar_counts():
    cfg, an = _config(), _analyzer(_config())
    s = _score(bars_per_track=4, n_tracks=2)
    s.tracks[1].bars = s.tracks[1].bars[:3]
    with pytest.raises(RequestValidationError, match="same number of bars"):
        validate_request(_req(), s, cfg, an)


def test_unknown_time_signature():
    cfg, an = _config(), _analyzer(_config())
    s = _score(ts=(7, 8))
    with pytest.raises(RequestValidationError, match="time signature"):
        validate_request(_req(), s, cfg, an)


def test_duplicate_track_id():
    cfg, an = _config(), _analyzer(_config())
    s = _score(n_tracks=1)
    r = _req(tracks=[TrackPrompt(id=0, bars=[0]),
                     TrackPrompt(id=0, bars=[1])])
    with pytest.raises(RequestValidationError, match="duplicate"):
        validate_request(r, s, cfg, an)


def test_track_id_out_of_range():
    cfg, an = _config(), _analyzer(_config())
    r = _req(tracks=[TrackPrompt(id=5, bars=[0])])
    with pytest.raises(RequestValidationError, match="out of range"):
        validate_request(r, _score(), cfg, an)


def test_bar_out_of_range():
    cfg, an = _config(), _analyzer(_config())
    r = _req(tracks=[TrackPrompt(id=0, bars=[99])])
    with pytest.raises(RequestValidationError, match="out of range"):
        validate_request(r, _score(), cfg, an)


def test_ar_and_ignore_mutually_exclusive():
    cfg, an = _config(), _analyzer(_config())
    r = _req(tracks=[TrackPrompt(id=0, bars=[], autoregressive=True, ignore=True)])
    with pytest.raises(RequestValidationError, match="autoregressive and ignore"):
        validate_request(r, _score(), cfg, an)


def test_ignore_must_not_specify_bars():
    cfg, an = _config(), _analyzer(_config())
    r = _req(tracks=[TrackPrompt(id=0, bars=[0], ignore=True)])
    with pytest.raises(RequestValidationError, match="ignored tracks"):
        validate_request(r, _score(), cfg, an)


def test_ar_bars_must_be_right_suffix():
    cfg, an = _config(), _analyzer(_config())
    # [0, 2] is not a contiguous right-suffix of a 4-bar track
    r = _req(tracks=[TrackPrompt(id=0, bars=[0, 2], autoregressive=True)])
    with pytest.raises(RequestValidationError, match="right-suffix"):
        validate_request(r, _score(), cfg, an)


def test_ar_full_suffix_accepted():
    cfg, an = _config(), _analyzer(_config())
    r = _req(tracks=[TrackPrompt(id=0, bars=[0, 1, 2, 3], autoregressive=True)])
    validate_request(r, _score(), cfg, an)


def test_unknown_attribute():
    cfg, an = _config(), _analyzer(_config())
    r = _req(tracks=[TrackPrompt(id=0, bars=[0],
                                 attributes={"not_a_real_attr": 1})])
    with pytest.raises(RequestValidationError, match="unknown attribute"):
        validate_request(r, _score(), cfg, an)


def test_attribute_value_out_of_range():
    cfg, an = _config(), _analyzer(_config())
    r = _req(tracks=[TrackPrompt(id=0, bars=[0],
                                 attributes={"note_density": 99999})])
    with pytest.raises(RequestValidationError, match="out of range"):
        validate_request(r, _score(), cfg, an)


def test_no_tracks_to_generate():
    cfg, an = _config(), _analyzer(_config())
    r = _req(tracks=[TrackPrompt(id=0, bars=[], ignore=True)])
    with pytest.raises(RequestValidationError, match="no tracks to generate"):
        validate_request(r, _score(), cfg, an)


def test_infill_with_no_infill_support():
    cfg = _config(CONFIG_NO_INFILL_JSON)
    an = _analyzer(cfg)
    r = _req(tracks=[TrackPrompt(id=0, bars=[0])])  # infill (not AR)
    with pytest.raises(RequestValidationError, match="supports_infill"):
        validate_request(r, _score(), cfg, an)


def test_empty_request_tracks():
    cfg, an = _config(), _analyzer(_config())
    r = GenerationRequest(tracks=[], config=SamplingConfig(max_attempts=1))
    with pytest.raises(RequestValidationError, match="empty"):
        validate_request(r, _score(), cfg, an)
