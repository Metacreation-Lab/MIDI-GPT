"""Tests for the stateless HTTP server (midigpt.http.server)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from midigpt.http.server import HttpServer, ModelEntry
from midigpt.inference.engine import InferenceEngine


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer) -> InferenceEngine:
    return InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)


def _client(engine: InferenceEngine, label: str = "test-ckpt") -> TestClient:
    entry = ModelEntry(id=label, engine=engine, label=label)
    server = HttpServer({label: entry}, default_model=label)
    return TestClient(server.app)


def _generate_body(score, track_id: int = 0, bars: list[int] | None = None) -> dict:
    if bars is None:
        bars = list(range(len(score.tracks[0].bars)))
    return {
        "score": score.to_dict(),
        "request": {
            "tracks": [{"id": track_id, "bars": bars}],
            "config": {"model_dim": 4, "seed": 0, "max_attempts": 1,
                       "novelty_check": False, "silence_check": False},
        },
    }


# --------------------------------------------------------------------------- #
#  /health
# --------------------------------------------------------------------------- #

def test_health(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    client = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer))
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["inflight"] == 0
    assert body["max_queue"] == 64


# --------------------------------------------------------------------------- #
#  /info
# --------------------------------------------------------------------------- #

def test_info_returns_checkpoint_label(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    client = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer), label="my-model")
    r = client.get("/info")
    assert r.status_code == 200
    assert r.json()["checkpoint"] == "my-model"


def test_info_has_capabilities_and_attributes(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    client = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer))
    data = client.get("/info").json()
    assert "capabilities" in data
    caps = data["capabilities"]
    for key in (
        "supports_token_mask", "supports_attention_mask",
        "supports_attention_approx", "supports_attention_skip", "supports_remove",
    ):
        assert key in caps, f"missing capability: {key}"
    assert "attributes" in data
    assert isinstance(data["attributes"], dict)
    assert caps["switchable_velocity"] is False and caps["switchable_microtiming"] is False
    assert caps["genres"] == []


def test_info_reports_attribute_track_types(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    data = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)).get("/info").json()
    assert data["attribute_track_types"] == ghost_analyzer.attribute_track_types()
    assert set(data["attribute_track_types"]) == set(data["attributes"])
    assert set(data["attribute_track_types"].values()) <= {"melodic", "drum", "both"}


@pytest.mark.parametrize("config, switchable, has_genres", [
    ("expressive_encoder.json", True, True),
    ("prism_encoder.json", False, True),
    ("yellow_config.json", False, False),
])
def test_info_reports_piece_controls(tiny_gpt2, ghost_tokenizer, ghost_analyzer, config, switchable, has_genres):
    """The controls a client may put in request.controls, read from the encoder config."""
    import midigpt._core as _core
    from types import SimpleNamespace
    from pathlib import Path
    from midigpt.attributes import AttributeAnalyzer
    from midigpt.tokenizer import Tokenizer

    cfg = _core.EncoderConfig.from_json((Path(__file__).resolve().parents[2] / "models" / config).read_text())
    analyzer = AttributeAnalyzer.from_config(cfg)
    engine = SimpleNamespace(_tokenizer=Tokenizer(cfg, analyzer), _analyzer=analyzer, _model=tiny_gpt2)
    server = HttpServer({"m": ModelEntry(id="m", engine=_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer), label="m")},
                        default_model="m")
    caps = server._capabilities(engine)
    assert caps["switchable_velocity"] is switchable
    assert caps["switchable_microtiming"] is switchable
    assert bool(caps["genres"]) is has_genres


# --------------------------------------------------------------------------- #
#  POST /generate  — happy path
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_generate_returns_score_and_timing(tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score):
    import torch
    torch.manual_seed(0)
    client = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer))
    r = client.post("/generate", json=_generate_body(simple_score))
    assert r.status_code == 200
    data = r.json()
    assert "score" in data
    assert "timing" in data
    for key in ("model_forward_s", "encode_s", "decode_s", "gen_count"):
        assert key in data["timing"], f"missing timing key: {key}"


def test_generate_reports_generated_bars(tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score):
    import torch
    torch.manual_seed(0)
    client = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer))
    body = _generate_body(simple_score)
    assert client.post("/generate", json=body).json()["generated_bars"] == {"0": [0, 1, 2, 3]}
    body["request"]["config"]["infill_probability"] = 0.5
    picked = client.post("/generate", json=body).json()["generated_bars"]["0"]
    assert len(picked) == 2 and set(picked) <= {0, 1, 2, 3}


def test_generate_400_on_bad_infill_probability(tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score):
    client = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer))
    body = _generate_body(simple_score)
    body["request"]["config"]["infill_probability"] = 0
    assert client.post("/generate", json=body).status_code in (400, 422)


@pytest.mark.slow
def test_generate_score_has_correct_track_count(tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score):
    import torch
    torch.manual_seed(0)
    client = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer))
    r = client.post("/generate", json=_generate_body(simple_score))
    assert r.status_code == 200
    result_tracks = r.json()["score"]["tracks"]
    assert len(result_tracks) == len(simple_score.tracks)


# --------------------------------------------------------------------------- #
#  POST /generate  — error paths
# --------------------------------------------------------------------------- #

def test_generate_400_on_malformed_score(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    client = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer))
    r = client.post("/generate", json={"score": {"bad": "data"}, "request": {}})
    assert r.status_code == 400


def test_generate_400_on_missing_fields(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    client = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer))
    r = client.post("/generate", json={"score": {}})
    # FastAPI/pydantic validation fires before our code
    assert r.status_code in (400, 422)


@pytest.mark.slow
def test_generate_semaphore_serialises_requests(tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score):
    """Two sequential requests should both succeed (semaphore releases correctly)."""
    import torch
    torch.manual_seed(0)
    client = _client(_engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer))
    body = _generate_body(simple_score)
    r1 = client.post("/generate", json=body)
    r2 = client.post("/generate", json=body)
    assert r1.status_code == 200
    assert r2.status_code == 200
