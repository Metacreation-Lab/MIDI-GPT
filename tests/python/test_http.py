"""Tests for the stateless HTTP server (midigpt.http.server)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from midigpt.http.server import HttpServer
from midigpt.inference.engine import InferenceEngine


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _engine(tiny_gpt2, ghost_tokenizer, ghost_analyzer) -> InferenceEngine:
    return InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)


def _client(engine: InferenceEngine, label: str = "test-ckpt") -> TestClient:
    server = HttpServer(engine, checkpoint_label=label)
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
    assert r.json() == {"status": "ok"}


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
