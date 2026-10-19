"""A non-streamed /generate whose client goes away is cancelled and gives its queue slot back, and
max_generation_seconds cancels a generation that runs too long, streamed or not."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from midigpt.http.server import HttpServer, ModelEntry
from midigpt.inference import GenerationCancelled
from midigpt.inference.engine import InferenceEngine
from midigpt.inference.session import SamplingSession


def _payload(score, stream: bool, request_id: str) -> dict:
    return {
        "score": score.to_dict(), "stream": stream, "request_id": request_id,
        "request": {"tracks": [{"id": 0, "bars": [0, 1]}],
                    "config": {"model_dim": 4, "seed": 0, "max_attempts": 1,
                               "novelty_check": False, "silence_check": False}},
    }


@pytest.fixture
def stopped(monkeypatch):
    """SamplingSession.run generates until the server cancels it; appends to the list when it stops."""
    calls = []

    def endless_run(self):
        while not self.cancel_event.is_set():
            if self.note_sink is not None:
                self.note_sink([])
            time.sleep(0.005)
        calls.append(True)
        raise GenerationCancelled(self._score, 0)

    monkeypatch.setattr(SamplingSession, "run", endless_run)
    return calls


def _server(tiny_gpt2, ghost_tokenizer, ghost_analyzer, **kw) -> HttpServer:
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    return HttpServer({"m": ModelEntry(id="m", engine=engine, label="m")}, default_model="m", **kw)


def test_disconnect_cancels_a_non_streamed_generation(tiny_gpt2, ghost_tokenizer, ghost_analyzer,
                                                       simple_score, stopped, monkeypatch):
    server = _server(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    monkeypatch.setattr(server, "_DISCONNECT_POLL_INTERVAL", 0.01)

    async def drive():
        body_sent = False
        gone_at = time.monotonic() + 0.2           # the client goes away mid-generation

        async def receive():
            # like uvicorn: a disconnect that already happened is returned at once,
            # otherwise wait (is_disconnected() cancels the wait straight away)
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                body = json.dumps(_payload(simple_score, False, "drop-me")).encode()
                return {"type": "http.request", "body": body, "more_body": False}
            while time.monotonic() < gone_at:
                await asyncio.sleep(0.01)
            return {"type": "http.disconnect"}

        async def send(msg):
            pass

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": "/generate", "raw_path": b"/generate", "query_string": b"",
                 "root_path": "", "headers": [(b"content-type", b"application/json")],
                 "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80)}
        await asyncio.wait_for(server.app(scope, receive, send), 10)

    asyncio.run(drive())
    assert stopped == [True]
    assert server._inflight == 0
    assert "drop-me" not in server._cancel_events


@pytest.mark.parametrize("stream", [False, True])
def test_max_generation_seconds_cancels(tiny_gpt2, ghost_tokenizer, ghost_analyzer, simple_score,
                                        stopped, stream):
    server = _server(tiny_gpt2, ghost_tokenizer, ghost_analyzer, max_generation_seconds=0.2)
    client = TestClient(server.app)
    r = client.post("/generate", json=_payload(simple_score, stream, "slow"))
    assert r.status_code == 200
    if stream:
        frames = [json.loads(line[len("data: "):]) for line in r.text.splitlines() if line.startswith("data: ")]
        assert frames[-1]["type"] == "cancelled"
    else:
        assert r.json()["status"] == "cancelled"
    assert stopped == [True]
    assert server._inflight == 0
