"""A client that drops a streamed /generate mid-stream gets its generation cancelled and its queue slot back.
Before the fix the cleanup's `await` was re-cancelled by anyio and the slot leaked: /health kept reporting
inflight=1 on an idle server."""

from __future__ import annotations

import asyncio
import json
import time

from midigpt.http.server import HttpServer, ModelEntry
from midigpt.inference import GenerationCancelled
from midigpt.inference.engine import InferenceEngine
from midigpt.inference.session import SamplingSession


def _body(score) -> bytes:
    return json.dumps({
        "score": score.to_dict(), "stream": True, "request_id": "drop-me",
        "request": {"tracks": [{"id": 0, "bars": [0, 1]}],
                    "config": {"model_dim": 4, "seed": 0, "max_attempts": 1,
                               "novelty_check": False, "silence_check": False}},
    }).encode()


def test_disconnect_mid_stream_releases_the_queue_slot(tiny_gpt2, ghost_tokenizer, ghost_analyzer,
                                                        simple_score, monkeypatch):
    stopped = []

    def endless_run(self):                     # streams until the server cancels it
        while not self.cancel_event.is_set():
            self.note_sink([])
            time.sleep(0.005)
        stopped.append(True)
        raise GenerationCancelled(self._score, 0)

    monkeypatch.setattr(SamplingSession, "run", endless_run)
    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    server = HttpServer({"m": ModelEntry(id="m", engine=engine, label="m")}, default_model="m")

    async def drive():
        frames = asyncio.Event()
        body_sent = False

        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": _body(simple_score), "more_body": False}
            await frames.wait()                # the client goes away once the stream has started
            return {"type": "http.disconnect"}

        async def send(msg):
            if msg["type"] == "http.response.body" and msg.get("body"):
                frames.set()

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": "/generate", "raw_path": b"/generate", "query_string": b"",
                 "root_path": "", "headers": [(b"content-type", b"application/json")],
                 "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80)}
        await asyncio.wait_for(server.app(scope, receive, send), 10)

    asyncio.run(drive())
    assert stopped == [True]
    assert server._inflight == 0
    assert "drop-me" not in server._cancel_events
