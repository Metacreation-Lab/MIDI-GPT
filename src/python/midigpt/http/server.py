"""Stateless HTTP server for MIDI-GPT generation.

Every request carries the full score + GenerationRequest; the server
holds no per-session state. The shared state is the loaded model(s)
(InferenceEngine, one or more), one semaphore per model (concurrency
within that model — see ModelEntry.max_parallel), and a server-wide
admitted-request counter that returns 503 instead of queueing forever
once too many requests are in flight at once (see HttpServer.max_queue).

Usage::

    # HuggingFace pretrained checkpoint (downloads and caches automatically)
    midigpt-http --pretrained yellow_medium --port 8000
    midigpt-http --pretrained prism_medium --port 8000
    midigpt-http --pretrained expressive_medium --port 8000

    # Custom HuggingFace repo
    midigpt-http --pretrained my_model --hf-repo myorg/myrepo --port 8000

    # Local checkpoint (.safetensors)
    midigpt-http --ckpt models/yellow_medium-final.safetensors --port 8000

    # Auto-shutdown after 10 minutes of inactivity
    midigpt-http --pretrained yellow_medium --idle-timeout 600

    # Multiple models in one server, selected per-request via the
    # "model" id (see _build_models_from_config for the YAML schema)
    midigpt-http --config models.yaml

Endpoints
---------
GET  /health       liveness probe (resets idle timer)
GET  /info         capabilities/attributes/resolution for one loaded model
                    (?model=<id>, defaults to the server's default model)
GET  /models       ids + labels of every model this server has loaded
POST /generate     score + request (+ optional "model" id) → result score
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue as queue_mod
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    import anyio
    import uvicorn
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "fastapi and uvicorn are required for the HTTP server. "
        "Install with: pip install midigpt[http]"
    ) from None

from midigpt._types import Score
from midigpt.inference import GenerationCancelled, GenerationRequest, InferenceEngine
from midigpt.inference.config import GRID_UNIT_FRACTIONS, PITCH_SCALE_PRESETS
from midigpt.inference.validation import RequestValidationError

log = logging.getLogger(__name__)

FAILED_REQUESTS_PATH = Path(
    os.environ.get("MIDIGPT_FAILED_REQUESTS_LOG", "failed_requests.jsonl")
)


def _log_failed_request(body: "_GenerateBody", exc: Exception) -> None:
    """Append the full request payload + error to a JSONL file for offline analysis."""
    record = {
        "timestamp": time.time(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "model": body.model,
        "score": body.score,
        "request": body.request,
    }
    try:
        with open(FAILED_REQUESTS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        log.exception("Could not write to failed-requests log at %s", FAILED_REQUESTS_PATH)


def _single_response(
    sess, result, request_id: str, status: str = "completed", model_id: str | None = None
) -> dict:
    return {
        "request_id": request_id,
        "model": model_id,
        "status": status,
        "score": result.to_dict() if result is not None else None,
        "seed": sess.resolved_seed,
        "generated_bars": {str(t): bars for t, bars in sess.generated_bars.items()},
        "dropped_context": sess.dropped_context,
        "timing": {
            "model_forward_s": sess.model_forward_time,
            "encode_s": sess.encode_time,
            "decode_s": sess.decode_time,
            "gen_count": sess.gen_count,
        },
        "tokens": _tokens_block(sess),
        "attempts": sess.attempt,
    }


def _candidates_response(
    sess, variations, num_candidates: int, request_id: str,
    status: str = "completed", model_id: str | None = None,
) -> dict:
    failures = [
        {"seed": v["seed"], "reason": v["error"]}
        for v in variations
        if v["error"] is not None
    ]
    return {
        "request_id": request_id,
        "model": model_id,
        "status": status,
        "base_seed": sess.resolved_seed,
        "generated_bars": {str(t): bars for t, bars in sess.generated_bars.items()},
        "dropped_context": sess.dropped_context,
        "candidates": [
            {
                "score": v["score"].to_dict() if v["score"] is not None else None,
                "seed": v["seed"],
                "gen_count": v["gen_count"],
                "error": v["error"],
                "truncated": v.get("truncated", False),
            }
            for v in variations
        ],
        "summary": {
            "requested": num_candidates,
            "succeeded": num_candidates - len(failures),
            "failed": len(failures),
            "failures": failures,
        },
        "timing": {
            "model_forward_s": sess.model_forward_time,
            "encode_s": sess.encode_time,
            "decode_s": sess.decode_time,
            "gen_count": sess.gen_count,
        },
        "tokens": _tokens_block(sess),
    }


def _tokens_block(sess) -> dict:
    """Token-budget summary for a completed generate() call: how much of
    the model's context window the prompt used, how many tokens got
    generated, and throughput. context_len/max_context_len are the same
    for every candidate in a num_candidates>1 request (shared prompt) —
    gen_count here is the SamplingSession-wide total (sum across candidates
    for the batched path); see each candidate's own "gen_count" for its
    individual count.

    "truncated": True means generation stopped because it hit the context
    budget (max_gen_tokens), NOT because the grammar/model reached a
    natural end — i.e. this may be cut off mid-bar/mid-track rather than
    actually finished. For num_candidates>1 this is True if ANY candidate
    was truncated — check each candidate's own "truncated" for which."""
    ctx = sess.context_len
    max_ctx = sess.max_context_len
    fwd_s = sess.model_forward_time
    return {
        "context_tokens": ctx,
        "generated_tokens": sess.gen_count,
        # the fullest the model's window got in any step (prompt + that step's tokens)
        "peak_context_tokens": sess.peak_window_tokens(),
        "max_context_tokens": max_ctx,
        "context_utilization": round(ctx / max_ctx, 4) if max_ctx else None,
        "tokens_per_second": round(sess.gen_count / fwd_s, 2) if fwd_s > 0 else None,
        "truncated": sess.truncated,
    }


class _GenerateBody(BaseModel):
    score: dict
    request: dict
    # Caller-supplied idempotency/correlation id. If omitted, the server
    # generates one (uuid4 hex) and returns it in the response body and the
    # X-Request-Id header — needed to target this specific generation with
    # POST /generate/{request_id}/cancel, and useful for log correlation
    # regardless (same shape as the request-id convention on LLM APIs).
    request_id: str | None = None
    # True: respond with a text/event-stream of newly-completed notes as
    # they're generated, ending with a "done"/"cancelled"/"error" event
    # carrying the full response. False (default): existing single JSON
    # response, unchanged.
    stream: bool = False
    # Which loaded model to run this request against — an id from
    # GET /models. Omit to use the server's default model; single-model
    # deployments never need to set this at all. Different models can have
    # different resolutions, capabilities (e.g. infill vs. autoregressive-only)
    # and control support — always cross-check against that model's own
    # GET /info rather than assuming they match another model's.
    model: str | None = None


@dataclass
class ModelEntry:
    """One loaded, warmed-up model this server can serve requests against."""

    id: str
    engine: InferenceEngine
    label: str
    # How many generate() calls against THIS model may run concurrently.
    # Each is still a full sequential decode (no cross-request batching —
    # see docs/MIDIGPT_HTTP_API.md "Concurrency" section for why) sharing
    # the box's CPU cores via the OS scheduler + torch's own intra-op
    # thread pool, not a throughput multiplier. Default 1 = old behavior
    # (fully serialized per model). Raise only after benchmarking on the
    # target machine — see the docs section for why more isn't free.
    max_parallel: int = 1


class HttpServer:
    """Stateless FastAPI server wrapping one or more InferenceEngines.

    Parameters
    ----------
    models:
        ``{model_id: ModelEntry}`` — every model this server can serve.
        Each model keeps its own tokenizer/resolution/capabilities, so
        clients must request generations against a specific ``model`` id
        (or omit it to use ``default_model``) and re-read ``GET /info``
        per model rather than assuming any two models agree on resolution,
        supported controls, or generation mode (infill vs. autoregressive).
    default_model:
        Id used when a request/`` GET /info`` call omits ``model``.
    idle_timeout:
        Seconds of inactivity after which the server shuts itself down.
        0 or None disables auto-shutdown.
    max_queue:
        Total requests allowed to be admitted (queued + actively running)
        across ALL models at once. Once this many are admitted, further
        ``POST /generate`` calls get an immediate ``503`` (with
        ``Retry-After``) instead of hanging on an unbounded wait — mirrors
        Ollama's ``OLLAMA_MAX_QUEUE`` backpressure. Default 64 is generous
        for a small deployment; it exists to fail fast under a burst rather
        than to be a normal operating limit.
    """

    _WATCHDOG_INTERVAL = 10  # seconds between idle checks

    def __init__(
        self,
        models: dict[str, ModelEntry],
        default_model: str,
        idle_timeout: float = 0,
        max_queue: int = 64,
    ) -> None:
        if not models:
            raise ValueError("HttpServer needs at least one model")
        if default_model not in models:
            raise ValueError(
                f"default_model {default_model!r} not among loaded models: {sorted(models)}"
            )
        self._models = models
        self._default_model = default_model
        # One semaphore PER model (sized by that model's max_parallel), not
        # one global semaphore — two requests against two different models
        # no longer wait on each other at all. Within one model, calls still
        # only overlap up to max_parallel; see ModelEntry.max_parallel.
        self._model_semaphores: dict[str, asyncio.Semaphore] = {
            mid: asyncio.Semaphore(max(1, entry.max_parallel)) for mid, entry in models.items()
        }
        self._max_queue = int(max_queue)
        # Requests currently admitted (queued behind a model's semaphore OR
        # actively running) — server-wide, across every model. Only ever
        # touched from the asyncio event loop (never a worker thread), so a
        # plain int is safe without a lock: no `await` happens between the
        # check and the increment in _admit().
        self._inflight = 0
        self._idle_timeout = float(idle_timeout) if idle_timeout else 0.0
        self._last_activity = time.monotonic()
        # request_id -> threading.Event, set by POST /generate/{id}/cancel and
        # polled from inside the (background-thread) decode loop. Entries are
        # removed as soon as their request finishes (success, error, or
        # cancelled) — see the `finally` blocks below — so this never grows
        # unbounded; it only ever holds in-flight requests.
        self._cancel_events: dict[str, threading.Event] = {}
        self._app = self._build_app()

    def _resolve_model(self, model_id: str | None) -> ModelEntry:
        mid = model_id or self._default_model
        entry = self._models.get(mid)
        if entry is None:
            raise HTTPException(
                status_code=400,
                detail=f"unknown model {mid!r}; available models: {sorted(self._models)}",
            )
        return entry

    def _admit(self) -> None:
        """Reserve one of the server's max_queue slots or raise 503.

        Called once per /generate call, before any work starts. Paired with
        exactly one _release() — see the handoff comment in generate().
        """
        if self._inflight >= self._max_queue:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"server busy: {self._inflight} request(s) already queued or "
                    f"running (max_queue={self._max_queue}); retry shortly"
                ),
                headers={"Retry-After": "5"},
            )
        self._inflight += 1

    def _release(self) -> None:
        self._inflight -= 1

    def _touch(self) -> None:
        self._last_activity = time.monotonic()

    async def _idle_watchdog(self) -> None:
        log.info("Idle watchdog started (timeout=%.0fs).", self._idle_timeout)
        while True:
            await asyncio.sleep(self._WATCHDOG_INTERVAL)
            elapsed = time.monotonic() - self._last_activity
            if elapsed >= self._idle_timeout:
                log.info(
                    "No activity for %.0fs (limit %.0fs) — shutting down.",
                    elapsed,
                    self._idle_timeout,
                )
                os._exit(0)

    @property
    def app(self) -> FastAPI:
        return self._app

    def _capabilities(self, engine: InferenceEngine) -> dict:
        ec = engine._tokenizer._vocab.config()
        ec_json = json.loads(ec.to_json())
        td_types = {d.get("type") for d in ec_json.get("token_domains", [])}
        ac_names = set(engine._analyzer.attribute_sizes().keys())
        genres = getattr(ec, "genre_grouping", None)
        return {
            "tension": "tension" in ac_names,
            "note_density": "note_density" in ac_names,
            "min_polyphony": "min_polyphony" in ac_names,
            "max_polyphony": "max_polyphony" in ac_names,
            "min_note_duration": "min_note_duration" in ac_names,
            "max_note_duration": "max_note_duration" in ac_names,
            "supports_token_mask": "MaskBar" in td_types,
            "supports_attention_mask": True,
            "supports_attention_approx": True,
            "supports_attention_skip": True,
            "supports_remove": True,
            "supports_pitch_mask": "NoteOnset" in td_types,
            "pitch_mask_scale_presets": sorted(PITCH_SCALE_PRESETS),
            "supports_rhythm_mask": "TimeAbsolutePos" in td_types,
            "rhythm_mask_grid_units": sorted(GRID_UNIT_FRACTIONS),
            "supports_remix": "TimeAbsolutePos" in td_types and "NoteOnset" in td_types,
            "supports_streaming": True,
            # window lengths (bars) the checkpoint was trained with: valid config.model_dim values
            "model_dim_options": sorted(ec_json.get("num_bars_map") or []),
            "max_context_tokens": engine._model.max_context(),
            # request.controls the checkpoint accepts (same rules as validation.py)
            "switchable_velocity": bool(ec_json.get("switchable_velocity", False)),
            "switchable_microtiming": bool(ec_json.get("switchable_microtiming", False)),
            "genres": [genres.decode(i) for i in range(genres.num_genres())] if genres is not None else [],
        }

    async def _stream_generate(
        self, body: "_GenerateBody", sess, req: GenerationRequest,
        request_id: str, cancel_event: threading.Event, model_id: str, on_finish,
    ) -> StreamingResponse:
        """text/event-stream response: one `data:` frame per newly-completed
        note as it's generated, ending with exactly one of "done" /
        "cancelled" / "error", each carrying the same response shape the
        non-streaming call would have returned.

        Scoped to the single-candidate path — num_candidates>1 needs each
        event tagged by candidate index to be unambiguous over one
        connection, which is a bigger design question than this covers.
        """
        num_candidates = getattr(req.config, "num_candidates", 1)
        if num_candidates > 1:
            raise HTTPException(
                status_code=400,
                detail="stream=true is not yet supported together with config.num_candidates > 1",
            )

        q: queue_mod.Queue = queue_mod.Queue()
        # Every frame carries progress, so a client can show throughput and
        # context use while the generation runs; "progress" frames come every
        # few tokens, between notes.
        def _progress() -> dict:
            return {
                "generated_tokens": sess.gen_count,
                "model_forward_s": sess.model_forward_time,
                "context_tokens": sess.context_len,
                # tokens generated in the current step: context_tokens + this is the window in use
                "step_generated_tokens": sess.gen_count - sess.step_gen_start,
                "max_context_tokens": sess.max_context_len,
                "attempt": sess.attempt,
            }
        sess.note_sink = lambda notes: q.put({"type": "notes", "notes": notes, "progress": _progress()})
        sess.progress_sink = lambda: q.put({"type": "progress", "progress": _progress()})
        loop = asyncio.get_running_loop()

        def _run() -> None:
            try:
                result = sess.run()
                q.put({
                    "type": "done",
                    "response": _single_response(sess, result, request_id, model_id=model_id),
                })
            except GenerationCancelled as exc:
                log.info("request %s cancelled (gen_count=%d)", request_id, exc.gen_count)
                q.put({
                    "type": "cancelled",
                    "response": _single_response(
                        sess, exc.partial, request_id, status="cancelled", model_id=model_id
                    ),
                })
            except Exception as exc:
                log.exception("Inference failed (streaming)")
                _log_failed_request(body, exc)
                q.put({"type": "error", "request_id": request_id, "error": str(exc)})
            finally:
                q.put(None)  # sentinel — always pushed exactly once

        async def event_source():
            async with self._model_semaphores[model_id]:
                gen_future = loop.run_in_executor(None, _run)
                try:
                    while True:
                        item = await loop.run_in_executor(None, q.get)
                        if item is None:
                            break
                        yield f"data: {json.dumps(item)}\n\n"
                finally:
                    # Reached either via normal completion above, or via
                    # Starlette closing this generator early on client
                    # disconnect. Either way: make sure the background
                    # generation actually stops and finishes BEFORE
                    # releasing the semaphore — otherwise a dropped
                    # connection could leave a generation running
                    # unbounded while blocking every other request behind
                    # the still-held semaphore.
                    #
                    # On a disconnect Starlette cancels this task, and anyio
                    # re-cancels every await in the finally: shield the wait,
                    # or it raises at once, the decode thread keeps running
                    # and on_finish() never runs (the queue slot leaks).
                    cancel_event.set()
                    try:
                        with anyio.CancelScope(shield=True):
                            await gen_future
                            import torch
                            if torch.backends.mps.is_available():
                                await loop.run_in_executor(None, torch.mps.empty_cache)
                    finally:
                        self._cancel_events.pop(request_id, None)
                        on_finish()  # release this request's _admit() queue slot

        return StreamingResponse(
            event_source(), media_type="text/event-stream", headers={"X-Request-Id": request_id}
        )

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            watchdog_task = None
            if self._idle_timeout > 0:
                watchdog_task = asyncio.create_task(self._idle_watchdog())
            yield
            if watchdog_task is not None:
                watchdog_task.cancel()

        app = FastAPI(
            title="MIDI-GPT HTTP Server",
            description="Stateless REST API for MIDI-GPT music generation.",
            version="0.2.3",
            lifespan=lifespan,
        )

        @app.get("/health", tags=["meta"])
        def health():
            self._touch()
            return {
                "status": "ok",
                "inflight": self._inflight,
                "max_queue": self._max_queue,
            }

        @app.get("/info", tags=["meta"])
        def info(model: str | None = None):
            entry = self._resolve_model(model)
            engine = entry.engine
            return {
                "model": entry.id,
                "default_model": self._default_model,
                "checkpoint": entry.label,
                "capabilities": self._capabilities(engine),
                "attributes": engine._analyzer.attribute_sizes(),
                # which tracks each attribute applies to: "melodic" | "drum" | "both"
                "attribute_track_types": engine._analyzer.attribute_track_types(),
                "resolution": engine._tokenizer._vocab.config().resolution,
            }

        @app.get("/models", tags=["meta"])
        def models():
            return {
                "default_model": self._default_model,
                "models": [
                    {"id": mid, "checkpoint": entry.label}
                    for mid, entry in self._models.items()
                ],
            }

        @app.post("/generate", tags=["generation"])
        async def generate(body: _GenerateBody, response: Response):
            self._touch()
            entry = self._resolve_model(body.model)
            self._admit()
            # For the streaming path, ownership of the release() call passes
            # to _stream_generate's event_source() finally block (the queue
            # slot must stay held until the stream actually finishes, not
            # until this function returns the StreamingResponse object) —
            # `handoff` tells the finally below to skip releasing in that case.
            handoff = False
            try:
                engine = entry.engine
                try:
                    score = Score.from_dict(body.score)
                    req = GenerationRequest.from_dict(body.request)
                except (KeyError, TypeError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid input: {exc}") from exc

                try:
                    sess = engine.session(score, req)
                except RequestValidationError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc

                if getattr(sess._request.config, "plan_only", False):   # a dry run: the steps, nothing generated
                    return {"request_id": body.request_id, "model": entry.id, "status": "planned",
                            "model_dim": sess._request.config.model_dim, "plan": sess.plan_summary(),
                            "dropped_context": sess.dropped_context}

                sess.enable_profiling = True
                num_candidates = getattr(req.config, "num_candidates", 1)

                # Caller-supplied or server-generated correlation id — needed to
                # target this specific in-flight generation with
                # POST /generate/{request_id}/cancel, echoed back either way so
                # a caller who didn't supply one can still find out what it was.
                request_id = body.request_id or uuid.uuid4().hex
                response.headers["X-Request-Id"] = request_id
                cancel_event = threading.Event()
                self._cancel_events[request_id] = cancel_event
                sess.cancel_event = cancel_event

                if body.stream:
                    handoff = True
                    return await self._stream_generate(
                        body, sess, req, request_id, cancel_event, entry.id, self._release
                    )

                async with self._model_semaphores[entry.id]:
                    loop = asyncio.get_running_loop()
                    try:
                        if num_candidates > 1:
                            variations: list = await loop.run_in_executor(None, sess.run_variations)
                        else:
                            result: Score = await loop.run_in_executor(None, sess.run)
                    except GenerationCancelled as exc:
                        log.info("request %s cancelled (gen_count=%d)", request_id, exc.gen_count)
                        if num_candidates > 1:
                            return _candidates_response(
                                sess, exc.partial, num_candidates, request_id,
                                status="cancelled", model_id=entry.id,
                            )
                        return _single_response(
                            sess, exc.partial, request_id, status="cancelled", model_id=entry.id
                        )
                    except Exception as exc:
                        log.exception("Inference failed")
                        _log_failed_request(body, exc)
                        raise HTTPException(status_code=500, detail=str(exc)) from exc
                    finally:
                        self._cancel_events.pop(request_id, None)
                        # Release cached-but-unused MPS buffers between requests.
                        # The KV cache grows via torch.cat every token (new tensor
                        # each time, old one freed) — over many requests this can
                        # fragment MPS's caching allocator and cause the *next*
                        # request's allocations to stall waiting on the driver,
                        # even though total system memory is fine. Cheap no-op on
                        # CPU/CUDA-only installs.
                        import torch
                        if torch.backends.mps.is_available():
                            await loop.run_in_executor(None, torch.mps.empty_cache)

                if num_candidates > 1:
                    return _candidates_response(sess, variations, num_candidates, request_id, model_id=entry.id)
                return _single_response(sess, result, request_id, model_id=entry.id)
            finally:
                if not handoff:
                    self._release()

        @app.post("/generate/{request_id}/cancel", tags=["generation"])
        async def cancel(request_id: str):
            self._touch()
            event = self._cancel_events.get(request_id)
            if event is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"no in-flight request with id {request_id!r} "
                    "(already finished, already cancelled, or never existed)",
                )
            event.set()
            return {"request_id": request_id, "status": "cancelling"}

        return app

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        uvicorn.run(self._app, host=host, port=port)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIDI-GPT stateless HTTP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "YAML config file describing one or more models to serve "
            "(see _build_models_from_config docstring / docs/MIDIGPT_HTTP_API.md "
            "for the schema). Mutually exclusive with --ckpt/--pretrained. "
            "host/port/idle-timeout/log-level in the file override the CLI "
            "flags of the same name if both are given."
        ),
    )
    model_grp = p.add_mutually_exclusive_group(required=False)
    model_grp.add_argument(
        "--ckpt",
        metavar="PATH",
        help="Path to a local .safetensors checkpoint file or checkpoint directory",
    )
    model_grp.add_argument(
        "--pretrained",
        metavar="NAME",
        help=(
            "Checkpoint filename prefix on HuggingFace (e.g. yellow_medium, prism_medium, "
            "expressive_medium). Downloads from --hf-repo (default: Metacreation/MIDI-GPT) "
            "and caches locally."
        ),
    )
    p.add_argument(
        "--hf-repo",
        metavar="REPO",
        default="Metacreation/MIDI-GPT",
        help="HuggingFace repo ID to download from (default: Metacreation/MIDI-GPT)",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help='Compute device: "cpu", "cuda", "mps", or "auto" (default: auto-detect)',
    )
    p.add_argument("--host", default="0.0.0.0", help="Host/IP to bind")
    p.add_argument("--port", type=int, default=8000, help="TCP port to listen on")
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=0,
        metavar="SECONDS",
        help="Shut down automatically after this many seconds of inactivity (0 = never)",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Max concurrent /generate calls against this model (single-model "
            "mode only — use 'max_parallel' per entry in --config for multi-model). "
            "Each is a full sequential decode sharing CPU/GPU with the others, not "
            "a throughput multiplier — benchmark before raising above 1."
        ),
    )
    p.add_argument(
        "--max-queue",
        type=int,
        default=64,
        metavar="N",
        help="Total requests admitted (queued+running) before returning 503",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args()
    if args.config and (args.ckpt or args.pretrained):
        p.error("--config cannot be combined with --ckpt/--pretrained")
    if not args.config and not (args.ckpt or args.pretrained):
        p.error("one of --config, --ckpt, or --pretrained is required")
    return args


def _load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"config file must be a YAML mapping at the top level: {path}")
    return cfg


def _build_models_from_config(cfg: dict) -> tuple[dict[str, ModelEntry], str]:
    """Load every model listed under the config's top-level ``models`` key.

    Schema (YAML)::

        host: 0.0.0.0            # optional, overrides --host
        port: 8000                # optional, overrides --port
        idle_timeout: 0           # optional, overrides --idle-timeout
        log_level: INFO           # optional, overrides --log-level
        max_queue: 64             # optional, overrides --max-queue (server-wide, all models)
        hf_repo: Metacreation/MIDI-GPT   # optional default for pretrained entries
        device: cpu               # optional default device for every entry
        max_parallel: 1           # optional default concurrency for every entry
        default_model: yellow_medium     # optional; defaults to the first entry's id

        models:
          - id: yellow_medium            # required, unique — this is the
            pretrained: yellow_medium    # "model" id clients pass to /generate
          - id: prism_medium
            pretrained: prism_medium
            device: cpu                 # per-entry override
            max_parallel: 2              # per-entry override — benchmark first, see ModelEntry
          - id: my_custom
            ckpt: /path/to/custom-final.safetensors
            hf_repo: myorg/myrepo       # per-entry override (pretrained entries only)

    Each entry needs exactly one of ``ckpt`` (local .safetensors path) or
    ``pretrained`` (HuggingFace checkpoint-filename prefix).
    """
    entries_cfg = cfg.get("models")
    if not entries_cfg:
        raise SystemExit("config file must have a non-empty top-level 'models' list")

    default_hf_repo = cfg.get("hf_repo", "Metacreation/MIDI-GPT")
    default_device = cfg.get("device")
    default_max_parallel = cfg.get("max_parallel", 1)

    models: dict[str, ModelEntry] = {}
    for spec in entries_cfg:
        mid = spec.get("id")
        if not mid:
            raise SystemExit(f"model entry missing required 'id': {spec}")
        if mid in models:
            raise SystemExit(f"duplicate model id in config: {mid!r}")

        ckpt = spec.get("ckpt")
        pretrained = spec.get("pretrained")
        if bool(ckpt) == bool(pretrained):
            raise SystemExit(f"model {mid!r}: specify exactly one of 'ckpt' or 'pretrained'")
        device = spec.get("device", default_device)
        max_parallel = int(spec.get("max_parallel", default_max_parallel))

        if ckpt:
            path = Path(ckpt)
            if not path.exists():
                raise SystemExit(f"model {mid!r}: checkpoint not found: {ckpt}")
            log.info("Loading model %r from checkpoint: %s (device=%s)", mid, ckpt, device or "auto")
            engine = InferenceEngine.from_checkpoint(str(path), device=device)
            label = ckpt
        else:
            hf_repo = spec.get("hf_repo", default_hf_repo)
            log.info(
                "Loading model %r pretrained=%s from %s (device=%s)",
                mid, pretrained, hf_repo, device or "auto",
            )
            engine = InferenceEngine.from_pretrained(pretrained, hf_repo=hf_repo, device=device)
            label = f"{hf_repo}/{pretrained}"

        models[mid] = ModelEntry(id=mid, engine=engine, label=label, max_parallel=max_parallel)

    default_model = cfg.get("default_model")
    if default_model is None:
        default_model = next(iter(models))
    elif default_model not in models:
        raise SystemExit(
            f"default_model {default_model!r} not among configured model ids: {sorted(models)}"
        )

    return models, default_model


def _load_single_model(args: argparse.Namespace) -> ModelEntry:
    if args.ckpt:
        path = Path(args.ckpt)
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {args.ckpt}")
        log.info("Loading checkpoint: %s (device=%s)", args.ckpt, args.device or "auto")
        engine = InferenceEngine.from_checkpoint(str(path), device=args.device)
        label = args.ckpt
        model_id = path.stem
    else:
        log.info(
            "Loading pretrained: %s from %s (device=%s)",
            args.pretrained, args.hf_repo, args.device or "auto",
        )
        engine = InferenceEngine.from_pretrained(
            args.pretrained, hf_repo=args.hf_repo, device=args.device
        )
        label = f"{args.hf_repo}/{args.pretrained}"
        model_id = args.pretrained
    return ModelEntry(id=model_id, engine=engine, label=label, max_parallel=args.max_parallel)


def main() -> None:
    args = _parse_args()

    cfg = _load_config(args.config) if args.config else None
    log_level = (cfg or {}).get("log_level", args.log_level)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if cfg is not None:
        host = cfg.get("host", args.host)
        port = cfg.get("port", args.port)
        idle_timeout = cfg.get("idle_timeout", args.idle_timeout)
        max_queue = cfg.get("max_queue", args.max_queue)
        models, default_model = _build_models_from_config(cfg)
    else:
        host, port, idle_timeout, max_queue = args.host, args.port, args.idle_timeout, args.max_queue
        entry = _load_single_model(args)
        models = {entry.id: entry}
        default_model = entry.id

    log.info(
        "Serving %d model(s): %s (default=%s, max_queue=%d)",
        len(models), sorted(models), default_model, max_queue,
    )
    server = HttpServer(models, default_model=default_model, idle_timeout=idle_timeout, max_queue=max_queue)
    log.info("Starting HTTP server on %s:%d", host, port)
    if idle_timeout:
        log.info("Auto-shutdown after %.0fs of inactivity.", idle_timeout)
    server.serve(host=host, port=port)


if __name__ == "__main__":
    main()
