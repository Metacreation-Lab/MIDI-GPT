"""Stateless HTTP server for MIDI-GPT generation.

Every request carries the full score + GenerationRequest; the server
holds no per-session state. The only shared state is the loaded model
(InferenceEngine) and a semaphore that serialises GPU work.

Usage::

    # Local checkpoint
    midigpt-http --ckpt models/yellow.pt --port 8000

    # HuggingFace pretrained name
    midigpt-http --pretrained yellow --port 8000

    # HuggingFace repo + filename
    midigpt-http --pretrained Metacreation/MIDI-GPT --hf-filename yellow.pt

Endpoints
---------
GET  /health       liveness probe
GET  /info         model capabilities and attribute sizes
POST /generate     score + request → result score
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "fastapi and uvicorn are required for the HTTP server. "
        "Install with: pip install midigpt[http]"
    ) from None

from midigpt._types import Score
from midigpt.inference import GenerationRequest, InferenceEngine
from midigpt.inference.validation import RequestValidationError

log = logging.getLogger(__name__)


class _GenerateBody(BaseModel):
    score: dict
    request: dict


class HttpServer:
    """Stateless FastAPI server wrapping an InferenceEngine.

    Parameters
    ----------
    engine:
        Loaded and warmed-up InferenceEngine.
    checkpoint_label:
        Human-readable label reported by ``GET /info`` (path or HF name).
    """

    def __init__(self, engine: InferenceEngine, checkpoint_label: str = "") -> None:
        self._engine = engine
        self._ckpt_label = checkpoint_label
        self._semaphore = asyncio.Semaphore(1)
        self._app = self._build_app()

    @property
    def app(self) -> FastAPI:
        return self._app

    def _capabilities(self) -> dict:
        ec = self._engine._tokenizer._vocab.config()
        td_types = {d.get("type") for d in json.loads(ec.to_json()).get("token_domains", [])}
        ac_names = set(self._engine._analyzer.attribute_sizes().keys())
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
        }

    def _build_app(self) -> FastAPI:
        app = FastAPI(
            title="MIDI-GPT HTTP Server",
            description="Stateless REST API for MIDI-GPT music generation.",
            version="0.2.3",
        )

        @app.get("/health", tags=["meta"])
        def health():
            return {"status": "ok"}

        @app.get("/info", tags=["meta"])
        def info():
            return {
                "checkpoint": self._ckpt_label,
                "capabilities": self._capabilities(),
                "attributes": self._engine._analyzer.attribute_sizes(),
            }

        @app.post("/generate", tags=["generation"])
        async def generate(body: _GenerateBody):
            try:
                score = Score.from_dict(body.score)
                req = GenerationRequest.from_dict(body.request)
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid input: {exc}") from exc

            try:
                sess = self._engine.session(score, req)
            except RequestValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            sess.enable_profiling = True

            async with self._semaphore:
                loop = asyncio.get_running_loop()
                try:
                    result: Score = await loop.run_in_executor(None, sess.run)
                except Exception as exc:
                    log.exception("Inference failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc

            return {
                "score": result.to_dict(),
                "timing": {
                    "model_forward_s": sess.model_forward_time,
                    "encode_s": sess.encode_time,
                    "decode_s": sess.decode_time,
                    "gen_count": sess.gen_count,
                },
            }

        return app

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        uvicorn.run(self._app, host=host, port=port)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIDI-GPT stateless HTTP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_grp = p.add_mutually_exclusive_group(required=True)
    model_grp.add_argument(
        "--ckpt",
        metavar="PATH",
        help="Path to a local .pt bundle or checkpoint directory",
    )
    model_grp.add_argument(
        "--pretrained",
        metavar="NAME_OR_REPO",
        help='Pretrained model name ("yellow", "ghost", "expressive") or HuggingFace repo ID',
    )
    p.add_argument(
        "--hf-filename",
        metavar="FILE",
        default=None,
        help="Filename within the HuggingFace repo (required when --pretrained is a repo ID)",
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.ckpt:
        path = Path(args.ckpt)
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {args.ckpt}")
        log.info("Loading checkpoint: %s (device=%s)", args.ckpt, args.device or "auto")
        engine = InferenceEngine.from_checkpoint(str(path), device=args.device)
        label = args.ckpt
    else:
        log.info("Loading pretrained: %s (device=%s)", args.pretrained, args.device or "auto")
        engine = InferenceEngine.from_pretrained(
            args.pretrained, filename=args.hf_filename, device=args.device
        )
        label = args.pretrained + (f"/{args.hf_filename}" if args.hf_filename else "")

    server = HttpServer(engine, checkpoint_label=label)
    log.info("Starting HTTP server on %s:%d", args.host, args.port)
    server.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
