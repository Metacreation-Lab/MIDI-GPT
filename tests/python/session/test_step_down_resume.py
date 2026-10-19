"""run(): a step-down keeps the finished steps and plans only what is left;
shuffle reorders infill steps only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import midigpt._core as _core
from midigpt._types import Bar, Note, Score, Track
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.session import SamplingSession
from midigpt.tokenizer import Tokenizer

CONFIG = Path(__file__).resolve().parents[3] / "models" / "yellow_config.json"


def _session(request, n_bars=16):
    tok = Tokenizer(_core.EncoderConfig.from_json(CONFIG.read_text()))
    sess = SamplingSession.__new__(SamplingSession)
    sess._engine = SimpleNamespace(_tokenizer=tok)
    sess._score = Score(tracks=[
        Track(bars=[Bar(notes=[Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=12)])
                    for _ in range(n_bars)])
        for _ in request.tracks
    ])
    sess._request = request
    sess.resolved_seed = 7
    sess.truncated = False
    sess.context_len = sess.max_context_len = 0
    return sess


def test_step_down_resumes_from_the_failing_step():
    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=list(range(16)), autoregressive=True), TrackPrompt(id=1, bars=[])],
        config=InferenceConfig(model_dim=8, bars_per_step=8),
    )
    sess = _session(req)
    ran = []

    def run_step(score, step, cfg):
        gen = sorted(b for _, b in step.bars_to_generate)
        sess.truncated = cfg.model_dim == 8 and gen[0] == 8      # the second 8-bar window runs out
        ran.append((cfg.model_dim, gen))
        return score

    sess._run_step = run_step
    getattr(sess, "_run", sess.run)()   # the planning loop itself (a session that drops ignored tracks wraps it in run())
    assert ran[0] == (8, list(range(8)))
    assert ran[1] == (8, list(range(8, 16)))                     # cut short: redone at model_dim 4
    redone = [b for d, gen in ran[2:] for b in gen]
    assert all(d == 4 for d, _ in ran[2:])
    assert sorted(redone) == list(range(8, 16))                  # bars 0-7 are kept, not regenerated


def test_shuffle_reorders_infill_steps_only():
    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=list(range(16)), autoregressive=True),
                TrackPrompt(id=1, bars=list(range(16)))],
        config=InferenceConfig(model_dim=4, bars_per_step=1, shuffle=True),
    )
    sess = _session(req)
    ran = []
    sess._run_step = lambda score, step, cfg: ran.append(
        (step.is_autoregressive, min(b for _, b in step.bars_to_generate))) or score
    getattr(sess, "_run", sess.run)()   # the planning loop itself (a session that drops ignored tracks wraps it in run())
    ar = [b for is_ar, b in ran if is_ar]
    infill = [b for is_ar, b in ran if not is_ar]
    assert ran[:len(ar)] == [(True, b) for b in ar] and ar == sorted(ar)
    assert sorted(infill) == list(range(16)) and infill != sorted(infill)
